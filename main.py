from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
)
import torch
import torch.nn as nn
from contextlib import asynccontextmanager
import json
import os


# --- Model paths ---
LAYER1_MODEL_PATH = "./layer1/layer1_model/checkpoint-420"
LAYER2_MODEL_PATH = "./layer2/layer2_contextual_model/checkpoint-1467"
LAYER2_MAPPINGS_PATH = "./layer2/layer2_contextual_model"

# --- Layer 1 label mapping ---
LAYER1_ID2LABEL = {0: "safe", 1: "dangerous"}


# --- Layer 2 custom model classes (copied from training notebook) ---

class ContextAwareLayer2Config(PretrainedConfig):
    model_type = "context_aware_layer2"

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        num_labels: int = 5,
        num_stages: int = 4,
        stage_embed_dim: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_labels = num_labels
        self.num_stages = num_stages
        self.stage_embed_dim = stage_embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout


class ContextAwareLayer2Classifier(PreTrainedModel):
    config_class = ContextAwareLayer2Config
    _no_split_modules = ["bert"]
    _keep_in_fp32_modules = []
    _tied_weights_keys = []

    @property
    def all_tied_weights_keys(self):
        return {}

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return False

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        try:
            test_tensor = torch.empty(0)
            in_meta_context = test_tensor.device.type == "meta"
        except Exception:
            in_meta_context = False

        if in_meta_context:
            bert_config = AutoConfig.from_pretrained(config.model_name)
            self.bert = AutoModel.from_config(bert_config)
        else:
            self.bert = AutoModel.from_pretrained(config.model_name)

        self.bert_hidden_size = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False

        self.stage_embedding = nn.Embedding(
            num_embeddings=config.num_stages,
            embedding_dim=config.stage_embed_dim,
        )

        combined_dim = self.bert_hidden_size + config.stage_embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        interview_stage: torch.Tensor,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = bert_output.last_hidden_state[:, 0, :]
        stage_emb = self.stage_embedding(interview_stage)
        combined = torch.cat([text_embedding, stage_emb], dim=-1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


# --- Global model storage ---
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Layer 1
    print("Loading Layer 1 model...")
    models["layer1_tokenizer"] = AutoTokenizer.from_pretrained(LAYER1_MODEL_PATH)
    models["layer1_model"] = AutoModelForSequenceClassification.from_pretrained(
        LAYER1_MODEL_PATH
    )
    models["layer1_model"].to(device)
    models["layer1_model"].eval()
    print("Layer 1 model loaded.")

    # Load Layer 2
    print("Loading Layer 2 model...")
    models["layer2_tokenizer"] = AutoTokenizer.from_pretrained(LAYER2_MAPPINGS_PATH)
    models["layer2_model"] = ContextAwareLayer2Classifier.from_pretrained(
        LAYER2_MODEL_PATH
    )
    models["layer2_model"].to(device)
    models["layer2_model"].eval()

    # Load label/stage mappings
    mappings_file = os.path.join(LAYER2_MAPPINGS_PATH, "label_mappings.json")
    with open(mappings_file, "r") as f:
        mappings = json.load(f)
    models["layer2_id2label"] = {int(k): v for k, v in mappings["id2label"].items()}
    models["layer2_stage2id"] = mappings["stage2id"]
    print("Layer 2 model loaded.")

    models["device"] = device

    yield

    models.clear()
    print("Models unloaded.")


app = FastAPI(
    title="Classification RAG Guardrail",
    description="Two-layer classification service for finance domain queries",
    version="2.0.0",
    lifespan=lifespan,
)


# --- Request / Response schemas ---

VALID_STAGES = {"opening", "technical_depth", "challenge", "closing"}


class ClassifyRequest(BaseModel):
    query: str
    prev_agent_response: str = ""
    interview_stage: str = "opening"

    @field_validator("interview_stage")
    @classmethod
    def validate_stage(cls, v: str) -> str:
        if v not in VALID_STAGES:
            raise ValueError(
                f"Invalid interview_stage '{v}'. Must be one of: {sorted(VALID_STAGES)}"
            )
        return v


LAYER1_BLOCK_THRESHOLD = 0.8


class Layer1Result(BaseModel):
    blocked: bool


class Layer2Result(BaseModel):
    label: str
    confidence: float


class ClassifyResponse(BaseModel):
    layer1: Layer1Result
    layer2: Layer2Result
    query: str
    prev_agent_response: str


# --- Inference helpers ---


def classify_layer1(query: str) -> Layer1Result:
    tokenizer = models["layer1_tokenizer"]
    model = models["layer1_model"]
    device = models["device"]

    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    dangerous_prob = probs[1].item()
    blocked = dangerous_prob >= LAYER1_BLOCK_THRESHOLD

    return Layer1Result(blocked=blocked)


def classify_layer2(
    query: str, prev_agent_response: str, interview_stage: str
) -> Layer2Result:
    tokenizer = models["layer2_tokenizer"]
    model = models["layer2_model"]
    device = models["device"]
    id2label = models["layer2_id2label"]
    stage2id = models["layer2_stage2id"]

    combined_text = f"{prev_agent_response} {tokenizer.sep_token} {query}"
    encoding = tokenizer(
        combined_text,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    stage_tensor = torch.tensor(
        [stage2id[interview_stage]], dtype=torch.long
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            interview_stage=stage_tensor,
        )
        probs = torch.softmax(outputs["logits"], dim=-1)[0]

    pred_id = probs.argmax().item()
    confidence = probs[pred_id].item()

    return Layer2Result(
        label=id2label[pred_id],
        confidence=round(confidence, 4),
    )


# --- Endpoints ---


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    """
    Classify a user message through both layers.

    - Layer 1: safe (non-finance) vs dangerous (finance)
    - Layer 2: 5-class intent classification with interview context
    """
    layer1_result = classify_layer1(request.query)
    layer2_result = classify_layer2(
        request.query, request.prev_agent_response, request.interview_stage
    )

    return ClassifyResponse(
        layer1=layer1_result,
        layer2=layer2_result,
        query=request.query,
        prev_agent_response=request.prev_agent_response,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": len(models) > 0}
