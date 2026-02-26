# Classification RAG Guardrail

A two-layer classification microservice that acts as a guardrail for AI-powered finance interview systems. It uses fine-tuned [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) models to ensure only relevant, safe, and compliant queries reach the main interview orchestrator.

---

## Architecture

```
                        User Query
                            |
                            v
                 +--------------------+
                 |      Layer 1       |
                 |  Domain Classifier |
                 |  (safe / dangerous)|
                 +--------+-----------+
                          |
                    Is Safe to Proceed?
                    /          \
                  No            Yes
                  |              |
                  v              v
              Flagged    +--------------------+
                         |      Layer 2       |
                         |  Intent Classifier |
                         |   (4 intents, 3 stages)   
                         +--------+-----------+
                                  |
                                  v
                          Agent Routing
                            Response
```

**Layer 1** is a fast binary classifier that determines whether a query is toxic or not. **Layer 2** is a context-aware multi-class classifier that categorizes the intent of the user query, gathering context from the previous agent response and the current interview stage.

## Example of why context is important**
Candidate message: "Can you explain what you mean by that?"

Without context — the classifier has to guess:

    Is this inquiry (asking for clarification on the question)?
    Is this small_talk?
    Is it answer_submission where they're quoting something back?

It's genuinely ambiguous in isolation.
---

## Classifications

### Layer 1 -- Domain

| Label | Description |
|---|---|
| `safe` | Query is **not** and attempt of red-teaming, prompt injection, or other malicious intent |
| `dangerous` | Query **is** an attempt of red-teaming, prompt injection, or other malicious intent |

### Layer 2 -- Intent

| Label | Description |
|---|---|
| `answer_submission` | Candidate providing an answer |
| `inquiry` | Asking for clarification |
| `off_topic` | Off-topic within finance context |
| `small_talk` | General conversation / small talk |

### Interview Stages (Layer 2 context)

| Stage | ID |
|---|---|
| `opening` | 0 |
| `technical_depth` | 1 |
| `closing` | 2 |

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
git clone <repository-url>
cd Classification-RAG-gaurdrail
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and update the model checkpoint paths to match your trained models:

```bash
cp .env.example .env
```

Then edit `.env` with the correct checkpoint paths:

```
LAYER1_MODEL_PATH=./layer1/layer1_model/checkpoint-XXX
LAYER2_MODEL_PATH=./layer2/layer2_contextual_model/checkpoint-XXX
```

### Training Models

If you don't have trained checkpoints yet:

1. Prepare training data in CSV format (see `layer1/` and `layer2/` directories for schema)
2. Run `layer1/layer1_training.ipynb` to train the Layer 1 model
3. Run `layer2/layer2_training.ipynb` to train the Layer 2 model
4. Update the checkpoint paths in your `.env`

### Running the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs at [`/docs`](http://127.0.0.1:8000/docs).

---

## API Reference

### `POST /classify`

Classify a user query through both layers.

**Request:**

```json
{
  "query": "What is the WACC formula?",
  "prev_agent_response": "Let's discuss valuation methods.",
  "interview_stage": "technical_depth"
}
```

**Response:**

```json
{
  "layer1": {
    "blocked": false
  },
  "layer2": {
    "label": "answer_submission",
    "confidence": 0.8543
  },
  "query": "What is the WACC formula?",
  "prev_agent_response": "Let's discuss valuation methods."
}
```

### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### Example -- cURL

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do you calculate DCF?",
    "prev_agent_response": "",
    "interview_stage": "opening"
  }'
```

## Docker

```bash
docker build -t guardrail-service .
docker run -p 8000:8000 guardrail-service
```

## Project Structure

```
Classification-RAG-gaurdrail/
├── main.py                          # FastAPI application
├── .env                             # Environment config (git-ignored)
├── .env.example                     # Template for .env
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container configuration
│
├── layer1/
│   ├── layer1_training.ipynb        # Layer 1 training notebook
│   ├── combined_data.csv            # Layer 1 training data
│   └── layer1_model/                # Trained checkpoints (git-ignored)
│       └── checkpoint-*/
│
└── layer2/
    ├── layer2_training.ipynb        # Layer 2 training notebook
    ├── layer2_contextual_data.csv   # Layer 2 training data
    └── layer2_contextual_model/     # Trained checkpoints (git-ignored)
        ├── checkpoint-*/
        └── label_mappings.json
```

---

## Model Details

| Property | Layer 1 | Layer 2 |
|---|---|---|
| Base Model | `answerdotai/ModernBERT-base` | `answerdotai/ModernBERT-base` |
| Task | Binary classification | Multi-class classification |
| Classes | 2 (`safe`, `dangerous`) | 5 (see table above) |
| Max Sequence Length | 128 tokens | 256 tokens |
| Context Inputs | Text only | Text + prev response + interview stage |
| Architecture | Standard classification head | Custom `ContextAwareLayer2Classifier` with 32-dim stage embedding |

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `LAYER1_MODEL_PATH` | Path to Layer 1 checkpoint | `./layer1/layer1_model/checkpoint-420` |
| `LAYER2_MODEL_PATH` | Path to Layer 2 checkpoint | `./layer2/layer2_contextual_model/checkpoint-1467` |
| `LAYER2_MAPPINGS_PATH` | Path to Layer 2 tokenizer + label mappings | `./layer2/layer2_contextual_model` |
| `LAYER1_BLOCK_THRESHOLD` | Probability threshold for flagging finance queries | `0.8` |
| `HOST` | Server bind address | `0.0.0.0` |
| `PORT` | Server port | `8000` |

---
