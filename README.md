# Production ML API

A production-ready machine learning system for real-time sentiment analysis, demonstrating end-to-end ML engineering: training, experiment tracking, containerized serving, and observability.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client                                │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Application                        │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │  /predict   │  │ /predict/    │  │  /health  /stats   │ │
│  │  (single)   │  │   batch      │  │  /metrics (prom.)  │ │
│  └──────┬──────┘  └──────┬───────┘  └────────────────────┘ │
│         └────────────────┘                                   │
│                    │                                         │
│         ┌──────────▼──────────┐                             │
│         │  Text Preprocessing  │  (clean → tokenize →       │
│         │     Pipeline         │   stopword removal)        │
│         └──────────┬──────────┘                             │
│                    │                                         │
│         ┌──────────▼──────────┐                             │
│         │  TF-IDF + Logistic   │  ← sklearn Pipeline        │
│         │  Regression Model    │                             │
│         └─────────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
          │                            │
          ▼                            ▼
┌──────────────────┐        ┌──────────────────────┐
│  MLflow Tracking │        │  Prometheus Metrics   │
│  - Experiments   │        │  - Latency histogram  │
│  - Parameters    │        │  - Request counter    │
│  - Metrics       │        │  - Label distribution │
│  - Model Registry│        └──────────────────────┘
└──────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| ML Pipeline | scikit-learn (TF-IDF + Logistic Regression) |
| Experiment Tracking | MLflow |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Observability | Prometheus metrics endpoint |
| Testing | pytest (unit + integration) |
| Data | HuggingFace Datasets (IMDB) |

## Quick Start

### Option 1: Docker (recommended)

```bash
# Clone and start all services
git clone https://github.com/BedirhanUlas/production-ml-api.git
cd production-ml-api

# Copy env config
cp .env.example .env

# Train model first (outside Docker, saves to ./models/)
pip install -r requirements.txt
make train

# Start full stack: API + MLflow + Redis
make docker-up

# API running at http://localhost:8000
# MLflow UI at http://localhost:5000
```

### Option 2: Local Dev

```bash
pip install -r requirements.txt

# Train model
make train

# Start API with hot reload
make run
```

## API Reference

### `POST /predict`

Classify the sentiment of a single text input.

**Request**
```json
{
  "text": "This product exceeded all my expectations!"
}
```

**Response**
```json
{
  "label": "positive",
  "confidence": 0.9432,
  "scores": {
    "negative": 0.0568,
    "positive": 0.9432
  },
  "text_length": 45
}
```

### `POST /predict/batch`

Classify up to 32 texts in a single request.

**Request**
```json
{
  "texts": [
    "Absolutely loved it!",
    "Worst purchase I've ever made.",
    "It was okay, nothing special."
  ]
}
```

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### `GET /metrics`

Prometheus-formatted metrics for scraping.

### `GET /stats`

In-memory prediction statistics since last restart.

## Model Performance

Trained on the IMDB dataset (50,000 reviews, binary sentiment).

| Metric | Score |
|---|---|
| Accuracy | ~92% |
| F1 (weighted) | ~92% |
| Inference latency (p50) | < 5ms |
| Inference latency (p99) | < 15ms |

## Project Structure

```
production-ml-api/
├── src/
│   ├── config.py            # Centralized settings
│   ├── data/
│   │   └── preprocessing.py # Text cleaning pipeline
│   ├── models/
│   │   └── classifier.py    # ML model + MLflow training
│   └── api/
│       ├── main.py          # FastAPI app + routes
│       └── schemas.py       # Pydantic request/response models
├── tests/
│   ├── test_preprocessing.py
│   └── test_api.py
├── scripts/
│   └── train.py             # Training entrypoint
├── .github/workflows/
│   └── ci.yml               # CI: test + lint + Docker build
├── Dockerfile               # Multi-stage production image
├── docker-compose.yml       # API + MLflow + Redis
├── Makefile                 # Convenience commands
└── requirements.txt
```

## Running Tests

```bash
make test

# Expected output:
# tests/test_preprocessing.py ................ PASSED
# tests/test_api.py ...................... PASSED
```

## MLflow Experiment Tracking

After training, open [http://localhost:5000](http://localhost:5000) to explore:
- Run parameters (max_features, C, ngram_range)
- Validation metrics (accuracy, F1, precision, recall)
- Registered model versions

## CI/CD Pipeline

Every push triggers:
1. **Test** — pytest across Python 3.10 and 3.11
2. **Lint** — ruff static analysis
3. **Docker build** — validates image builds on merge to main

## License

MIT
