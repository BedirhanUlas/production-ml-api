import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)
from src.config import settings
from src.data.preprocessing import preprocess
from src.models.classifier import SentimentClassifier

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/sentiment_model.pkl")

PREDICTION_COUNTER = Counter(
    "predictions_total", "Total predictions made", ["label"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds"
)
REQUEST_COUNTER = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

_model = SentimentClassifier()
_stats: dict = defaultdict(int)
_confidence_sum: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        _model.load(MODEL_PATH)
        logger.info("Model loaded from disk.")
    else:
        logger.warning("No pre-trained model found. Train and save a model first.")
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Production-grade sentiment analysis API with MLflow tracking and Prometheus metrics.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Process-Time"] = str(round(duration, 4))
    REQUEST_COUNTER.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    return response


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=_model._is_fitted,
        version=settings.VERSION,
    )


@app.get("/metrics", tags=["ops"])
async def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(body: PredictRequest):
    if not _model._is_fitted:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train first.")

    with PREDICTION_LATENCY.time():
        cleaned = preprocess(body.text)
        result = _model.predict(cleaned)

    PREDICTION_COUNTER.labels(label=result["label"]).inc()
    _stats[result["label"]] += 1
    global _confidence_sum
    _confidence_sum += result["confidence"]

    return PredictResponse(
        label=result["label"],
        confidence=result["confidence"],
        scores=result["scores"],
        text_length=len(body.text),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
async def predict_batch(body: BatchPredictRequest):
    if not _model._is_fitted:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train first.")

    results = []
    for text in body.texts:
        cleaned = preprocess(text)
        result = _model.predict(cleaned)
        PREDICTION_COUNTER.labels(label=result["label"]).inc()
        results.append(
            PredictResponse(
                label=result["label"],
                confidence=result["confidence"],
                scores=result["scores"],
                text_length=len(text),
            )
        )

    return BatchPredictResponse(results=results, count=len(results))


@app.get("/stats", response_model=MetricsResponse, tags=["ops"])
async def stats():
    total = sum(_stats.values())
    avg_conf = round(_confidence_sum / total, 4) if total > 0 else None
    return MetricsResponse(
        total_predictions=total,
        avg_confidence=avg_conf,
        label_distribution=dict(_stats),
    )
