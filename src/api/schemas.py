from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Text to classify")

    @validator("text")
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace only")
        return v.strip()


class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: Dict[str, float]
    text_length: int


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=32)


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    total_predictions: int
    avg_confidence: Optional[float]
    label_distribution: Dict[str, int]
