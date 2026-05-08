from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    request_id: str | None = None


class Prediction(BaseModel):
    label: str
    score: float
    model_version: str
    worker_id: str


class GatewayPrediction(BaseModel):
    prediction: Prediction
    cached: bool
    worker_url: str
    attempts: int
    latency_ms: float
