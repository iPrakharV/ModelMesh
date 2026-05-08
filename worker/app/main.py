from __future__ import annotations

import os
import random
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import load_model


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    request_id: str | None = None


app = FastAPI(title="ModelMesh Worker", version="0.1.0")
model = load_model(os.getenv("MODEL_ARTIFACT"))
worker_id = os.getenv("WORKER_ID", "worker-local")
delay_ms = int(os.getenv("SIMULATE_DELAY_MS", "0"))
fail_rate = float(os.getenv("FAIL_RATE", "0"))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "worker_id": worker_id, "model_version": model.version}


@app.post("/predict")
async def predict(payload: PredictRequest) -> dict[str, str | float]:
    if delay_ms > 0:
        time.sleep(delay_ms / 1000)

    if fail_rate > 0 and random.random() < fail_rate:
        raise HTTPException(status_code=503, detail="simulated worker failure")

    result = model.predict(payload.text)
    return {**result, "worker_id": worker_id}
