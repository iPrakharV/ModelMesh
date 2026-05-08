from __future__ import annotations

import os
import random
import asyncio
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import load_model


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    request_id: str | None = None


class ControlUpdate(BaseModel):
    delay_ms: int | None = Field(default=None, ge=0, le=5000)
    fail_rate: float | None = Field(default=None, ge=0, le=1)


@dataclass
class WorkerRuntime:
    delay_ms: int
    fail_rate: float

    def snapshot(self) -> dict[str, int | float | str]:
        return {
            "worker_id": worker_id,
            "model_version": model.version,
            "delay_ms": self.delay_ms,
            "fail_rate": self.fail_rate,
        }


app = FastAPI(title="ModelMesh Worker", version="0.1.0")
model = load_model(os.getenv("MODEL_ARTIFACT"))
worker_id = os.getenv("WORKER_ID", "worker-local")
runtime = WorkerRuntime(
    delay_ms=int(os.getenv("SIMULATE_DELAY_MS", "0")),
    fail_rate=float(os.getenv("FAIL_RATE", "0")),
)


@app.get("/health")
async def health() -> dict[str, str | int | float]:
    return {"status": "ok", **runtime.snapshot()}


@app.get("/config")
async def config() -> dict[str, str | int | float]:
    return runtime.snapshot()


@app.post("/control")
async def control(update: ControlUpdate) -> dict[str, str | int | float]:
    if update.delay_ms is not None:
        runtime.delay_ms = update.delay_ms
    if update.fail_rate is not None:
        runtime.fail_rate = update.fail_rate
    return runtime.snapshot()


@app.post("/predict")
async def predict(payload: PredictRequest) -> dict[str, str | float]:
    if runtime.delay_ms > 0:
        await asyncio.sleep(runtime.delay_ms / 1000)

    if runtime.fail_rate > 0 and random.random() < runtime.fail_rate:
        raise HTTPException(status_code=503, detail="simulated worker failure")

    result = model.predict(payload.text)
    return {**result, "worker_id": worker_id}
