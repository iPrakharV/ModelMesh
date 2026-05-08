from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException

from .cache import PredictionCache
from .metrics import MetricsStore
from .router import WorkerPool
from .schemas import GatewayPrediction, PredictRequest


def parse_worker_urls(raw: str | None) -> list[str]:
    if not raw:
        return ["http://localhost:8011"]
    return [item.strip() for item in raw.split(",") if item.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.client = httpx.AsyncClient()
    yield
    await app.state.client.aclose()


app = FastAPI(title="ModelMesh Gateway", version="0.1.0", lifespan=lifespan)
router = WorkerPool(parse_worker_urls(os.getenv("WORKER_URLS")))
cache = PredictionCache(os.getenv("REDIS_URL"))
metrics = MetricsStore()
request_timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "1.0"))


@app.get("/health")
async def health() -> dict[str, str | int]:
    return {"status": "ok", "workers": len(router.workers)}


@app.get("/workers")
async def workers() -> list[dict[str, str | int | float | None]]:
    return router.snapshot()


@app.get("/metrics")
async def metrics_snapshot() -> dict[str, float | int]:
    return metrics.snapshot()


@app.post("/predict", response_model=GatewayPrediction)
async def predict(payload: PredictRequest) -> GatewayPrediction:
    started = time.perf_counter()
    cache_key = cache.key_for(payload.model_dump())
    cached = cache.get(cache_key)

    if cached is not None:
        latency_ms = (time.perf_counter() - started) * 1000
        metrics.record_request(latency_ms, cached=True)
        return GatewayPrediction(
            prediction=cached["prediction"],
            cached=True,
            worker_url=cached["worker_url"],
            attempts=0,
            latency_ms=round(latency_ms, 2),
        )

    errors: list[str] = []
    attempts = 0
    client: httpx.AsyncClient = app.state.client

    for worker in router.candidates(attempts=2):
        attempts += 1
        router.mark_start(worker)
        worker_started = time.perf_counter()
        try:
            response = await client.post(
                f"{worker.url}/predict",
                json=payload.model_dump(),
                timeout=request_timeout,
            )
            response.raise_for_status()
            prediction = response.json()
            worker_latency_ms = (time.perf_counter() - worker_started) * 1000
            router.mark_success(worker, worker_latency_ms)
            latency_ms = (time.perf_counter() - started) * 1000
            result = {
                "prediction": prediction,
                "worker_url": worker.url,
                "attempts": attempts,
                "latency_ms": round(latency_ms, 2),
            }
            cache.set(cache_key, result)
            metrics.record_request(latency_ms, cached=False)
            return GatewayPrediction(cached=False, **result)
        except httpx.HTTPError as exc:
            error = exc.__class__.__name__
            router.mark_failure(worker, error)
            errors.append(f"{worker.url}: {error}")

    latency_ms = (time.perf_counter() - started) * 1000
    metrics.record_request(latency_ms, cached=False, worker_error=True)
    raise HTTPException(status_code=503, detail={"message": "all workers failed", "errors": errors})
