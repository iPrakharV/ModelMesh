from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .cache import PredictionCache
from .dashboard import DASHBOARD_HTML
from .metrics import MetricsStore
from .router import WorkerPool
from .schemas import GatewayPrediction, PredictRequest, WorkerControlRequest


def env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


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
router = WorkerPool(
    parse_worker_urls(os.getenv("WORKER_URLS")),
    strategy=os.getenv("ROUTER_STRATEGY", "load_aware"),
)
cache = PredictionCache(os.getenv("REDIS_URL"), enabled=env_flag("CACHE_ENABLED", default=True))
metrics = MetricsStore()
request_timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "1.0"))


@app.get("/health")
async def health() -> dict[str, str | int]:
    return {"status": "ok", "workers": len(router.workers), "router_strategy": router.strategy}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    return DASHBOARD_HTML


@app.get("/workers")
async def workers() -> list[dict[str, str | int | float | None]]:
    client: httpx.AsyncClient = app.state.client
    rows = router.snapshot()
    for index, row in enumerate(rows):
        row["index"] = index
        try:
            response = await client.get(f"{row['url']}/config", timeout=0.4)
            response.raise_for_status()
            config = response.json()
            row["worker_id"] = config.get("worker_id")
            row["model_version"] = config.get("model_version")
            row["delay_ms"] = config.get("delay_ms")
            row["fail_rate"] = config.get("fail_rate")
            row["control_status"] = "ok"
        except httpx.HTTPError:
            row["control_status"] = "offline"
    return rows


@app.post("/workers/{worker_index}/control")
async def control_worker(worker_index: int, update: WorkerControlRequest) -> dict[str, object]:
    if worker_index < 0 or worker_index >= len(router.workers):
        raise HTTPException(status_code=404, detail="worker not found")

    worker = router.workers[worker_index]
    client: httpx.AsyncClient = app.state.client
    try:
        response = await client.post(
            f"{worker.url}/control",
            json=update.model_dump(exclude_none=True),
            timeout=request_timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"worker control failed: {exc.__class__.__name__}") from exc

    return {"worker_index": worker_index, "worker_url": worker.url, "config": response.json()}


@app.get("/metrics")
async def metrics_snapshot() -> dict[str, float | int]:
    return metrics.snapshot()


@app.post("/predict", response_model=GatewayPrediction)
async def predict(payload: PredictRequest) -> GatewayPrediction:
    started = time.perf_counter()
    cache_key = cache.key_for(text=payload.text)
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
