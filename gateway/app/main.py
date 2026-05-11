from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .cache import PredictionCache
from .dashboard import DASHBOARD_HTML
from .metrics import MetricsStore
from .router import Worker, WorkerPool
from .schemas import GatewayPrediction, PredictRequest, WorkerControlRequest
from .settings import GatewaySettings

MAX_WORKER_ATTEMPTS = 2
JsonObject = dict[str, Any]
WorkerRow = dict[str, str | int | float | None]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.client = httpx.AsyncClient()
    yield
    await app.state.client.aclose()


app = FastAPI(title="ModelMesh Gateway", version="0.1.0", lifespan=lifespan)
settings = GatewaySettings.from_env()
router = WorkerPool(
    settings.worker_urls,
    strategy=settings.router_strategy,
)
cache = PredictionCache(
    settings.redis_url,
    enabled=settings.cache_enabled,
)
metrics = MetricsStore()
request_timeout = settings.request_timeout_seconds


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def gateway_client() -> httpx.AsyncClient:
    return app.state.client


def worker_for_index(worker_index: int) -> Worker:
    if worker_index < 0 or worker_index >= len(router.workers):
        raise HTTPException(status_code=404, detail="worker not found")
    return router.workers[worker_index]


async def read_worker_config(worker_url: str) -> JsonObject | None:
    try:
        response = await gateway_client().get(f"{worker_url}/config", timeout=0.4)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    return response.json()


def add_worker_config(
    row: WorkerRow,
    config: JsonObject | None,
) -> None:
    if config is None:
        row["control_status"] = "offline"
        return

    row["worker_id"] = config.get("worker_id")
    row["model_version"] = config.get("model_version")
    row["delay_ms"] = config.get("delay_ms")
    row["fail_rate"] = config.get("fail_rate")
    row["control_status"] = "ok"


def cached_prediction(cached: JsonObject, latency_ms: float) -> GatewayPrediction:
    return GatewayPrediction(
        prediction=cached["prediction"],
        cached=True,
        worker_url=cached["worker_url"],
        attempts=0,
        latency_ms=round(latency_ms, 2),
    )


async def control_remote_worker(
    worker: Worker,
    update: WorkerControlRequest,
) -> JsonObject:
    try:
        response = await gateway_client().post(
            f"{worker.url}/control",
            json=update.model_dump(exclude_none=True),
            timeout=request_timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        detail = f"worker control failed: {exc.__class__.__name__}"
        raise HTTPException(status_code=502, detail=detail) from exc
    return response.json()


async def request_prediction(worker: Worker, payload: PredictRequest) -> JsonObject:
    response = await gateway_client().post(
        f"{worker.url}/predict",
        json=payload.model_dump(),
        timeout=request_timeout,
    )
    response.raise_for_status()
    return response.json()


@app.get("/health")
async def health() -> dict[str, str | int]:
    return {
        "status": "ok",
        "workers": len(router.workers),
        "router_strategy": router.strategy,
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    return DASHBOARD_HTML


@app.get("/workers")
async def workers() -> list[WorkerRow]:
    rows = router.snapshot()
    for index, row in enumerate(rows):
        row["index"] = index
        add_worker_config(row, await read_worker_config(str(row["url"])))
    return rows


@app.post("/workers/{worker_index}/control")
async def control_worker(
    worker_index: int,
    update: WorkerControlRequest,
) -> dict[str, object]:
    worker = worker_for_index(worker_index)
    config = await control_remote_worker(worker, update)
    return {"worker_index": worker_index, "worker_url": worker.url, "config": config}


@app.get("/metrics")
async def metrics_snapshot() -> dict[str, float | int]:
    return metrics.snapshot()


@app.post("/predict", response_model=GatewayPrediction)
async def predict(payload: PredictRequest) -> GatewayPrediction:
    started = time.perf_counter()
    cache_key = cache.key_for(text=payload.text)
    cached = cache.get(cache_key)

    if cached is not None:
        latency_ms = elapsed_ms(started)
        metrics.record_request(latency_ms, cached=True)
        return cached_prediction(cached, latency_ms)

    errors: list[str] = []
    attempts = 0

    for worker in router.candidates(attempts=MAX_WORKER_ATTEMPTS):
        attempts += 1
        router.mark_start(worker)
        worker_started = time.perf_counter()
        try:
            prediction = await request_prediction(worker, payload)
            worker_latency_ms = elapsed_ms(worker_started)
            router.mark_success(worker, worker_latency_ms)
            latency_ms = elapsed_ms(started)
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

    latency_ms = elapsed_ms(started)
    metrics.record_request(latency_ms, cached=False, worker_error=True)
    raise HTTPException(
        status_code=503,
        detail={"message": "all workers failed", "errors": errors},
    )
