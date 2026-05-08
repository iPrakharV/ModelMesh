from __future__ import annotations

from dataclasses import dataclass, field
from time import time


@dataclass
class Worker:
    url: str
    in_flight: int = 0
    requests: int = 0
    failures: int = 0
    latency_ema_ms: float = 0.0
    last_error: str | None = None
    last_seen: float = field(default_factory=time)


class WorkerPool:
    def __init__(self, worker_urls: list[str]) -> None:
        if not worker_urls:
            raise ValueError("at least one worker URL is required")

        self.workers = [Worker(url=url.rstrip("/")) for url in worker_urls]

    def candidates(self, attempts: int) -> list[Worker]:
        count = min(max(attempts, 1), len(self.workers))
        ranked = sorted(self.workers, key=self._score)
        return ranked[:count]

    def mark_start(self, worker: Worker) -> None:
        worker.in_flight += 1

    def mark_success(self, worker: Worker, latency_ms: float) -> None:
        worker.in_flight = max(worker.in_flight - 1, 0)
        worker.requests += 1
        worker.last_error = None
        worker.last_seen = time()
        if worker.latency_ema_ms == 0:
            worker.latency_ema_ms = latency_ms
        else:
            worker.latency_ema_ms = worker.latency_ema_ms * 0.8 + latency_ms * 0.2

    def mark_failure(self, worker: Worker, error: str) -> None:
        worker.in_flight = max(worker.in_flight - 1, 0)
        worker.requests += 1
        worker.failures += 1
        worker.last_error = error
        worker.last_seen = time()

    def snapshot(self) -> list[dict[str, str | int | float | None]]:
        return [
            {
                "url": worker.url,
                "in_flight": worker.in_flight,
                "requests": worker.requests,
                "failures": worker.failures,
                "latency_ema_ms": round(worker.latency_ema_ms, 2),
                "last_error": worker.last_error,
            }
            for worker in self.workers
        ]

    def _score(self, worker: Worker) -> tuple[int, int, float, str]:
        return (worker.failures, worker.in_flight, worker.latency_ema_ms, worker.url)
