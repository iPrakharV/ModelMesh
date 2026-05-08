from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    index = round((len(ordered) - 1) * percent)
    return ordered[index]


@dataclass
class MetricsStore:
    window_size: int = 500
    started_at: float = field(default_factory=time.time)
    requests: int = 0
    cache_hits: int = 0
    worker_errors: int = 0
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=500))

    def record_request(self, latency_ms: float, *, cached: bool, worker_error: bool = False) -> None:
        self.requests += 1
        self.latencies_ms.append(latency_ms)
        if cached:
            self.cache_hits += 1
        if worker_error:
            self.worker_errors += 1

    def snapshot(self) -> dict[str, float | int]:
        latencies = list(self.latencies_ms)
        uptime = max(time.time() - self.started_at, 0.001)
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "worker_errors": self.worker_errors,
            "cache_hit_rate": round(self.cache_hits / self.requests, 4) if self.requests else 0,
            "requests_per_second": round(self.requests / uptime, 2),
            "p50_latency_ms": round(percentile(latencies, 0.50), 2),
            "p95_latency_ms": round(percentile(latencies, 0.95), 2),
        }
