from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle


@dataclass(frozen=True)
class Worker:
    url: str


class RoundRobinRouter:
    def __init__(self, worker_urls: list[str]) -> None:
        if not worker_urls:
            raise ValueError("at least one worker URL is required")

        self.workers = [Worker(url=url.rstrip("/")) for url in worker_urls]
        self._cycle = cycle(self.workers)

    def candidates(self, attempts: int) -> list[Worker]:
        count = min(max(attempts, 1), len(self.workers))
        return [next(self._cycle) for _ in range(count)]

    def snapshot(self) -> list[dict[str, str]]:
        return [{"url": worker.url} for worker in self.workers]
