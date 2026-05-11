from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import environ


def parse_non_negative_int(name: str, raw: str | None, *, default: int) -> int:
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return value


def parse_probability(name: str, raw: str | None, *, default: float) -> float:
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc

    if value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True)
class WorkerSettings:
    worker_id: str = "worker-local"
    model_artifact: str | None = None
    delay_ms: int = 0
    fail_rate: float = 0.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> WorkerSettings:
        values = env or environ
        return cls(
            worker_id=values.get("WORKER_ID", "worker-local"),
            model_artifact=values.get("MODEL_ARTIFACT"),
            delay_ms=parse_non_negative_int(
                "SIMULATE_DELAY_MS",
                values.get("SIMULATE_DELAY_MS"),
                default=0,
            ),
            fail_rate=parse_probability("FAIL_RATE", values.get("FAIL_RATE"), default=0.0),
        )
