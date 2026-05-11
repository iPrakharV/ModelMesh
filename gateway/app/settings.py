from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import environ

DEFAULT_WORKER_URL = "http://localhost:8011"
ROUTER_STRATEGIES = {"load_aware", "round_robin"}
FALSE_VALUES = {"0", "false", "no", "off"}
TRUE_VALUES = {"1", "true", "yes", "on"}


def parse_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ValueError(f"expected a boolean value, got {raw!r}")


def parse_worker_hostports(env: Mapping[str, str] | None = None) -> list[str]:
    values = env or environ
    urls = []
    for key in sorted(values):
        if key.startswith("WORKER_") and key.endswith("_HOSTPORT"):
            hostport = values[key].strip()
            if hostport:
                urls.append(f"http://{hostport}")
    return urls


def parse_worker_urls(raw: str | None, env: Mapping[str, str] | None = None) -> list[str]:
    if raw:
        return [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    return parse_worker_hostports(env) or [DEFAULT_WORKER_URL]


def parse_timeout(raw: str | None, *, default: float) -> float:
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"REQUEST_TIMEOUT_SECONDS must be a number, got {raw!r}") from exc

    if value <= 0:
        raise ValueError("REQUEST_TIMEOUT_SECONDS must be greater than zero")
    return value


@dataclass(frozen=True)
class GatewaySettings:
    worker_urls: list[str]
    router_strategy: str = "load_aware"
    redis_url: str | None = None
    cache_enabled: bool = True
    request_timeout_seconds: float = 1.0
    controls_enabled: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> GatewaySettings:
        values = env or environ
        router_strategy = values.get("ROUTER_STRATEGY", "load_aware")
        if router_strategy not in ROUTER_STRATEGIES:
            raise ValueError("ROUTER_STRATEGY must be load_aware or round_robin")

        return cls(
            worker_urls=parse_worker_urls(values.get("WORKER_URLS"), values),
            router_strategy=router_strategy,
            redis_url=values.get("REDIS_URL"),
            cache_enabled=parse_bool(values.get("CACHE_ENABLED"), default=True),
            request_timeout_seconds=parse_timeout(
                values.get("REQUEST_TIMEOUT_SECONDS"),
                default=1.0,
            ),
            controls_enabled=parse_bool(values.get("GATEWAY_CONTROLS_ENABLED"), default=True),
        )
