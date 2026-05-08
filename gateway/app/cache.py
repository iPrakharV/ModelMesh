from __future__ import annotations

import hashlib
import json
from typing import Any

from redis import Redis
from redis.exceptions import RedisError


class PredictionCache:
    def __init__(self, redis_url: str | None, ttl_seconds: int = 300, enabled: bool = True) -> None:
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self.memory: dict[str, dict[str, Any]] = {}
        self.redis: Redis | None = None

        if enabled and redis_url:
            try:
                client = Redis.from_url(redis_url, decode_responses=True)
                client.ping()
                self.redis = client
            except RedisError:
                self.redis = None

    def key_for(self, *, text: str) -> str:
        normalized = json.dumps({"text": text}, sort_keys=True, separators=(",", ":"))
        return "prediction:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if self.redis is not None:
            try:
                value = self.redis.get(key)
                return json.loads(value) if value else None
            except (RedisError, json.JSONDecodeError):
                return None
        return self.memory.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self.redis is not None:
            try:
                self.redis.setex(key, self.ttl_seconds, json.dumps(value))
                return
            except RedisError:
                pass
        self.memory[key] = value
