from gateway.app.cache import PredictionCache


def test_memory_cache_round_trip() -> None:
    cache = PredictionCache(redis_url=None)
    key = cache.key_for({"text": "hello", "request_id": None})
    value = {"prediction": {"label": "healthy"}, "worker_url": "http://worker"}

    cache.set(key, value)

    assert cache.get(key) == value


def test_cache_key_is_stable_for_field_order() -> None:
    cache = PredictionCache(redis_url=None)

    first = cache.key_for({"text": "hello", "request_id": "1"})
    second = cache.key_for({"request_id": "1", "text": "hello"})

    assert first == second
