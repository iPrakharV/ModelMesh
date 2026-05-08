from gateway.app.cache import PredictionCache


def test_memory_cache_round_trip() -> None:
    cache = PredictionCache(redis_url=None)
    key = cache.key_for(text="hello")
    value = {"prediction": {"label": "healthy"}, "worker_url": "http://worker"}

    cache.set(key, value)

    assert cache.get(key) == value


def test_cache_key_ignores_request_metadata() -> None:
    cache = PredictionCache(redis_url=None)

    first = cache.key_for(text="hello")
    second = cache.key_for(text="hello")

    assert first == second


def test_disabled_cache_is_a_noop() -> None:
    cache = PredictionCache(redis_url=None, enabled=False)
    key = cache.key_for(text="hello")

    cache.set(key, {"prediction": {"label": "healthy"}})

    assert cache.get(key) is None
