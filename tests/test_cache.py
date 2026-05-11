from gateway.app.cache import PredictionCache
from gateway.app.settings import parse_worker_hostports, parse_worker_urls


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


def test_worker_hostport_env_vars_are_render_friendly() -> None:
    env = {
        "WORKER_A_HOSTPORT": "worker-a:9000",
        "WORKER_B_HOSTPORT": "worker-b:9000",
        "OTHER_VALUE": "ignored",
    }

    assert parse_worker_hostports(env) == ["http://worker-a:9000", "http://worker-b:9000"]


def test_worker_urls_still_accepts_explicit_list() -> None:
    assert parse_worker_urls("http://a:9000,http://b:9000") == ["http://a:9000", "http://b:9000"]
