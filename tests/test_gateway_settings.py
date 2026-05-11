import pytest

from gateway.app.settings import GatewaySettings, parse_bool, parse_worker_urls


def test_gateway_settings_reads_explicit_worker_urls() -> None:
    settings = GatewaySettings.from_env({
        "WORKER_URLS": "http://a:9000/, http://b:9000",
        "ROUTER_STRATEGY": "round_robin",
        "CACHE_ENABLED": "false",
        "REQUEST_TIMEOUT_SECONDS": "2.5",
        "REDIS_URL": "redis://cache:6379/0",
    })

    assert settings.worker_urls == ["http://a:9000", "http://b:9000"]
    assert settings.router_strategy == "round_robin"
    assert settings.cache_enabled is False
    assert settings.request_timeout_seconds == 2.5
    assert settings.redis_url == "redis://cache:6379/0"


def test_gateway_settings_reads_render_hostports() -> None:
    settings = GatewaySettings.from_env({
        "WORKER_B_HOSTPORT": "worker-b:9000",
        "WORKER_A_HOSTPORT": "worker-a:9000",
    })

    assert settings.worker_urls == ["http://worker-a:9000", "http://worker-b:9000"]


def test_gateway_settings_defaults_to_local_worker() -> None:
    assert parse_worker_urls(None, {}) == ["http://localhost:8011"]


def test_gateway_settings_rejects_unknown_router_strategy() -> None:
    with pytest.raises(ValueError, match="ROUTER_STRATEGY"):
        GatewaySettings.from_env({"ROUTER_STRATEGY": "random"})


def test_gateway_settings_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="REQUEST_TIMEOUT_SECONDS"):
        GatewaySettings.from_env({"REQUEST_TIMEOUT_SECONDS": "0"})


def test_parse_bool_rejects_ambiguous_value() -> None:
    with pytest.raises(ValueError, match="boolean"):
        parse_bool("maybe", default=True)
