import pytest

from worker.app.settings import WorkerSettings


def test_worker_settings_reads_runtime_env() -> None:
    settings = WorkerSettings.from_env({
        "WORKER_ID": "worker-a",
        "MODEL_ARTIFACT": "/tmp/model.json",
        "SIMULATE_DELAY_MS": "250",
        "FAIL_RATE": "0.25",
    })

    assert settings.worker_id == "worker-a"
    assert settings.model_artifact == "/tmp/model.json"
    assert settings.delay_ms == 250
    assert settings.fail_rate == 0.25


def test_worker_settings_uses_local_defaults() -> None:
    settings = WorkerSettings.from_env({})

    assert settings.worker_id == "worker-local"
    assert settings.model_artifact is None
    assert settings.delay_ms == 0
    assert settings.fail_rate == 0.0


def test_worker_settings_rejects_negative_delay() -> None:
    with pytest.raises(ValueError, match="SIMULATE_DELAY_MS"):
        WorkerSettings.from_env({"SIMULATE_DELAY_MS": "-1"})


def test_worker_settings_rejects_invalid_fail_rate() -> None:
    with pytest.raises(ValueError, match="FAIL_RATE"):
        WorkerSettings.from_env({"FAIL_RATE": "1.5"})
