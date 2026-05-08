from fastapi.testclient import TestClient

from worker.app.main import app, runtime


def test_worker_control_updates_runtime() -> None:
    previous_delay = runtime.delay_ms
    previous_fail_rate = runtime.fail_rate

    try:
        with TestClient(app) as client:
            response = client.post("/control", json={"delay_ms": 123, "fail_rate": 0.25})

        assert response.status_code == 200
        assert response.json()["delay_ms"] == 123
        assert response.json()["fail_rate"] == 0.25
        assert runtime.delay_ms == 123
        assert runtime.fail_rate == 0.25
    finally:
        runtime.delay_ms = previous_delay
        runtime.fail_rate = previous_fail_rate


def test_worker_control_rejects_invalid_fail_rate() -> None:
    with TestClient(app) as client:
        response = client.post("/control", json={"fail_rate": 2})

    assert response.status_code == 422
