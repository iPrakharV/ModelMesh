from dataclasses import replace

from fastapi.testclient import TestClient

from gateway.app import main as gateway_main
from gateway.app.main import app


def test_gateway_controls_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        gateway_main,
        "settings",
        replace(gateway_main.settings, controls_enabled=False),
    )

    with TestClient(app) as client:
        response = client.post("/workers/0/control", json={"delay_ms": 0})

    assert response.status_code == 403
    assert response.json()["detail"] == "worker controls are disabled"
