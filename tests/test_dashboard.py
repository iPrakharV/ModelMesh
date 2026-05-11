from dataclasses import replace

from fastapi.testclient import TestClient

from gateway.app import main as gateway_main
from gateway.app.main import app


def test_dashboard_serves_html() -> None:
    with TestClient(app) as client:
        response = client.get("/dashboard")

    assert response.status_code == 200
    assert "ModelMesh" in response.text
    assert "Gateway metrics" in response.text


def test_dashboard_marks_controls_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        gateway_main,
        "settings",
        replace(gateway_main.settings, controls_enabled=False),
    )

    with TestClient(app) as client:
        response = client.get("/dashboard")

    assert response.status_code == 200
    assert "const controlsEnabled = false;" in response.text
