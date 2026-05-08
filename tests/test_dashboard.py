from fastapi.testclient import TestClient

from gateway.app.main import app


def test_dashboard_serves_html() -> None:
    with TestClient(app) as client:
        response = client.get("/dashboard")

    assert response.status_code == 200
    assert "ModelMesh" in response.text
    assert "Gateway metrics" in response.text
