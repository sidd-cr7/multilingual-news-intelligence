from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200 and response.json()["status"] == "ok"


def test_empty_text_returns_validation_error() -> None:
    response = client.post("/api/analyze/text", json={"text": ""})
    assert response.status_code == 422


def test_bad_file_type_rejected() -> None:
    response = client.post("/api/analyze/pdf", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415
