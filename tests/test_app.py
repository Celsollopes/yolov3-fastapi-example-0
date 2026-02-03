from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home():
    """Test GET / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Congratulations" in response.text


def test_health():
    """Test GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_missing_file():
    """Test POST /predict without file."""
    response = client.post("/predict", data={"model": "yolov3-tiny"})
    assert response.status_code in (400, 422)
