from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_invalid_image() -> None:
    response = client.post(
        "/api/v1/predict",
        files={"image": ("medicines.jpg", b"not-an-image", "image/jpeg")},
    )

    assert response.status_code == 400
