from fastapi.testclient import TestClient

from app.main import app


def test_chat_page_is_available() -> None:
    response = TestClient(app).get("/api/v1/chat")

    assert response.status_code == 200
    assert "Medicine AI Chat" in response.text
    assert "/api/v1/gemini/vision" in response.text
