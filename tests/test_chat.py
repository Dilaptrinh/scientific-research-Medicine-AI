from fastapi.testclient import TestClient

from app.main import app


def test_chat_page_is_available() -> None:
    response = TestClient(app).get("/api/v1/chat")

    assert response.status_code == 200
    assert "Thuốc AI" in response.text
    assert '"/gemini/vision"' in response.text
    assert "clearImage" in response.text
    assert "Máy chủ trả lỗi" in response.text
    assert "localStorage" in response.text
    assert "restoreConversation" in response.text


def test_user_message_is_rendered_before_image_validation() -> None:
    page = TestClient(app).get("/").text
    ask_body = page[page.index("async function ask()"):]

    assert ask_body.index('add(text, "user")') < ask_body.index("if (!file)")


def test_web_page_is_served_at_root() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Thuốc AI" in response.text


def test_gemini_rejects_non_image_upload() -> None:
    response = TestClient(app).post(
        "/api/v1/gemini/vision",
        data={"question": "thuốc gì"},
        files={"image": ("notes.txt", b"not-an-image", "text/plain")},
    )

    assert response.status_code == 415
