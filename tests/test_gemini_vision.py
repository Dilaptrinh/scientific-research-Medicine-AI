import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import gemini_vision


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _payload(text: str) -> bytes:
    return json.dumps(
        {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    ).encode("utf-8")


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gemini_vision.GeminiVisionService()


def test_analyze_image_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gemini_vision.request, "urlopen", lambda *a, **k: _FakeResponse(_payload("Paracetamol"))
    )
    service = gemini_vision.GeminiVisionService(api_key="key", model="test-model")

    answer = service.analyze_image(b"image", "image/png", "thuoc gi")

    assert answer == "Paracetamol"
    assert service.model == "test-model"


def test_read_timeout_becomes_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TimeoutResponse(_FakeResponse):
        def read(self) -> bytes:
            raise TimeoutError("timed out")

    monkeypatch.setattr(
        gemini_vision.request, "urlopen", lambda *a, **k: _TimeoutResponse(b"")
    )
    service = gemini_vision.GeminiVisionService(api_key="key", timeout=1)

    with pytest.raises(RuntimeError, match="did not respond"):
        service.analyze_image(b"image", "image/png", "thuoc gi")


def test_response_without_candidates_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gemini_vision.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(b'{"promptFeedback": {"blockReason": "SAFETY"}}'),
    )
    service = gemini_vision.GeminiVisionService(api_key="key")

    with pytest.raises(RuntimeError, match="no text"):
        service.analyze_image(b"image", "image/png", "thuoc gi")


def test_endpoint_returns_answer_and_forwards_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    captured: dict[str, bytes] = {}

    def fake_urlopen(http_request, *args, **kwargs):
        captured["body"] = http_request.data
        return _FakeResponse(_payload("Paracetamol"))

    monkeypatch.setattr(gemini_vision.request, "urlopen", fake_urlopen)

    response = TestClient(app).post(
        "/api/v1/gemini/vision",
        data={
            "question": "thuốc gì",
            "history": json.dumps([{"role": "user", "parts": [{"text": "chào"}]}]),
        },
        files={"image": ("screen.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Paracetamol"
    assert response.json()["model"]
    sent = json.loads(captured["body"])
    assert [entry["role"] for entry in sent["contents"]] == ["user", "user"]


def test_endpoint_returns_503_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    response = TestClient(app).post(
        "/api/v1/gemini/vision",
        data={"question": "thuốc gì"},
        files={"image": ("screen.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]
