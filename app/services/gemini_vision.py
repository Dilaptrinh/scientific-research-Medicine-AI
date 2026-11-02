import base64
import http.client
import json
import logging
import mimetypes
import os
from pathlib import Path
from urllib import error, request
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_TIMEOUT_SECONDS = 120


class GeminiVisionService:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self._api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        self._model = (
            model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        ).strip()
        try:
            self._timeout = timeout or int(
                os.getenv("GEMINI_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
            )
        except ValueError as exc:
            raise RuntimeError("GEMINI_TIMEOUT must be an integer number of seconds.") from exc
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Set it in the terminal environment."
            )

    @property
    def model(self) -> str:
        return self._model

    def analyze_image(
        self,
        image: bytes,
        mime_type: str,
        question: str,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        if not image:
            raise ValueError("Image must not be empty.")
        if not question.strip():
            raise ValueError("Question must not be empty.")

        contents = list(history or [])
        contents.append({
            "role": "user",
            "parts": [
                {"text": question},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(image).decode("ascii"),
                    }
                },
            ],
        })
        payload = {
            "system_instruction": {
                "parts": [{
                    "text": (
                        "Bạn là chatbot dự đoán và tư vấn thông tin thuốc bằng tiếng Việt. "
                        "Trả lời trực tiếp, tự nhiên và ngắn gọn như đang trò chuyện. "
                        "Nếu ảnh đủ rõ, hãy nêu tên thuốc dự đoán, dấu hiệu nhận biết, "
                        "mức độ tin cậy và trả lời đúng câu hỏi. Nếu ảnh chưa đủ rõ, "
                        "nói rõ chưa thể xác định và hỏi người dùng chụp thêm mặt vỉ, "
                        "nhãn hoặc ký hiệu trên viên. Không liệt kê hàng loạt tên thuốc "
                        "đoán mò. Chỉ nhắc xác minh với dược sĩ/bác sĩ một lần, thật ngắn, "
                        "khi thông tin không đủ hoặc câu hỏi liên quan liều dùng/an toàn. "
                        "Không lặp lại cảnh báo dài dòng ở mọi lượt."
                    )
                }]
            },
            "contents": contents,
        }
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent"
        )
        body = json.dumps(payload).encode("utf-8")
        try:
            http_request = request.Request(
                endpoint,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self._api_key,
                },
                method="POST",
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Could not build Gemini request for model '{self._model}': {exc}"
            ) from exc

        try:
            with request.urlopen(http_request, timeout=self._timeout) as response:
                raw_body = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API returned HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Could not reach Gemini API: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Gemini API did not respond within {self._timeout}s. "
                "Increase GEMINI_TIMEOUT or use a smaller image."
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            raise RuntimeError(f"Gemini API connection failed: {exc}") from exc

        try:
            result = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Gemini API returned a non-JSON response.") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"Gemini API returned no text: {result}")

        candidates = result.get("candidates") or []
        parts = (
            ((candidates[0] or {}).get("content") or {}).get("parts") or []
            if candidates
            else []
        )
        texts = [
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        if not texts:
            reason = result.get("promptFeedback") or result.get("error") or result
            raise RuntimeError(f"Gemini API returned no text: {reason}")
        return "\n".join(texts).strip()

    def analyze_file(
        self,
        path: Path,
        question: str,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"Image was not found: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return self.analyze_image(path.read_bytes(), mime_type, question, history)
