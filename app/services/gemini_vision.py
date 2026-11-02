import base64
import json
import mimetypes
import os
from pathlib import Path
from urllib import error, request


class GeminiVisionService:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 60,
    ) -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self._timeout = timeout
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Set it in the terminal environment."
            )

    @property
    def model(self) -> str:
        return self._model

    def analyze_image(self, image: bytes, mime_type: str, question: str) -> str:
        if not image:
            raise ValueError("Image must not be empty.")
        if not question.strip():
            raise ValueError("Question must not be empty.")

        payload = {
            "contents": [{
                "parts": [
                    {"text": question},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image).decode("ascii"),
                        }
                    },
                ]
            }]
        }
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={self._api_key}"
        )
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self._timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API returned HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Could not reach Gemini API: {exc.reason}") from exc

        try:
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Gemini API returned no text: {result}") from exc

    def analyze_file(self, path: Path, question: str) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"Image was not found: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return self.analyze_image(path.read_bytes(), mime_type, question)
