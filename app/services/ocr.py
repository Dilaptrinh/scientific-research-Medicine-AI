from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float | None = None


class MedicineOcr:
    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as error:
            raise RuntimeError(
                "OCR requires rapidocr-onnxruntime. Install it with "
                "pip install rapidocr-onnxruntime."
            ) from error
        self._engine = RapidOCR(text_score=0.5)

    def read(self, image: np.ndarray) -> OcrResult:
        result = self._engine(image)
        if isinstance(result, tuple):
            result = result[0]
        if not result:
            return OcrResult("")
        texts = [str(item[1]) for item in result if len(item) >= 2]
        scores = [float(item[2]) for item in result if len(item) >= 3]
        return OcrResult(" ".join(texts).strip(), sum(scores) / len(scores) if scores else None)
