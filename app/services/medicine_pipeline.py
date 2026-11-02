from dataclasses import dataclass
import os
from pathlib import Path

import cv2
import numpy as np

from app.schemas.prediction import Detection
from app.services.classification import MobileNetTypeClassifier
from app.services.detection import YoloDetector
from app.services.ocr import MedicineOcr
from app.services.pill_rag import PillRagService, PillRagResult


@dataclass(frozen=True)
class ObjectResult:
    detection: Detection
    image_rag: PillRagResult | None = None
    ocr_text: str | None = None


class MedicinePipeline:
    """YOLO -> type CNN -> OCR/Image RAG for every detected object."""

    def __init__(
        self,
        detector: YoloDetector,
        classifier: MobileNetTypeClassifier,
        ocr: MedicineOcr,
        image_rag: PillRagService,
    ) -> None:
        self.detector = detector
        self.classifier = classifier
        self.ocr = ocr
        self.image_rag = image_rag

    @classmethod
    def from_root(cls, root: Path) -> "MedicinePipeline":
        yolo_path = Path(os.getenv("YOLO_MODEL_PATH", root / "models" / "yolo" / "best.pt"))
        type_model_path = Path(os.getenv(
            "TYPE_MODEL_PATH",
            root / "models" / "mobilenet" / "type_classifier.pth",
        ))
        if not yolo_path.is_absolute():
            yolo_path = root / yolo_path
        if not type_model_path.is_absolute():
            type_model_path = root / type_model_path
        return cls(
            detector=YoloDetector(yolo_path),
            classifier=MobileNetTypeClassifier(type_model_path),
            ocr=MedicineOcr(),
            image_rag=PillRagService.from_paths(
                root / "data" / "pill_index" / "pills.index",
                root / "data" / "pill_index" / "pills.json",
                knowledge_path=root / "data" / "medicine_knowledge" / "medicines.json",
                result_top_k=3,
            ),
        )

    def run(self, image: np.ndarray, image_id: str) -> list[ObjectResult]:
        results: list[ObjectResult] = []
        for detection in self.detector.detect(image, image_id):
            crop = cv2.imread(detection.crop_path or "")
            if crop is None:
                raise OSError(f"Could not read YOLO crop: {detection.crop_path}")
            classification = self.classifier.classify(crop)
            if classification.label in {"hop", "vi"}:
                ocr_result = self.ocr.read(crop)
                classification = classification.model_copy(
                    update={
                        "ocr_text": ocr_result.text or None,
                        "ocr_confidence": ocr_result.confidence,
                    }
                )
                results.append(ObjectResult(detection.model_copy(
                    update={"classification": classification}
                ), ocr_text=ocr_result.text or None))
            else:
                results.append(ObjectResult(
                    detection.model_copy(update={"classification": classification}),
                    image_rag=self.image_rag.search(crop),
                ))
        return results
