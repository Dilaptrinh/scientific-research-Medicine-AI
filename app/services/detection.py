from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.schemas.prediction import Detection


@dataclass(frozen=True)
class YoloDetector:
    """YOLO adapter that converts model results into project schemas."""

    model_path: Path
    confidence_threshold: float = 0.25
    crop_directory: Path = Path("data/processed")

    def __post_init__(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YOLO model was not found: {self.model_path}. "
                "Copy best.pt there or configure YOLO_MODEL_PATH."
            )

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "YOLO support requires the 'ml' dependencies. "
                "Install them with: pip install -e \".[ml,dev]\""
            ) from error

        object.__setattr__(self, "_model", YOLO(str(self.model_path)))

    def detect(self, image: np.ndarray, image_id: str) -> list[Detection]:
        results = self._model(
            image,
            verbose=False,
            conf=self.confidence_threshold,
        )
        result = results[0]
        self.crop_directory.mkdir(parents=True, exist_ok=True)

        detections: list[Detection] = []
        for index, box in enumerate(result.boxes):
            x1, y1, x2, y2 = (
                int(value) for value in box.xyxy[0].cpu().tolist()
            )
            class_id = int(box.cls[0].cpu().item())
            confidence = float(box.conf[0].cpu().item())
            label = str(self._model.names[class_id])
            crop_path = self.crop_directory / f"{image_id}_crop_{index}.jpg"

            crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size == 0:
                continue

            import cv2

            if not cv2.imwrite(str(crop_path), crop):
                raise OSError(f"Could not save YOLO crop: {crop_path}")

            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bounding_box=(x1, y1, x2, y2),
                    crop_path=str(crop_path),
                )
            )

        return detections
