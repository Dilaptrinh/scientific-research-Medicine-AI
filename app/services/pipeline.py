from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.schemas.prediction import PredictionResponse
from app.services.classification import MobileNetTypeClassifier
from app.services.detection import YoloDetector


class MedicinePipeline(Protocol):
    def run(self, image: np.ndarray, image_id: str) -> PredictionResponse:
        """Run YOLO detection and MobileNet type classification."""


@dataclass
class YoloMobileNetPipeline:
    detector: YoloDetector
    classifier: MobileNetTypeClassifier

    def run(self, image: np.ndarray, image_id: str) -> PredictionResponse:
        detections = self.detector.detect(image, image_id)
        for detection in detections:
            if not detection.crop_path:
                continue

            import cv2

            crop = cv2.imread(detection.crop_path)
            if crop is None:
                raise OSError(f"Could not read YOLO crop: {detection.crop_path}")
            detection.classification = self.classifier.classify(crop)

        return PredictionResponse(image_id=image_id, detections=detections)
