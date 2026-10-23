from pathlib import Path
from typing import Protocol

from app.schemas.prediction import PredictionResponse


class MedicinePipeline(Protocol):
    def run(self, image_path: Path) -> PredictionResponse:
        """Run detection, classification, retrieval, and VLM reasoning."""
