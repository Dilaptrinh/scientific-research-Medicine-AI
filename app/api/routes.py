from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.prediction import PredictionResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)) -> PredictionResponse:
    import cv2
    import numpy as np

    from app.services.classification import MobileNetTypeClassifier
    from app.services.detection import YoloDetector
    from app.services.pipeline import YoloMobileNetPipeline

    content = await image.read()
    decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    pipeline = YoloMobileNetPipeline(
        detector=YoloDetector(Path("models/yolo/best.pt")),
        classifier=MobileNetTypeClassifier(
            Path("models/mobilenet/type_classifier.pth")
        ),
    )
    return pipeline.run(decoded, uuid4().hex)
