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

    from app.services.detection import YoloDetector

    content = await image.read()
    decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    image_id = uuid4().hex
    detections = YoloDetector(Path("models/yolo/best.pt")).detect(decoded, image_id)
    return PredictionResponse(image_id=image_id, detections=detections)
