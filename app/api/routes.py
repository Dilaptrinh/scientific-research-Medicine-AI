from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.rag import (
    PillRagResponse,
    RagCandidateResponse,
    RagKnowledgeResponse,
)
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


@router.post("/rag/predict", response_model=PillRagResponse)
async def predict_pill_crop(image: UploadFile = File(...)) -> PillRagResponse:
    """Run RAG on one image crop produced by an upstream detector."""
    import cv2
    import numpy as np

    from app.services.pill_rag import PillRagService

    content = await image.read()
    decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        service = PillRagService.from_paths(
            index_path=Path("data/pill_index/pills.index"),
            metadata_path=Path("data/pill_index/pills.json"),
            knowledge_path=Path("data/medicine_knowledge/medicines.json"),
        )
        result = service.search(decoded)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    candidates = []
    for candidate in result.candidates:
        knowledge = candidate.knowledge
        candidates.append(
            RagCandidateResponse(
                medicine_id=candidate.medicine_id,
                medicine_name=candidate.medicine_name,
                similarity=candidate.similarity,
                average_similarity=candidate.average_similarity,
                matched_image_count=candidate.matched_image_count,
                source_document_ids=list(candidate.source_document_ids),
                knowledge=(
                    RagKnowledgeResponse(
                        active_ingredient=knowledge.active_ingredient,
                        strength=knowledge.strength,
                        indication=knowledge.indication,
                        warning=knowledge.warning,
                        source_document_id=knowledge.source_document_id,
                        source_url=knowledge.source_url,
                        source_note=knowledge.source_note,
                    )
                    if knowledge is not None
                    else None
                ),
            )
        )
    return PillRagResponse(
        status=result.status,
        threshold=result.threshold,
        answer=result.answer,
        candidates=candidates,
    )
