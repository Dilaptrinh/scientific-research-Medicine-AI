from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.prediction import PredictionResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
)
async def predict(image: UploadFile = File(...)) -> PredictionResponse:
    del image
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="The YOLO and MobileNet pipeline has not been implemented yet.",
    )
