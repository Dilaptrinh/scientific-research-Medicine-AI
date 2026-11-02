from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    bounding_box: tuple[float, float, float, float]
    crop_path: str | None = None


class PredictionResponse(BaseModel):
    image_id: str
    annotated_image_url: str | None = None
    detections: list[Detection]
