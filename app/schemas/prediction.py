from pydantic import BaseModel, Field


class Classification(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    ocr_text: str | None = None
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    bounding_box: tuple[float, float, float, float]
    crop_path: str | None = None
    classification: Classification | None = None


class PredictionResponse(BaseModel):
    image_id: str
    annotated_image_url: str | None = None
    detections: list[Detection]
