from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    bounding_box: tuple[float, float, float, float]
    crop_path: str | None = None
    classification: "Classification | None" = None


class Classification(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    image_id: str
    annotated_image_url: str | None = None
    detections: list[Detection]


Detection.model_rebuild()
