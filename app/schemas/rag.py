from pydantic import BaseModel, Field


class RagKnowledgeResponse(BaseModel):
    active_ingredient: str | None = None
    strength: str | None = None
    indication: str | None = None
    warning: str | None = None
    source_document_id: str | None = None
    source_url: str | None = None
    source_note: str | None = None


class RagCandidateResponse(BaseModel):
    medicine_id: str
    medicine_name: str
    similarity: float = Field(ge=-1, le=1)
    average_similarity: float = Field(ge=-1, le=1)
    matched_image_count: int = Field(ge=1)
    source_document_ids: list[str]
    knowledge: RagKnowledgeResponse | None = None


class PillRagResponse(BaseModel):
    status: str
    threshold: float = Field(ge=0, le=1)
    answer: str
    candidates: list[RagCandidateResponse]


class TextRagRequest(BaseModel):
    question: str = Field(min_length=1)


class TextRagResponse(BaseModel):
    answer: str
    medicine_name: str | None = None
    source: str | None = None


class FullRagResponse(TextRagResponse):
    image_status: str
    image_similarity: float | None = None
