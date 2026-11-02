import logging
import json
import os
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app import config
from app.web import chat_page, local_chat_page

from app.schemas.rag import (
    FullRagResponse,
    GeminiVisionResponse,
    PillRagResponse,
    RagCandidateResponse,
    RagKnowledgeResponse,
    TextRagRequest,
    TextRagResponse,
)
from app.schemas.prediction import PredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter()
MAX_GEMINI_IMAGE_BYTES = 10 * 1024 * 1024


@router.get("/chat", include_in_schema=False)
def chat() -> object:
    return chat_page()


@router.get("/local-chat", include_in_schema=False)
def local_chat() -> object:
    return local_chat_page()


def _text_rag_service():
    from app.services.text_rag import TextRagService

    return TextRagService.from_paths(
        index_path=Path("data/text_index/text.index"),
        metadata_path=Path("data/text_index/text.json"),
    )


def _graph_rag_context(medicine_name: str | None) -> str:
    from app.services.graph_rag import GraphRagService

    return GraphRagService.from_csv(
        Path("docs/graph-neo4j/support.csv")
    ).context(medicine_name)


def _local_image_results(decoded: object, root: Path) -> list[tuple[object, object]]:
    import cv2

    from app.services.detection import YoloDetector
    from app.services.pill_rag import PillRagService

    detections = YoloDetector(
        Path(os.getenv("YOLO_MODEL_PATH", root / "models" / "yolo" / "best.pt"))
    ).detect(decoded, uuid4().hex)
    if not detections:
        raise RuntimeError("YOLO không phát hiện được viên thuốc trong ảnh.")

    image_rag = PillRagService.from_paths(
        index_path=root / "data" / "pill_index" / "pills.index",
        metadata_path=root / "data" / "pill_index" / "pills.json",
        knowledge_path=root / "data" / "medicine_knowledge" / "medicines.json",
        result_top_k=3,
    )
    results = []
    for detection in detections:
        crop = cv2.imread(detection.crop_path or "")
        if crop is None:
            continue
        results.append((detection, image_rag.search(crop)))
    if not results:
        raise RuntimeError("Image RAG không đọc được vùng thuốc do YOLO tạo ra.")
    return results


def _select_local_predictions(
    results: list[tuple[object, object]],
) -> tuple[list[object], list[str]]:
    from app.services.graph_rag import GraphRagService

    graph = GraphRagService.from_csv(Path("docs/graph-neo4j/support.csv"))
    selected = [
        result.candidates[0] if result.candidates else None
        for _, result in results
    ]
    warnings: list[str] = []
    for index, (_, result) in enumerate(results):
        if not result.candidates:
            continue
        best = selected[index]
        best_score = best.similarity if best else -1
        for candidate in result.candidates[1:]:
            support_count = sum(
                graph.supports(candidate.medicine_name, other.medicine_name)
                for position, other in enumerate(selected)
                if position != index and other is not None
            )
            adjusted_score = candidate.similarity + 0.05 * support_count
            if adjusted_score > best_score:
                best = candidate
                best_score = adjusted_score
                warnings.append(
                    f"Crop {index + 1}: Graph RAG chuyển sang "
                    f"{candidate.medicine_name} trong top-3."
                )
        selected[index] = best

    names = [candidate.medicine_name for candidate in selected if candidate]
    if len(names) > 1 and not graph.relations_between(names):
        warnings.append(
            "Graph RAG không tìm thấy quan hệ hỗ trợ giữa các thuốc được chọn; "
            "giữ thứ hạng Image RAG và cần xác minh thêm."
        )
    return selected, warnings


def _prediction_payload(
    index: int,
    result: object,
    candidate: object | None,
) -> dict[str, object]:
    if candidate is None:
        top_k = []
    else:
        top_k = [
            {"medicine_name": item.medicine_name, "similarity": item.similarity}
            for item in result.candidates
        ]
    return {
        "crop": index + 1,
        "medicine_name": candidate.medicine_name if candidate else None,
        "similarity": candidate.similarity if candidate else None,
        "top_k": top_k,
    }


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


@router.post("/text-rag/query", response_model=TextRagResponse)
async def text_rag_query(request: TextRagRequest) -> TextRagResponse:
    try:
        answer, medicine_name, source = _text_rag_service().answer(request.question)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return TextRagResponse(
        answer=answer,
        medicine_name=medicine_name or None,
        source=source or None,
    )


@router.post("/full-rag/predict", response_model=FullRagResponse)
async def full_rag_predict(
    question: str = Form(...),
    image: UploadFile = File(...),
) -> FullRagResponse:
    import cv2
    import numpy as np

    content = await image.read()
    decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    try:
        from app.services.pill_rag import PillRagService

        image_result = PillRagService.from_paths(
            index_path=Path("data/pill_index/pills.index"),
            metadata_path=Path("data/pill_index/pills.json"),
            knowledge_path=Path("data/medicine_knowledge/medicines.json"),
        ).search(decoded)
        candidate = image_result.candidates[0] if image_result.candidates else None
        answer, medicine_name, source = _text_rag_service().answer(
            question, candidate.medicine_name if candidate else None
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return FullRagResponse(
        answer=answer,
        medicine_name=medicine_name or (candidate.medicine_name if candidate else None),
        source=source or None,
        image_status=image_result.status,
        image_similarity=candidate.similarity if candidate else None,
    )


@router.post("/gemini/vision", response_model=GeminiVisionResponse)
async def gemini_vision(
    question: str = Form(...),
    image: UploadFile = File(...),
    history: str = Form("[]"),
) -> GeminiVisionResponse:
    from app.services.gemini_vision import GeminiVisionService

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(content) > MAX_GEMINI_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be smaller than 10 MB.")
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Uploaded file must be an image.")
    try:
        parsed_history = json.loads(history)
        if not isinstance(parsed_history, list):
            raise ValueError("history must be a JSON array.")
        service = GeminiVisionService()
        answer = service.analyze_image(
            content,
            image.content_type or "application/octet-stream",
            question,
            parsed_history,
        )
    except (RuntimeError, ValueError, json.JSONDecodeError) as error:
        logger.warning("Gemini vision request failed: %s", error)
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001 - never leak an opaque 500
        logger.exception("Unexpected Gemini vision failure")
        raise HTTPException(
            status_code=503,
            detail=f"Gemini vision failed: {type(error).__name__}: {error}",
        ) from error
    return GeminiVisionResponse(answer=answer, model=service.model)


@router.post("/rag-chat", response_model=GeminiVisionResponse)
async def rag_chat(
    question: str = Form(...),
    image: UploadFile = File(...),
    history: str = Form("[]"),
) -> GeminiVisionResponse:
    """Run the local vision/text RAG pipeline, then let an LLM phrase its result."""
    import cv2
    import numpy as np

    content = await image.read()
    decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    try:
        parsed_history = json.loads(history)
        if not isinstance(parsed_history, list):
            raise ValueError("history must be a JSON array.")

        from app.services.gemini_vision import GeminiVisionService

        root = Path(__file__).resolve().parents[2]
        results = _local_image_results(decoded, root)
        selected, graph_warnings = _select_local_predictions(results)
        text_sections = []
        text_sources = []
        for candidate in selected:
            if candidate is None:
                continue
            text_answer, _, text_source = _text_rag_service().answer(
                question, candidate.medicine_name
            )
            text_sections.append(f"{candidate.medicine_name}:\n{text_answer}")
            if text_source:
                text_sources.append(text_source)
        medicine_name = selected[0].medicine_name if selected and selected[0] else None
        context = (
            f"YOLO detections: {len(results)} vùng thuốc.\n"
            "Image RAG top-3 và lựa chọn sau Graph RAG:\n"
            + "\n".join(
                str(_prediction_payload(index, result, selected[index]))
                for index, (_, result) in enumerate(results)
            )
            + "\nText RAG results:\n"
            + "\n".join(text_sections)
            + "\nGraph warnings:\n"
            + "\n".join(graph_warnings or ["Không có cảnh báo."])
        )
        service = GeminiVisionService()
        answer = service.generate_from_context(question, context, parsed_history)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.warning("Local RAG chat failed: %s", error)
        raise HTTPException(status_code=503, detail=str(error)) from error
    return GeminiVisionResponse(answer=answer, model=service.model)


@router.post("/local-rag-chat", response_model=FullRagResponse)
async def local_rag_chat(
    question: str = Form(...),
    image: UploadFile | None = File(None),
) -> FullRagResponse:
    """Run YOLO, Image RAG, and Text RAG without an LLM."""
    import cv2
    import numpy as np

    if image is None:
        try:
            answer, medicine_name, source = _text_rag_service().answer(question)
            graph_context = _graph_rag_context(medicine_name)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return FullRagResponse(
            answer=f"[Text RAG]\n{answer}\n\n[Graph RAG]\n{graph_context}",
            medicine_name=medicine_name or None,
            source=source or None,
            image_status="not_run",
            image_similarity=None,
        )

    content = await image.read()
    decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    try:
        root = Path(__file__).resolve().parents[2]
        results = _local_image_results(decoded, root)
        selected, graph_warnings = _select_local_predictions(results)
        text_sections = []
        text_sources = []
        for candidate in selected:
            if candidate is None:
                continue
            text_answer, _, text_source = _text_rag_service().answer(
                question, candidate.medicine_name
            )
            text_sections.append(f"{candidate.medicine_name}:\n{text_answer}")
            if text_source:
                text_sources.append(text_source)
        names = [candidate.medicine_name for candidate in selected if candidate]
        answer = (
            "[Image RAG - top-3 từng vùng]\n"
            + "\n".join(
                str(_prediction_payload(index, result, selected[index]))
                for index, (_, result) in enumerate(results)
            )
            + "\n\n[Text RAG - từng thuốc]\n"
            + "\n\n".join(text_sections)
            + "\n\n[Graph RAG]\n"
            + "\n".join(graph_warnings or ["Không có cảnh báo."])
        )
        medicine_name = names[0] if names else None
        text_medicine = medicine_name
        source = text_sources[0] if text_sources else None
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.warning("Local RAG chat failed: %s", error)
        raise HTTPException(status_code=503, detail=str(error)) from error
    candidate = selected[0] if selected and selected[0] else None
    return FullRagResponse(
        answer=answer,
        medicine_name=text_medicine or medicine_name,
        source=source or None,
        image_status="match" if candidate else "unknown",
        image_similarity=candidate.similarity if candidate else None,
        predictions=[
            _prediction_payload(index, result, selected[index])
            for index, (_, result) in enumerate(results)
        ],
        graph_warnings=graph_warnings,
    )
