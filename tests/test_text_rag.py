from pathlib import Path

from app.services.text_rag import TextRagService


def test_text_rag_answers_question_from_local_csv() -> None:
    service = TextRagService.from_csv(Path("docs/rag-text/thuoc_48.csv"))

    answer, medicine_name, source = service.answer(
        "Becacold-E có thành phần chính là gì?"
    )

    assert medicine_name == "Becacold-E"
    assert "Paracetamol" in answer
    assert source


def test_text_rag_can_use_image_candidate_name() -> None:
    service = TextRagService.from_csv(Path("docs/rag-text/thuoc_48.csv"))

    answer, medicine_name, _ = service.answer(
        "Thuốc này dùng để điều trị bệnh gì?",
        medicine_name="Coxileb 200",
    )

    assert medicine_name == "Coxileb 200"
    assert "Chỉ định" in answer
