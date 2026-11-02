import numpy as np

from app.services.text_rag import (
    TextChunk,
    TextEmbeddingModel,
    TextFaissIndex,
    TextRagService,
)


class FakeEmbedder(TextEmbeddingModel):
    def __init__(self) -> None:
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [
                [1, 0, 0] if "becacold" in text.lower() else [0, 1, 0]
                for text in texts
            ],
            dtype=np.float32,
        )


def service() -> TextRagService:
    chunks = [
        TextChunk(
            "Becacold-E",
            "Hoạt chất",
            "Paracetamol, Cafein, Phenylephrin HCl",
            "Tờ hướng dẫn",
        ),
        TextChunk(
            "Coxileb 200",
            "Chỉ định",
            "Giảm đau, kháng viêm xương khớp",
            "Tờ hướng dẫn",
        ),
    ]
    index = TextFaissIndex(3)
    index.add(np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32), chunks)
    return TextRagService(FakeEmbedder(), index)


def test_text_rag_answers_question_from_local_csv() -> None:
    answer, medicine_name, source = service().answer(
        "Becacold-E có thành phần chính là gì?"
    )

    assert medicine_name == "Becacold-E"
    assert "Paracetamol" in answer
    assert source


def test_text_rag_can_use_image_candidate_name() -> None:
    answer, medicine_name, _ = service().answer(
        "Thuốc này dùng để điều trị bệnh gì?",
        medicine_name="Coxileb 200",
    )

    assert medicine_name == "Coxileb 200"
    assert "Chỉ định" in answer
