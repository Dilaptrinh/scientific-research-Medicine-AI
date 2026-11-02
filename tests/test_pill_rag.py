import json

import numpy as np
import pytest

from app.services.pill_rag import (
    MedicineKnowledgeStore,
    group_matches,
)
from app.services.retrieval import PillMatch, PillReference


def match(medicine_id: str, similarity: float) -> PillMatch:
    return PillMatch(
        reference=PillReference(
            image_path=f"{medicine_id}/sample.jpg",
            medicine_id=medicine_id,
            medicine_name=medicine_id.replace("_", " "),
        ),
        similarity=similarity,
    )


def test_group_matches_deduplicates_medicines_and_applies_threshold() -> None:
    result = group_matches(
        [match("medicine_a", 0.80), match("medicine_a", 0.92), match("medicine_b", 0.88)],
        top_k=2,
        similarity_threshold=0.90,
    )

    assert result.status == "match"
    assert [candidate.medicine_id for candidate in result.candidates] == [
        "medicine_a",
        "medicine_b",
    ]
    assert result.candidates[0].matched_image_count == 2
    assert result.candidates[0].similarity == pytest.approx(0.92)


def test_group_matches_returns_unknown_below_threshold() -> None:
    result = group_matches([match("medicine_a", 0.74)], similarity_threshold=0.75)

    assert result.status == "unknown"
    assert "Chưa đủ cơ sở" in result.answer


def test_knowledge_store_loads_metadata(tmp_path) -> None:
    path = tmp_path / "medicines.json"
    path.write_text(
        json.dumps(
            [
                {
                    "medicine_id": "medicine_a",
                    "medicine_name": "Medicine A",
                    "active_ingredient": "Ingredient A",
                }
            ]
        ),
        encoding="utf-8",
    )

    store = MedicineKnowledgeStore.load(path)

    assert store.get("medicine_a").active_ingredient == "Ingredient A"
