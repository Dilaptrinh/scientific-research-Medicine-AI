import numpy as np
import pytest

from app.services.retrieval import FaissPillIndex, PillReference


faiss = pytest.importorskip("faiss")


def test_faiss_returns_closest_pill_and_persists_metadata(tmp_path) -> None:
    index = FaissPillIndex(dimension=3)
    references = [
        PillReference(
            image_path="paracetamol/sample_1.jpg",
            medicine_id="paracetamol_500",
            medicine_name="Paracetamol 500mg",
            source_document_id="doc-001",
        ),
        PillReference(
            image_path="amoxicillin/sample_1.jpg",
            medicine_id="amoxicillin_500",
            medicine_name="Amoxicillin 500mg",
            source_document_id="doc-002",
        ),
    ]
    index.add(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        references,
    )

    matches = index.search(np.array([0.95, 0.05, 0.0], dtype=np.float32), top_k=1)

    assert len(matches) == 1
    assert matches[0].reference.medicine_id == "paracetamol_500"
    assert matches[0].similarity > 0.9

    index_path = tmp_path / "pill.index"
    metadata_path = tmp_path / "pill.json"
    index.save(index_path, metadata_path)
    loaded = FaissPillIndex.load(index_path, metadata_path)

    assert loaded.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1)[
        0
    ].reference.source_document_id == "doc-001"
