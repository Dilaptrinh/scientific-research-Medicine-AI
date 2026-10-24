import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PillReference:
    image_path: str
    medicine_id: str
    medicine_name: str
    source_document_id: str | None = None


@dataclass(frozen=True)
class PillMatch:
    reference: PillReference
    similarity: float


class FaissPillIndex:
    """FAISS cosine-similarity index with JSON metadata beside the index."""

    def __init__(self, dimension: int) -> None:
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError(
                'FAISS is required. Install it with: pip install -e ".[retrieval]"'
            ) from error

        if dimension <= 0:
            raise ValueError("FAISS vector dimension must be positive.")

        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dimension)
        self._references: list[PillReference] = []

    def add(
        self,
        embeddings: np.ndarray,
        references: list[PillReference],
    ) -> None:
        vectors = self._validate_embeddings(embeddings)
        if len(vectors) != len(references):
            raise ValueError("Each embedding must have one metadata reference.")
        self._index.add(vectors)
        self._references.extend(references)

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[PillMatch]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self._index.ntotal == 0:
            return []

        query = self._validate_embeddings(np.asarray(embedding).reshape(1, -1))
        scores, indices = self._index.search(
            query,
            min(top_k, self._index.ntotal),
        )
        return [
            PillMatch(
                reference=self._references[int(index)],
                similarity=float(score),
            )
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    def save(self, index_path: Path, metadata_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(index_path))
        metadata_path.write_text(
            json.dumps(
                [asdict(reference) for reference in self._references],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "FaissPillIndex":
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError(
                'FAISS is required. Install it with: pip install -e ".[retrieval]"'
            ) from error

        index = faiss.read_index(str(index_path))
        references = [
            PillReference(**item)
            for item in json.loads(metadata_path.read_text(encoding="utf-8"))
        ]
        if index.ntotal != len(references):
            raise ValueError("FAISS index and metadata entry counts do not match.")

        result = cls(index.d)
        result._index = index
        result._references = references
        return result

    def _validate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._index.d:
            raise ValueError(
                f"Expected embeddings with shape (N, {self._index.d}), "
                f"got {vectors.shape}."
            )
        self._faiss.normalize_L2(vectors)
        return vectors
