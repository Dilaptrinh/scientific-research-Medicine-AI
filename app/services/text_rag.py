import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TextChunk:
    medicine_name: str
    section: str
    text: str
    source: str


class TextEmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                'Install Text RAG dependencies with: pip install -e ".[text-rag]"'
            ) from error
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True),
            dtype=np.float32,
        )


class TextFaissIndex:
    def __init__(self, dimension: int) -> None:
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError(
                'FAISS is required. Install it with: pip install -e ".[retrieval]"'
            ) from error
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[TextChunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[TextChunk]) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._index.d:
            raise ValueError(f"Expected embeddings with dimension {self._index.d}.")
        if len(vectors) != len(chunks):
            raise ValueError("Each embedding must have one text chunk.")
        self._faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(self, embedding: np.ndarray, top_k: int = 6) -> list[tuple[TextChunk, float]]:
        query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        self._faiss.normalize_L2(query)
        scores, indices = self._index.search(query, min(top_k, self._index.ntotal))
        return [
            (self._chunks[int(index)], float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    def save(self, index_path: Path, metadata_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(index_path))
        metadata_path.write_text(
            json.dumps([asdict(chunk) for chunk in self._chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "TextFaissIndex":
        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Text FAISS index is missing: {index_path} and {metadata_path}"
            )
        import faiss

        result = cls(faiss.read_index(str(index_path)).d)
        result._index = faiss.read_index(str(index_path))
        result._chunks = [
            TextChunk(**item)
            for item in json.loads(metadata_path.read_text(encoding="utf-8"))
        ]
        if result._index.ntotal != len(result._chunks):
            raise ValueError("Text FAISS index and metadata counts do not match.")
        return result


class TextRagService:
    def __init__(self, embedder: TextEmbeddingModel, index: TextFaissIndex) -> None:
        self._embedder = embedder
        self._index = index

    @classmethod
    def from_paths(
        cls,
        index_path: Path,
        metadata_path: Path,
        model_name: str = "BAAI/bge-m3",
    ) -> "TextRagService":
        return cls(TextEmbeddingModel(model_name), TextFaissIndex.load(index_path, metadata_path))

    def answer(
        self, question: str, medicine_name: str | None = None
    ) -> tuple[str, str, str]:
        if not question.strip():
            raise ValueError("Question must not be empty.")
        query = f"{medicine_name}: {question}" if medicine_name else question
        matches = self._index.search(self._embedder.encode([query])[0])
        if not matches:
            return "Không tìm thấy tài liệu phù hợp.", medicine_name or "", ""
        context = "\n".join(
            f"{chunk.medicine_name} - {chunk.section}: {chunk.text}"
            for chunk, _ in matches
        )
        best = matches[0][0]
        return context, best.medicine_name, best.source


def chunks_from_csv(path: Path) -> list[TextChunk]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        fields = (
            ("hoat_chat", "Hoạt chất"),
            ("dang_bao_che", "Dạng bào chế"),
            ("chi_dinh", "Chỉ định"),
            ("lieu_dung", "Liều dùng"),
            ("chong_chi_dinh", "Chống chỉ định"),
            ("tac_dung_phu", "Tác dụng phụ"),
            ("luu_y", "Lưu ý"),
        )
        chunks = []
        for row in rows:
            name = (row.get("ten_thuoc") or "").strip()
            if name:
                chunks.extend(
                    TextChunk(name, label, (row.get(key) or "").strip(), row.get("nguon", ""))
                    for key, label in fields
                    if (row.get(key) or "").strip()
                )
        return chunks
