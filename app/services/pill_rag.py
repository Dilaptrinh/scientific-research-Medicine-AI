import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.pill_embedding import (
    MobileNetPillEmbedder,
    PillEmbeddingConfig,
    resolve_checkpoint_path,
)
from app.services.retrieval import FaissPillIndex, PillMatch


@dataclass(frozen=True)
class MedicineKnowledge:
    medicine_id: str
    medicine_name: str
    active_ingredient: str | None = None
    strength: str | None = None
    indication: str | None = None
    warning: str | None = None
    source_document_id: str | None = None
    source_url: str | None = None
    source_note: str | None = None


@dataclass(frozen=True)
class RagCandidate:
    medicine_id: str
    medicine_name: str
    similarity: float
    average_similarity: float
    matched_image_count: int
    source_document_ids: tuple[str, ...]
    knowledge: MedicineKnowledge | None


@dataclass(frozen=True)
class PillRagResult:
    status: str
    threshold: float
    candidates: tuple[RagCandidate, ...]

    @property
    def answer(self) -> str:
        if self.status == "unknown" or not self.candidates:
            return (
                "Chưa đủ cơ sở xác định thuốc từ ảnh. "
                "Vui lòng chụp lại ảnh rõ hơn hoặc xác minh với dược sĩ."
            )

        candidate = self.candidates[0]
        knowledge = candidate.knowledge
        lines = [
            f"Thuốc có khả năng gần nhất: {candidate.medicine_name}.",
            f"Độ tương đồng hình ảnh: {candidate.similarity:.4f}.",
        ]
        if knowledge is not None:
            if knowledge.active_ingredient:
                lines.append(f"Hoạt chất: {knowledge.active_ingredient}.")
            if knowledge.strength:
                lines.append(f"Hàm lượng: {knowledge.strength}.")
            if knowledge.indication:
                lines.append(f"Thông tin sử dụng: {knowledge.indication}.")
            if knowledge.warning:
                lines.append(f"Cảnh báo: {knowledge.warning}.")
            if knowledge.source_url:
                lines.append(f"Nguồn: {knowledge.source_url}")
            elif knowledge.source_note:
                lines.append(f"Thông tin nguồn: {knowledge.source_note}.")
        lines.append("Kết quả chỉ mang tính tham khảo, không thay thế tư vấn y tế.")
        return "\n".join(lines)


class MedicineKnowledgeStore:
    def __init__(self, medicines: dict[str, MedicineKnowledge]) -> None:
        self._medicines = medicines

    @classmethod
    def load(cls, path: Path) -> "MedicineKnowledgeStore":
        if not path.is_file():
            raise FileNotFoundError(
                f"Medicine knowledge base was not found: {path}. "
                "Run scripts/build_medicine_knowledge_template.py and fill "
                "the fields from official sources."
            )

        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Medicine knowledge base must contain a JSON list.")

        medicines: dict[str, MedicineKnowledge] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("Each medicine knowledge entry must be an object.")
            medicine_id = item.get("medicine_id")
            medicine_name = item.get("medicine_name")
            if not isinstance(medicine_id, str) or not medicine_id:
                raise ValueError("Each medicine entry needs a non-empty medicine_id.")
            if not isinstance(medicine_name, str) or not medicine_name:
                raise ValueError(
                    f"Medicine {medicine_id!r} needs a non-empty medicine_name."
                )
            if medicine_id in medicines:
                raise ValueError(f"Duplicate medicine_id in knowledge base: {medicine_id}")
            medicines[medicine_id] = MedicineKnowledge(
                medicine_id=medicine_id,
                medicine_name=medicine_name,
                active_ingredient=item.get("active_ingredient"),
                strength=item.get("strength"),
                indication=item.get("indication"),
                warning=item.get("warning"),
                source_document_id=item.get("source_document_id"),
                source_url=item.get("source_url"),
                source_note=item.get("source_note"),
            )
        return cls(medicines)

    def get(self, medicine_id: str) -> MedicineKnowledge | None:
        return self._medicines.get(medicine_id)


def group_matches(
    matches: list[PillMatch],
    knowledge_store: MedicineKnowledgeStore | None = None,
    top_k: int = 5,
    similarity_threshold: float = 0.75,
) -> PillRagResult:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1.")

    grouped: dict[str, list[PillMatch]] = defaultdict(list)
    for match in matches:
        grouped[match.reference.medicine_id].append(match)

    candidates = []
    for medicine_id, medicine_matches in grouped.items():
        best = max(medicine_matches, key=lambda match: match.similarity)
        source_ids = tuple(
            sorted(
                {
                    match.reference.source_document_id
                    for match in medicine_matches
                    if match.reference.source_document_id
                }
            )
        )
        candidates.append(
            RagCandidate(
                medicine_id=medicine_id,
                medicine_name=best.reference.medicine_name,
                similarity=best.similarity,
                average_similarity=sum(
                    match.similarity for match in medicine_matches
                )
                / len(medicine_matches),
                matched_image_count=len(medicine_matches),
                source_document_ids=source_ids,
                knowledge=knowledge_store.get(medicine_id)
                if knowledge_store
                else None,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.similarity,
            candidate.average_similarity,
            candidate.matched_image_count,
        ),
        reverse=True,
    )
    selected = tuple(candidates[:top_k])
    status = (
        "match"
        if selected and selected[0].similarity >= similarity_threshold
        else "unknown"
    )
    return PillRagResult(
        status=status,
        threshold=similarity_threshold,
        candidates=selected,
    )


class PillRagService:
    def __init__(
        self,
        embedder: MobileNetPillEmbedder,
        index: FaissPillIndex,
        knowledge_store: MedicineKnowledgeStore | None = None,
        similarity_threshold: float = 0.75,
        retrieval_top_k: int = 15,
        result_top_k: int = 5,
    ) -> None:
        if retrieval_top_k <= 0 or result_top_k <= 0:
            raise ValueError("Retrieval and result top_k values must be positive.")
        self._embedder = embedder
        self._index = index
        self._knowledge_store = knowledge_store
        self._similarity_threshold = similarity_threshold
        self._retrieval_top_k = retrieval_top_k
        self._result_top_k = result_top_k

    @classmethod
    def from_paths(
        cls,
        index_path: Path,
        metadata_path: Path,
        checkpoint_path: Path | None = None,
        knowledge_path: Path | None = None,
        similarity_threshold: float = 0.75,
        result_top_k: int = 5,
    ) -> "PillRagService":
        checkpoint = resolve_checkpoint_path(checkpoint_path)
        embedder = MobileNetPillEmbedder(
            PillEmbeddingConfig(checkpoint_path=checkpoint)
        )
        index = FaissPillIndex.load(index_path, metadata_path)
        if index.dimension != embedder.embedding_dim:
            raise ValueError(
                f"Index dimension ({index.dimension}) does not match model "
                f"dimension ({embedder.embedding_dim}). Rebuild the index."
            )
        knowledge_store = (
            MedicineKnowledgeStore.load(knowledge_path)
            if knowledge_path is not None
            else None
        )
        return cls(
            embedder=embedder,
            index=index,
            knowledge_store=knowledge_store,
            similarity_threshold=similarity_threshold,
            result_top_k=result_top_k,
        )

    def search(self, image: np.ndarray) -> PillRagResult:
        embedding = self._embedder.embed(image)
        matches = self._index.search(embedding, top_k=self._retrieval_top_k)
        return group_matches(
            matches,
            knowledge_store=self._knowledge_store,
            top_k=self._result_top_k,
            similarity_threshold=self._similarity_threshold,
        )
