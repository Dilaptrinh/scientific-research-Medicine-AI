import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SupportRelation:
    medicine: str
    supported_by: str
    reason: str


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


class GraphRagService:
    """Local graph retrieval over medicine support relationships."""

    def __init__(self, relations: list[SupportRelation]) -> None:
        self._by_medicine: dict[str, list[SupportRelation]] = {}
        for relation in relations:
            self._by_medicine.setdefault(_normalize(relation.medicine), []).append(relation)

    @classmethod
    def from_csv(cls, path: Path) -> "GraphRagService":
        if not path.is_file():
            raise FileNotFoundError(f"Graph RAG data was not found: {path}")
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = csv.DictReader(file)
            relations = [
                SupportRelation(
                    medicine=row["Tên thuốc"].strip(),
                    supported_by=row["Thuốc hỗ trợ"].strip(),
                    reason=row["Cách thức hỗ trợ cụ thể"].strip(),
                )
                for row in rows
                if row.get("Tên thuốc") and row.get("Thuốc hỗ trợ")
            ]
        return cls(relations)

    def related_to(self, medicine_name: str | None) -> list[SupportRelation]:
        if not medicine_name:
            return []
        query = _normalize(medicine_name)
        exact = self._by_medicine.get(query)
        if exact:
            return exact
        return [
            relation
            for key, relations in self._by_medicine.items()
            if key in query or query in key
            for relation in relations
        ]

    def context(self, medicine_name: str | None) -> str:
        relations = self.related_to(medicine_name)
        if not relations:
            return "Graph RAG: không có quan hệ thuốc hỗ trợ phù hợp."
        lines = ["Graph RAG - các quan hệ hỗ trợ tìm thấy:"]
        lines.extend(
            f"- {item.medicine} -> {item.supported_by}: {item.reason}"
            for item in relations
        )
        return "\n".join(lines)
