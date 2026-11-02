import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


@dataclass(frozen=True)
class MedicineDocument:
    name: str
    fields: dict[str, str]
    source: str


class TextRagService:
    """Local, source-grounded retrieval over the medicine CSV."""

    def __init__(self, documents: list[MedicineDocument]) -> None:
        if not documents:
            raise ValueError("Text RAG knowledge base is empty.")
        self._documents = documents

    @classmethod
    def from_csv(cls, path: Path) -> "TextRagService":
        if not path.is_file():
            raise FileNotFoundError(f"Text RAG knowledge base was not found: {path}")
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = {"ten_thuoc", "hoat_chat", "chi_dinh", "lieu_dung",
                        "chong_chi_dinh", "tac_dung_phu", "luu_y", "nguon"}
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(f"Text RAG CSV is missing columns: {sorted(required)}")
            documents = [
                MedicineDocument(
                    name=row["ten_thuoc"].strip(),
                    fields={key: (row.get(key) or "").strip() for key in required},
                    source=(row.get("nguon") or "").strip(),
                )
                for row in reader
                if (row.get("ten_thuoc") or "").strip()
            ]
        return cls(documents)

    def answer(self, question: str, medicine_name: str | None = None) -> tuple[str, str, str]:
        if not question.strip():
            raise ValueError("Question must not be empty.")
        document = self._find_document(question, medicine_name)
        if document is None:
            return (
                "Chưa xác định được thuốc trong câu hỏi. Hãy nêu tên thuốc hoặc "
                "gửi ảnh qua endpoint full RAG.",
                "",
                "",
            )
        field, label = self._find_field(question)
        if field is None:
            answer = self._summary(document)
            field = "summary"
        else:
            answer = f"{document.name} - {label}: {document.fields[field]}."
        return answer, document.name, document.source

    def _find_document(
        self, question: str, medicine_name: str | None
    ) -> MedicineDocument | None:
        haystack = _normalize(f"{medicine_name or ''} {question}")
        matches = [
            document for document in self._documents
            if _normalize(document.name) in haystack
        ]
        return max(matches, key=lambda item: len(_normalize(item.name)), default=None)

    @staticmethod
    def _find_field(question: str) -> tuple[str | None, str]:
        normalized = _normalize(question)
        rules = (
            (("hoat chat", "thanh phan", "thanh phan chinh"), "hoat_chat", "Hoạt chất"),
            (("dieu tri", "cong dung", "chi dinh", "dung de lam gi"), "chi_dinh", "Chỉ định"),
            (("lieu", "uống bao nhieu", "su dung the nao"), "lieu_dung", "Liều dùng"),
            (("chong chi dinh", "khong duoc dung"), "chong_chi_dinh", "Chống chỉ định"),
            (("tac dung phu", "phan ung phu"), "tac_dung_phu", "Tác dụng phụ"),
            (("luu y", "can than", "canh bao"), "luu_y", "Lưu ý"),
        )
        for keywords, field, label in rules:
            if any(keyword in normalized for keyword in keywords):
                return field, label
        return None, ""

    @staticmethod
    def _summary(document: MedicineDocument) -> str:
        return (
            f"{document.name}. Hoạt chất: {document.fields['hoat_chat']}. "
            f"Chỉ định: {document.fields['chi_dinh']}. "
            f"Lưu ý: {document.fields['luu_y']}."
        )
