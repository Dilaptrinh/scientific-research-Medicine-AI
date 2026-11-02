from pathlib import Path
import sys

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Medicine AI Research API",
    version="0.1.0",
)
app.include_router(router, prefix="/api/v1")


def run_text_rag_cli() -> None:
    from app.services.text_rag import TextRagService

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    project_root = Path(__file__).resolve().parents[1]
    index_dir = project_root / "data" / "text_index"
    service = TextRagService.from_paths(
        index_path=index_dir / "text.index",
        metadata_path=index_dir / "text.json",
    )
    print("Text RAG local - nhập 'thoat' để dừng.")
    while True:
        question = input("\nCâu hỏi: ").strip()
        if not question or question.lower() in {"thoat", "exit", "quit"}:
            print("Đã dừng.")
            return
        answer, medicine_name, source = service.answer(question)
        print(f"\nThuốc: {medicine_name or 'chưa xác định'}")
        print(f"Trả lời:\n{answer}")
        if source:
            print(f"Nguồn: {source}")


if __name__ == "__main__":
    run_text_rag_cli()
