"""Run image retrieval, grouping, thresholding, and knowledge lookup."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from app.services.pill_rag import PillRagService


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read image: {path}")
    return image


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run RAG retrieval for a pill crop.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--index", type=Path, default=Path("data/pill_index/pills.index"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/pill_index/pills.json")
    )
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=Path("data/medicine_knowledge/medicines.json"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    service = PillRagService.from_paths(
        index_path=args.index,
        metadata_path=args.metadata,
        checkpoint_path=args.checkpoint,
        knowledge_path=args.knowledge,
        similarity_threshold=args.threshold,
        result_top_k=args.top_k,
    )
    result = service.search(read_image(args.image))
    print(json.dumps(
        {
            "status": result.status,
            "threshold": result.threshold,
            "answer": result.answer,
            "candidates": [
                {
                    "medicine_id": candidate.medicine_id,
                    "medicine_name": candidate.medicine_name,
                    "similarity": candidate.similarity,
                    "average_similarity": candidate.average_similarity,
                    "matched_image_count": candidate.matched_image_count,
                    "source_document_ids": candidate.source_document_ids,
                    "knowledge": (
                        candidate.knowledge.__dict__
                        if candidate.knowledge is not None
                        else None
                    ),
                }
                for candidate in result.candidates
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
