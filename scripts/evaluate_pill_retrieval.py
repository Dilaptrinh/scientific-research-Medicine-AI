"""Evaluate pill retrieval accuracy on labelled query images."""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.services.pill_embedding import (
    MobileNetPillEmbedder,
    PillEmbeddingConfig,
    resolve_checkpoint_path,
)
from app.services.retrieval import FaissPillIndex


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def collect_queries(query_directory: Path) -> list[tuple[Path, str]]:
    queries = [
        (path, path.parent.name)
        for path in sorted(query_directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not queries:
        raise FileNotFoundError(
            f"No query images found in labelled subfolders: {query_directory}"
        )
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Top-1 and Top-k pill retrieval accuracy."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/pill_queries"),
        help="Folder containing one labelled subfolder per medicine.",
    )
    parser.add_argument("--index", type=Path, default=Path("data/pill_index/pills.index"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/pill_index/pills.json")
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    queries = collect_queries(args.queries)
    checkpoint = resolve_checkpoint_path(args.checkpoint)
    embedder = MobileNetPillEmbedder(
        PillEmbeddingConfig(checkpoint_path=checkpoint)
    )
    index = FaissPillIndex.load(args.index, args.metadata)
    if index.dimension != embedder.embedding_dim:
        raise ValueError(
            f"Index dimension ({index.dimension}) does not match model "
            f"dimension ({embedder.embedding_dim}). Rebuild the index first."
        )

    top1_correct = 0
    top_k_correct = 0
    skipped = 0
    class_counts: Counter[str] = Counter()

    print(f"Evaluating {len(queries)} images with Top-{args.top_k} retrieval...")
    for query_path, expected_id in queries:
        image = read_image(query_path)
        if image is None:
            raise OSError(f"Could not read query image: {query_path}")

        matches = index.search(embedder.embed(image), top_k=args.top_k)
        predicted_ids = [match.reference.medicine_id for match in matches]
        if not predicted_ids:
            skipped += 1
            print(f"SKIP {query_path}: no matches")
            continue

        top1 = predicted_ids[0] == expected_id
        top_k = expected_id in predicted_ids
        top1_correct += int(top1)
        top_k_correct += int(top_k)
        class_counts[expected_id] += 1
        status = "OK" if top1 else "MISS"
        print(
            f"{status} {query_path} | expected={expected_id} | "
            f"top1={predicted_ids[0]} | top-{args.top_k}={predicted_ids}"
        )

    evaluated = len(queries) - skipped
    if evaluated == 0:
        raise RuntimeError("No query image produced a retrieval result.")

    print("\nSummary")
    print(f"- Images: {evaluated}/{len(queries)}")
    print(f"- Medicines: {len(class_counts)}")
    print(f"- Top-1 accuracy: {top1_correct / evaluated:.2%}")
    print(f"- Top-{args.top_k} accuracy: {top_k_correct / evaluated:.2%}")
    if skipped:
        print(f"- Skipped: {skipped}")


if __name__ == "__main__":
    main()
