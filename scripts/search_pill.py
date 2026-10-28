import argparse
from pathlib import Path

import cv2
import numpy as np

from app.services.pill_embedding import (
    MobileNetPillEmbedder,
    PillEmbeddingConfig,
    resolve_checkpoint_path,
)
from app.services.retrieval import FaissPillIndex


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a pill image in FAISS.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--index", type=Path, default=Path("data/pill_index/pills.index"))
    parser.add_argument("--metadata", type=Path, default=Path("data/pill_index/pills.json"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    image = read_image(args.image)
    if image is None:
        raise OSError(f"Could not read image: {args.image}")

    resolved_checkpoint = resolve_checkpoint_path(args.checkpoint)
    embedder = MobileNetPillEmbedder(
        PillEmbeddingConfig(checkpoint_path=resolved_checkpoint)
    )
    index = FaissPillIndex.load(args.index, args.metadata)
    matches = index.search(embedder.embed(image), top_k=args.top_k)
    for rank, match in enumerate(matches, start=1):
        print(
            f"{rank}. {match.reference.medicine_name} "
            f"(similarity={match.similarity:.4f}, "
            f"image={match.reference.image_path})"
        )


if __name__ == "__main__":
    main()
