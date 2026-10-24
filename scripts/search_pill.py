import argparse
from pathlib import Path

import cv2

from app.services.pill_embedding import (
    MobileNetPillEmbedder,
    PillEmbeddingConfig,
)
from app.services.retrieval import FaissPillIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a pill image in FAISS.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--index", type=Path, default=Path("data/pill_index/pills.index"))
    parser.add_argument("--metadata", type=Path, default=Path("data/pill_index/pills.json"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise OSError(f"Could not read image: {args.image}")

    embedder = MobileNetPillEmbedder(
        PillEmbeddingConfig(checkpoint_path=args.checkpoint)
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
