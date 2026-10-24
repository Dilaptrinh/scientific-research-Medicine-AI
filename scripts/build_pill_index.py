import argparse
from pathlib import Path

import cv2
import numpy as np

from app.services.pill_embedding import (
    MobileNetPillEmbedder,
    PillEmbeddingConfig,
)
from app.services.retrieval import FaissPillIndex, PillReference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from medicine-labelled pill folders."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/pill_dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/pill_index"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    image_paths = sorted(
        path
        for path in args.dataset.glob("*/*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No pill images found in medicine subfolders: {args.dataset}"
        )

    embedder = MobileNetPillEmbedder(
        PillEmbeddingConfig(checkpoint_path=args.checkpoint)
    )
    embeddings = []
    references = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise OSError(f"Could not read image: {image_path}")
        embeddings.append(embedder.embed(image))
        medicine_id = image_path.parent.name
        references.append(
            PillReference(
                image_path=str(image_path),
                medicine_id=medicine_id,
                medicine_name=medicine_id.replace("_", " "),
            )
        )

    index = FaissPillIndex(embedder.embedding_dim)
    index.add(np.stack(embeddings), references)
    index.save(args.output / "pills.index", args.output / "pills.json")
    print(f"Indexed {len(references)} images into {args.output}")


if __name__ == "__main__":
    main()
