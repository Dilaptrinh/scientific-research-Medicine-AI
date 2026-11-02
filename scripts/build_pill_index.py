import argparse
from pathlib import Path

import cv2
import numpy as np

from app.services.pill_embedding import (
    MobileNetPillEmbedder,
    PillEmbeddingConfig,
    resolve_checkpoint_path,
)
from app.services.retrieval import FaissPillIndex, PillReference


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def build_index(
    dataset: Path,
    output: Path,
    checkpoint: Path | None = None,
    embedder: MobileNetPillEmbedder | None = None,
) -> None:
    image_paths = sorted(
        path
        for path in dataset.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No pill images found in medicine subfolders: {dataset}"
        )

    if embedder is None:
        resolved_checkpoint = resolve_checkpoint_path(checkpoint)
        embedder = MobileNetPillEmbedder(
            PillEmbeddingConfig(checkpoint_path=resolved_checkpoint)
        )
    embeddings = []
    references = []
    for image_path in image_paths:
        image = read_image(image_path)
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
    index.save(output / "pills.index", output / "pills.json")
    print(f"Indexed {len(references)} images into {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from medicine-labelled pill folders."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/pill_dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/pill_index"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    build_index(args.dataset, args.output, args.checkpoint)


if __name__ == "__main__":
    main()
