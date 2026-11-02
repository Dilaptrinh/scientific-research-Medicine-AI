"""Build the local pill index and search one local path or image URL."""

import argparse
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np

from app.services.pill_embedding import (
    MobileNetPillEmbedder,
    PillEmbeddingConfig,
    resolve_checkpoint_path,
)
from app.services.retrieval import FaissPillIndex


def download_image(url: str) -> Path:
    suffix = Path(urlparse(url).path).suffix or ".jpg"
    temporary_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    request = Request(url, headers={"User-Agent": "medicine-ai-research/0.1"})
    try:
        with urlopen(request, timeout=30) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS if needed, then search a pill image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Local image path or direct public image URL. "
        "If omitted, the script asks for one.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    image_input = (args.image or input("Image path or URL: ").strip()).strip('"')
    if not image_input:
        raise ValueError("An image path or URL is required.")

    output_directory = Path("data/pill_index")
    index_path = output_directory / "pills.index"
    metadata_path = output_directory / "pills.json"
    resolved_checkpoint = resolve_checkpoint_path(args.checkpoint)
    embedder = MobileNetPillEmbedder(
        PillEmbeddingConfig(checkpoint_path=resolved_checkpoint)
    )

    rebuild_index = not index_path.is_file() or not metadata_path.is_file()
    if not rebuild_index:
        existing_index = FaissPillIndex.load(index_path, metadata_path)
        rebuild_index = existing_index.dimension != embedder.embedding_dim
        if rebuild_index:
            print(
                "FAISS index dimension does not match the embedding model. "
                "Rebuilding..."
            )

    if rebuild_index:
        print("Building FAISS index from data/pill_dataset...")
        from build_pill_index import build_index

        build_index(
            dataset=Path("data/pill_dataset"),
            output=output_directory,
            checkpoint=args.checkpoint,
            embedder=embedder,
        )

    temporary_path: Path | None = None
    image_path = Path(image_input)
    if image_input.lower().startswith(("http://", "https://")):
        print("Downloading image from URL...")
        temporary_path = download_image(image_input)
        image_path = temporary_path

    try:
        image = read_image(image_path)
        if image is None:
            raise OSError(f"Không đọc được ảnh: {image_input}")
        index = FaissPillIndex.load(index_path, metadata_path)
        matches = index.search(embedder.embed(image), top_k=args.top_k)
        if not matches:
            print("No matches found in the FAISS index.")
            return
        print("\nNearest matches:")
        for rank, match in enumerate(matches, start=1):
            print(
                f"{rank}. {match.reference.medicine_name} "
                f"(similarity={match.similarity:.4f})"
            )
            print(f"   reference: {match.reference.image_path}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
