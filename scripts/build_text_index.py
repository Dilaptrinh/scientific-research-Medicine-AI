import argparse
from pathlib import Path

import numpy as np

from app.services.text_rag import TextEmbeddingModel, TextFaissIndex, chunks_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local FAISS Text RAG index.")
    parser.add_argument("--source", type=Path, default=Path("docs/rag-text/thuoc_48.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/text_index"))
    args = parser.parse_args()

    chunks = chunks_from_csv(args.source)
    if not chunks:
        raise ValueError(f"No text chunks found in {args.source}")
    embedder = TextEmbeddingModel()
    embeddings = np.asarray(
        embedder.fit(
            [
                f"{chunk.medicine_name} {chunk.section} "
                f"{'thành phần chính' if chunk.section == 'Hoạt chất' else ''} "
                f"{'công dụng điều trị' if chunk.section == 'Chỉ định' else ''} "
                f"{chunk.text}"
                for chunk in chunks
            ]
        )
    )
    index = TextFaissIndex(embeddings.shape[1])
    index.add(embeddings, chunks)
    index.save(args.output / "text.index", args.output / "text.json")
    embedder.save(args.output / "text_model.json")
    print(f"Indexed {len(chunks)} text chunks into {args.output}")


if __name__ == "__main__":
    main()
