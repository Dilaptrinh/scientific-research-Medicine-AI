"""Create a medicine knowledge template from FAISS metadata."""

import argparse
import json
from pathlib import Path


def build_template(metadata_path: Path, output_path: Path) -> None:
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    medicines: dict[str, dict[str, str | None]] = {}
    for reference in raw:
        medicine_id = reference["medicine_id"]
        medicines.setdefault(
            medicine_id,
            {
                "medicine_id": medicine_id,
                "medicine_name": reference["medicine_name"],
                "active_ingredient": None,
                "strength": None,
                "indication": None,
                "warning": None,
                "source_document_id": reference.get("source_document_id"),
                "source_url": None,
                "source_note": None,
            },
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(list(medicines.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Created {len(medicines)} medicine entries at {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a medicine knowledge JSON template."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/pill_index/pills.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/medicine_knowledge/medicines.json"),
    )
    args = parser.parse_args()
    build_template(args.metadata, args.output)


if __name__ == "__main__":
    main()
