"""Train and benchmark a one-class YOLO detector.

The detector is intentionally responsible only for finding an ``object``.
Classification of the crop into box, blister, or pill belongs to MobileNet
and is not included in the YOLO metrics reported by this script.

Example:

    python scripts/train_yolo.py --data data/yolo/data.yaml --model yolov9s.pt

The script writes ``metrics.json``, ``metrics.csv``, and
``hyperparameters.csv`` into the run directory. Ultralytics also writes
the confusion matrices, curves, and weights there.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer.")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and benchmark a one-class YOLO detector."
    )
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml")
    parser.add_argument("--model", default="yolov9s.pt", help="Base YOLO weights")
    parser.add_argument("--project", type=Path, default=Path("runs/yolo"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size; 16 is safer than 32 on a 16 GB T4.",
    )
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds")
    parser.add_argument("--name-prefix", default="object_detector")
    parser.add_argument(
        "--val-split",
        choices=("val", "test"),
        default="val",
        help="Split used for the final mAP and speed report.",
    )
    parser.add_argument(
        "--class-name",
        default="object",
        help="Expected single class name in the dataset.",
    )
    return parser.parse_args()


def count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.model.parameters())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_one_class(names: Any, expected_name: str) -> None:
    if isinstance(names, dict):
        labels = list(names.values())
    else:
        labels = list(names)
    if labels != [expected_name]:
        raise ValueError(
            "This script requires exactly one class named "
            f"{expected_name!r}; dataset contains {labels!r}."
        )


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML was not found: {args.data}")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch == 0:
        raise ValueError("epochs and imgsz must be positive; batch cannot be zero.")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "YOLO training requires the ml dependencies. "
            'Install them with: pip install -e ".[ml]"'
        ) from error

    seeds = parse_seeds(args.seeds)
    run_rows: list[dict[str, Any]] = []
    hyperparameter_rows = [
        {"hyperparameter": "model", "value": args.model},
        {"hyperparameter": "data", "value": str(args.data)},
        {"hyperparameter": "epochs", "value": args.epochs},
        {"hyperparameter": "imgsz", "value": args.imgsz},
        {"hyperparameter": "batch", "value": args.batch},
        {"hyperparameter": "patience", "value": args.patience},
        {"hyperparameter": "workers", "value": args.workers},
        {"hyperparameter": "device", "value": args.device},
        {"hyperparameter": "val_split", "value": args.val_split},
        {"hyperparameter": "class_name", "value": args.class_name},
        {"hyperparameter": "seeds", "value": ",".join(map(str, seeds))},
    ]

    for seed in seeds:
        model = YOLO(args.model)
        parameters = count_parameters(model)
        run_name = f"{args.name_prefix}_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}"

        started = time.perf_counter()
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=seed,
            deterministic=True,
            patience=args.patience,
            project=str(args.project),
            name=run_name,
            exist_ok=False,
            plots=True,
            save=True,
            val=True,
            verbose=True,
        )
        training_minutes = (time.perf_counter() - started) / 60
        save_dir = Path(str(model.trainer.save_dir))
        best_path = save_dir / "weights" / "best.pt"
        if not best_path.is_file():
            raise FileNotFoundError(f"Best checkpoint was not created: {best_path}")

        best_model = YOLO(str(best_path))
        validate_one_class(best_model.names, args.class_name)
        metrics = best_model.val(
            data=str(args.data),
            split=args.val_split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=True,
            verbose=True,
        )
        box_metrics = metrics.box
        speed = metrics.speed
        inference_ms = float(speed.get("inference", 0.0))
        row = {
            "seed": seed,
            "run_dir": str(save_dir),
            "best_checkpoint": str(best_path),
            "mAP50": float(box_metrics.map50),
            "mAP75": float(box_metrics.map75),
            "mAP50-95": float(box_metrics.map),
            "training_minutes": round(training_minutes, 3),
            "inference_ms_per_image": round(inference_ms, 3),
            "inference_images_per_second": round(
                1000 / inference_ms, 3
            )
            if inference_ms > 0
            else None,
            "parameters": parameters,
            "parameters_million": round(parameters / 1_000_000, 3),
        }
        run_rows.append(row)
        (save_dir / "metrics.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        write_csv(save_dir / "metrics.csv", [row])
        write_csv(save_dir / "hyperparameters.csv", hyperparameter_rows)
        print(json.dumps(row, indent=2))

    write_csv(args.project / "summary_metrics.csv", run_rows)
    write_csv(args.project / "hyperparameters.csv", hyperparameter_rows)


if __name__ == "__main__":
    main()
