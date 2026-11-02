"""Run YOLO detection and save crops from a terminal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import cv2

from yolo_detector import YoloDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detection on one image.")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/yolo/best.pt"),
        help="YOLO checkpoint path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Directory for cropped detections",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image was not found: {args.image}")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1.")

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Input file is not a valid image: {args.image}")

    detector = YoloDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        crop_directory=args.output,
    )
    image_id = uuid4().hex
    detections = detector.detect(image, image_id)
    print(json.dumps({
        "image_id": image_id,
        "detections": detections,
    }, indent=2))


if __name__ == "__main__":
    main()
