"""
Generic inference script used to generate qualitative object-detection
results during the thesis.

The script runs inference with a trained YOLO or YOLO-World model on
images, folders of images, or video files and saves the annotated outputs.

Quantitative evaluation is handled separately by the scripts provided
in the evaluation module.

Example:
    python run_inference.py \
        --model models/finetuned/yoloworld/model_best.pt \
        --source datasets/spontaneous/images \
        --model-type yoloworld \
        --conf 0.35
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO, YOLOWorld


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a YOLO or YOLO-World model."
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint.",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image, image directory, video, webcam index, or other supported source.",
    )

    parser.add_argument(
        "--model-type",
        choices=["yolo", "yoloworld"],
        default="yolo",
        help="Model architecture to load.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for detections.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold used for non-maximum suppression.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Inference device, e.g. 0, 1, cpu, or mps.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/inference"),
        help="Directory where inference outputs are stored.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional inference run name.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate model
    # ------------------------------------------------------------------
    model_path = args.model.resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    model_name = model_path.stem

    run_name = args.name or model_name

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if args.model_type == "yoloworld":
        model = YOLOWorld(str(model_path))
    else:
        model = YOLO(str(model_path))

    print("\n" + "=" * 60)
    print("Object Detection Inference")
    print("=" * 60)
    print(f"Model      : {model_path}")
    print(f"Model type : {args.model_type}")
    print(f"Source     : {args.source}")
    print(f"Confidence : {args.conf}")
    print(f"Device     : {args.device}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        project=str(output_dir),
        name=run_name,
        verbose=True,
    )

    results_dir = output_dir / run_name

    print("\n" + "=" * 60)
    print("Inference completed")
    print("=" * 60)
    print(f"Results saved to: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()