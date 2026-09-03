"""
Generalized evaluation script used to reproduce the quantitative
object-detection evaluations conducted during the thesis.

The script evaluates a trained YOLO or YOLO-World checkpoint on a
labeled validation or test dataset and reports standard object-detection
metrics, including Precision, Recall, mAP@0.5, and mAP@0.5-0.95.

Ultralytics validation plots, including the confusion matrix and
precision-recall curves, are automatically generated when enabled.

Inference on individual images or videos is handled separately by the
scripts provided in the inference module.

Example:
    python evaluate_model.py \
        --model models/finetuned/yoloworld/model_best.pt \
        --data datasets/spontaneous/data.yaml \
        --model-type yoloworld \
        --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO, YOLOWorld


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLO or YOLO-World detection model."
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the dataset YAML file.",
    )

    parser.add_argument(
        "--model-type",
        choices=["yolo", "yoloworld"],
        default="yolo",
        help="Model architecture to load.",
    )

    parser.add_argument(
        "--split",
        choices=["val", "test", "train"],
        default="test",
        help="Dataset split used for evaluation.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Evaluation image size.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold used for NMS.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Evaluation device, e.g. 0, 1, or cpu.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/evaluation"),
        help="Directory where evaluation outputs are stored.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    model_path = args.model.resolve()
    data_yaml = args.data.resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}"
        )

    model_name = model_path.stem

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{model_name}_{args.split}"

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if args.model_type == "yoloworld":
        model = YOLOWorld(str(model_path))
    else:
        model = YOLO(str(model_path))

    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print(f"Model      : {model_path}")
    print(f"Model type : {args.model_type}")
    print(f"Dataset    : {data_yaml}")
    print(f"Split      : {args.split}")
    print(f"Device     : {args.device}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=True,
        plots=True,
        save_json=True,
        project=str(output_dir),
        name=run_name,
    )

    # ------------------------------------------------------------------
    # Extract metrics
    # ------------------------------------------------------------------
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    results = {
        "model": str(model_path),
        "model_type": args.model_type,
        "dataset": str(data_yaml),
        "split": args.split,
        "image_size": args.imgsz,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
    }

    # ------------------------------------------------------------------
    # Save metrics summary
    # ------------------------------------------------------------------
    results_dir = output_dir / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "metrics_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"mAP@0.5         : {map50:.4f}")
    print(f"mAP@0.5-0.95    : {map50_95:.4f}")
    print(f"Results saved   : {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()