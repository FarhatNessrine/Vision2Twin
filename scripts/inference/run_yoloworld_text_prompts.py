"""
YOLO-World text-prompt inference script used to reproduce the
open-vocabulary detection experiments conducted during the thesis.

The script loads a trained YOLO-World checkpoint, applies custom text
prompts using the YOLO-World open-vocabulary interface, and performs
inference on images, image folders, or videos.

Quantitative evaluation is handled separately by the evaluation module.

Example:
    python run_yoloworld_text_prompts.py \
        --model models/finetuned/yoloworld/model_best.pt \
        --source examples/videos/test_video.mkv \
        --prompts "tidal turbine" "assembled hub" "rear cap" \
        --conf 0.60
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLOWorld


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run YOLO-World inference using custom text prompts."
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained YOLO-World checkpoint.",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help=(
            "Input source for inference: image, image directory, "
            "video, webcam index, or other Ultralytics-supported source."
        ),
    )

    parser.add_argument(
        "--prompts",
        nargs="+",
        required=True,
        help="Text prompts/classes to detect.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.60,
        help="Detection confidence threshold.",
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
        default=Path("runs/inference/text_prompts"),
        help="Directory where inference results are stored.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="yoloworld_text_prompts",
        help="Name of the inference run.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate model checkpoint
    # ------------------------------------------------------------------
    model_path = args.model.resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO-World checkpoint not found: {model_path}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load YOLO-World model
    # ------------------------------------------------------------------
    model = YOLOWorld(str(model_path))

    # ------------------------------------------------------------------
    # Register custom text prompts
    # ------------------------------------------------------------------
    model.set_classes(args.prompts)

    print("\n" + "=" * 60)
    print("YOLO-World Text-Prompt Inference")
    print("=" * 60)
    print(f"Model      : {model_path}")
    print(f"Source     : {args.source}")
    print(f"Prompts    : {', '.join(args.prompts)}")
    print(f"Confidence : {args.conf}")
    print(f"IoU        : {args.iou}")
    print(f"Image size : {args.imgsz}")
    print(f"Device     : {args.device}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        project=str(output_dir),
        name=args.name,
        verbose=True,
    )

    results_dir = output_dir / args.name

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Inference completed")
    print("=" * 60)
    print(f"Results saved to: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()