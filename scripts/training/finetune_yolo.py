"""
Generalized YOLO fine-tuning script used to reproduce the YOLOv8/YOLOv9
training experiments conducted during the thesis.

This script was refactored from the original experimental code to remove
machine-specific paths and provide reusable command-line arguments while
preserving the original training configuration.

Compatible with Ultralytics YOLO object-detection models.

Example:
    python finetune_yolo.py \
        --model yolov9s.pt \
        --data datasets/synthetic/data.yaml \
        --name yolov9_synthetic
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune an Ultralytics YOLO object-detection model."
    )

    # Required experiment inputs
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Pretrained model or checkpoint, e.g. yolov8s.pt or yolov9s.pt.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the YOLO dataset YAML file.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name, e.g. yolov9_synthetic.",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)

    # Output locations
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/training"),
        help="Directory for Ultralytics training outputs.",
    )

    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("models/finetuned"),
        help="Directory where the best trained model is copied.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    data_yaml = args.data.resolve()

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    # ------------------------------------------------------------------
    # Experiment name
    # ------------------------------------------------------------------
    model_name = Path(args.model).stem

    if args.name:
        run_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}_{timestamp}"

    project_dir = args.project.resolve()
    weights_dir = args.weights_dir.resolve()

    project_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("YOLO Fine-Tuning")
    print("=" * 60)
    print(f"Model      : {args.model}")
    print(f"Dataset    : {data_yaml}")
    print(f"Run        : {run_name}")
    print(f"Device     : {args.device}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = YOLO(args.model)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    start_time = time.time()

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        optimizer=args.optimizer,
        lr0=args.lr0,
        project=str(project_dir),
        name=run_name,
        save=True,
        patience=args.patience,
        batch=args.batch,
        workers=args.workers,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
        deterministic=True,
    )

    training_time = time.time() - start_time

    # ------------------------------------------------------------------
    # Locate best checkpoint
    # ------------------------------------------------------------------
    best_checkpoint = project_dir / run_name / "weights" / "best.pt"

    if not best_checkpoint.exists():
        raise FileNotFoundError(
            f"Training completed but best.pt was not found: {best_checkpoint}"
        )

    # ------------------------------------------------------------------
    # Validate best checkpoint
    # ------------------------------------------------------------------
    best_model = YOLO(str(best_checkpoint))

    metrics = best_model.val(
        data=str(data_yaml),
        device=args.device,
    )

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    # ------------------------------------------------------------------
    # Store best model
    # ------------------------------------------------------------------
    saved_model = weights_dir / f"{run_name}_best.pt"
    shutil.copy2(best_checkpoint, saved_model)

    # ------------------------------------------------------------------
    # Save experiment metadata
    # ------------------------------------------------------------------
    metadata = {
        "run_name": run_name,
        "base_model": args.model,
        "dataset": str(data_yaml),
        "epochs": args.epochs,
        "batch_size": args.batch,
        "image_size": args.imgsz,
        "optimizer": args.optimizer,
        "learning_rate": args.lr0,
        "patience": args.patience,
        "device": args.device,
        "seed": args.seed,
        "training_time_seconds": round(training_time, 2),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "best_model": str(saved_model),
    }

    metadata_path = weights_dir / f"{run_name}_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training completed")
    print("=" * 60)
    print(f"Run              : {run_name}")
    print(f"mAP@50           : {map50:.4f}")
    print(f"mAP@50-95        : {map50_95:.4f}")
    print(f"Training time    : {training_time / 3600:.2f} h")
    print(f"Best checkpoint  : {saved_model}")
    print(f"Metadata         : {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()