"""
Generalized YOLO-World fine-tuning script used to reproduce the
YOLO-World training experiments conducted during the thesis.

This script was refactored from the original experimental code to remove
machine-specific paths and provide reusable command-line arguments while
preserving the original training configuration.

Evaluation and inference are handled separately by the dedicated scripts
provided in the evaluation and inference modules.

Example:
    python finetune_yoloworld.py \
        --model yolov8m-worldv2.pt \
        --data datasets/mixed/data.yaml \
        --name yoloworld_mixed_120k
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO-World object-detection model."
    )

    # ------------------------------------------------------------------
    # Input configuration
    # ------------------------------------------------------------------
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Pretrained YOLO-World model, e.g. yolov8m-worldv2.pt.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the training/validation dataset YAML file.",
    )

    parser.add_argument(
        "--test-data",
        type=Path,
        default=None,
        help="Optional YAML file for independent real-world evaluation.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name, e.g. yoloworld_mixed_120k.",
    )

    # ------------------------------------------------------------------
    # Training parameters
    # ------------------------------------------------------------------
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.0005)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--freeze", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", type=str, default="0")

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/yoloworld"),
        help="Directory containing Ultralytics training outputs.",
    )

    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("models/finetuned/yoloworld"),
        help="Directory where the selected best model is stored.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------
    data_yaml = args.data.resolve()

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Training dataset YAML not found: {data_yaml}"
        )

    test_yaml = None

    if args.test_data is not None:
        test_yaml = args.test_data.resolve()

        if not test_yaml.exists():
            raise FileNotFoundError(
                f"Test dataset YAML not found: {test_yaml}"
            )

    # ------------------------------------------------------------------
    # Experiment configuration
    # ------------------------------------------------------------------
    if args.name:
        run_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"yoloworld_{timestamp}"

    project_dir = args.project.resolve()
    weights_dir = args.weights_dir.resolve()

    project_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("YOLO-World Fine-Tuning")
    print("=" * 60)
    print(f"Model       : {args.model}")
    print(f"Dataset     : {data_yaml}")
    print(f"Test data   : {test_yaml if test_yaml else 'Not specified'}")
    print(f"Run         : {run_name}")
    print(f"Device      : {args.device}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch size  : {args.batch}")
    print(f"Freeze      : {args.freeze}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # YOLO-World data configuration
    # ------------------------------------------------------------------
    data_config = {
        "train": {
            "yolo_data": [str(data_yaml)],
        },
        "val": {
            "yolo_data": [str(data_yaml)],
        },
    }

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = YOLOWorld(args.model)

    # ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
start_time = time.time()

model.train(
    data=data_config,
    epochs=args.epochs,
    batch=args.batch,
    imgsz=args.imgsz,
    lr0=args.lr0,
    optimizer=args.optimizer,
    freeze=args.freeze,
    augment=True,
    dropout=0.2,
    workers=args.workers,
    device=args.device,
    patience=args.patience,
    project=str(project_dir),
    name=run_name,
    trainer=WorldTrainerFromScratch,
    save=True,
    plots=True,
)

training_time = time.time() - start_time

# ------------------------------------------------------------------
# Locate and save best checkpoint
# ------------------------------------------------------------------
best_checkpoint = project_dir / run_name / "weights" / "best.pt"

if not best_checkpoint.exists():
    raise FileNotFoundError(
        f"Training completed but best.pt was not found at: "
        f"{best_checkpoint}"
    )

saved_model = weights_dir / f"{run_name}_best.pt"
shutil.copy2(best_checkpoint, saved_model)

# ------------------------------------------------------------------
# Save experiment metadata
# ------------------------------------------------------------------
metadata = {
    "run_name": run_name,
    "base_model": args.model,
    "training_dataset": str(data_yaml),
    "epochs": args.epochs,
    "batch_size": args.batch,
    "image_size": args.imgsz,
    "learning_rate": args.lr0,
    "optimizer": args.optimizer,
    "freeze_layers": args.freeze,
    "patience": args.patience,
    "device": args.device,
    "training_time_seconds": round(training_time, 2),
    "best_model": str(saved_model),
}

metadata_path = weights_dir / f"{run_name}_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\n" + "=" * 60)
print("Training completed")
print("=" * 60)
print(f"Run           : {run_name}")
print(f"Training time : {training_time / 3600:.2f} h")
print(f"Best model    : {saved_model}")
print(f"Metadata      : {metadata_path}")
print("=" * 60)