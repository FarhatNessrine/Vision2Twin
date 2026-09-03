"""
Second-stage YOLO-World fine-tuning script used to reproduce the
spontaneous-data adaptation experiments conducted during the thesis.

The input model is a YOLO-World checkpoint previously trained on the
mixed dataset. The model is subsequently fine-tuned on a smaller
spontaneous real-world dataset to improve adaptation to real production
conditions.

This script was refactored from the original experimental code to remove
machine-specific paths and provide reusable command-line arguments while
preserving the original training configuration.

Evaluation and inference are handled separately by the dedicated scripts
provided in the evaluation and inference modules.

Example:
    python finetune_yoloworld_spontaneous.py \
        --model models/finetuned/yoloworld_mixed_best.pt \
        --data datasets/spontaneous/data.yaml \
        --name yoloworld_mixed_to_spontaneous
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
        description=(
            "Fine-tune a previously trained YOLO-World model "
            "on spontaneous real-world data."
        )
    )

    # ------------------------------------------------------------------
    # Input configuration
    # ------------------------------------------------------------------
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Path to the YOLO-World checkpoint previously trained "
            "on the mixed dataset."
        ),
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the spontaneous dataset YAML file.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name, e.g. yoloworld_mixed_to_spontaneous.",
    )

    # ------------------------------------------------------------------
    # Training parameters
    # ------------------------------------------------------------------
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--optimizer", type=str, default="Adam")

    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Number of initial model layers to freeze.",
    )

    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")

    # Augmentation parameters used in the original experiment
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--mixup", type=float, default=0.2)

    # ------------------------------------------------------------------
    # Output configuration
    # ------------------------------------------------------------------
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/yoloworld_spontaneous"),
        help="Directory containing Ultralytics training outputs.",
    )

    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("models/finetuned/yoloworld"),
        help="Directory where the selected best checkpoint is stored.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate dataset
    # ------------------------------------------------------------------
    data_yaml = args.data.resolve()

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Spontaneous dataset YAML not found: {data_yaml}"
        )

    # ------------------------------------------------------------------
    # Experiment configuration
    # ------------------------------------------------------------------
    if args.name:
        run_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"yoloworld_spontaneous_{timestamp}"

    project_dir = args.project.resolve()
    weights_dir = args.weights_dir.resolve()

    project_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("YOLO-World Spontaneous-Data Fine-Tuning")
    print("=" * 60)
    print(f"Initial model : {args.model}")
    print(f"Dataset       : {data_yaml}")
    print(f"Run           : {run_name}")
    print(f"Device        : {args.device}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch}")
    print(f"Learning rate : {args.lr0}")
    print(f"Freeze layers : {args.freeze}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Dataset configuration
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
    # Load previously trained mixed-data model
    # ------------------------------------------------------------------
    model = YOLOWorld(args.model)

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------
    print("Starting fine-tuning on spontaneous data...\n")

    start_time = time.time()

    model.train(
        data=data_config,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        freeze=args.freeze,
        patience=args.patience,
        augment=True,
        mosaic=args.mosaic,
        mixup=args.mixup,
        workers=args.workers,
        device=args.device,
        project=str(project_dir),
        name=run_name,
        plots=True,
        save=True,
        trainer=WorldTrainerFromScratch,
    )

    training_time = time.time() - start_time

    # ------------------------------------------------------------------
    # Locate best checkpoint
    # ------------------------------------------------------------------
    best_checkpoint = (
        project_dir
        / run_name
        / "weights"
        / "best.pt"
    )

    if not best_checkpoint.exists():
        raise FileNotFoundError(
            f"Training completed but best.pt was not found at: "
            f"{best_checkpoint}"
        )

    # ------------------------------------------------------------------
    # Save selected checkpoint
    # ------------------------------------------------------------------
    saved_model = weights_dir / f"{run_name}_best.pt"

    shutil.copy2(
        best_checkpoint,
        saved_model,
    )

    # ------------------------------------------------------------------
    # Save experiment metadata
    # ------------------------------------------------------------------
    metadata = {
        "training_stage": "spontaneous_data_finetuning",
        "run_name": run_name,
        "initial_checkpoint": args.model,
        "training_dataset": str(data_yaml),
        "epochs": args.epochs,
        "batch_size": args.batch,
        "image_size": args.imgsz,
        "learning_rate": args.lr0,
        "optimizer": args.optimizer,
        "freeze_layers": args.freeze,
        "patience": args.patience,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "device": args.device,
        "training_time_seconds": round(training_time, 2),
        "best_model": str(saved_model),
    }

    metadata_path = (
        weights_dir
        / f"{run_name}_metadata.json"
    )

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Fine-tuning completed")
    print("=" * 60)
    print(f"Run           : {run_name}")
    print(f"Training time : {training_time / 3600:.2f} h")
    print(f"Best model    : {saved_model}")
    print(f"Metadata      : {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()