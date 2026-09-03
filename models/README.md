## Base Models

The experiments were initialized from publicly available pretrained checkpoints.

- **YOLOv8s** — `yolov8s.pt`  
  Official Ultralytics documentation:  
  https://docs.ultralytics.com/models/yolov8/

- **YOLOv9s** — `yolov9s.pt`  
  Official Ultralytics documentation:  
  https://docs.ultralytics.com/models/yolov9/

- **YOLO-World v2 (medium)** — `yolov8m-worldv2.pt`  
  Official Ultralytics documentation:  
  https://docs.ultralytics.com/models/yolo-world/

These original pretrained checkpoints are not redistributed in this repository.
They can be obtained through the official Ultralytics model distribution and are
automatically downloaded by Ultralytics when referenced by model name, when supported.

## Finetuned Models

This directory contains the main model checkpoints obtained during the thesis
experiments.

The models correspond to different training configurations, including:

- Synthetic-data training
- Controlled-data training
- Mixed synthetic and controlled data
- YOLO-World mixed-data training
- Second-stage fine-tuning on spontaneous real-world data

The selected checkpoints correspond to the best-performing models retained
from the corresponding experiments.

Model evaluation results are documented separately in:

```text
scripts/evaluation/