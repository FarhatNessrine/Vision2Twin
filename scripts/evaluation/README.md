# Evaluation Scripts

This folder contains the quantitative evaluation tools used to assess the object-detection models developed during the thesis.

The evaluation workflow is separated from training and inference to keep the experimental pipeline modular and reproducible.

## Available Script

### `evaluate_model.py`

Generic evaluation script compatible with both standard Ultralytics YOLO models and YOLO-World models.

The script reports:

- Precision
- Recall
- mAP@0.5
- mAP@0.5-0.95
- Confusion matrix
- Precision-Recall curves
- Additional Ultralytics validation plots

Evaluation results are stored under:

```text
runs/evaluation/