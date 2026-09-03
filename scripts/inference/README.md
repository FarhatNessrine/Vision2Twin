# Inference Scripts

This folder contains the inference scripts used to generate qualitative
object-detection results during the thesis.

Inference is separated from model training and quantitative evaluation to
maintain a modular and reproducible workflow.

## Available Scripts

### `run_inference.py`

Generic inference script compatible with both standard Ultralytics YOLO models
and YOLO-World models.

It can be used with:

- Individual images
- Image directories
- Video files
- Webcam streams
- Other input sources supported by Ultralytics

### Example — YOLO

```bash
python scripts/inference/run_inference.py \
    --model models/finetuned/yolo/yolov9_best.pt \
    --source examples/images \
    --model-type yolo \
    --conf 0.35
'''
### `run_yoloworld_text_prompts.py'


This script reproduces the open-vocabulary YOLO-World inference experiments
conducted during the thesis.

YOLO-World allows the detection vocabulary to be modified at inference time
using text prompts. The prompts are provided through the --prompts argument
instead of being hard-coded in the Python script.

Basic Example
python scripts/inference/run_yoloworld_text_prompts.py \
    --model models/finetuned/yoloworld/yoloworld_synthetic_best.pt \
    --source examples/videos/test_video.mkv \
    --prompts "tidal turbine" "assembled hub" "rear cap" \
    --conf 0.60    