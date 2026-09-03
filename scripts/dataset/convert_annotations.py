import json
import os
import random
import shutil

# -----------------------------
# CONFIGURATION PARAMETERS
# -----------------------------
json_dir = "path/to/annotation_json"
rgb_dir = "path/to/images"
output_dir = "path/to/data_folder"

# Dataset split ratios
train_ratio = 0.8
valid_ratio = 0.2
test_ratio = 0  # Set to 0 if not needed

# Image resolution (adjust to your real image size)
image_width = 1280
image_height = 720

# -----------------------------
# CREATE OUTPUT DIRECTORIES
# -----------------------------
for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

# -----------------------------
# LOAD ALL JSON CAPTURES
# -----------------------------
all_captures = []
for filename in os.listdir(json_dir):
    if filename.startswith("captures") and filename.endswith(".json"):
        json_path = os.path.join(json_dir, filename)
        with open(json_path, 'r') as f:
            data = json.load(f)
        for capture in data['captures']:
            image_name = os.path.basename(capture['filename'])
            image_id = os.path.splitext(image_name)[0]
            all_captures.append((capture, image_name, image_id))

random.shuffle(all_captures)

# -----------------------------
# SPLIT DATASETS
# -----------------------------
total = len(all_captures)
train_idx = int(total * train_ratio)
valid_idx = train_idx + int(total * valid_ratio)

train_captures = all_captures[:train_idx]
valid_captures = all_captures[train_idx:valid_idx]
test_captures = all_captures[valid_idx:]


# -----------------------------
# FUNCTION: PROCESS CAPTURES
# -----------------------------
def process_captures(captures, split):
    label_map = {}

    for capture, image_name, image_id in captures:
        src_img = os.path.join(rgb_dir, image_name)
        if not os.path.exists(src_img):
            print(f" Missing image: {src_img}")
            continue

        dst_img = os.path.join(output_dir, split, 'images', image_name)
        shutil.copy(src_img, dst_img)

        bbox_annotations = []

        for ann in capture.get('annotations', []):
            if ann['annotation_definition'] == 'bounding box':
                for value in ann['values']:
                    if all(k in value for k in ['x', 'y', 'width', 'height']):
                        label_id = value['label_id']
                        label_name = value['label_name']

                        # FIX HERE: Convert class IDs from 1-based to 0-based indexing
                        label_id = label_id - 1

                        # Save mapping (optional for YAML generation)
                        label_map[label_id] = label_name

                        # Normalized YOLO coordinates
                        x = value['x']
                        y = value['y']
                        width = value['width']
                        height = value['height']

                        xc = (x + width / 2) / image_width
                        yc = (y + height / 2) / image_height
                        w_norm = width / image_width
                        h_norm = height / image_height

                        bbox_annotations.append(f"{label_id} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Write YOLO label file
        txt_path = os.path.join(output_dir, split, 'labels', f"{image_id}.txt")
        with open(txt_path, 'w') as f_txt:
            f_txt.write('\n'.join(bbox_annotations))

    return label_map


# -----------------------------
# PROCESS SPLITS
# -----------------------------
print("Processing training set...")
train_labels = process_captures(train_captures, 'train')

print("Processing validation set...")
valid_labels = process_captures(valid_captures, 'valid')

if test_ratio > 0:
    print("Processing test set...")
    test_labels = process_captures(test_captures, 'test')
else:
    test_labels = {}

# -----------------------------
# GENERATE YAML CONFIG
# -----------------------------
all_labels = {**train_labels, **valid_labels, **test_labels}
sorted_labels = sorted(all_labels.items(), key=lambda x: x[0])
class_names = [name for _, name in sorted_labels]

yaml_path = os.path.join(output_dir, 'data.yaml')
with open(yaml_path, 'w') as f_yaml:
    f_yaml.write(f"train: {os.path.join(output_dir, 'train', 'images')}\n")
    f_yaml.write(f"val: {os.path.join(output_dir, 'valid', 'images')}\n")
    f_yaml.write(f"\nnc: {len(class_names)}\n")
    f_yaml.write(f"names: {class_names}\n")

print("\n YOLO dataset generated successfully!")
print(f" Output directory: {output_dir}")
print(f" Number of classes: {len(class_names)}")
print(f" YAML file created at: {yaml_path}")
