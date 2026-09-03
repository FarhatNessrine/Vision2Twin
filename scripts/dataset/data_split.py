import os
import random
import shutil

# =============================
# PATHS
# =============================

SOURCE_IMAGES = "path/to/data/images"
SOURCE_LABELS = "path/to/data/labels"

OUTPUT_DIR = "/mnt/storage/admindi/home/nfarhat/object_detection/spont_data_split2"

TRAIN_RATIO = 0.8

# =============================
# CREATE OUTPUT FOLDERS
# ============================= 

for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# =============================
# LIST IMAGES
# =============================

images = [f for f in os.listdir(SOURCE_IMAGES) if f.endswith((".jpg", ".png", ".jpeg"))]

random.shuffle(images)

train_size = int(len(images) * TRAIN_RATIO)

train_images = images[:train_size]
val_images = images[train_size:]

print(f"Total images: {len(images)}")
print(f"Train: {len(train_images)}")
print(f"Val: {len(val_images)}")

# =============================
# COPY FILES
# =============================

def copy_files(image_list, split):

    for img in image_list:

        img_path = os.path.join(SOURCE_IMAGES, img)
        label_name = os.path.splitext(img)[0] + ".txt"
        label_path = os.path.join(SOURCE_LABELS, label_name)

        dst_img = os.path.join(OUTPUT_DIR, split, "images", img)
        dst_lbl = os.path.join(OUTPUT_DIR, split, "labels", label_name)

        shutil.copy(img_path, dst_img)

        if os.path.exists(label_path):
            shutil.copy(label_path, dst_lbl)
        else:
            print(f"⚠ Missing label: {label_name}")

# run copy
copy_files(train_images, "train")
copy_files(val_images, "val")

print("Dataset split completed.")
