import os
import shutil

# === PATHS ===
synthetic_path = "path/to/synthetic_data"
real_path = "path/to/real_data"
merged_path = "path/to/new_data"

splits = ["train", "valid"]

def ensure_dirs():
    for split in splits:
        os.makedirs(os.path.join(merged_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(merged_path, split, "labels"), exist_ok=True)

def copy_dataset(source_path, prefix):
    for split in splits:
        img_src = os.path.join(source_path, split, "images")
        lbl_src = os.path.join(source_path, split, "labels")

        img_dst = os.path.join(merged_path, split, "images")
        lbl_dst = os.path.join(merged_path, split, "labels")

        for filename in os.listdir(img_src):
            name, ext = os.path.splitext(filename)

            new_name = f"{prefix}_{name}{ext}"
            shutil.copy(
                os.path.join(img_src, filename),
                os.path.join(img_dst, new_name)
            )

            label_file = name + ".txt"
            if os.path.exists(os.path.join(lbl_src, label_file)):
                shutil.copy(
                    os.path.join(lbl_src, label_file),
                    os.path.join(lbl_dst, f"{prefix}_{label_file}")
                )
            else:
                print(f"Missing label for {filename}")

def verify_dataset():
    print("\nVerifying dataset consistency...\n")

    for split in splits:
        img_dir = os.path.join(merged_path, split, "images")
        lbl_dir = os.path.join(merged_path, split, "labels")

        images = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
        labels = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir)}

        missing_labels = images - labels
        missing_images = labels - images

        print(f"Split: {split}")
        print(f"Images: {len(images)}")
        print(f"Labels: {len(labels)}")

        if missing_labels:
            print(f"Missing labels: {len(missing_labels)}")
        if missing_images:
            print(f"Missing images: {len(missing_images)}")

        if not missing_labels and not missing_images:
            print("Dataset is consistent\n")

# === RUN ===
ensure_dirs()
copy_dataset(synthetic_path, "syn")
copy_dataset(real_path, "real")
verify_dataset()
