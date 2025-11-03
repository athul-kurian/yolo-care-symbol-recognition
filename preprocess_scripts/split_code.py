import os
import shutil
from pathlib import Path

# Paths
base_dir = "."
train_images = Path(base_dir) / "train/images"
train_labels = Path(base_dir) / "train/labels"
valid_images = Path(base_dir) / "valid/images"
valid_labels = Path(base_dir) / "valid/labels"

# Make sure destination directories exist
valid_images.mkdir(parents=True, exist_ok=True)
valid_labels.mkdir(parents=True, exist_ok=True)

# Load image names
with open("./validation_image_names.txt", "r") as f:
    image_names = [line.strip() for line in f.readlines()]

# Move files
for name in image_names:
    img_src = train_images / f"{name}.jpg"
    lbl_src = train_labels / f"{name}.txt"
    img_dst = valid_images / f"{name}.jpg"
    lbl_dst = valid_labels / f"{name}.txt"

    if img_src.exists():
        shutil.move(str(img_src), str(img_dst))
    if lbl_src.exists():
        shutil.move(str(lbl_src), str(lbl_dst))