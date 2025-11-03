# script to apply augmentations to images and oversample images with underrepresented classes

import os
import cv2
import random
import albumentations as A
import numpy as np

IMAGES_DIR = "./images"
LABELS_DIR = "./labels"
OUT_IMAGES_DIR = "./aug_images"
OUT_LABELS_DIR = "./aug_labels"

os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUT_LABELS_DIR, exist_ok=True)

UNDERREPRESENTED_CLASSES = {} 

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.3),
    A.Blur(blur_limit=3, p=0.2),
    A.MotionBlur(p=0.2),
    A.RandomGamma(p=0.3),
    A.Affine(scale=(0.9, 1.1), rotate=(-10, 10), shear=(-5, 5), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RandomResizedCrop(640, 640, scale=(0.8, 1.0), p=0.4),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def load_yolo_labels(label_path):
    if not os.path.exists(label_path):
        return [], []
    boxes, class_labels = [], []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = map(float, parts)
            boxes.append([x, y, w, h])
            class_labels.append(int(cls))
    return boxes, class_labels


def save_yolo_labels(label_path, boxes, class_labels):
    with open(label_path, 'w') as f:
        for cls, (x, y, w, h) in zip(class_labels, boxes):
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def contains_underrepresented_class(class_labels):
    return any(cls in UNDERREPRESENTED_CLASSES for cls in class_labels)


image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_file in image_files:
    img_path = os.path.join(IMAGES_DIR, img_file)
    label_path = os.path.join(LABELS_DIR, os.path.splitext(img_file)[0] + '.txt')

    # Load image and labels
    image = cv2.imread(img_path)
    if image is None:
        continue

    boxes, class_labels = load_yolo_labels(label_path)
    if len(boxes) == 0:
        continue

    # Determine augmentation count
    aug_count = 6 if contains_underrepresented_class(class_labels) else 2

    for i in range(aug_count):
        transformed = transform(image=image, bboxes=boxes, class_labels=class_labels)

        aug_img = transformed['image']
        aug_boxes = transformed['bboxes']
        aug_classes = transformed['class_labels']

        if len(aug_boxes) == 0:
            continue  # skip if no boxes remain after augmentation

        # Save augmented image
        out_img_name = f"{os.path.splitext(img_file)[0]}_aug{i}.jpg"
        out_label_name = f"{os.path.splitext(img_file)[0]}_aug{i}.txt"
        cv2.imwrite(os.path.join(OUT_IMAGES_DIR, out_img_name), aug_img)

        # Save corresponding label
        save_yolo_labels(os.path.join(OUT_LABELS_DIR, out_label_name), aug_boxes, aug_classes)
