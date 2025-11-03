import os
import cv2
import numpy as np

def yolo_to_pixel(box, img_w, img_h):
    x_c, y_c, w, h = box
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return x1, y1, x2, y2

def process_yolo_folder(folder, target_class=14):
    for file in os.listdir(os.path.join(folder, "images")):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(folder, "images", file)
        label_path = os.path.join(folder, "labels", os.path.splitext(file)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to read {image_path}")
            continue
        h, w = image.shape[:2]

        # Load all annotations
        with open(label_path, "r") as f:
            lines = f.readlines()

        boxes = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            boxes.append((class_id, coords))

        # Create masks
        all_mask = np.zeros((h, w), dtype=np.uint8)
        z_mask = np.zeros((h, w), dtype=np.uint8)
        kept_lines = []

        for class_id, coords in boxes:
            x1, y1, x2, y2 = yolo_to_pixel(coords, w, h)
            box_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(box_mask, (x1, y1), (x2, y2), 1, -1)

            if class_id == target_class:
                z_mask = cv2.bitwise_or(z_mask, box_mask)
            else:
                all_mask = cv2.bitwise_or(all_mask, box_mask)
                kept_lines.append(f"{class_id} {' '.join(map(str, coords))}\n")

        # Compute z-only regions (z_mask minus overlapping)
        z_only = cv2.bitwise_and(z_mask, cv2.bitwise_not(all_mask))

        # White out pixels in z-only regions
        image[z_only == 1] = 255

        # Save modified image and updated label file
        cv2.imwrite(image_path, image)
        with open(label_path, "w") as f:
            f.writelines(kept_lines)

        print(f"Processed: {file}")

process_yolo_folder("./train")