# script to convert all images to grascale 640x640 images

import cv2
import os
import numpy as np

def preprocess_image(img, target_size=640):
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create black canvas and center image
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas


def process_folder(input_folder, output_folder, target_size=640):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Skipping unreadable image: {filename}")
            continue

        processed = preprocess_image(img, target_size)
        cv2.imwrite(output_path, processed)
        print(f"Processed: {filename}")

    print("\nAll images processed and saved to:", output_folder)

if __name__ == "__main__":
    input_dir = "./images_raw"
    output_dir = "./images_gray_640"  
    process_folder(input_dir, output_dir, target_size=640)
