<p align="center">
  <img src="https://github.com/athul-kurian/yolo-care-symbol-recognition/blob/main/assets/banner_collage.png" alt="Textile Symbol Detection Banner" style="width:100%; height:auto;"/>
</p>

# 👕 YOLO-Based Detection of Textile Care Symbols

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Ultralytics](https://img.shields.io/badge/YOLOv11-Ultralytics-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-blue)
![Albumentations](https://img.shields.io/badge/Albumentations-red)
![License](https://img.shields.io/badge/License-MIT-green)

> **A deep learning–based object detection system for recognizing textile care symbols using the YOLO architecture.**  
> The model detects 14 ISO 3758:2023 textile care symbols commonly printed on garment labels.

### 📃 [Read our Paper](https://github.com/athul-kurian/yolo-care-symbol-recognition/blob/main/paper.pdf)

---

## 🧠 Overview

This project implements a **YOLO-based detection pipeline** to identify and classify textile care (laundry) symbols from clothing label images.

The system was designed for small, low-resolution, and visually cluttered images where multiple overlapping care symbols may appear together.  
It uses a YOLOv11m model trained on a carefully annotated dataset to deliver high detection accuracy.

**Highlights:**
- 🎯 The system detects **14 standardized ISO 3758:2023 care symbols** that include washing, bleaching, drying, ironing, and dry-cleaning icons.
- ⚙️ The model was trained and validated on **1,227 labeled images** containing **6,244 annotated symbol instances**.
- 🧩 The final trained model is **YOLOv11m**, which achieved a **mean Average Precision (mAP@0.5) of 90.9%** on the validation dataset.
- 🖼️ The project includes a [**visualization tool**](https://github.com/athul-kurian/yolo-care-symbol-recognition/blob/main/visualize_outputs.py) that interprets YOLO outputs and overlays bounding boxes and legends for easier inspection of predictions.

---

## 🚀 How to Use

Clone the repository:
```bash
git clone https://github.com/athul-kurian/yolo-textile-care.git
```

Once the repository is cloned, navigate to the [`Final Product/`](https://github.com/athul-kurian/yolo-care-symbol-recognition/tree/main/Final%20Product) folder.  
This folder contains:
- The trained model file: `model.pt`
- A ready-to-use Jupyter notebook: [`test_model.ipynb`](https://github.com/athul-kurian/yolo-care-symbol-recognition/blob/main/Final%20Product/test_model.ipynb)
- 4 sample images to test on

To run inference:

1. Open [`test_model.ipynb`](https://github.com/athul-kurian/yolo-care-symbol-recognition/blob/main/Final%20Product/test_model.ipynb).
2. Follow the notebook cells to load the trained YOLO model, run inference on sample label images, and visualize detections.

---

## 🧩 Methodology

### 1. Dataset & Annotation
- Images were collected from garment labels and public sources.
- Annotations were prepared in YOLO format with bounding boxes for 14 symbol classes.
- Any symbols other than the 14 target classes were masked out to avoid confusing the model.

### 2. Preprocessing (OpenCV)
- Images were convereted to **grayscale**
- They were were resized and padded to **640×640**: black borders were added as needed to standardize input dimensions while maintaining aspect ratio.

### 3. Data Augmentation & Balancing
- Augmentations were performed using **Albumentations**.
- Included transformations such as rotations, flips, affine distortions, brightness/contrast shifts, and Gaussian blur.
- Images containing **underrepresented classes** were oversampled to address class imbalance.
- Final training set contained **1,227 images** and **6,244 annotated instances** (symbols).

### 4. Model Training
- Trained and compared multiple YOLO variants (v3, v4, v5, v8, v11); YOLOv11m achieved the best performance.
- Trained on NVIDIA A100 GPU.

### 5. Evaluation
- Evaluation Metric used was **mAP@0.5**: Mean Average Precision (mAP) at an Intersection of Union (IoU) threshold of 0.5
- Achieved **mAP@0.5 = 90.9%** on validation.
<img src="https://github.com/athul-kurian/yolo-care-symbol-recognition/blob/main/yolov11m%20training/metrics.png" style="width:50%;height:50%"/>
<img src="https://github.com/athul-kurian/yolo-care-symbol-recognition/blob/main/yolov11m%20training/care-symbols-yolov11m-v5/PR_curve.png" style="width:70%;height:70%"/>

### 6. Visualization
- Developed a [`draw_YOLO_boxes()`](https://github.com/athul-kurian/yolo-care-symbol-recognition/blob/main/visualize_outputs.py) utility using **Pillow (PIL)**.

---

## 🖼️ Samples
<img src="https://github.com/athul-kurian/yolo-care-symbol-recognition/blob/main/sample2.png" style="width:50%;height:50%"/>
<img src="https://github.com/athul-kurian/yolo-care-symbol-recognition/blob/main/sample1.png" style="width:50%;height:50%"/>
<img src="https://github.com/athul-kurian/yolo-care-symbol-recognition/blob/main/sample3.png" style="width:50%;height:50%"/>
<img src="https://github.com/athul-kurian/yolo-care-symbol-recognition/blob/main/sample4.png" style="width:50%;height:50%"/>





