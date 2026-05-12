# NYCU Computer Vision 2026 HW3

- **Student ID:** 314551178  
- **Name:** 陳鎮成  

---

## Introduction

The model uses a **Mask R-CNN** architecture with a **CBAM-integrated ResNet-50 backbone** to detect and segment four types of cells from medical images.

---

## Environment Setup

It is recommended to use **Python 3.9 or higher**.

To install the required dependencies, please run:

```bash
pip install -r requirements.txt
```

> **Note:** Please ensure PyTorch is installed correctly with GPU support for your CUDA version before installing other dependencies.

---

## Usage

### 1. Data Preparation

Before running the scripts, please update the file paths in the source code to point to your local directories and desired file names.

#### In `train.py`

Update:

- `TRAIN_DIR`

#### In `inference.py`

Update:

- `TEST_DIR` (test images path)
- `JSON_PATH`
- `MODEL_PATHS` (paths to trained weights)

---

### 2. Training

To train the model, execute:

```bash
python train.py
```

This script trains the model for **100 epochs** using:

- `CosineAnnealingLR`
- Automatic Mixed Precision (AMP)

---

### 3. Inference

To run inference and generate final predictions on the test set:

```bash
python inference.py
```

The script performs:

- Ensemble inference
- Test-Time Augmentation (TTA)
- Segmentation prediction generation
- RLE format encoding

The final predictions will be saved to the specified output JSON file.

---

## Performance Snapshot
<img width="1443" height="376" alt="image" src="https://github.com/user-attachments/assets/0ff3ffde-b55a-4329-98b3-1bdd06ba7beb" />
