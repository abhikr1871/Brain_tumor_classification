# 🧠 Brain Tumor Classification from MRI Images
### *Image Processing & Computer Vision Semester Project*

An end-to-end framework combining **Classical Image Processing** (skull stripping / extreme contour extraction, CLAHE, bilateral filtering, gamma adjustment, GLCM Haralick & LBP texture analysis), **Classical Machine Learning** (SVM RBF, Random Forest), and **Deep Learning / Transfer Learning** (Custom 4-Stage CNN, ResNet18, ResNet50, DenseNet121, EfficientNet-B0) with **Explainable AI (Grad-CAM)** and an interactive **Streamlit Diagnostic Web App**.

---

## 📁 Project Structure

```text
Brain_tumor_classification/
├── app/
│   └── streamlit_app.py           # Interactive web UI (Inference + Preprocessing + Grad-CAM + Benchmarks)
├── data/
│   ├── raw/                       # Original MRI dataset (Training/ & Testing/ across 4 classes)
│   ├── processed/                 # Preprocessed (contour-cropped, CLAHE, 224x224) PNG images
│   ├── augmented/                 # Optional offline augmented samples
│   └── splits/                    # Reproducible train.csv, val.csv, test.csv metadata
├── models_checkpoints/            # Saved PyTorch (.pth) & Scikit-Learn (.pkl) model weights
├── notebooks/
│   └── 01_project_walkthrough.ipynb # End-to-end visualization, Grad-CAM & benchmark notebook
├── outputs/
│   ├── figures/                   # Training curves, confusion matrices, ROC curves
│   ├── logs/                      # Per-epoch training history JSON logs
│   └── reports/                   # Quantitative test evaluation JSON reports
├── src/
│   ├── config.py                  # Global paths, classes, hyperparameters, device management
│   ├── data/
│   │   ├── dataset.py             # PyTorch Dataset, DataLoaders, and stratified split builder
│   │   ├── download.py            # Kaggle API automated downloader
│   │   └── transforms.py          # Preprocessing & stochastic augmentation transforms
│   ├── explainability/
│   │   └── gradcam.py             # Hook-based Grad-CAM heatmap generator & overlay visualizer
│   ├── image_processing/
│   │   ├── enhancement.py         # LAB CLAHE, Homomorphic FFT, Bilateral, Gamma, Canny/Sobel edges
│   │   ├── feature_extraction.py  # GLCM Haralick features, LBP histogram, intensity moments
│   │   ├── segmentation.py        # Otsu + morphological tumor segmentation & shape descriptors
│   │   └── skull_stripping.py     # Gaussian blur, thresholding, morphology & contour crop
│   ├── models/
│   │   ├── classical_ml.py        # Scaled SVM (RBF) & Random Forest pipelines
│   │   ├── custom_cnn.py          # Custom 4-stage CNN with BatchNorm & Dropout
│   │   └── transfer_models.py     # Pretrained ResNet18/50, EfficientNet-B0, DenseNet121
│   └── training/
│       ├── evaluate.py            # Classification report, confusion matrix, OvR ROC-AUC & curves
│       ├── loss_functions.py      # Cross-Entropy & Focal Loss
│       └── train.py               # Training loop, AMP, ReduceLROnPlateau, early stopping
├── tests/
│   └── test_pipeline.py           # Unit tests across IPCV, models, Grad-CAM, and splits
├── run_pipeline.py                # Unified CLI orchestrator for all pipeline stages
└── requirements.txt               # Python dependencies
```

---

## 📊 Dataset & Experimental Results

* **Dataset:** [Masoud Nickparvar Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (v2 — **7,200** scans across `glioma`, `meningioma`, `notumor`, `pituitary`, 1,800 images per class)
* **Splits:** `4,760` Train / `840` Val (85/15 stratified split of `Training/`) and `1,600` held-out Test (`Testing/`)

| Model | Approach | Test Accuracy | Macro F1 | Macro ROC-AUC |
| :--- | :--- | :---: | :---: | :---: |
| Custom CNN (15 epochs) | Deep Learning (from scratch) | 78.25% | 0.7814 | 0.9353 |
| SVM (RBF) | Classical ML (GLCM + LBP + Stats) | 83.19% | 0.8268 | — |
| Random Forest | Classical ML (GLCM + LBP + Stats) | 84.38% | 0.8386 | — |
| ResNet18 (15 epochs) | Transfer Learning (ImageNet) | 95.69% | 0.9563 | 0.9899 |
| **ResNet18 + TTA** | **Transfer Learning + Flip TTA** | **95.94%** | **0.9588** | **0.9906** |

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Sanity Check & Unit Tests
```bash
python run_pipeline.py --mode test
python -m unittest discover -s tests -v
```

### 3. Download & Preprocess Dataset
Download via Kaggle API (`python src/data/download.py`) or `kagglehub` into `data/raw/`, then run preprocessing once:
```bash
python run_pipeline.py --mode preprocess
```

### 4. Train & Evaluate Models
```bash
# Classical ML baselines (SVM & Random Forest)
python run_pipeline.py --mode train-ml

# Deep Learning (custom_cnn | resnet18 | resnet50 | efficientnet_b0 | densenet121)
python run_pipeline.py --mode train-dl --model resnet18 --epochs 15 --batch-size 16
python run_pipeline.py --mode evaluate --model resnet18 --tta
```
*(Optional: pass `--loss focal` to train with Focal Loss, or omit `--tta` for single-pass evaluation.)*

### 5. Launch Interactive Streamlit App
```bash
python run_pipeline.py --mode app
```

---

## 🔬 Syllabus & Technical Mapping

| Topic Domain | Techniques Implemented | File Reference |
| :--- | :--- | :--- |
| **Classical Image Processing** | Gaussian Blur, Thresholding, Morphology, Extreme Contours | [`src/image_processing/skull_stripping.py`](src/image_processing/skull_stripping.py) |
| **Image Enhancement & Edges** | LAB CLAHE, Homomorphic FFT, Bilateral, Gamma, Canny/Sobel | [`src/image_processing/enhancement.py`](src/image_processing/enhancement.py) |
| **Morphological Segmentation** | Otsu + Morphological Masking & 2D Shape Descriptors | [`src/image_processing/segmentation.py`](src/image_processing/segmentation.py) |
| **Handcrafted Features** | GLCM (Haralick), Local Binary Patterns (LBP), Moments | [`src/image_processing/feature_extraction.py`](src/image_processing/feature_extraction.py) |
| **Classical ML Baselines** | StandardScaler + SVM (RBF), Random Forest Classifier | [`src/models/classical_ml.py`](src/models/classical_ml.py) |
| **Deep Learning** | Custom 4-stage CNN, ResNet18/50, EfficientNet-B0, DenseNet121 | [`src/models/custom_cnn.py`](src/models/custom_cnn.py), [`src/models/transfer_models.py`](src/models/transfer_models.py) |
| **Loss & Regularization** | Cross-Entropy, Focal Loss, AMP, ReduceLROnPlateau, Early Stopping | [`src/training/loss_functions.py`](src/training/loss_functions.py), [`src/training/train.py`](src/training/train.py) |
| **Explainable AI (XAI)** | Hook-based Grad-CAM Saliency Maps & Overlays | [`src/explainability/gradcam.py`](src/explainability/gradcam.py) |
| **Web Deployment** | Interactive Streamlit Diagnostic & Benchmark Dashboard | [`app/streamlit_app.py`](app/streamlit_app.py) |
