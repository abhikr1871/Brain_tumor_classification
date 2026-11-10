# 🧠 Brain Tumor Classification from MRI Images
### *Image Processing & Computer Vision Semester Project*

An end-to-end framework combining **Classical Image Processing** (skull stripping, contour extraction, CLAHE, GLCM texture analysis) and **Modern Deep Learning / Transfer Learning** (Custom CNN, ResNet50, DenseNet121) with **Explainable AI (Grad-CAM)** and an interactive **Streamlit Diagnostic Web App**.

---

## 📁 Project Structure

```text
Brain-tumor-classification/
├── app/
│   └── streamlit_app.py           # Interactive web UI (Inference + Preprocessing + Grad-CAM)
├── data/
│   ├── raw/                       # Original MRI datasets (glioma, meningioma, notumor, pituitary)
│   ├── processed/                 # Cropped, skull-stripped, and contrast-enhanced images
│   ├── augmented/                 # Offline augmented samples
│   └── splits/                    # Train/Val/Test CSV split metadata
├── models_checkpoints/            # Saved PyTorch (.pth) model weights
├── notebooks/                     # Exploratory analysis & experiments
├── outputs/
│   ├── figures/                   # Confusion matrices, ROC curves, Grad-CAM overlays
│   ├── logs/                      # Training loss/accuracy logs
│   └── reports/                   # Quantitative metrics & evaluation reports
├── src/
│   ├── config.py                  # Global configurations, paths, hyperparams, device
│   ├── data/
│   │   ├── dataset.py             # PyTorch Dataset and DataLoader builders
│   │   ├── download.py            # Kaggle API automated downloader
│   │   └── transforms.py          # Data augmentation & image preprocessing transforms
│   ├── explainability/
│   │   └── gradcam.py             # Grad-CAM heatmap generator & overlay visualizer
│   ├── image_processing/
│   │   ├── enhancement.py         # CLAHE, Bilateral filtering, Gamma adjustment
│   │   ├── feature_extraction.py  # GLCM Haralick features, LBP, statistical moments
│   │   └── skull_stripping.py     # Otsu thresholding, contour finding, extreme point crop
│   ├── models/
│   │   ├── classical_ml.py        # SVM (RBF) & Random Forest on handcrafted features
│   │   ├── custom_cnn.py          # Custom 4-stage convolutional neural network
│   │   └── transfer_models.py     # Pretrained ResNet50, EfficientNet, DenseNet121
│   ├── training/
│   │   ├── evaluate.py            # Metrics: Confusion matrix, Precision, Recall, F1
│   │   ├── loss_functions.py      # Cross-Entropy & Focal Loss
│   │   └── train.py               # AMP training loop, early stopping, scheduler
│   └── utils/                     # Helper routines, logging, visualization utilities
├── requirements.txt               # Complete Python dependencies
└── .gitignore                     # Git hygiene for data and large model weights
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
You can download the recommended 4-class dataset (Masoud Nickparvar / Sartaj Bhuvaji) automatically via Kaggle API:
```bash
python src/data/download.py
```
*Or manually download from Kaggle (`masoudnickparvar/brain-tumor-mri-dataset`) and unzip into `data/raw/`.*

### 3. Launch Interactive Streamlit App
```bash
streamlit run app/streamlit_app.py
```

---

## 🔬 Syllabus & Technical Mapping

| Topic Domain | Techniques Implemented | File Reference |
| :--- | :--- | :--- |
| **Classical Image Processing** | Otsu Thresholding, Morphological Ops, Extreme Contours | [`src/image_processing/skull_stripping.py`](file:///c:/Users/abhij/OneDrive/Desktop/Brain-tumor-%20classification/src/image_processing/skull_stripping.py) |
| **Image Enhancement** | CLAHE, Bilateral Filtering, Gamma Correction | [`src/image_processing/enhancement.py`](file:///c:/Users/abhij/OneDrive/Desktop/Brain-tumor-%20classification/src/image_processing/enhancement.py) |
| **Handcrafted Features** | GLCM (Haralick), Local Binary Patterns (LBP) | [`src/image_processing/feature_extraction.py`](file:///c:/Users/abhij/OneDrive/Desktop/Brain-tumor-%20classification/src/image_processing/feature_extraction.py) |
| **Classical ML Baselines** | SVM (RBF), Random Forest Classifier | [`src/models/classical_ml.py`](file:///c:/Users/abhij/OneDrive/Desktop/Brain-tumor-%20classification/src/models/classical_ml.py) |
| **Deep Learning** | Custom 4-stage CNN, ResNet50, DenseNet121 | [`src/models/custom_cnn.py`](file:///c:/Users/abhij/OneDrive/Desktop/Brain-tumor-%20classification/src/models/custom_cnn.py), [`src/models/transfer_models.py`](file:///c:/Users/abhij/OneDrive/Desktop/Brain-tumor-%20classification/src/models/transfer_models.py) |
| **Loss & Regularization** | Focal Loss, Mixed Precision (AMP), Early Stopping | [`src/training/loss_functions.py`](file:///c:/Users/abhij/OneDrive/Desktop/Brain-tumor-%20classification/src/training/loss_functions.py), [`src/training/train.py`](file:///c:/Users/abhij/OneDrive/Desktop/Brain-tumor-%20classification/src/training/train.py) |
| **Explainable AI (XAI)** | Grad-CAM Saliency Maps & Overlay Heatmaps | [`src/explainability/gradcam.py`](file:///c:/Users/abhij/OneDrive/Desktop/Brain-tumor-%20classification/src/explainability/gradcam.py) |
| **Web Deployment** | Streamlit Interactive Diagnostic Application | [`app/streamlit_app.py`](file:///c:/Users/abhij/OneDrive/Desktop/Brain-tumor-%20classification/app/streamlit_app.py) |
