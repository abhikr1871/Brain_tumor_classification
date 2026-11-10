"""
Central Configuration for Brain Tumor Classification Project.
Defines paths, image processing parameters, model hyperparameters, and execution device.
"""

from pathlib import Path
import torch

# Base Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_SPLITS = DATA_DIR / "splits"

CHECKPOINTS_DIR = PROJECT_ROOT / "models_checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure runtime directories exist
for p in [DATA_RAW, DATA_PROCESSED, DATA_SPLITS, CHECKPOINTS_DIR, FIGURES_DIR, LOGS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Dataset & Label Specifications (4-Class Kaggle / Figshare standard)
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

# Image Processing Parameters
IMG_SIZE = (224, 224)         # Standard input size for modern CNNs & ViTs
CLAHE_CLIP_LIMIT = 2.0         # Adaptive Histogram Equalization clip limit
CLAHE_GRID_SIZE = (8, 8)       # Tile grid size for CLAHE
MEDIAN_BLUR_KSIZE = 3          # Denoising kernel size

# Training Hyperparameters (Default tuned for local 4GB GPU & Google Colab)
SEED = 42
BATCH_SIZE = 16               # Conservative batch size to prevent CUDA OOM on 4GB VRAM
NUM_WORKERS = 2               # DataLoader parallel workers (safe default for Windows)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 25
PATIENCE = 7                  # Early stopping patience

# Hardware / Device Management
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()  # Automatic Mixed Precision for faster training & lower VRAM
