"""
Dataset Downloader & Extraction Script.
Supports automated download via Kaggle API or manual placement instructions.
Primary Recommended Dataset:
- Masoud Nickparvar Brain Tumor MRI Dataset (7,023 images, 4 classes: glioma, meningioma, notumor, pituitary)
  Kaggle slug: 'masoudnickparvar/brain-tumor-mri-dataset'
Alternative Option:
- Sartaj Bhuvaji Brain Tumor Classification (3,264 images, 4 classes)
  Kaggle slug: 'sartajbhuvaji/brain-tumor-classification-mri'
"""

import os
import shutil
import zipfile
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_RAW

def download_kaggle_dataset(dataset_slug: str = "masoudnickparvar/brain-tumor-mri-dataset"):
    """
    Downloads and extracts dataset from Kaggle using the kaggle python package.
    Requires kaggle.json in ~/.kaggle/ or KAGGLE_USERNAME & KAGGLE_KEY environment variables.
    """
    print(f"[*] Checking Kaggle API credentials for dataset '{dataset_slug}'...")
    try:
        import kaggle
        print(f"[*] Downloading {dataset_slug} to {DATA_RAW}...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_slug, path=DATA_RAW, unzip=True)
        print(f"[+] Successfully downloaded and unzipped into {DATA_RAW}")
    except Exception as e:
        print(f"[-] Automated download failed: {e}")
        print("\n--- Manual Download Instructions ---")
        print("1. Go to: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        print("2. Click 'Download' (archive.zip).")
        print(f"3. Extract the contents directly into: {DATA_RAW}")
        print("   Expected structure:")
        print(f"   {DATA_RAW}/Training/glioma/...")
        print(f"   {DATA_RAW}/Training/meningioma/...")
        print(f"   {DATA_RAW}/Training/notumor/...")
        print(f"   {DATA_RAW}/Training/pituitary/...")
        print(f"   {DATA_RAW}/Testing/...")

if __name__ == "__main__":
    download_kaggle_dataset()
