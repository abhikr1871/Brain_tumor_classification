"""
PyTorch Dataset and DataLoader Builders for Brain Tumor MRI Scans.
Supports directory structures with train/test splits or automated stratified k-fold splits.
"""

from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.config import DATA_RAW, CLASSES, CLASS_TO_IDX, BATCH_SIZE, NUM_WORKERS, IMG_SIZE
from src.data.transforms import get_train_transforms, get_val_transforms

class BrainTumorDataset(Dataset):
    """
    Custom Dataset for MRI image classification.
    """
    def __init__(self, file_paths: List[Path], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_dataset_paths(root_dir: Path = DATA_RAW) -> Tuple[List[Path], List[int]]:
    """
    Scans directory recursively to collect all image paths and their corresponding labels.
    Expected folder structure:
      root_dir/
         glioma/...
         meningioma/...
         notumor/...
         pituitary/...
      or Training / Testing subfolders.
    """
    file_paths = []
    labels = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}

    for cls_name, cls_idx in CLASS_TO_IDX.items():
        # Match case-insensitively across directories
        found = False
        for p in root_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in valid_extensions:
                # Check if class name is in the parent path
                parts_lower = [part.lower() for part in p.parts]
                if cls_name.lower() in parts_lower:
                    file_paths.append(p)
                    labels.append(cls_idx)
                    found = True

    return file_paths, labels

def get_dataloaders(
    root_dir: Path = DATA_RAW,
    batch_size: int = BATCH_SIZE,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    num_workers: int = NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits data into Stratified Train, Validation, and Test sets and returns DataLoaders.
    """
    paths, labels = load_dataset_paths(root_dir)
    
    if len(paths) == 0:
        raise FileNotFoundError(
            f"No images found under {root_dir}. Please run src/data/download.py or place dataset in data/raw."
        )

    # First split: Train + Val vs Test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, test_size=test_split, random_state=seed, stratify=labels
    )

    # Second split: Train vs Val
    adjusted_val_ratio = val_split / (1.0 - test_split)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=adjusted_val_ratio, random_state=seed, stratify=train_val_labels
    )

    # Build datasets
    train_ds = BrainTumorDataset(train_paths, train_labels, transform=get_train_transforms(IMG_SIZE))
    val_ds = BrainTumorDataset(val_paths, val_labels, transform=get_val_transforms(IMG_SIZE))
    test_ds = BrainTumorDataset(test_paths, test_labels, transform=get_val_transforms(IMG_SIZE))

    # Build loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader
