"""
PyTorch Dataset and DataLoader Builders for Brain Tumor MRI Scans.
Supports directory structures with train/test splits or automated stratified k-fold splits.
"""

from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_RAW,
    DATA_PROCESSED,
    DATA_SPLITS,
    CLASSES,
    CLASS_TO_IDX,
    BATCH_SIZE,
    NUM_WORKERS,
    IMG_SIZE,
    SEED,
)
from src.data.transforms import get_train_transforms, get_val_transforms

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}


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
    file_paths: List[Path] = []
    labels: List[int] = []

    if not root_dir.exists():
        return file_paths, labels

    for p in sorted(root_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in VALID_EXTENSIONS:
            continue
        rel_parts_lower = [part.lower() for part in p.relative_to(root_dir).parts[:-1]]
        for cls_name, cls_idx in CLASS_TO_IDX.items():
            if cls_name.lower() in rel_parts_lower:
                file_paths.append(p)
                labels.append(cls_idx)
                break

    return file_paths, labels


def _find_subdir(root_dir: Path, name: str) -> Optional[Path]:
    """Finds a direct child directory matching name case-insensitively."""
    if not root_dir.exists():
        return None
    for child in root_dir.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def resolve_dataset_root(prefer_processed: bool = True) -> Tuple[Path, bool]:
    """
    Returns (dataset_root, is_preprocessed).
    Uses DATA_PROCESSED when populated; otherwise falls back to DATA_RAW.
    """
    if prefer_processed and DATA_PROCESSED.exists():
        for p in DATA_PROCESSED.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                return DATA_PROCESSED, True
    return DATA_RAW, False


def _save_split_csv(paths: List[Path], labels: List[int], root_dir: Path, csv_path: Path) -> None:
    rows = [
        {
            "relative_path": str(p.relative_to(root_dir)),
            "label": label,
            "class_name": CLASSES[label],
        }
        for p, label in zip(paths, labels)
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def create_data_splits(
    root_dir: Path = DATA_RAW,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = SEED,
    splits_dir: Path = DATA_SPLITS,
) -> Tuple[Tuple[List[Path], List[int]], Tuple[List[Path], List[int]], Tuple[List[Path], List[int]]]:
    """
    Builds reproducible Train/Val/Test splits and saves CSV metadata to data/splits/.
    Respects existing Training/ and Testing/ subfolders when present.
    """
    train_dir = _find_subdir(root_dir, "Training")
    test_dir = _find_subdir(root_dir, "Testing")

    if train_dir is not None and test_dir is not None:
        train_val_paths, train_val_labels = load_dataset_paths(train_dir)
        test_paths, test_labels = load_dataset_paths(test_dir)
        if not train_val_paths or not test_paths:
            raise FileNotFoundError(f"Expected class images under both {train_dir} and {test_dir}.")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths,
            train_val_labels,
            test_size=val_split,
            random_state=seed,
            stratify=train_val_labels,
        )
    else:
        paths, labels = load_dataset_paths(root_dir)
        if not paths:
            raise FileNotFoundError(
                f"No images found under {root_dir}. Please run src/data/download.py or place dataset in data/raw."
            )
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            paths, labels, test_size=test_split, random_state=seed, stratify=labels
        )
        adjusted_val_ratio = val_split / (1.0 - test_split)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths,
            train_val_labels,
            test_size=adjusted_val_ratio,
            random_state=seed,
            stratify=train_val_labels,
        )

    splits_dir.mkdir(parents=True, exist_ok=True)
    _save_split_csv(train_paths, train_labels, root_dir, splits_dir / "train.csv")
    _save_split_csv(val_paths, val_labels, root_dir, splits_dir / "val.csv")
    _save_split_csv(test_paths, test_labels, root_dir, splits_dir / "test.csv")

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def get_dataloaders(
    root_dir: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = SEED,
    num_workers: int = NUM_WORKERS,
    apply_preprocessing: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits data into Stratified Train, Validation, and Test sets and returns DataLoaders.
    """
    if root_dir is None:
        root_dir, is_preprocessed = resolve_dataset_root()
    else:
        root_dir = Path(root_dir)
        is_preprocessed = root_dir.resolve() == DATA_PROCESSED.resolve()

    if apply_preprocessing is None:
        apply_preprocessing = not is_preprocessed

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = create_data_splits(
        root_dir=root_dir, val_split=val_split, test_split=test_split, seed=seed
    )

    # Build datasets
    train_ds = BrainTumorDataset(
        train_paths,
        train_labels,
        transform=get_train_transforms(IMG_SIZE, apply_preprocessing=apply_preprocessing),
    )
    val_ds = BrainTumorDataset(
        val_paths,
        val_labels,
        transform=get_val_transforms(IMG_SIZE, apply_preprocessing=apply_preprocessing),
    )
    test_ds = BrainTumorDataset(
        test_paths,
        test_labels,
        transform=get_val_transforms(IMG_SIZE, apply_preprocessing=apply_preprocessing),
    )

    # Build loaders
    pin_mem = torch.cuda.is_available()
    persist = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persist,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persist,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persist,
    )

    return train_loader, val_loader, test_loader
