"""
Data Augmentations and Image Transformations for MRI Scans.
Includes both classical image processing integration (CLAHE, skull crop)
and modern stochastic PyTorch augmentations.
"""

from torchvision import transforms
from PIL import Image
import numpy as np
import torch

from src.image_processing.skull_stripping import crop_brain_contour
from src.image_processing.enhancement import apply_clahe

class BrainContourCropTransform:
    """Torchvision-compatible transform for Otsu extreme contour cropping."""
    def __call__(self, img: Image.Image) -> Image.Image:
        np_img = np.array(img)
        cropped = crop_brain_contour(np_img)
        return Image.fromarray(cropped)

class CLAHETransform:
    """Torchvision-compatible transform for Contrast Limited Adaptive Histogram Equalization."""
    def __init__(self, clip_limit: float = 2.0):
        self.clip_limit = clip_limit

    def __call__(self, img: Image.Image) -> Image.Image:
        np_img = np.array(img)
        enhanced = apply_clahe(np_img, clip_limit=self.clip_limit, color_space="RGB")
        return Image.fromarray(enhanced)

def get_train_transforms(img_size: tuple = (224, 224), apply_preprocessing: bool = True) -> transforms.Compose:
    """Data augmentation pipeline for training."""
    transform_list = []
    if apply_preprocessing:
        transform_list.extend([
            BrainContourCropTransform(),
            CLAHETransform(clip_limit=2.0)
        ])
        
    transform_list.extend([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transforms.Compose(transform_list)

def get_val_transforms(img_size: tuple = (224, 224), apply_preprocessing: bool = True) -> transforms.Compose:
    """Deterministic validation & inference transformation pipeline."""
    transform_list = []
    if apply_preprocessing:
        transform_list.extend([
            BrainContourCropTransform(),
            CLAHETransform(clip_limit=2.0)
        ])
        
    transform_list.extend([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transforms.Compose(transform_list)
