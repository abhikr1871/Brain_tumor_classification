"""
Classical Feature Extraction for MRI Scans.
Extracts handcrafted texture and morphological features for Classical ML Baselines (SVM, RF):
- Gray-Level Co-occurrence Matrix (GLCM) / Haralick Features
- Local Binary Patterns (LBP)
- First-order Intensity Statistics (Mean, Variance, Skewness, Kurtosis)
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis
from typing import Dict, List

def extract_glcm_features(gray_image: np.ndarray, distances: List[int] = [1, 3], angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> Dict[str, float]:
    """
    Computes Haralick texture features using Gray-Level Co-occurrence Matrix.
    Metrics: Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.
    """
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    features = {
        "contrast": float(graycoprops(glcm, 'contrast').mean()),
        "dissimilarity": float(graycoprops(glcm, 'dissimilarity').mean()),
        "homogeneity": float(graycoprops(glcm, 'homogeneity').mean()),
        "energy": float(graycoprops(glcm, 'energy').mean()),
        "correlation": float(graycoprops(glcm, 'correlation').mean()),
        "ASM": float(graycoprops(glcm, 'ASM').mean()),
    }
    return features

def extract_lbp_features(gray_image: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    """
    Computes normalized histogram of Local Binary Patterns (LBP).
    Captures micro-texture variations in brain lesions.
    """
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        
    lbp = local_binary_pattern(gray_image, P=P, R=R, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_intensity_statistics(gray_image: np.ndarray) -> Dict[str, float]:
    """
    Calculates first-order statistical moments from pixel intensities.
    """
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        
    pixels = gray_image.flatten()
    return {
        "mean_intensity": float(np.mean(pixels)),
        "std_intensity": float(np.std(pixels)),
        "skewness": float(skew(pixels)),
        "kurtosis": float(kurtosis(pixels))
    }

def extract_all_classical_features(image: np.ndarray) -> np.ndarray:
    """
    Combines GLCM, LBP, and Intensity moments into a unified 1D feature vector.
    """
    glcm_dict = extract_glcm_features(image)
    stat_dict = extract_intensity_statistics(image)
    lbp_hist = extract_lbp_features(image)
    
    vector = list(glcm_dict.values()) + list(stat_dict.values()) + lbp_hist.tolist()
    return np.array(vector, dtype=np.float32)
