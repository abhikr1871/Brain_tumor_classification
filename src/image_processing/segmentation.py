"""
Classical Image Processing: Morphological Tumor Segmentation & Shape Features.
Isolates hyperintense lesion candidates inside the cerebral region using:
- Otsu Thresholding + Interior Brain Masking (excludes bright skull rim)
- Morphological Opening & Closing
- Contour Shape Analysis (Area, Perimeter, Circularity, Solidity, Eccentricity)
"""

from typing import Dict, Tuple

import cv2
import numpy as np


def _to_gray(image: np.ndarray, is_rgb: bool = False) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    code = cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, code)


def segment_tumor_region(
    image: np.ndarray,
    is_rgb: bool = False,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.35,
) -> np.ndarray:
    """
    Segments candidate hyperintense tumor regions inside the brain interior.
    Returns a binary uint8 mask (0 or 255) matching image height and width.
    """
    gray = _to_gray(image, is_rgb=is_rgb)
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Build interior brain mask to ignore bright outer skull boundaries
    _, rough_brain = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(rough_brain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8)

    brain_mask = np.zeros((h, w), dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(brain_mask, [largest], -1, 255, thickness=cv2.FILLED)

    erode_k = max(5, (min(h, w) // 24) | 1)
    interior_mask = cv2.erode(
        brain_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    )

    brain_vals = blurred[interior_mask > 0]
    if brain_vals.size < 50 or float(brain_vals.std()) < 5.0:
        return np.zeros((h, w), dtype=np.uint8)

    # Combine Otsu threshold and statistical hyperintensity threshold
    otsu_val, _ = cv2.threshold(
        brain_vals.reshape(-1, 1).astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    stat_thresh = float(brain_vals.mean() + 1.5 * brain_vals.std())
    thresh_val = min(245.0, max(float(otsu_val), stat_thresh))

    candidate = np.where((blurred >= thresh_val) & (interior_mask > 0), 255, 0).astype(np.uint8)

    # Morphological opening & closing to remove speckles and fill internal gaps
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k_open)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k_close)

    cand_contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brain_area = float(np.count_nonzero(interior_mask))
    min_area = min_area_ratio * brain_area
    max_area = max_area_ratio * brain_area

    final_mask = np.zeros((h, w), dtype=np.uint8)
    valid = [c for c in cand_contours if min_area <= cv2.contourArea(c) <= max_area]
    if valid:
        best = max(valid, key=cv2.contourArea)
        cv2.drawContours(final_mask, [best], -1, 255, thickness=cv2.FILLED)

    return final_mask


def extract_morphological_features(mask: np.ndarray) -> Dict[str, float]:
    """
    Computes 2D morphological shape descriptors from a binary segmentation mask.
    """
    h, w = mask.shape[:2]
    total_pixels = float(max(1, h * w))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "tumor_area_px": 0.0,
            "tumor_area_ratio": 0.0,
            "tumor_perimeter_px": 0.0,
            "circularity": 0.0,
            "solidity": 0.0,
            "eccentricity": 0.0,
            "aspect_ratio": 0.0,
        }

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perimeter = float(cv2.arcLength(c, True))
    circularity = min(1.0, (4.0 * np.pi * area) / (perimeter ** 2)) if perimeter > 1e-6 else 0.0

    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 1e-6 else 0.0

    _, _, bw, bh = cv2.boundingRect(c)
    aspect_ratio = float(bw) / float(bh) if bh > 0 else 0.0

    eccentricity = 0.0
    if len(c) >= 5:
        _, (axis1, axis2), _ = cv2.fitEllipse(c)
        major = max(axis1, axis2)
        minor = min(axis1, axis2)
        if major > 1e-6:
            eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor / major) ** 2)))

    return {
        "tumor_area_px": area,
        "tumor_area_ratio": area / total_pixels,
        "tumor_perimeter_px": perimeter,
        "circularity": float(circularity),
        "solidity": float(solidity),
        "eccentricity": eccentricity,
        "aspect_ratio": aspect_ratio,
    }


def draw_segmentation_overlay(
    image: np.ndarray, mask: np.ndarray, is_rgb: bool = False, alpha: float = 0.35
) -> np.ndarray:
    """
    Overlays the segmented lesion mask and contour boundary onto the MRI and returns RGB.
    """
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif is_rgb:
        rgb = image.copy()
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if np.count_nonzero(mask) == 0:
        return rgb

    colored = rgb.copy()
    colored[mask > 0] = (0, 220, 130)
    blended = cv2.addWeighted(colored, alpha, rgb, 1.0 - alpha, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 255, 160), thickness=2)
    return blended

