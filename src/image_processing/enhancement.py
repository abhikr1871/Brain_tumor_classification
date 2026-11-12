"""
Contrast Enhancement and Image Denoising for MRI Scans.
Includes:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Bilateral Filtering (Edge-preserving noise reduction)
- Gamma Correction
"""

import cv2
import numpy as np

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
    color_space: str = "BGR",
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Increases local contrast in MRI scans without amplifying background noise.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 3:
        # Convert to LAB color space, apply CLAHE to L-channel, and merge back
        is_rgb = color_space.upper() == "RGB"
        to_lab = cv2.COLOR_RGB2LAB if is_rgb else cv2.COLOR_BGR2LAB
        from_lab = cv2.COLOR_LAB2RGB if is_rgb else cv2.COLOR_LAB2BGR
        lab = cv2.cvtColor(image, to_lab)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        enhanced_lab = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(enhanced_lab, from_lab)
    else:
        return clahe.apply(image)

def apply_bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """
    Applies Bilateral Filter to suppress noise while preserving tumor edges sharply.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def adjust_gamma(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """
    Applies Gamma correction for non-linear luminance/contrast adjustment.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
