"""
Contrast Enhancement, Frequency-Domain Filtering, Edge Extraction, and Denoising for MRI Scans.
Includes:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Bilateral Filtering (Edge-preserving noise reduction)
- Gamma Correction
- Homomorphic Frequency-Domain Filtering (Butterworth high-pass emphasis)
- Canny / Sobel Edge Extraction
"""

from typing import Callable

import cv2
import numpy as np


def _apply_to_luminance(image: np.ndarray, fn: Callable[[np.ndarray], np.ndarray], color_space: str = "BGR") -> np.ndarray:
    """Applies a single-channel uint8 enhancement function to grayscale or LAB L-channel."""
    if len(image.shape) == 3:
        is_rgb = color_space.upper() == "RGB"
        to_lab = cv2.COLOR_RGB2LAB if is_rgb else cv2.COLOR_BGR2LAB
        from_lab = cv2.COLOR_LAB2RGB if is_rgb else cv2.COLOR_LAB2BGR
        lab = cv2.cvtColor(image, to_lab)
        l_chan, a_chan, b_chan = cv2.split(lab)
        enhanced_l = fn(l_chan)
        return cv2.cvtColor(cv2.merge((enhanced_l, a_chan, b_chan)), from_lab)
    return fn(image)


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
    return _apply_to_luminance(image, clahe.apply, color_space=color_space)


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


def apply_homomorphic_filter(
    image: np.ndarray,
    cutoff: float = 30.0,
    gamma_low: float = 0.6,
    gamma_high: float = 1.5,
    color_space: str = "BGR",
) -> np.ndarray:
    """
    Applies 2D FFT Homomorphic Filtering to normalize low-frequency MRI bias fields
    while boosting high-frequency lesion boundaries.
    """
    def _filter_channel(channel: np.ndarray) -> np.ndarray:
        rows, cols = channel.shape
        log_img = np.log1p(channel.astype(np.float32))
        fft_shifted = np.fft.fftshift(np.fft.fft2(log_img))

        u = np.arange(rows) - rows / 2.0
        v = np.arange(cols) - cols / 2.0
        vv, uu = np.meshgrid(v, u)
        d2 = uu ** 2 + vv ** 2
        h_filt = (gamma_high - gamma_low) * (1.0 - np.exp(-d2 / (2.0 * (cutoff ** 2)))) + gamma_low

        filtered = np.real(np.fft.ifft2(np.fft.ifftshift(fft_shifted * h_filt)))
        exp_img = np.expm1(filtered)
        exp_min, exp_max = float(exp_img.min()), float(exp_img.max())
        if exp_max - exp_min > 1e-6:
            scaled = (exp_img - exp_min) / (exp_max - exp_min) * 255.0
        else:
            scaled = np.zeros_like(exp_img)
        return np.clip(scaled, 0, 255).astype(np.uint8)

    return _apply_to_luminance(image, _filter_channel, color_space=color_space)


def extract_edge_map(
    image: np.ndarray,
    method: str = "canny",
    low_thresh: int = 50,
    high_thresh: int = 150,
    color_space: str = "BGR",
) -> np.ndarray:
    """
    Extracts structural edge maps using Canny or Sobel gradient magnitude.
    """
    if len(image.shape) == 3:
        code = cv2.COLOR_RGB2GRAY if color_space.upper() == "RGB" else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(image, code)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    if method.lower() == "sobel":
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        max_val = float(mag.max())
        if max_val > 1e-6:
            return np.clip((mag / max_val) * 255.0, 0, 255).astype(np.uint8)
        return np.zeros_like(gray, dtype=np.uint8)

    return cv2.Canny(blurred, low_thresh, high_thresh)
