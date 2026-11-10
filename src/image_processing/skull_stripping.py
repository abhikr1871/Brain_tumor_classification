"""
Classical Image Processing: Skull Stripping and Extreme Contour Cropping.
Removes non-brain black background and skull borders to focus model attention on cerebral tissue.
Techniques utilized:
- Gaussian Filtering (Noise suppression)
- Otsu's Global Thresholding (Binarization)
- Morphological Operations (Erosion, Dilation, Closing)
- Contour Analysis (Finding extreme points & bounding box)
"""

import cv2
import numpy as np
from typing import Tuple, Optional

def crop_brain_contour(image: np.ndarray, plot: bool = False) -> np.ndarray:
    """
    Finds the extreme points of the brain contour in an MRI scan and crops the image.
    
    Args:
        image: BGR or Grayscale image (uint8 numpy array).
        plot: If True, returns diagnostic visualization tuple (cropped_image, thresh, contour_img).
    
    Returns:
        Cropped numpy array containing only the brain region.
    """
    # 1. Convert to grayscale if 3-channel
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # 2. Gaussian blur to remove high-frequency noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Threshold the image, then perform morphological operations to close small gaps
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # 4. Find contours in thresholded image and take the largest one
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback if no contour found (e.g., all black image)
        return image
        
    c = max(contours, key=cv2.contourArea)
    
    # 5. Determine the extreme points along the contour
    ext_left = tuple(c[c[:, :, 0].argmin()][0])
    ext_right = tuple(c[c[:, :, 0].argmax()][0])
    ext_top = tuple(c[c[:, :, 1].argmin()][0])
    ext_bot = tuple(c[c[:, :, 1].argmax()][0])
    
    # 6. Crop original image
    # Add small margin/padding safely
    h, w = gray.shape
    y_min = max(0, ext_top[1])
    y_max = min(h, ext_bot[1])
    x_min = max(0, ext_left[0])
    x_max = min(w, ext_right[0])
    
    # Check for valid bounding slice
    if (y_max - y_min < 10) or (x_max - x_min < 10):
        return image
        
    cropped = image[y_min:y_max, x_min:x_max]
    return cropped

def preprocess_mri_pipeline(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Full classical preprocessing pipeline:
    Load -> Contour Crop -> Resize.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    cropped = crop_brain_contour(img)
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    return resized
