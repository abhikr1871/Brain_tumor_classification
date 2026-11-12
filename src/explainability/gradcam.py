"""
Explainable AI (XAI): Grad-CAM Heatmap Visualizer.
Highlights discriminative regions within MRI scans that prompted the model's decision.
Essential for medical diagnostic credibility.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Optional

class GradCAM:
    """
    Hook-based Grad-CAM implementation compatible with PyTorch CNNs & Transfer Models.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Removes registered forward and backward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generates 2D heatmap [0, 1] for target class.
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        # Channel-wise mean of gradients (importance weights)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = cam.squeeze(0).squeeze(0).detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

def overlay_cam_on_image(
    original_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    is_rgb: bool = False,
) -> np.ndarray:
    """
    Overlays Grad-CAM heatmap onto the original RGB/BGR image and returns RGB.
    """
    h, w = original_image.shape[:2]
    heatmap = cv2.resize(cam, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    if len(original_image.shape) == 2:
        code = cv2.COLOR_GRAY2RGB if is_rgb else cv2.COLOR_GRAY2BGR
        original_image = cv2.cvtColor(original_image, code)

    if is_rgb:
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(heatmap_colored, alpha, original_image, 1 - alpha, 0)

    overlay = cv2.addWeighted(heatmap_colored, alpha, original_image, 1 - alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
