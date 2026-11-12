"""Unit tests for Brain Tumor Classification IPCV & Deep Learning pipeline."""

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASSES
from src.data.dataset import create_data_splits
from src.explainability.gradcam import GradCAM, overlay_cam_on_image
from src.image_processing.enhancement import (
    adjust_gamma,
    apply_bilateral_filter,
    apply_clahe,
    apply_homomorphic_filter,
    extract_edge_map,
)
from src.image_processing.feature_extraction import extract_all_classical_features
from src.image_processing.segmentation import (
    draw_segmentation_overlay,
    extract_morphological_features,
    segment_tumor_region,
)
from src.image_processing.skull_stripping import crop_brain_contour
from src.models.custom_cnn import BrainTumorCNN
from src.models.transfer_models import build_transfer_model
from src.training.evaluate import evaluate_model
from src.training.loss_functions import FocalLoss


def _make_synthetic_mri() -> np.ndarray:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.ellipse(img, (128, 128), (85, 105), 0, 0, 360, (140, 140, 140), -1)
    cv2.circle(img, (150, 100), 20, (230, 230, 230), -1)
    return img


class TestImageProcessing(unittest.TestCase):
    def test_crop_brain_contour_reduces_black_borders(self):
        img = _make_synthetic_mri()
        cropped = crop_brain_contour(img)
        self.assertLess(cropped.shape[0], img.shape[0])
        self.assertLess(cropped.shape[1], img.shape[1])

    def test_crop_brain_contour_all_black_fallback(self):
        black = np.zeros((128, 128, 3), dtype=np.uint8)
        cropped = crop_brain_contour(black)
        self.assertEqual(cropped.shape, black.shape)

    def test_enhancement_preserves_shape_and_dtype(self):
        img = _make_synthetic_mri()
        for cs in ("BGR", "RGB"):
            out = apply_clahe(img, color_space=cs)
            self.assertEqual(out.shape, img.shape)
            self.assertEqual(out.dtype, np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.assertEqual(apply_clahe(gray).shape, gray.shape)
        self.assertEqual(apply_bilateral_filter(img).shape, img.shape)
        self.assertEqual(adjust_gamma(img, gamma=1.2).shape, img.shape)

        homo = apply_homomorphic_filter(img, color_space="RGB")
        self.assertEqual(homo.shape, img.shape)
        self.assertEqual(homo.dtype, np.uint8)

        for method in ("canny", "sobel"):
            edges = extract_edge_map(img, method=method, color_space="RGB")
            self.assertEqual(edges.shape, gray.shape)
            self.assertEqual(edges.dtype, np.uint8)

    def test_morphological_segmentation_and_shape_features(self):
        img = _make_synthetic_mri()
        mask = segment_tumor_region(img, is_rgb=False)
        self.assertEqual(mask.shape, img.shape[:2])
        self.assertGreater(int(np.count_nonzero(mask)), 0)

        feats = extract_morphological_features(mask)
        self.assertGreater(feats["tumor_area_px"], 0.0)
        self.assertGreater(feats["circularity"], 0.5)
        self.assertTrue(0.0 <= feats["eccentricity"] <= 1.0)

        overlay = draw_segmentation_overlay(img, mask, is_rgb=True)
        self.assertEqual(overlay.shape, img.shape)

        empty_feats = extract_morphological_features(np.zeros((64, 64), dtype=np.uint8))
        self.assertEqual(empty_feats["tumor_area_px"], 0.0)

    def test_classical_feature_vector(self):
        img = _make_synthetic_mri()
        vec = extract_all_classical_features(img)
        self.assertEqual(vec.ndim, 1)
        self.assertEqual(len(vec), 20)
        self.assertTrue(np.all(np.isfinite(vec)))


class TestModelsAndExplainability(unittest.TestCase):
    def test_custom_cnn_forward_and_gradcam(self):
        model = BrainTumorCNN(num_classes=len(CLASSES))
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (1, len(CLASSES)))
        self.assertTrue(hasattr(model, "target_gradcam_layer"))

        cam_gen = GradCAM(model, model.target_gradcam_layer)
        try:
            cam = cam_gen.generate_heatmap(x, target_class=0)
        finally:
            cam_gen.remove_hooks()

        self.assertEqual(cam.ndim, 2)
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertLessEqual(float(cam.max()), 1.0)
        self.assertEqual(len(cam_gen._handles), 0)

        overlay_bgr = overlay_cam_on_image(np.zeros((224, 224, 3), dtype=np.uint8), cam, is_rgb=False)
        overlay_rgb = overlay_cam_on_image(np.zeros((224, 224, 3), dtype=np.uint8), cam, is_rgb=True)
        self.assertEqual(overlay_bgr.shape, (224, 224, 3))
        self.assertEqual(overlay_rgb.shape, (224, 224, 3))

    def test_transfer_models_build(self):
        for arch in ("resnet18", "resnet50", "densenet121", "efficientnet_b0"):
            model = build_transfer_model(arch, num_classes=len(CLASSES), pretrained=False)
            self.assertTrue(hasattr(model, "target_gradcam_layer"))

    def test_focal_loss_scalar(self):
        loss_fn = FocalLoss()
        logits = torch.randn(4, len(CLASSES), requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(logits, targets)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_evaluate_model_with_tta(self):
        model = BrainTumorCNN(num_classes=len(CLASSES))
        ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 2, 3]))
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            import src.training.evaluate as eval_mod
            orig_reports, orig_figures = eval_mod.REPORTS_DIR, eval_mod.FIGURES_DIR
            eval_mod.REPORTS_DIR = Path(tmp)
            eval_mod.FIGURES_DIR = Path(tmp)
            try:
                res = evaluate_model(
                    model, loader, device=torch.device("cpu"), save_figures=False, model_name="unit", use_tta=True
                )
            finally:
                eval_mod.REPORTS_DIR = orig_reports
                eval_mod.FIGURES_DIR = orig_figures
            self.assertEqual(len(res["predictions"]), 4)
            self.assertTrue((Path(tmp) / "unit_tta_report.json").exists())


class TestDatasetSplits(unittest.TestCase):
    def test_create_data_splits_respects_training_testing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            splits_dir = Path(tmp) / "splits"
            dummy = np.zeros((32, 32, 3), dtype=np.uint8)

            for cls_name in CLASSES:
                (root / "Training" / cls_name).mkdir(parents=True, exist_ok=True)
                (root / "Testing" / cls_name).mkdir(parents=True, exist_ok=True)
                for i in range(5):
                    cv2.imwrite(str(root / "Training" / cls_name / f"tr_{i}.png"), dummy)
                for i in range(2):
                    cv2.imwrite(str(root / "Testing" / cls_name / f"te_{i}.png"), dummy)

            (tr_p, _), (va_p, _), (te_p, _) = create_data_splits(
                root_dir=root, val_split=0.2, seed=42, splits_dir=splits_dir
            )

            self.assertEqual(len(tr_p), 16)
            self.assertEqual(len(va_p), 4)
            self.assertEqual(len(te_p), 8)
            self.assertTrue(set(tr_p).isdisjoint(set(va_p)))
            self.assertTrue(set(tr_p).isdisjoint(set(te_p)))
            self.assertTrue(set(va_p).isdisjoint(set(te_p)))
            self.assertTrue((splits_dir / "train.csv").exists())
            self.assertTrue((splits_dir / "val.csv").exists())
            self.assertTrue((splits_dir / "test.csv").exists())


if __name__ == "__main__":
    unittest.main()

