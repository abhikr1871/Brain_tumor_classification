"""
Unified Orchestrator for Brain Tumor Classification Project.
Provides a single, intuitive entrypoint for all pipeline phases.

Usage:
    python run_pipeline.py --mode test            # Rapid pipeline sanity check (no download required)
    python run_pipeline.py --mode preprocess      # Run skull-stripping & CLAHE on data/raw/
    python run_pipeline.py --mode train-ml        # Train classical ML baseline (GLCM + SVM / RF)
    python run_pipeline.py --mode train-dl        # Train deep learning model (Custom CNN or ResNet50)
    python run_pipeline.py --mode evaluate        # Evaluate trained model on test split
    python run_pipeline.py --mode app             # Launch interactive Streamlit diagnostic UI
"""

import argparse
import json
import pickle
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

# Ensure UTF-8 output encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import cv2
import torch
from tqdm import tqdm

from src.config import (
    PROJECT_ROOT,
    DATA_RAW,
    DATA_PROCESSED,
    CLASSES,
    IMG_SIZE,
    CLAHE_CLIP_LIMIT,
    CLAHE_GRID_SIZE,
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    DEVICE,
)
from src.image_processing.skull_stripping import crop_brain_contour
from src.image_processing.enhancement import apply_clahe
from src.image_processing.feature_extraction import extract_all_classical_features
from src.models.custom_cnn import BrainTumorCNN
from src.models.transfer_models import build_transfer_model
from src.explainability.gradcam import GradCAM, overlay_cam_on_image

def generate_synthetic_sample(output_path: Path):
    """Generates an anatomically structured synthetic brain slice for pipeline verification."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Outer skull ellipse
    cv2.ellipse(img, (128, 128), (95, 115), 0, 0, 360, (70, 70, 70), -1)
    # Cerebral cortex / brain tissue
    cv2.ellipse(img, (128, 128), (85, 105), 0, 0, 360, (140, 140, 140), -1)
    # Ventricles (dark fluid)
    cv2.ellipse(img, (110, 125), (12, 35), 10, 0, 360, (30, 30, 30), -1)
    cv2.ellipse(img, (146, 125), (12, 35), -10, 0, 360, (30, 30, 30), -1)
    # Simulated hyperintense tumor mass (Glioma / Meningioma)
    cv2.circle(img, (160, 95), 22, (235, 235, 235), -1)
    cv2.imwrite(str(output_path), img)
    return img

def run_sanity_test():
    """Runs a complete end-to-end sanity check without requiring full dataset download."""
    print("=" * 65)
    print("[*] BRAIN TUMOR CLASSIFICATION: PIPELINE SANITY TEST")
    print("=" * 65)

    test_dir = PROJECT_ROOT / "outputs" / "test_run"
    test_dir.mkdir(parents=True, exist_ok=True)
    sample_img_path = test_dir / "sample_brain_mri.png"

    # Step 1: Create and verify synthetic MRI scan
    print("\n[1/5] Generating sample brain MRI scan...")
    img = generate_synthetic_sample(sample_img_path)
    print(f"      --> Saved synthetic test scan at: {sample_img_path}")

    # Step 2: Classical Image Processing (Skull Stripping & CLAHE)
    print("\n[2/5] Testing Classical Image Processing Pipeline...")
    cropped = crop_brain_contour(img)
    enhanced = apply_clahe(cropped, clip_limit=2.0)
    cv2.imwrite(str(test_dir / "01_cropped.png"), cropped)
    cv2.imwrite(str(test_dir / "02_clahe_enhanced.png"), enhanced)
    print(f"      --> Original shape: {img.shape} -> Cropped shape: {cropped.shape}")
    print(f"      --> Otsu Extreme Contour Crop & CLAHE: SUCCESS")

    # Step 3: Handcrafted Feature Extraction (GLCM + LBP)
    print("\n[3/5] Testing Classical Feature Extraction (GLCM Haralick + LBP)...")
    features = extract_all_classical_features(enhanced)
    print(f"      --> Extracted 1D feature vector length: {len(features)} descriptors")
    print(f"      --> Feature extraction: SUCCESS")

    # Step 4: Deep Learning Model Forward Pass
    print(f"\n[4/5] Testing Neural Network Forward Pass on {DEVICE}...")
    model = BrainTumorCNN(num_classes=len(CLASSES)).to(DEVICE)
    resized = cv2.resize(enhanced, IMG_SIZE)
    input_tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    print(f"      --> Output logits shape: {output.shape}")
    print(f"      --> Class probabilities: {dict(zip(CLASSES, [round(float(p), 4) for p in probs]))}")
    print(f"      --> Model forward pass: SUCCESS")

    # Step 5: Explainable AI (Grad-CAM)
    print("\n[5/5] Testing Explainable AI (Grad-CAM Saliency Map)...")
    target_layer = model.target_gradcam_layer
    gradcam = GradCAM(model, target_layer)
    try:
        cam = gradcam.generate_heatmap(input_tensor, target_class=0)
    finally:
        gradcam.remove_hooks()
    overlay = overlay_cam_on_image(resized, cam)
    cv2.imwrite(str(test_dir / "03_gradcam_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"      --> Generated Grad-CAM heatmap overlay: SUCCESS")
    print(f"      --> Saved test visualization artifacts in: {test_dir}")

    print("\n" + "=" * 65)
    print("[+] ALL PIPELINE MODULES VERIFIED & OPERATIONAL!")
    print("=" * 65)


def _preprocess_bgr_image(image: np.ndarray) -> np.ndarray:
    """Applies brain contour cropping, LAB CLAHE enhancement, and resizing."""
    cropped = crop_brain_contour(image)
    enhanced = apply_clahe(
        cropped,
        clip_limit=CLAHE_CLIP_LIMIT,
        tile_grid_size=CLAHE_GRID_SIZE,
        color_space="BGR",
    )
    return cv2.resize(enhanced, IMG_SIZE, interpolation=cv2.INTER_AREA)


def _build_model(model_name: str, pretrained: bool) -> torch.nn.Module:
    """Instantiates the requested neural network architecture."""
    if model_name == "custom_cnn":
        return BrainTumorCNN(num_classes=len(CLASSES))
    return build_transfer_model(model_name, num_classes=len(CLASSES), pretrained=pretrained)


def run_preprocessing() -> None:
    """Preprocesses all raw MRI images into data/processed/ and generates split CSVs."""
    from src.data.dataset import load_dataset_paths, create_data_splits

    raw_paths, _ = load_dataset_paths(DATA_RAW)
    if not raw_paths:
        raise FileNotFoundError(f"No images found under {DATA_RAW}. Download the dataset first.")

    print(f"[*] Preprocessing {len(raw_paths)} MRI scans from {DATA_RAW} -> {DATA_PROCESSED}...")
    saved, skipped = 0, 0
    for src_path in tqdm(raw_paths, desc="Preprocessing MRIs"):
        img = cv2.imread(str(src_path))
        if img is None:
            skipped += 1
            continue
        processed = _preprocess_bgr_image(img)
        dst_path = (DATA_PROCESSED / src_path.relative_to(DATA_RAW)).with_suffix(".png")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_path), processed)
        saved += 1

    (train_split, _), (val_split, _), (test_split, _) = create_data_splits(root_dir=DATA_PROCESSED)
    print(f"[+] Preprocessed {saved} images ({skipped} skipped) into {DATA_PROCESSED}")
    print(f"[+] Split counts -> Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")


def _extract_features_for_paths(paths: List[Path], apply_preprocessing: bool, desc: str) -> np.ndarray:
    """Extracts 1D handcrafted feature vectors for a list of MRI image paths."""
    features = []
    for img_path in tqdm(paths, desc=desc):
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        if apply_preprocessing:
            img = _preprocess_bgr_image(img)
        features.append(extract_all_classical_features(img))
    return np.vstack(features)


def run_classical_ml() -> None:
    """Trains and evaluates SVM (RBF) and Random Forest baselines on handcrafted features."""
    from src.data.dataset import resolve_dataset_root, create_data_splits
    from src.models.classical_ml import train_and_eval_classical
    from src.training.evaluate import save_confusion_matrix_plot

    root_dir, is_preprocessed = resolve_dataset_root()
    print(f"[*] Loading dataset splits from {root_dir} (preprocessed={is_preprocessed})...")
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = create_data_splits(
        root_dir=root_dir
    )

    fit_paths = train_paths + val_paths
    fit_labels = np.array(train_labels + val_labels)
    y_test = np.array(test_labels)

    X_train = _extract_features_for_paths(
        fit_paths, apply_preprocessing=not is_preprocessed, desc="Extracting train features"
    )
    X_test = _extract_features_for_paths(
        test_paths, apply_preprocessing=not is_preprocessed, desc="Extracting test features"
    )

    results = train_and_eval_classical(X_train, fit_labels, X_test, y_test, class_names=CLASSES)

    summary = {}
    for name, res in results.items():
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        ckpt_path = CHECKPOINTS_DIR / f"{slug}.pkl"
        with open(ckpt_path, "wb") as f:
            pickle.dump(res["model"], f)
        save_confusion_matrix_plot(
            res["confusion_matrix"],
            FIGURES_DIR / f"{slug}_confusion_matrix.png",
            title=f"{name} Confusion Matrix",
        )
        summary[name] = {"accuracy": res["accuracy"], "classification_report": res["report"]}
        print(f"[+] Saved {name} model -> {ckpt_path}")

    report_path = REPORTS_DIR / "classical_ml_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Classical ML comparison report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Classification Pipeline Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["test", "preprocess", "train-ml", "train-dl", "evaluate", "app"],
        default="test",
        help="Pipeline phase to execute"
    )
    parser.add_argument(
        "--model",
        choices=["custom_cnn", "resnet18", "resnet50", "efficientnet_b0", "densenet121"],
        default="resnet50",
        help="Architecture for DL training",
    )
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy", help="Loss function for DL training")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")

    args = parser.parse_args()

    if args.mode == "test":
        run_sanity_test()

    elif args.mode == "preprocess":
        run_preprocessing()

    elif args.mode == "train-ml":
        run_classical_ml()

    elif args.mode == "app":
        print("[*] Launching Streamlit Web Application...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])

    elif args.mode == "train-dl":
        from src.data.dataset import get_dataloaders, resolve_dataset_root
        from src.training.train import run_training

        root_dir, is_preprocessed = resolve_dataset_root()
        print(f"[*] Preparing DataLoaders from {root_dir} (preprocessed={is_preprocessed})...")
        train_loader, val_loader, _ = get_dataloaders(root_dir=root_dir, batch_size=args.batch_size)

        print(f"[*] Initializing model architecture: {args.model}...")
        model = _build_model(args.model, pretrained=True)
        run_training(
            model,
            train_loader,
            val_loader,
            model_name=args.model,
            num_epochs=args.epochs,
            loss_name=args.loss,
        )

    elif args.mode == "evaluate":
        from src.data.dataset import get_dataloaders, resolve_dataset_root
        from src.training.evaluate import evaluate_model

        root_dir, _ = resolve_dataset_root()
        _, _, test_loader = get_dataloaders(root_dir=root_dir, batch_size=args.batch_size)
        weights_path = CHECKPOINTS_DIR / f"{args.model}_best.pth"
        if not weights_path.exists():
            print(f"[!] Checkpoint not found at {weights_path}. Train the model first using --mode train-dl.")
            return

        model = _build_model(args.model, pretrained=False)
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        evaluate_model(model, test_loader, device=DEVICE, model_name=args.model)

if __name__ == "__main__":
    main()
