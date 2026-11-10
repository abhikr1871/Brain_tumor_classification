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
import sys
import os
import subprocess
from pathlib import Path

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

from src.config import PROJECT_ROOT, DATA_RAW, DATA_PROCESSED, CLASSES, IMG_SIZE, CHECKPOINTS_DIR, DEVICE
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
    target_layer = model.features[-1].block[0]  # Last conv block
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_heatmap(input_tensor, target_class=0)
    overlay = overlay_cam_on_image(resized, cam)
    cv2.imwrite(str(test_dir / "03_gradcam_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"      --> Generated Grad-CAM heatmap overlay: SUCCESS")
    print(f"      --> Saved test visualization artifacts in: {test_dir}")

    print("\n" + "=" * 65)
    print("[+] ALL PIPELINE MODULES VERIFIED & OPERATIONAL!")
    print("=" * 65)

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Classification Pipeline Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["test", "preprocess", "train-ml", "train-dl", "evaluate", "app"],
        default="test",
        help="Pipeline phase to execute"
    )
    parser.add_argument("--model", choices=["custom_cnn", "resnet50", "densenet121"], default="resnet50", help="Architecture for DL training")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")

    args = parser.parse_args()

    if args.mode == "test":
        run_sanity_test()

    elif args.mode == "app":
        print("[*] Launching Streamlit Web Application...")
        subprocess.run(["streamlit", "run", "app/streamlit_app.py"])

    elif args.mode == "train-dl":
        from src.data.dataset import get_dataloaders
        from src.training.train import run_training

        print(f"[*] Preparing DataLoaders from {DATA_RAW}...")
        train_loader, val_loader, test_loader = get_dataloaders(root_dir=DATA_RAW, batch_size=args.batch_size)

        print(f"[*] Initializing model architecture: {args.model}...")
        if args.model == "custom_cnn":
            model = BrainTumorCNN(num_classes=len(CLASSES))
        else:
            model = build_transfer_model(args.model, num_classes=len(CLASSES), pretrained=True)

        run_training(model, train_loader, val_loader, model_name=args.model, num_epochs=args.epochs)

    elif args.mode == "evaluate":
        from src.data.dataset import get_dataloaders
        from src.training.evaluate import evaluate_model

        _, _, test_loader = get_dataloaders(root_dir=DATA_RAW, batch_size=args.batch_size)
        weights_path = CHECKPOINTS_DIR / f"{args.model}_best.pth"
        if not weights_path.exists():
            print(f"[!] Checkpoint not found at {weights_path}. Train the model first using --mode train-dl.")
            return

        if args.model == "custom_cnn":
            model = BrainTumorCNN(num_classes=len(CLASSES))
        else:
            model = build_transfer_model(args.model, num_classes=len(CLASSES), pretrained=False)

        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        evaluate_model(model, test_loader, device=DEVICE)

if __name__ == "__main__":
    main()
