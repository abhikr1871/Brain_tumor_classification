"""
Evaluation & Diagnostic Metrics Engine.
Computes:
- Confusion Matrix & normalized confusion matrix
- Per-class Precision, Recall, F1-Score (Classification Report)
- Multi-class ROC curves & Area Under Curve (AUC)
- Saves diagnostic visual figures to outputs/figures/
"""

import json
import sys
from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import CLASSES, FIGURES_DIR, IMG_SIZE, REPORTS_DIR, DEVICE
from src.explainability.gradcam import GradCAM, overlay_cam_on_image


def save_confusion_matrix_plot(cm: np.ndarray, out_path: Path, title: str = "Brain Tumor Confusion Matrix") -> None:
    """Saves a labeled confusion matrix heatmap."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth Label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_error_analysis_grid(
    model: torch.nn.Module,
    file_paths: List[Path],
    targets: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    out_path: Path,
    device: torch.device = DEVICE,
    max_samples: int = 8,
) -> None:
    """Saves a grid of misclassified MRI samples with Grad-CAM saliency overlays."""
    wrong_indices = np.where(preds != targets)[0]
    if len(wrong_indices) == 0:
        return

    # Prioritize highest-confidence errors so failure modes are easy to inspect
    wrong_sorted = sorted(
        wrong_indices.tolist(),
        key=lambda idx: float(probs[idx, preds[idx]]),
        reverse=True,
    )[:max_samples]

    cols = min(4, len(wrong_sorted))
    rows = int(np.ceil(len(wrong_sorted) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    infer_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    target_layer = getattr(model, "target_gradcam_layer", None)
    cam_gen = GradCAM(model, target_layer) if target_layer is not None else None
    try:
        for ax, idx in zip(axes_flat, wrong_sorted):
            img_bgr = cv2.imread(str(file_paths[idx]))
            if img_bgr is None:
                ax.axis("off")
                continue
            img_rgb = cv2.cvtColor(cv2.resize(img_bgr, IMG_SIZE), cv2.COLOR_BGR2RGB)
            if cam_gen is not None:
                tensor = infer_tf(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
                with torch.enable_grad():
                    cam = cam_gen.generate_heatmap(tensor, target_class=int(preds[idx]))
                vis = overlay_cam_on_image(img_rgb, cam, is_rgb=True)
            else:
                vis = img_rgb
            ax.imshow(vis)
            conf = float(probs[idx, preds[idx]] * 100)
            ax.set_title(
                f"True: {CLASSES[int(targets[idx])]}\nPred: {CLASSES[int(preds[idx])]} ({conf:.1f}%)",
                fontsize=10,
            )
            ax.axis("off")
    finally:
        if cam_gen is not None:
            cam_gen.remove_hooks()

    for ax in axes_flat[len(wrong_sorted):]:
        ax.axis("off")

    plt.suptitle("Misclassification Error Analysis (Grad-CAM on Predicted Class)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close(fig)


@torch.no_grad()
def evaluate_model(
    model,
    test_loader,
    device=DEVICE,
    save_figures=True,
    model_name: str = "model",
    use_tta: bool = False,
):
    model = model.to(device)
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for images, labels in test_loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1)
        if use_tta:
            flipped_probs = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
            probs = 0.5 * (probs + flipped_probs)
        _, preds = probs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    report_key = f"{model_name}_tta" if use_tta else model_name

    # Classification Report
    report = classification_report(all_targets, all_preds, target_names=CLASSES, output_dict=False)
    report_dict = classification_report(all_targets, all_preds, target_names=CLASSES, output_dict=True)
    print(f"\n--- Test Evaluation Report ({report_key}) ---")
    print(report)

    macro_roc_auc = None
    try:
        macro_roc_auc = float(roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro"))
        print(f"Macro ROC-AUC (OvR): {macro_roc_auc:.4f}")
    except ValueError as exc:
        print(f"[!] Could not compute ROC-AUC: {exc}")

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)

    if save_figures:
        cm_path = FIGURES_DIR / f"{report_key}_confusion_matrix.png"
        save_confusion_matrix_plot(cm, cm_path, title=f"{report_key} Confusion Matrix")
        print(f"[+] Confusion matrix plot saved to {cm_path}")

        if macro_roc_auc is not None:
            y_bin = label_binarize(all_targets, classes=list(range(len(CLASSES))))
            plt.figure(figsize=(7, 6))
            for idx, cls_name in enumerate(CLASSES):
                fpr, tpr, _ = roc_curve(y_bin[:, idx], all_probs[:, idx])
                plt.plot(fpr, tpr, label=f"{cls_name} (AUC = {auc(fpr, tpr):.3f})")
            plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
            plt.title(f"{report_key} Multi-Class ROC Curves")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.tight_layout()
            roc_path = FIGURES_DIR / f"{report_key}_roc_curves.png"
            plt.savefig(roc_path, dpi=300)
            plt.close()
            print(f"[+] ROC curve plot saved to {roc_path}")

        file_paths = getattr(test_loader.dataset, "file_paths", None)
        if file_paths and len(file_paths) == len(all_targets):
            err_path = FIGURES_DIR / f"{report_key}_error_analysis.png"
            save_error_analysis_grid(
                model, file_paths, all_targets, all_preds, all_probs, err_path, device=device
            )
            if err_path.exists():
                print(f"[+] Error analysis grid saved to {err_path}")

    report_path = REPORTS_DIR / f"{report_key}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": report_key,
                "use_tta": use_tta,
                "macro_roc_auc": macro_roc_auc,
                "classification_report": report_dict,
            },
            f,
            indent=2,
        )
    print(f"[+] Evaluation metrics saved to {report_path}")

    return {
        "classification_report": report,
        "classification_report_dict": report_dict,
        "macro_roc_auc": macro_roc_auc,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "targets": all_targets,
        "probabilities": all_probs,
    }
