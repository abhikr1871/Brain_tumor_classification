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
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import CLASSES, FIGURES_DIR, REPORTS_DIR, DEVICE


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


@torch.no_grad()
def evaluate_model(model, test_loader, device=DEVICE, save_figures=True, model_name: str = "model"):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Classification Report
    report = classification_report(all_targets, all_preds, target_names=CLASSES, output_dict=False)
    report_dict = classification_report(all_targets, all_preds, target_names=CLASSES, output_dict=True)
    print("\n--- Test Evaluation Report ---")
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
        cm_path = FIGURES_DIR / f"{model_name}_confusion_matrix.png"
        save_confusion_matrix_plot(cm, cm_path, title=f"{model_name} Confusion Matrix")
        print(f"[+] Confusion matrix plot saved to {cm_path}")

        if macro_roc_auc is not None:
            y_bin = label_binarize(all_targets, classes=list(range(len(CLASSES))))
            plt.figure(figsize=(7, 6))
            for idx, cls_name in enumerate(CLASSES):
                fpr, tpr, _ = roc_curve(y_bin[:, idx], all_probs[:, idx])
                plt.plot(fpr, tpr, label=f"{cls_name} (AUC = {auc(fpr, tpr):.3f})")
            plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
            plt.title(f"{model_name} Multi-Class ROC Curves")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.tight_layout()
            roc_path = FIGURES_DIR / f"{model_name}_roc_curves.png"
            plt.savefig(roc_path, dpi=300)
            plt.close()
            print(f"[+] ROC curve plot saved to {roc_path}")

    report_path = REPORTS_DIR / f"{model_name}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": model_name, "macro_roc_auc": macro_roc_auc, "classification_report": report_dict},
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
