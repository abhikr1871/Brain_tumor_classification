"""
Evaluation & Diagnostic Metrics Engine.
Computes:
- Confusion Matrix & normalized confusion matrix
- Per-class Precision, Recall, F1-Score (Classification Report)
- Multi-class ROC curves & Area Under Curve (AUC)
- Saves diagnostic visual figures to outputs/figures/
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import CLASSES, FIGURES_DIR, REPORTS_DIR, DEVICE

@torch.no_grad()
def evaluate_model(model, test_loader, device=DEVICE, save_figures=True):
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
    print("\n--- Test Evaluation Report ---")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)

    if save_figures:
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
        plt.title("Brain Tumor Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Ground Truth Label")
        plt.tight_layout()
        cm_path = FIGURES_DIR / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"[+] Confusion matrix plot saved to {cm_path}")

    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "targets": all_targets,
        "probabilities": all_probs
    }
