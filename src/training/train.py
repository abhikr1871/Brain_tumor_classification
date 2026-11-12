"""
Training Engine for Brain Tumor Classification.
Features:
- Automatic Mixed Precision (AMP) for speed and VRAM reduction
- Early Stopping and Best Model Checkpointing
- Learning Rate Scheduling (ReduceLROnPlateau)
- Progress reporting via tqdm
"""

import json
import sys
from pathlib import Path
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICE,
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    USE_AMP,
    SEED,
)
from src.training.loss_functions import FocalLoss

def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp: bool):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training Batch", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Validation Batch", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def _save_training_curves(history: dict, model_name: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc")
    axes[1].set_title(f"{model_name} - Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model_name}_training_curves.png", dpi=200)
    plt.close(fig)

def run_training(
    model: nn.Module,
    train_loader,
    val_loader,
    model_name: str = "brain_tumor_model",
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    device: torch.device = DEVICE,
    loss_name: str = "cross_entropy",
):
    torch.manual_seed(SEED)
    model = model.to(device)
    criterion = FocalLoss() if loss_name == "focal" else nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights_path = CHECKPOINTS_DIR / f"{model_name}_best.pth"

    print(f"\n[+] Starting training on {device} (AMP: {use_amp}, Loss: {loss_name})")
    print(f"[+] Model checkpoint will be saved to: {best_weights_path}\n")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != prev_lr:
            print(f"  --> Learning rate reduced: {prev_lr:.2e} -> {new_lr:.2e}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc
            }, best_weights_path)
            print(f"  --> Saved new best checkpoint (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[!] Early stopping triggered after {epoch} epochs.")
                break

    history_path = LOGS_DIR / f"{model_name}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    _save_training_curves(history, model_name)

    print(f"\n[+] Training complete. Best model saved at {best_weights_path}")
    print(f"[+] Training history saved to {history_path}")
    return history
