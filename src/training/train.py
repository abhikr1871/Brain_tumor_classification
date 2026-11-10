"""
Training Engine for Brain Tumor Classification.
Features:
- Automatic Mixed Precision (AMP) for speed and VRAM reduction
- Early Stopping and Best Model Checkpointing
- Learning Rate Scheduling (ReduceLROnPlateau)
- Progress reporting via tqdm
"""

import sys
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import DEVICE, CHECKPOINTS_DIR, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, USE_AMP
from src.training.loss_functions import FocalLoss

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training Batch", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if USE_AMP and device.type == "cuda":
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

def run_training(
    model: nn.Module,
    train_loader,
    val_loader,
    model_name: str = "brain_tumor_model",
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    device: torch.device = DEVICE
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)
    scaler = torch.amp.GradScaler(enabled=USE_AMP and device.type == "cuda")

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights_path = CHECKPOINTS_DIR / f"{model_name}_best.pth"

    print(f"\n[+] Starting training on {device} (AMP: {USE_AMP})")
    print(f"[+] Model checkpoint will be saved to: {best_weights_path}\n")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        scheduler.step(val_loss)

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

    print(f"\n[+] Training complete. Best model saved at {best_weights_path}")
    return history
