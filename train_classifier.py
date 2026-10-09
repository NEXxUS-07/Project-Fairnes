"""
train_classifier.py — EfficientNet-B3 Training on HAM10000
===========================================================
Training features:
  • AdamW optimizer with layer-wise LR decay (backbone LR = base_lr * 0.1)
  • Cosine annealing with warm restarts (CosineAnnealingWarmRestarts)
  • Linear warm-up for first ``--warmup_epochs`` epochs
  • Mixed-precision training (torch.cuda.amp)
  • Gradient clipping (max_norm=1.0)
  • Early stopping (patience=10 on val macro-F1)
  • Saves best model to ``best_model.pth`` whenever val macro-F1 improves
  • Saves full ``training_history.json`` at end of training
  • Optional: starts with frozen backbone for first N epochs (gradual unfreeze)

Usage:
    # Basic (real HAM10000, no balancing):
    python train_classifier.py \\
        --train_csv splits/train.csv \\
        --val_csv   splits/val.csv \\
        --image_dir HAM10000_images/ \\
        --output_dir results/

    # Fairness-balanced dataset:
    python train_classifier.py \\
        --train_csv balanced_dataset.csv \\
        --val_csv   splits/val.csv \\
        --image_dir HAM10000_images/ \\
        --output_dir results/balanced/ \\
        --run_name   balanced

    # Resume from checkpoint:
    python train_classifier.py \\
        --train_csv splits/train.csv \\
        --val_csv   splits/val.csv \\
        --image_dir HAM10000_images/ \\
        --resume    results/best_model.pth

Tech stack: Python, PyTorch, timm, scikit-learn
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
)

from classifier import (
    EfficientNetB3Classifier,
    build_class_weights,
    build_loss,
    save_checkpoint,
    load_classifier,
    HAM_CLASSES, HAM_LABEL_MAP, NATIVE_IMAGE_SIZE,
)
from dataset import HAMDataset

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stop training if the monitored metric does not improve for ``patience``
    consecutive epochs.

    Parameters
    ----------
    patience   : int    epochs to wait before stopping.
    min_delta  : float  minimum change to be counted as improvement.
    mode       : "max" (higher = better) or "min" (lower = better).
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best       = float("-inf") if mode == "max" else float("inf")
        self.counter    = 0
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Update state and return True if training should stop.
        """
        improved = (
            value > self.best + self.min_delta if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, "
            f"best={self.best:.4f}, counter={self.counter})"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET ADAPTER  (balanced_dataset.csv → HAMDataset compatible)
# ═══════════════════════════════════════════════════════════════════════════

class BalancedCSVDataset(torch.utils.data.Dataset):
    """
    Loads a balanced_dataset.csv produced by fairness_balancer.py.
    Columns: image_path, label, label_idx, source, split.

    Returns the same 4-tuple as HAMDataset:
        (image_tensor, label_idx, lesion_id, skin_tone)

    Uses albumentations if available; otherwise torchvision transforms.
    """

    def __init__(
        self,
        csv_path:   str,
        split:      str = "train",
        image_size: int = NATIVE_IMAGE_SIZE,
    ):
        self.df    = pd.read_csv(csv_path)
        self.df.columns = [c.strip().lower() for c in self.df.columns]
        if "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.image_size = image_size
        self.split      = split
        self._build_transform()

    def _build_transform(self):
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            if self.split == "train":
                self.tfm = A.Compose([
                    A.Resize(self.image_size + 32, self.image_size + 32),
                    A.RandomCrop(self.image_size, self.image_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=30, p=0.6),
                    A.ColorJitter(brightness=0.15, contrast=0.15,
                                  saturation=0.15, hue=0.05, p=0.5),
                    A.CoarseDropout(max_holes=8, max_height=24,
                                    max_width=24, p=0.3),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
            else:
                self.tfm = A.Compose([
                    A.Resize(self.image_size + 32, self.image_size + 32),
                    A.CenterCrop(self.image_size, self.image_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
            self._use_albu = True
        except ImportError:
            import torchvision.transforms as T
            if self.split == "train":
                self.tfm = T.Compose([
                    T.Resize((self.image_size + 32, self.image_size + 32)),
                    T.RandomCrop(self.image_size),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.15, contrast=0.15,
                                  saturation=0.15, hue=0.05),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
            else:
                self.tfm = T.Compose([
                    T.Resize((self.image_size + 32, self.image_size + 32)),
                    T.CenterCrop(self.image_size),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
            self._use_albu = False

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        from PIL import Image
        import numpy as np
        row   = self.df.iloc[idx]
        label = int(row.get("label_idx", HAM_LABEL_MAP.get(str(row["label"]), 0)))
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), 0)

        if self._use_albu:
            t = self.tfm(image=np.array(img))["image"]
        else:
            t = self.tfm(img)
        return t, label, str(row.get("lesion_id", "")), str(row.get("skin_tone", ""))


# ═══════════════════════════════════════════════════════════════════════════
#  METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_epoch_metrics(
    all_preds:   np.ndarray,
    all_targets: np.ndarray,
    prefix:      str = "val",
) -> dict:
    """Compute Accuracy, macro-Precision, macro-Recall, macro-F1 per epoch."""
    acc    = accuracy_score(all_targets, all_preds)
    prec   = precision_score(all_targets, all_preds, average="macro",
                             zero_division=0)
    rec    = recall_score(all_targets, all_preds, average="macro",
                          zero_division=0)
    f1     = f1_score(all_targets, all_preds, average="macro",
                      zero_division=0)
    f1_per = f1_score(all_targets, all_preds, average=None,
                      labels=list(range(len(HAM_CLASSES))), zero_division=0)
    return {
        f"{prefix}_accuracy":      round(float(acc),  6),
        f"{prefix}_precision_macro": round(float(prec), 6),
        f"{prefix}_recall_macro":    round(float(rec),  6),
        f"{prefix}_f1_macro":        round(float(f1),   6),
        f"{prefix}_f1_per_class":    {
            HAM_CLASSES[i]: round(float(f1_per[i]), 6)
            for i in range(len(f1_per))
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  ONE EPOCH RUNNERS
# ═══════════════════════════════════════════════════════════════════════════

def run_train_epoch(
    model:     EfficientNetB3Classifier,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn:   nn.Module,
    scaler:    GradScaler,
    device:    torch.device,
    clip_grad: float = 1.0,
) -> tuple[float, dict]:
    """Run one training epoch. Returns (avg_loss, metrics_dict)."""
    model.train()
    total_loss   = 0.0
    all_preds    = []
    all_targets  = []

    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss  += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    metrics  = compute_epoch_metrics(
        np.array(all_preds), np.array(all_targets), prefix="train"
    )
    return avg_loss, metrics


@torch.no_grad()
def run_val_epoch(
    model:   EfficientNetB3Classifier,
    loader:  DataLoader,
    loss_fn: nn.Module,
    device:  torch.device,
) -> tuple[float, dict]:
    """Run one validation epoch. Returns (avg_loss, metrics_dict)."""
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = loss_fn(logits, labels)

        total_loss  += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    metrics  = compute_epoch_metrics(
        np.array(all_preds), np.array(all_targets), prefix="val"
    )
    return avg_loss, metrics


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    log.info("Device: %s", device)

    # ── Datasets ─────────────────────────────────────────────────────────
    use_balanced = args.train_csv.endswith("balanced_dataset.csv") or \
                   not args.image_dir
    if use_balanced and not args.image_dir:
        # balanced_dataset.csv has absolute image_path column
        train_ds = BalancedCSVDataset(args.train_csv, split="train",
                                      image_size=args.image_size)
        val_ds   = BalancedCSVDataset(args.val_csv,   split="val",
                                      image_size=args.image_size)
    else:
        train_ds = HAMDataset(args.image_dir, args.train_csv,
                              split="train", image_size=args.image_size)
        val_ds   = HAMDataset(args.image_dir, args.val_csv,
                              split="val",   image_size=args.image_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    log.info(
        "Train: %d samples (%d batches) | Val: %d samples (%d batches)",
        len(train_ds), len(train_loader), len(val_ds), len(val_loader),
    )

    # ── Class weights ────────────────────────────────────────────────────
    # Read label distribution from train CSV
    train_meta = pd.read_csv(args.train_csv)
    train_meta.columns = [c.strip().lower() for c in train_meta.columns]
    class_weights = build_class_weights(
        train_meta, label_col="label", device=device
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = EfficientNetB3Classifier(
        num_classes   = len(HAM_CLASSES),
        pretrained    = not args.no_pretrain,
        dropout_rate  = args.dropout,
        backbone_name = args.backbone,
    ).to(device)

    start_epoch = 1
    history     = []
    best_f1     = float("-inf")

    if args.resume:
        model, ckpt = load_classifier(args.resume, device=device, strict=False)
        model = model.to(device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_f1     = ckpt.get("metrics", {}).get("val_f1_macro", float("-inf"))
        log.info("Resuming from epoch %d, best val_f1=%.4f", start_epoch - 1, best_f1)

    # Optional: freeze backbone initially for gradual unfreezing
    if args.freeze_epochs > 0:
        model.freeze_backbone()

    log.info(model)

    # ── Loss ─────────────────────────────────────────────────────────────
    loss_fn = build_loss(
        weights         = class_weights,
        label_smoothing = args.label_smoothing,
    )

    # ── Optimizer (layer-wise LR) ─────────────────────────────────────────
    backbone_params = list(model.backbone.parameters())
    head_params     = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=args.weight_decay)

    # ── LR Schedule: linear warmup + cosine annealing ────────────────────
    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = args.warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler       = GradScaler(enabled=(device.type == "cuda"))
    early_stop   = EarlyStopping(patience=args.patience, mode="max")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()

        # Gradual backbone unfreeze
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            model.unfreeze_backbone()

        # Train
        train_loss, train_m = run_train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device,
            clip_grad=args.clip_grad,
        )
        # Validate
        val_loss, val_m = run_val_epoch(model, val_loader, loss_fn, device)
        # LR step
        scheduler.step()

        cur_lr  = optimizer.param_groups[1]["lr"]   # head LR
        elapsed = time.perf_counter() - t0
        val_f1  = val_m["val_f1_macro"]

        epoch_record = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "lr":         round(cur_lr, 8),
            **train_m,
            **val_m,
        }
        history.append(epoch_record)

        log.info(
            "Epoch %4d/%4d  "
            "loss=%.4f/%.4f  "
            "acc=%.4f/%.4f  "
            "f1=%.4f/%.4f  "
            "lr=%.2e  %.1fs",
            epoch, args.epochs,
            train_loss, val_loss,
            train_m["train_accuracy"], val_m["val_accuracy"],
            train_m["train_f1_macro"], val_f1,
            cur_lr, elapsed,
        )

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = out_dir / "best_model.pth"
            save_checkpoint(
                model=model, optimizer=optimizer, epoch=epoch,
                metrics={**val_m, "val_loss": round(val_loss, 6)},
                path=ckpt_path, scaler=scaler,
            )
            log.info("  ★  New best val_f1=%.4f — saved.", best_f1)

        # Early stopping
        if early_stop(val_f1):
            log.info(
                "Early stopping triggered at epoch %d (patience=%d, best=%.4f)",
                epoch, args.patience, early_stop.best,
            )
            break

    # ── Save history ──────────────────────────────────────────────────────
    history_path = out_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({"run_name": args.run_name, "history": history}, f, indent=2)
    log.info("Training history → %s", history_path)

    log.info("Training complete.  Best val macro-F1 = %.4f", best_f1)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train EfficientNet-B3 on HAM10000",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--train_csv",  required=True, help="CSV with image_path/label columns")
    p.add_argument("--val_csv",    required=True)
    p.add_argument("--image_dir",  default=None,  help="Image root (required for HAMDataset)")
    p.add_argument("--image_size", type=int, default=NATIVE_IMAGE_SIZE)
    p.add_argument("--num_workers",type=int, default=4)

    # Model
    p.add_argument("--backbone",   default="tf_efficientnet_b3.ns")
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--no_pretrain",action="store_true")

    # Training
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--backbone_lr_mult", type=float, default=0.1,
                   help="Backbone LR = lr * this multiplier")
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--clip_grad",  type=float, default=1.0)
    p.add_argument("--warmup_epochs",type=int,  default=3)
    p.add_argument("--freeze_epochs",type=int,  default=0,
                   help="Freeze backbone for first N epochs")

    # Early stopping
    p.add_argument("--patience",   type=int,   default=10)

    # Output
    p.add_argument("--output_dir", default="results")
    p.add_argument("--run_name",   default="baseline")
    p.add_argument("--resume",     default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
