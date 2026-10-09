"""
train_ft.py — Fine-tuning with frozen backbone warm-up
=======================================================
Strategy:
  1. Epochs 1-5: Freeze backbone, train only head with high LR (1e-3)
  2. Epochs 6+: Unfreeze backbone, train all with lower LR (1e-4)

This is a standard approach for fine-tuning large pretrained models.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from classifier import EfficientNetB3Classifier, HAM_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train_ft")

IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class SkinDataset:
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), 0)
        if self.transform:
            img = self.transform(img)
        return img, int(row["label_idx"])


def get_transforms(split):
    import torchvision.transforms as T
    if split == "train":
        return T.Compose([
            T.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16)),
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([
        T.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16)),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1


def set_trainable(model, freeze_backbone=True):
    """Freeze or unfreeze backbone parameters."""
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = not freeze_backbone
        else:
            param.requires_grad = True


def main(args):
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    log.info("Device: CPU (%d threads)", args.threads)

    # Load data
    train_df = pd.read_csv("splits/train.csv")
    val_df = pd.read_csv("splits/val.csv")
    test_df = pd.read_csv("splits/test.csv")

    train_ds = SkinDataset(train_df, get_transforms("train"))
    val_ds = SkinDataset(val_df, get_transforms("val"))
    test_ds = SkinDataset(test_df, get_transforms("val"))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    # Model
    log.info("Building EfficientNet-B3...")
    model = EfficientNetB3Classifier(
        num_classes=7, pretrained=True, dropout_rate=0.2, backbone_name="efficientnet_b3",
    ).to(device)

    # Phase 1: Train head only
    log.info("━━━ Phase 1: Train head only (epochs 1-%d) ━━━", args.warmup_epochs)
    set_trainable(model, freeze_backbone=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.head_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.warmup_epochs, eta_min=1e-5)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_f1 = -1.0
    best_path = Path("models/best_model.pth")

    for epoch in range(1, args.warmup_epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        elapsed = time.perf_counter() - t0
        log.info(
            "Phase1 Epoch %2d  tr_loss=%.4f  val_loss=%.4f  "
            "tr_acc=%.3f  val_acc=%.3f  tr_f1=%.3f  val_f1=%.3f  lr=%.2e  %.0fs",
            epoch, tr_loss, val_loss, tr_acc, val_acc, tr_f1, val_f1,
            optimizer.param_groups[0]["lr"], elapsed,
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_f1_macro": val_f1, "val_accuracy": val_acc,
                "backbone": "efficientnet_b3", "ham_classes": HAM_CLASSES,
                "image_size": IMAGE_SIZE,
            }, best_path)
            log.info("  ★ Saved best (F1=%.4f)", best_f1)

    # Phase 2: Fine-tune all
    log.info("━━━ Phase 2: Fine-tune all (epochs %d-%d) ━━━", args.warmup_epochs + 1, args.epochs)
    set_trainable(model, freeze_backbone=False)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": args.ft_lr},
        {"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": args.head_lr},
    ], weight_decay=1e-4)
    remaining = args.epochs - args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)

    for epoch in range(args.warmup_epochs + 1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        elapsed = time.perf_counter() - t0
        lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
        log.info(
            "Phase2 Epoch %2d  tr_loss=%.4f  val_loss=%.4f  "
            "tr_acc=%.3f  val_acc=%.3f  tr_f1=%.3f  val_f1=%.3f  lrs=[%s]  %.0fs",
            epoch, tr_loss, val_loss, tr_acc, val_acc, tr_f1, val_f1,
            ", ".join(lrs), elapsed,
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_f1_macro": val_f1, "val_accuracy": val_acc,
                "backbone": "efficientnet_b3", "ham_classes": HAM_CLASSES,
                "image_size": IMAGE_SIZE,
            }, best_path)
            log.info("  ★ Saved best (F1=%.4f)", best_f1)

    # Test
    log.info("Evaluating on test set...")
    model.load_state_dict(torch.load(best_path)["model_state"])
    test_loss, test_acc, test_f1 = validate(model, test_loader, loss_fn, device)
    log.info("TEST: loss=%.4f  acc=%.3f  F1=%.4f", test_loss, test_acc, test_f1)
    log.info("DONE. Best val F1: %.4f, Test F1: %.4f", best_f1, test_f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--ft_lr", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    main(args)
