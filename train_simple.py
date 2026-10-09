"""
train_simple.py — Simplified training for HAM10000
====================================================
Focuses on getting the model to actually learn rather than
over-engineering the training pipeline.

Key differences from train_fast.py:
  • Higher learning rate (1e-3 for head, 1e-4 for backbone)
  • No MixUp (harmful on balanced data)
  • Minimal label smoothing (0.05)
  • Simpler augmentations
  • Longer training (30 epochs) with cosine LR
  • More frequent evaluation
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

from classifier import (
    EfficientNetB3Classifier,
    build_class_weights,
    build_loss,
    HAM_CLASSES, HAM_LABEL_MAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train_simple")

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


def main(args):
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    log.info("Device: CPU (%d threads)", args.threads)

    # Load data
    train_df = pd.read_csv("splits/train.csv")
    val_df = pd.read_csv("splits/val.csv")
    test_df = pd.read_csv("splits/test.csv")
    log.info("Data: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    # Datasets
    train_ds = SkinDataset(train_df, get_transforms("train"))
    val_ds = SkinDataset(val_df, get_transforms("val"))
    test_ds = SkinDataset(test_df, get_transforms("val"))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
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
        num_classes=7,
        pretrained=True,
        dropout_rate=0.2,
        backbone_name="efficientnet_b3",
    ).to(device)

    # Differential learning rates
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)

    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Class weights (uniform for balanced data)
    class_weights = torch.ones(7, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # Training
    log.info("Starting training: %d epochs, batch=%d, lr=%.2e", args.epochs, args.batch_size, args.lr)
    best_f1 = -1.0
    best_path = Path("models/best_model.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        log.info(
            "Epoch %2d/%d  tr_loss=%.4f  val_loss=%.4f  "
            "tr_acc=%.3f  val_acc=%.3f  tr_f1=%.3f  val_f1=%.3f  lr=%.2e  %.0fs",
            epoch, args.epochs, tr_loss, val_loss, tr_acc, val_acc, tr_f1, val_f1, lr, elapsed,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_f1_macro": val_f1,
                "val_accuracy": val_acc,
                "backbone": "efficientnet_b3",
                "ham_classes": HAM_CLASSES,
                "image_size": IMAGE_SIZE,
            }, best_path)
            log.info("  ★ Saved best model (F1=%.4f)", best_f1)

    # Final test evaluation
    log.info("Evaluating on test set...")
    model.load_state_dict(torch.load(best_path)["model_state"])
    test_loss, test_acc, test_f1 = validate(model, test_loader, loss_fn, device)
    log.info("TEST: loss=%.4f  acc=%.3f  F1=%.4f", test_loss, test_acc, test_f1)

    log.info("DONE. Best val F1: %.4f, Test F1: %.4f", best_f1, test_f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    main(args)
