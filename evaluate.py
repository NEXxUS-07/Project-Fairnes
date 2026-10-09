"""
evaluate.py — Standalone evaluation script for HAM10000 checkpoints
====================================================================
Evaluates a trained classifier on the test split and prints:
  - Overall accuracy, precision, recall, F1 (macro)
  - Per-class F1 scores
  - Confusion matrix (printed + optional plot)
  - Inference latency statistics

Usage:
    python evaluate.py \
        --checkpoint models/best_model.pth \
        --test_csv splits/test.csv \
        --image_dir HAM10000_images \
        --batch_size 32 \
        --threads 4

Requirements:
    torch torchvision timm scikit-learn pandas pillow matplotlib seaborn
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from classifier import (
    EfficientNetB3Classifier,
    load_classifier,
    HAM_CLASSES, HAM_LABEL_MAP, NATIVE_IMAGE_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("evaluate")

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, image_size=224):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), 0)
        if self.transform:
            img = self.transform(img)
        return img, int(row["label_idx"])


def build_transforms(image_size=224):
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((image_size + 16, image_size + 16)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    latencies = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        t0 = time.perf_counter()
        logits = model(images)
        torch.cuda.synchronize() if device.type == "cuda" else None
        latencies.append(time.perf_counter() - t0)

        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    prec_macro = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec_macro = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1_per = f1_score(all_targets, all_preds, average=None,
                      labels=list(range(len(HAM_CLASSES))), zero_division=0)

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(HAM_CLASSES))))

    avg_latency = np.mean(latencies) * 1000 if latencies else 0
    p95_latency = np.percentile(latencies, 95) * 1000 if latencies else 0

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_per_class": {HAM_CLASSES[i]: float(f1_per[i]) for i in range(len(HAM_CLASSES))},
        "confusion_matrix": cm.tolist(),
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "classification_report": classification_report(
            all_targets, all_preds, target_names=HAM_CLASSES, zero_division=0
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    log.info("Device: CPU (%d threads)", args.threads)

    # ── Load test data ───────────────────────────────────────────────────
    test_csv = Path(args.test_csv)
    if not test_csv.exists():
        log.error("Test CSV not found: %s", test_csv)
        sys.exit(1)

    test_df = pd.read_csv(test_csv)
    log.info("Test set: %d images", len(test_df))

    # Validate paths
    missing = sum(1 for p in test_df["image_path"] if not Path(p).exists())
    if missing:
        log.warning("%d/%d test image paths missing", missing, len(test_df))

    test_ds = EvalDataset(test_df, build_transforms(args.image_size), args.image_size)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # ── Load model ───────────────────────────────────────────────────────
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        log.error("Checkpoint not found: %s", ckpt)
        sys.exit(1)

    log.info("Loading model from %s …", ckpt)
    model, meta = load_classifier(ckpt, device=device)
    model.eval()

    # ── Evaluate ─────────────────────────────────────────────────────────
    log.info("Evaluating on test set …")
    results = evaluate(model, test_loader, device)

    # ── Print results ────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("TEST SET RESULTS")
    log.info("=" * 60)
    log.info("Accuracy:            %.4f", results["accuracy"])
    log.info("F1 (macro):          %.4f", results["f1_macro"])
    log.info("Precision (macro):   %.4f", results["precision_macro"])
    log.info("Recall (macro):      %.4f", results["recall_macro"])
    log.info("Avg latency:         %.1f ms", results["avg_latency_ms"])
    log.info("P95 latency:         %.1f ms", results["p95_latency_ms"])
    log.info("")
    log.info("Per-class F1:")
    for cls, f1 in results["f1_per_class"].items():
        log.info("  %-10s %.4f", cls, f1)
    log.info("")
    log.info("Confusion Matrix:")
    cm = np.array(results["confusion_matrix"])
    header = "       " + "  ".join(f"{c:>5}" for c in HAM_CLASSES)
    log.info(header)
    for i, row in enumerate(cm):
        log.info(f"{HAM_CLASSES[i]:>5}  " + "  ".join(f"{v:>5}" for v in row))
    log.info("")
    log.info("Classification Report:")
    log.info("\n" + results["classification_report"])

    # ── Save results ─────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": str(ckpt),
            "test_csv": str(test_csv),
            "results": {
                "accuracy": results["accuracy"],
                "f1_macro": results["f1_macro"],
                "precision_macro": results["precision_macro"],
                "recall_macro": results["recall_macro"],
                "f1_per_class": results["f1_per_class"],
                "confusion_matrix": results["confusion_matrix"],
                "avg_latency_ms": results["avg_latency_ms"],
                "p95_latency_ms": results["p95_latency_ms"],
            },
        }, f, indent=2)
    log.info("Results saved → %s", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--test_csv",   required=True, help="Path to test.csv")
    p.add_argument("--image_dir",  default="HAM10000_images", help="Image directory")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--threads",    type=int, default=4)
    p.add_argument("--output",     default="test_results.json")
    args = p.parse_args()
    main(args)
