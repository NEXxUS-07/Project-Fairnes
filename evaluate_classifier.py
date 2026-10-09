"""
evaluate_classifier.py — Comprehensive Evaluation + Baseline vs. Balanced Comparison
======================================================================================
Computes and visualises:
  • Per-class: Accuracy, Precision, Recall, F1
  • Per-class AUC-ROC (one-vs-rest)
  • Confusion matrix (raw counts + normalised)
  • Training curves (loss, F1) from training_history.json
  • Side-by-side comparison: baseline model vs. fairness-balanced model

All plots use a premium dark theme and are saved to ``results/`` folder.

Usage:
    # Evaluate a single model:
    python evaluate_classifier.py \\
        --checkpoint results/best_model.pth \\
        --test_csv   splits/test.csv \\
        --image_dir  HAM10000_images/ \\
        --output_dir results/

    # Full comparison (baseline vs. balanced):
    python evaluate_classifier.py \\
        --baseline_ckpt  results/baseline/best_model.pth \\
        --balanced_ckpt  results/balanced/best_model.pth \\
        --test_csv       splits/test.csv \\
        --image_dir      HAM10000_images/ \\
        --output_dir     results/comparison/ \\
        --compare

    # Also plot training curves:
    python evaluate_classifier.py \\
        --checkpoint results/best_model.pth \\
        --test_csv   splits/test.csv \\
        --image_dir  HAM10000_images/ \\
        --history    results/training_history.json \\
        --output_dir results/

Dependencies: matplotlib, seaborn, scikit-learn, torch, torchvision
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from classifier import (
    EfficientNetB3Classifier,
    load_classifier,
    HAM_CLASSES, HAM_FULL_NAMES, NATIVE_IMAGE_SIZE,
)
from dataset import HAMDataset

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT THEME
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#e6edf3"
ACCENT     = "#58a6ff"

# Colour-blind-safe palette for 7 classes
CLASS_COLORS = [
    "#0072B2", "#D55E00", "#009E73",
    "#CC79A7", "#E69F00", "#56B4E9", "#F0E442",
]

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  GRID_COLOR,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model:      EfficientNetB3Classifier,
    loader:     DataLoader,
    device:     torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on a DataLoader.

    Returns
    -------
    all_probs   : (N, num_classes) float — softmax probabilities
    all_preds   : (N,) int          — predicted class indices
    all_targets : (N,) int          — ground-truth class indices
    """
    from torch.cuda.amp import autocast
    model.to(device).eval()

    all_probs, all_preds, all_targets = [], [], []

    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        with autocast(enabled=(device.type == "cuda")):
            logits = model(images)

        import torch.nn.functional as F
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

        all_probs.append(probs)
        all_preds.append(preds)
        all_targets.extend(labels.cpu().numpy().tolist())

    return (
        np.vstack(all_probs),
        np.concatenate(all_preds),
        np.array(all_targets, dtype=int),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_full_metrics(
    probs:   np.ndarray,
    preds:   np.ndarray,
    targets: np.ndarray,
) -> dict:
    """
    Compute comprehensive per-class and macro metrics.

    Returns
    -------
    dict with keys: accuracy, per_class (dict), macro, auc_roc_per_class
    """
    n_cls    = len(HAM_CLASSES)
    labels_b = label_binarize(targets, classes=list(range(n_cls)))

    # Per-class metrics
    prec_pc  = precision_score(targets, preds, average=None,
                               labels=list(range(n_cls)), zero_division=0)
    rec_pc   = recall_score(targets, preds, average=None,
                            labels=list(range(n_cls)), zero_division=0)
    f1_pc    = f1_score(targets, preds, average=None,
                        labels=list(range(n_cls)), zero_division=0)

    # Per-class AUC-ROC (one-vs-rest)
    auc_pc = {}
    for i, cls in enumerate(HAM_CLASSES):
        if labels_b[:, i].sum() > 0:
            auc_pc[cls] = float(roc_auc_score(labels_b[:, i], probs[:, i]))
        else:
            auc_pc[cls] = float("nan")

    per_class = {
        HAM_CLASSES[i]: {
            "precision": round(float(prec_pc[i]), 6),
            "recall":    round(float(rec_pc[i]),  6),
            "f1":        round(float(f1_pc[i]),   6),
            "auc_roc":   round(auc_pc[HAM_CLASSES[i]], 6)
                         if not np.isnan(auc_pc[HAM_CLASSES[i]]) else None,
            "support":   int((targets == i).sum()),
        }
        for i in range(n_cls)
    }

    macro_auc = float(np.nanmean(list(auc_pc.values())))

    return {
        "accuracy":         round(float(accuracy_score(targets, preds)), 6),
        "precision_macro":  round(float(precision_score(targets, preds, average="macro", zero_division=0)), 6),
        "recall_macro":     round(float(recall_score(targets, preds, average="macro", zero_division=0)), 6),
        "f1_macro":         round(float(f1_score(targets, preds, average="macro", zero_division=0)), 6),
        "auc_roc_macro":    round(macro_auc, 6),
        "per_class":        per_class,
        "auc_roc_per_class": {k: round(v, 6) if not np.isnan(v) else None
                              for k, v in auc_pc.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    log.info("Saved → %s", path)


def plot_confusion_matrix(
    preds:    np.ndarray,
    targets:  np.ndarray,
    out_dir:  Path,
    title:    str = "Confusion Matrix",
    prefix:   str = "",
):
    """Save raw-count and row-normalised confusion matrices."""
    cm     = confusion_matrix(targets, preds, labels=list(range(len(HAM_CLASSES))))
    cm_n   = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)   # row-normalised

    short  = HAM_CLASSES
    full   = [HAM_FULL_NAMES[c] for c in HAM_CLASSES]

    for mat, norm_label, fname, fmt in [
        (cm,   "Counts",      f"{prefix}confusion_matrix_counts.png",      "d"),
        (cm_n, "Recall Rate", f"{prefix}confusion_matrix_normalised.png",  ".2f"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(DARK_BG)
        sns.heatmap(
            mat, annot=True, fmt=fmt, ax=ax,
            xticklabels=short, yticklabels=short,
            cmap="Blues", linewidths=0.5, linecolor=GRID_COLOR,
            cbar_kws={"label": norm_label},
        )
        ax.set_xlabel("Predicted Class", fontsize=11, labelpad=8)
        ax.set_ylabel("True Class",      fontsize=11, labelpad=8)
        ax.set_title(f"{title}  [{norm_label}]", fontsize=13,
                     fontweight="bold", pad=12)
        ax.tick_params(labelsize=9)
        _save(fig, out_dir / fname)


def plot_per_class_metrics(
    metrics:  dict,
    out_dir:  Path,
    title:    str = "Per-Class Metrics",
    prefix:   str = "",
):
    """Bar chart: Precision, Recall, F1, AUC-ROC per class."""
    pc  = metrics["per_class"]
    cls = HAM_CLASSES
    metric_keys = ["precision", "recall", "f1", "auc_roc"]
    metric_labels = ["Precision", "Recall", "F1", "AUC-ROC"]
    colors = [ACCENT, "#3fb950", "#f78166", "#d2a8ff"]

    x   = np.arange(len(cls))
    w   = 0.2    # bar width

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    for i, (key, label, col) in enumerate(zip(metric_keys, metric_labels, colors)):
        vals = [pc[c].get(key) or 0.0 for c in cls]
        bars = ax.bar(x + i * w, vals, w, label=label, color=col,
                      edgecolor="white", linewidth=0.4, alpha=0.92)
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7, color="white", fontweight="bold")

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels([f"{c}\n({HAM_FULL_NAMES[c][:10]}…)"
                        if len(HAM_FULL_NAMES[c]) > 10 else f"{c}\n({HAM_FULL_NAMES[c]})"
                        for c in cls], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.7)
    _save(fig, out_dir / f"{prefix}per_class_metrics.png")


def plot_roc_curves(
    probs:   np.ndarray,
    targets: np.ndarray,
    out_dir: Path,
    title:   str = "ROC Curves (One-vs-Rest)",
    prefix:  str = "",
):
    """One ROC curve per class, all on one plot."""
    labels_b = label_binarize(targets, classes=list(range(len(HAM_CLASSES))))
    fig, ax  = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    for i, (cls, col) in enumerate(zip(HAM_CLASSES, CLASS_COLORS)):
        if labels_b[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(labels_b[:, i], probs[:, i])
        auc_val      = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{cls} — {HAM_FULL_NAMES[cls]} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color=GRID_COLOR, lw=1.5, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    ax.grid(color=GRID_COLOR, linewidth=0.4, linestyle="--", alpha=0.6)
    _save(fig, out_dir / f"{prefix}roc_curves.png")


def plot_training_curves(
    history_json: Path,
    out_dir:      Path,
    prefix:       str = "",
):
    """Plot loss + macro-F1 training curves from training_history.json."""
    with open(history_json) as f:
        data = json.load(f)
    hist = data.get("history", data)
    if not hist:
        log.warning("Empty training history — skipping curves.")
        return

    epochs         = [r["epoch"]     for r in hist]
    train_loss     = [r.get("train_loss", 0) for r in hist]
    val_loss       = [r.get("val_loss",   0) for r in hist]
    train_f1       = [r.get("train_f1_macro", 0) for r in hist]
    val_f1         = [r.get("val_f1_macro",   0) for r in hist]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)

    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.grid(color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.6)

    # Loss
    axes[0].plot(epochs, train_loss, color=ACCENT,   lw=2, label="Train")
    axes[0].plot(epochs, val_loss,   color="#f78166", lw=2, label="Val")
    axes[0].set_title("Loss",  fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=9)

    # F1
    axes[1].plot(epochs, train_f1, color=ACCENT,   lw=2, label="Train")
    axes[1].plot(epochs, val_f1,   color="#f78166", lw=2, label="Val")
    best_ep = epochs[int(np.argmax(val_f1))]
    best_f1_v = max(val_f1)
    axes[1].axvline(best_ep, color="#d2a8ff", lw=1.5, linestyle="--",
                    label=f"Best epoch={best_ep} (F1={best_f1_v:.4f})")
    axes[1].set_title("Macro-F1", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1")
    axes[1].legend(fontsize=9)

    fig.suptitle(f"Training Curves  [{data.get('run_name', '')}]",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR)
    fig.tight_layout(pad=2)
    _save(fig, out_dir / f"{prefix}training_curves.png")


def plot_comparison(
    baseline_metrics: dict,
    balanced_metrics: dict,
    out_dir:          Path,
):
    """
    Side-by-side bar charts comparing baseline vs. balanced model
    on per-class F1, per-class AUC-ROC, and macro summary.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    cls        = HAM_CLASSES
    x          = np.arange(len(cls))
    w          = 0.37
    base_color = "#58a6ff"    # blue
    fair_color = "#3fb950"    # green

    # ── 1. Per-class F1 ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(PANEL_BG)
    base_f1 = [baseline_metrics["per_class"][c]["f1"] for c in cls]
    fair_f1 = [balanced_metrics["per_class"][c]["f1"] for c in cls]
    ax1.bar(x - w / 2, base_f1, w, label="Baseline",  color=base_color,
            edgecolor="white", linewidth=0.4, alpha=0.9)
    ax1.bar(x + w / 2, fair_f1, w, label="Balanced",  color=fair_color,
            edgecolor="white", linewidth=0.4, alpha=0.9)
    for xi, (b, f) in enumerate(zip(base_f1, fair_f1)):
        ax1.annotate(f"{b:.2f}", (xi - w / 2, b + 0.01), ha="center",
                     fontsize=7.5, color=base_color, fontweight="bold")
        ax1.annotate(f"{f:.2f}", (xi + w / 2, f + 0.01), ha="center",
                     fontsize=7.5, color=fair_color, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}\n{HAM_FULL_NAMES[c]}" for c in cls], fontsize=8)
    ax1.set_ylabel("F1 Score"); ax1.set_ylim(0, 1.15)
    ax1.set_title("Per-Class F1 Score  —  Baseline vs. Fairness-Balanced",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", color=GRID_COLOR, linewidth=0.4, linestyle="--", alpha=0.6)

    # ── 2. Per-class AUC-ROC ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(PANEL_BG)
    base_auc = [baseline_metrics["per_class"][c].get("auc_roc") or 0. for c in cls]
    fair_auc = [balanced_metrics["per_class"][c].get("auc_roc") or 0. for c in cls]
    ax2.bar(x - w / 2, base_auc, w, label="Baseline", color=base_color,
            edgecolor="white", linewidth=0.4, alpha=0.9)
    ax2.bar(x + w / 2, fair_auc, w, label="Balanced", color=fair_color,
            edgecolor="white", linewidth=0.4, alpha=0.9)
    ax2.set_xticks(x); ax2.set_xticklabels(cls, fontsize=9)
    ax2.set_ylabel("AUC-ROC"); ax2.set_ylim(0, 1.1)
    ax2.set_title("Per-Class AUC-ROC", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", color=GRID_COLOR, linewidth=0.4, linestyle="--", alpha=0.6)

    # ── 3. Macro summary table ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(PANEL_BG)
    ax3.axis("off")

    macro_keys = ["accuracy", "precision_macro", "recall_macro",
                  "f1_macro", "auc_roc_macro"]
    macro_labels = ["Accuracy", "Precision (macro)", "Recall (macro)",
                    "F1 (macro)", "AUC-ROC (macro)"]

    rows = []
    for key, label in zip(macro_keys, macro_labels):
        b_val = baseline_metrics.get(key, float("nan"))
        f_val = balanced_metrics.get(key, float("nan"))
        delta = f_val - b_val if not (np.isnan(b_val) or np.isnan(f_val)) else float("nan")
        rows.append([label, f"{b_val:.4f}", f"{f_val:.4f}",
                     f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"])

    tbl = ax3.table(
        cellText=rows,
        colLabels=["Metric", "Baseline", "Balanced", "Δ"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(PANEL_BG if r > 0 else DARK_BG)
        cell.set_text_props(color=TEXT_COLOR,
                            fontweight="bold" if r == 0 else "normal")
        cell.set_edgecolor(GRID_COLOR)
        if r > 0 and c == 3:
            val = rows[r - 1][3]
            cell.set_text_props(
                color="#3fb950" if val.startswith("+") else "#f78166"
            )
    ax3.set_title("Macro Metrics Summary", fontsize=11,
                  fontweight="bold", pad=10)

    fig.suptitle("Baseline vs. Fairness-Balanced Model Comparison",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.01)
    _save(fig, out_dir / "comparison_summary.png")


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(
    ckpt_path:   Union[str, Path],
    test_csv:    Union[str, Path],
    image_dir:   Optional[Union[str, Path]],
    out_dir:     Path,
    image_size:  int  = NATIVE_IMAGE_SIZE,
    batch_size:  int  = 64,
    num_workers: int  = 4,
    device_str:  str  = "cuda",
    prefix:      str  = "",
    history_json: Optional[Path] = None,
) -> dict:
    """
    Evaluate one model checkpoint on the test set.

    Parameters
    ----------
    ckpt_path    : path to best_model.pth
    test_csv     : path to test split CSV
    image_dir    : root image directory (None if CSV has absolute paths)
    out_dir      : where to save plots
    prefix       : filename prefix for saved plots (e.g. "baseline_" / "balanced_")
    history_json : optional path to training_history.json for curve plots

    Returns
    -------
    dict of computed metrics
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available()
                          or device_str == "cpu" else "cpu")

    # ── Load model ───────────────────────────────────────────────────────
    model, _ = load_classifier(ckpt_path, device=device)

    # ── Dataset ──────────────────────────────────────────────────────────
    if image_dir:
        test_ds = HAMDataset(image_dir, test_csv, split="test",
                             image_size=image_size)
    else:
        from train_classifier import BalancedCSVDataset
        test_ds = BalancedCSVDataset(test_csv, split="test",
                                     image_size=image_size)

    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    log.info("Evaluating on %d test samples…", len(test_ds))

    # ── Inference ────────────────────────────────────────────────────────
    probs, preds, targets = run_inference(model, loader, device)

    # ── Metrics ──────────────────────────────────────────────────────────
    metrics = compute_full_metrics(probs, preds, targets)
    log.info(
        "Results: acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
        metrics["accuracy"], metrics["precision_macro"],
        metrics["recall_macro"], metrics["f1_macro"],
        metrics["auc_roc_macro"],
    )
    # Print per-class summary
    log.info("\n%s", pd.DataFrame(metrics["per_class"]).T.to_string())

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_confusion_matrix(preds, targets, out_dir,
                          title=f"Confusion Matrix [{prefix.rstrip('_')}]",
                          prefix=prefix)
    plot_per_class_metrics(metrics, out_dir,
                           title=f"Per-Class Metrics [{prefix.rstrip('_')}]",
                           prefix=prefix)
    plot_roc_curves(probs, targets, out_dir,
                    title=f"ROC Curves [{prefix.rstrip('_')}]",
                    prefix=prefix)
    if history_json and history_json.exists():
        plot_training_curves(history_json, out_dir, prefix=prefix)

    # ── Save metrics JSON ─────────────────────────────────────────────────
    mpath = out_dir / f"{prefix}metrics.json"
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info("Metrics saved → %s", mpath)

    return metrics


def evaluate_and_compare(
    baseline_ckpt:  Union[str, Path],
    balanced_ckpt:  Union[str, Path],
    test_csv:       Union[str, Path],
    image_dir:      Optional[Union[str, Path]],
    out_dir:        Path,
    image_size:     int = NATIVE_IMAGE_SIZE,
    batch_size:     int = 64,
    num_workers:    int = 4,
    device_str:     str = "cuda",
    baseline_history: Optional[Path] = None,
    balanced_history: Optional[Path] = None,
):
    """
    Evaluate both baseline and balanced models and produce comparison plots.
    """
    log.info("══ Evaluating BASELINE model ══════════════════════════")
    base_metrics = evaluate_model(
        ckpt_path=baseline_ckpt, test_csv=test_csv,
        image_dir=image_dir, out_dir=out_dir,
        image_size=image_size, batch_size=batch_size,
        num_workers=num_workers, device_str=device_str,
        prefix="baseline_", history_json=baseline_history,
    )

    log.info("══ Evaluating BALANCED model ══════════════════════════")
    fair_metrics = evaluate_model(
        ckpt_path=balanced_ckpt, test_csv=test_csv,
        image_dir=image_dir, out_dir=out_dir,
        image_size=image_size, batch_size=batch_size,
        num_workers=num_workers, device_str=device_str,
        prefix="balanced_", history_json=balanced_history,
    )

    log.info("══ Generating comparison plot ══════════════════════════")
    plot_comparison(base_metrics, fair_metrics, out_dir)

    # Combined report
    combined = {
        "baseline": base_metrics,
        "balanced": fair_metrics,
        "delta": {
            k: round(fair_metrics[k] - base_metrics[k], 6)
            for k in ["accuracy", "precision_macro", "recall_macro",
                      "f1_macro", "auc_roc_macro"]
        },
    }
    with open(out_dir / "comparison_report.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)
    log.info("Comparison report → %s", out_dir / "comparison_report.json")

    # ── Summary to console ────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  BASELINE vs. FAIRNESS-BALANCED — FINAL COMPARISON")
    print("═" * 65)
    for key in ["accuracy", "precision_macro", "recall_macro",
                "f1_macro", "auc_roc_macro"]:
        lb = key.replace("_macro", " (macro)").replace("_", " ").title()
        b  = base_metrics[key]
        f  = fair_metrics[key]
        d  = f - b
        sym = "▲" if d > 0 else "▼"
        print(f"  {lb:<28}  {b:.4f} → {f:.4f}  "
              f"({sym} {abs(d):.4f})")
    print("═" * 65)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate EfficientNet-B3 on HAM10000 + comparison plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Mode: single or comparison
    p.add_argument("--compare", action="store_true",
                   help="Compare baseline vs. balanced models side by side")
    # Single model
    p.add_argument("--checkpoint",  default=None,
                   help="Checkpoint path for single-model evaluation")
    p.add_argument("--history",     default=None,
                   help="training_history.json for curve plots")
    # Comparison mode
    p.add_argument("--baseline_ckpt",   default=None)
    p.add_argument("--balanced_ckpt",   default=None)
    p.add_argument("--baseline_history",default=None)
    p.add_argument("--balanced_history",default=None)
    # Shared
    p.add_argument("--test_csv",    required=True)
    p.add_argument("--image_dir",   default=None)
    p.add_argument("--output_dir",  default="results")
    p.add_argument("--image_size",  type=int, default=NATIVE_IMAGE_SIZE)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.compare:
        if not (args.baseline_ckpt and args.balanced_ckpt):
            raise SystemExit(
                "Comparison mode requires --baseline_ckpt and --balanced_ckpt"
            )
        evaluate_and_compare(
            baseline_ckpt    = args.baseline_ckpt,
            balanced_ckpt    = args.balanced_ckpt,
            test_csv         = args.test_csv,
            image_dir        = args.image_dir,
            out_dir          = out,
            image_size       = args.image_size,
            batch_size       = args.batch_size,
            num_workers      = args.num_workers,
            device_str       = args.device,
            baseline_history = Path(args.baseline_history) if args.baseline_history else None,
            balanced_history = Path(args.balanced_history) if args.balanced_history else None,
        )
    else:
        if not args.checkpoint:
            raise SystemExit("Single-model mode requires --checkpoint")
        evaluate_model(
            ckpt_path    = args.checkpoint,
            test_csv     = args.test_csv,
            image_dir    = args.image_dir,
            out_dir      = out,
            image_size   = args.image_size,
            batch_size   = args.batch_size,
            num_workers  = args.num_workers,
            device_str   = args.device,
            history_json = Path(args.history) if args.history else None,
        )
