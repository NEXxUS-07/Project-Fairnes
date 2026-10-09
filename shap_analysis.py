"""
shap_analysis.py — SHAP Gradient Explainer for EfficientNet-B3
================================================================
Implements image-level SHAP explanation using ``shap.GradientExplainer``.

GradientExplainer (Integrated Gradients variant) computes:
    φᵢ(f, x) ≈ (xᵢ - x̄ᵢ) · ∂f/∂xᵢ   averaged over background samples

For a per-class explanation, we call it once per class of interest.

Main API:
    shap_exp = SHAPExplainer(model, background_loader)
    maps, top3 = shap_exp.explain(images, class_indices=None)
    fig = shap_exp.plot_shap_maps(image, maps, title="…")
    agg  = shap_exp.aggregate_importance(val_loader)   → batch report

Outputs:
    • (H, W)   mean-|SHAP| heatmap per class
    • top-3 contributing regions per prediction
    • batch aggregate maps (mean |SHAP| across dataset, per class)

Dependencies:
    pip install shap
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from classifier import EfficientNetB3Classifier, HAM_CLASSES, HAM_FULL_NAMES

log = logging.getLogger(__name__)

try:
    import shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False
    log.warning("shap not installed.  Install with: pip install shap")

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
TEXT_COL  = "#e6edf3"

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_COLORS = [
    "#0072B2", "#D55E00", "#009E73",
    "#CC79A7", "#E69F00", "#56B4E9", "#F0E442",
]


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _denorm(t: torch.Tensor) -> np.ndarray:
    """CHW tensor → HWC uint8."""
    arr = t.cpu().numpy().transpose(1, 2, 0)
    arr = arr * _STD + _MEAN
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def _shap_to_heatmap(shap_vals: np.ndarray) -> np.ndarray:
    """
    Convert (C, H, W) SHAP values → (H, W) float ∈ [0, 1] importance map.
    Aggregates channels by taking the mean over |SHAP|.
    """
    importance = np.abs(shap_vals).mean(axis=0)      # (H, W)
    mn, mx = importance.min(), importance.max()
    return (importance - mn) / (mx - mn + 1e-8)


def _find_top3_regions(
    heatmap:    np.ndarray,
    patch_size: int = 32,
) -> list[dict]:
    """
    Divide the heatmap into non-overlapping patches and return the top-3
    highest mean-importance patches as bounding boxes.

    Parameters
    ----------
    heatmap    : (H, W) float
    patch_size : spatial size of each patch in pixels

    Returns
    -------
    List of 3 dicts: {"x", "y", "w", "h", "importance", "rank"}
    """
    H, W = heatmap.shape
    patches = []

    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            patch = heatmap[y:y + patch_size, x:x + patch_size]
            patches.append({
                "x": x, "y": y,
                "w": patch_size, "h": patch_size,
                "importance": float(patch.mean()),
            })

    patches.sort(key=lambda p: p["importance"], reverse=True)
    for rank, p in enumerate(patches[:3], start=1):
        p["rank"] = rank
    return patches[:3]


# ═══════════════════════════════════════════════════════════════════════════
#  SHAP EXPLAINER
# ═══════════════════════════════════════════════════════════════════════════

class SHAPExplainer:
    """
    SHAP GradientExplainer for EfficientNet-B3 skin lesion classification.

    Parameters
    ----------
    model              : EfficientNetB3Classifier (eval mode)
    background_loader  : DataLoader yielding (images, labels, ...) batches.
                         Up to ``n_background`` images are used as the
                         baseline distribution for Integrated Gradients.
    n_background       : maximum number of background samples (default 100).
    device             : torch device.

    Notes
    -----
    • GradientExplainer is O(n_background × n_explain) forward+backward passes.
      Keep n_background ≤ 200 for reasonable runtime.
    • For very large inputs (300×300), SHAP maps will be computed at the
      input resolution but averaged over colour channels for visualisation.
    """

    def __init__(
        self,
        model:             EfficientNetB3Classifier,
        background_loader,
        n_background:      int  = 100,
        device:            Union[str, torch.device] = "cpu",
    ):
        if not _SHAP_OK:
            raise ImportError("shap is not installed.  pip install shap")

        self.model  = model
        self.device = torch.device(device)
        self.model.to(self.device).eval()

        # ── Collect background samples ────────────────────────────────────
        log.info("Collecting %d background samples for SHAP…", n_background)
        bg_tensors = []
        for batch in background_loader:
            imgs = batch[0]
            bg_tensors.append(imgs)
            if sum(t.shape[0] for t in bg_tensors) >= n_background:
                break
        background = torch.cat(bg_tensors, dim=0)[:n_background].to(self.device)
        log.info("Background tensor: %s", tuple(background.shape))

        # ── Wrap model for SHAP ───────────────────────────────────────────
        # shap.GradientExplainer needs the model to accept a single tensor
        # and return a tensor.
        self._explainer = shap.GradientExplainer(
            model    = _ModelWrapper(self.model),
            data     = background,
        )

        log.info("SHAPExplainer ready.  background_size=%d", len(background))

    # ─────────────────────────────────────────────────────────────────────
    #  EXPLAIN A BATCH
    # ─────────────────────────────────────────────────────────────────────

    def explain(
        self,
        images:        torch.Tensor,
        class_indices: Optional[list[int]] = None,
        ranked_outputs: int = 3,
    ) -> tuple[list[np.ndarray], list[dict]]:
        """
        Compute per-image SHAP explanation maps.

        Parameters
        ----------
        images         : (B, 3, H, W) ImageNet-normalised tensor on any device.
        class_indices  : target class per image.  None → predicted class.
        ranked_outputs : how many ranked outputs to return SHAP values for.

        Returns
        -------
        shap_maps  : list[np.ndarray]  — length B,
                     each entry is (H, W) float importance map for the target class.
        summaries  : list[dict]  — per-image summary with top-3 regions.
        """
        imgs = images.to(self.device)

        # Get predictions if class_indices not given
        with torch.no_grad():
            logits = self.model(imgs)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()

        if class_indices is None:
            class_indices = logits.argmax(dim=-1).cpu().tolist()

        # shap_values returns:
        #   list of length num_classes, each entry (B, 3, H, W)  if ranked_outputs=None
        # or tuple (shap_vals, indices) of shapes (B, ranked, 3, H, W) and (B, ranked)
        log.info("Computing SHAP values for batch of %d (this may take a while)…",
                 len(imgs))
        shap_vals, shap_indices = self._explainer.shap_values(
            imgs,
            ranked_outputs=ranked_outputs,
            output_rank_order="max",
        )
        # shap_vals : (B, ranked_outputs, 3, H, W)
        # shap_indices : (B, ranked_outputs)

        B = images.shape[0]
        shap_maps = []
        summaries = []

        for i in range(B):
            cls_idx    = class_indices[i]
            cls_code   = HAM_CLASSES[cls_idx]
            confidence = float(probs[i, cls_idx])

            # Find which ranked output corresponds to target class
            ranked_cls = shap_indices[i].tolist()
            if cls_idx in ranked_cls:
                rank_pos = ranked_cls.index(cls_idx)
                sv = shap_vals[i, rank_pos]    # (3, H, W)
            else:
                # Target class not in top-ranked; use rank-0
                sv = shap_vals[i, 0]

            heat = _shap_to_heatmap(sv)       # (H, W)
            top3_regions = _find_top3_regions(heat)

            top3_classes_sorted = np.argsort(probs[i])[::-1][:3]
            top3_classes = [
                {"class": HAM_CLASSES[j],
                 "full_name": HAM_FULL_NAMES[HAM_CLASSES[j]],
                 "confidence": round(float(probs[i, j]), 6)}
                for j in top3_classes_sorted
            ]

            shap_maps.append(heat)
            summaries.append({
                "predicted_class":   cls_code,
                "full_name":         HAM_FULL_NAMES[cls_code],
                "confidence":        round(confidence, 6),
                "top_3_classes":     top3_classes,
                "top_3_regions":     top3_regions,
            })

        return shap_maps, summaries

    # ─────────────────────────────────────────────────────────────────────
    #  VISUALISATION
    # ─────────────────────────────────────────────────────────────────────

    def plot_shap_maps(
        self,
        image_tensor:  torch.Tensor,
        shap_map:      np.ndarray,
        summary:       dict,
        top3_regions:  Optional[list[dict]] = None,
    ) -> Image.Image:
        """
        Render a 2-row SHAP explanation figure for one image.

        Row 1: [Original] | [SHAP heatmap overlay] | [Top-3 region boxes]
        Row 2: Top-3 class confidence bar chart

        Parameters
        ----------
        image_tensor : (3, H, W) tensor
        shap_map     : (H, W) float ∈ [0, 1]
        summary      : dict from explain()

        Returns
        -------
        PIL.Image
        """
        img_np    = _denorm(image_tensor)        # (H, W, 3) uint8
        H, W      = img_np.shape[:2]

        # SHAP heatmap → RGB
        heat_uint = (shap_map * 255).astype(np.uint8)
        try:
            heat_col = cv2.applyColorMap(heat_uint, cv2.COLORMAP_INFERNO)
            heat_col = cv2.cvtColor(heat_col, cv2.COLOR_BGR2RGB)
        except Exception:
            cmap_fn  = plt.get_cmap("inferno")
            heat_col = (cmap_fn(shap_map) * 255).astype(np.uint8)[:, :, :3]
        heat_col = cv2.resize(heat_col, (W, H), interpolation=cv2.INTER_LINEAR)

        # Alpha blend
        overlay = np.clip(
            0.55 * img_np.astype(np.float32) +
            0.45 * heat_col.astype(np.float32),
            0, 255
        ).astype(np.uint8)

        fig = plt.figure(figsize=(14, 7))
        fig.patch.set_facecolor(DARK_BG)
        gs  = gridspec.GridSpec(2, 3, figure=fig,
                                hspace=0.45, wspace=0.25,
                                height_ratios=[3, 1])

        # ── Row 1 panels ──────────────────────────────────────────────────
        for col, (panel, title) in enumerate([
            (img_np,   "Original"),
            (overlay,  f"SHAP Overlay  [{summary['predicted_class']}]"),
            (img_np,   "Top-3 Contributing Regions"),
        ]):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(panel)
            ax.set_facecolor(PANEL_BG)
            ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight="bold")
            ax.axis("off")

            # Draw bounding boxes on the 3rd panel
            if col == 2 and top3_regions:
                box_colors = ["#ff4444", "#ffaa00", "#44ff88"]
                for reg, bc in zip(top3_regions, box_colors):
                    rect = mpatches.FancyBboxPatch(
                        (reg["x"], reg["y"]), reg["w"], reg["h"],
                        boxstyle="round,pad=1",
                        linewidth=2, edgecolor=bc,
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    ax.text(
                        reg["x"] + 2, reg["y"] + 12,
                        f"#{reg['rank']}",
                        color=bc, fontsize=9, fontweight="bold",
                    )

        # ── Row 2: confidence bar ─────────────────────────────────────────
        ax_bar = fig.add_subplot(gs[1, :])
        ax_bar.set_facecolor(PANEL_BG)
        top3  = summary["top_3_classes"]
        names = [f"{t['class']}  {t['full_name']}" for t in top3]
        vals  = [t["confidence"] for t in top3]
        colors = ["#58a6ff", "#3fb950", "#d2a8ff"]

        bars = ax_bar.barh(names[::-1], vals[::-1], color=colors[::-1],
                           edgecolor="white", linewidth=0.4, height=0.55)
        for bar, v in zip(bars, vals[::-1]):
            ax_bar.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{v*100:.1f}%", va="center", fontsize=9, color=TEXT_COL)
        ax_bar.set_xlim(0, 1.15)
        ax_bar.set_xlabel("Confidence", color=TEXT_COL, fontsize=9)
        ax_bar.tick_params(colors=TEXT_COL, labelsize=9)
        for sp in ax_bar.spines.values():
            sp.set_edgecolor("#21262d")
        ax_bar.set_title("Class Confidences (Top-3)", color=TEXT_COL,
                          fontsize=9, pad=3)

        fig.suptitle(
            f"SHAP Explanation — Predicted: {summary['full_name']} "
            f"(conf={summary['confidence']:.3f})",
            color=TEXT_COL, fontsize=12, fontweight="bold", y=1.01,
        )

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        pil = Image.fromarray(buf)
        plt.close(fig)
        return pil

    # ─────────────────────────────────────────────────────────────────────
    #  AGGREGATE IMPORTANCE (batch-level report)
    # ─────────────────────────────────────────────────────────────────────

    def aggregate_importance(
        self,
        loader,
        max_batches: int   = 20,
        device:      Union[str, torch.device] = "cpu",
    ) -> dict[str, np.ndarray]:
        """
        Compute mean |SHAP| importance map per class across a validation set.

        For each class, averages the per-pixel importance maps over all
        validation images where that class was predicted.

        Parameters
        ----------
        loader      : DataLoader
        max_batches : maximum number of batches to process (for speed)

        Returns
        -------
        dict[class_code → (H, W) float]  — mean importance per class
        """
        device = torch.device(device)
        self.model.to(device)

        accumulator  = {c: [] for c in HAM_CLASSES}
        batches_done = 0

        for batch in tqdm(loader, desc="Aggregate SHAP", total=max_batches):
            if batches_done >= max_batches:
                break
            images = batch[0].to(device)
            try:
                shap_maps, summaries = self.explain(images)
                for i, (sm, summary) in enumerate(zip(shap_maps, summaries)):
                    cls = summary["predicted_class"]
                    accumulator[cls].append(sm)
            except Exception as e:
                log.warning("SHAP batch failed: %s", e)
            batches_done += 1

        mean_maps: dict[str, np.ndarray] = {}
        for cls, maps in accumulator.items():
            if maps:
                mean_maps[cls] = np.stack(maps).mean(axis=0)
                log.info("Class '%s': aggregated %d maps.", cls, len(maps))
            else:
                log.warning("Class '%s': no SHAP maps collected.", cls)

        return mean_maps

    def plot_aggregate_report(
        self,
        mean_maps: dict[str, np.ndarray],
        out_path:  Path,
    ) -> Path:
        """
        Save a grid of per-class mean |SHAP| importance maps.

        The grid shows one panel per class with a colour-coded importance map
        (brighter = model focused here more strongly for that class).
        """
        n_cls = len(HAM_CLASSES)
        cols  = 4
        rows  = (n_cls + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols,
                                  figsize=(cols * 4, rows * 4))
        fig.patch.set_facecolor(DARK_BG)
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, cls in enumerate(HAM_CLASSES):
            ax = axes_flat[i]
            ax.set_facecolor(PANEL_BG)
            if cls in mean_maps:
                im = ax.imshow(mean_maps[cls], cmap="inferno",
                               vmin=0, vmax=1)
                plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color=TEXT_COL)
            ax.set_title(f"{cls}\n{HAM_FULL_NAMES[cls]}",
                         color=TEXT_COL, fontsize=9, fontweight="bold")
            ax.axis("off")

        # Hide unused panels
        for j in range(n_cls, len(axes_flat)):
            axes_flat[j].axis("off")
            axes_flat[j].set_facecolor(DARK_BG)

        fig.suptitle("Aggregate SHAP Importance Maps — Per Class",
                     color=TEXT_COL, fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout(pad=1.5)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=DARK_BG)
        plt.close(fig)
        log.info("Aggregate SHAP report → %s", out_path)
        return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL WRAPPER (needed for shap.GradientExplainer)
# ─────────────────────────────────────────────────────────────────────────────

class _ModelWrapper(nn.Module):
    """
    Thin wrapper to expose only the forward pass to shap.GradientExplainer.
    Returns raw logits (not softmax) — SHAP works with unnormalised outputs.
    """
    def __init__(self, model: EfficientNetB3Classifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
