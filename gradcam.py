"""
gradcam.py — Grad-CAM for EfficientNet-B3 Skin Lesion Classifier
=================================================================
Implements Grad-CAM (Selvaraju et al. 2017) using PyTorch backward hooks.
Optionally delegates to the ``pytorch-grad-cam`` library for additional
methods (GradCAM++, EigenCAM, ScoreCAM).

Target layer:
    EfficientNet-B3 (timm) → model.backbone.conv_head
    This is a 1×1 Conv2d outputting 1536 channels — the final feature map
    before GlobalAveragePooling.  Grad-CAM here gives semantically rich,
    spatially coarse (10×10 for 300×300 input) activation maps.

Main API:
    cam  = GradCAM(model, target_layer="auto")
    out  = cam.generate(images, class_indices=None)   # (B, H, W) heatmaps
    grids = cam.overlay_batch(images_pil, out)        # list of PIL composites

Dependency:
    pip install opencv-python  (optional but recommended for bilinear resize)
    pip install grad-cam       (optional, for extra CAM methods)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from classifier import EfficientNetB3Classifier, HAM_CLASSES, HAM_FULL_NAMES

log = logging.getLogger(__name__)

# ImageNet denormalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert an ImageNet-normalised CHW tensor to a uint8 HWC numpy array.

    Parameters
    ----------
    tensor : (3, H, W)  float32 in normalised range

    Returns
    -------
    (H, W, 3) uint8  in [0, 255]
    """
    img = tensor.cpu().numpy().transpose(1, 2, 0)   # HWC
    img = img * _STD + _MEAN                         # undo normalisation
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def heatmap_to_rgb(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Convert a float heatmap (H, W) ∈ [0, 1] to a uint8 RGB image.
    Uses OpenCV colormap; falls back to matplotlib if cv2 unavailable.
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    try:
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    except Exception:
        # Matplotlib fallback
        cmap_fn = plt.get_cmap("jet")
        heatmap_color = (cmap_fn(heatmap) * 255).astype(np.uint8)[:, :, :3]
    return heatmap_color


def overlay_heatmap(
    image_rgb:   np.ndarray,
    heatmap_rgb: np.ndarray,
    alpha:       float = 0.45,
) -> np.ndarray:
    """
    Alpha-blend heatmap over the original image.

    Parameters
    ----------
    image_rgb   : (H, W, 3) uint8
    heatmap_rgb : (H, W, 3) uint8  (same spatial dims after resize)
    alpha       : heatmap opacity

    Returns
    -------
    (H, W, 3) uint8 composite
    """
    if heatmap_rgb.shape[:2] != image_rgb.shape[:2]:
        heatmap_rgb = cv2.resize(
            heatmap_rgb,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    blended = (1 - alpha) * image_rgb.astype(np.float32) + \
              alpha * heatmap_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _auto_target_layer(model: EfficientNetB3Classifier) -> nn.Module:
    """
    Automatically identify the Grad-CAM target layer for EfficientNet-B3.

    Priority:
      1. backbone.conv_head  — final 1×1 conv → 1536 ch  (preferred)
      2. backbone.blocks[-1] — last MBConv block group
    """
    backbone = model.backbone
    if hasattr(backbone, "conv_head"):
        log.info("GradCAM target: backbone.conv_head")
        return backbone.conv_head
    # Fallback: last block in the last stage
    if hasattr(backbone, "blocks"):
        last_stage = backbone.blocks[-1]
        if hasattr(last_stage, "__iter__"):
            last_block = list(last_stage)[-1]
        else:
            last_block = last_stage
        log.info("GradCAM target: backbone.blocks[-1][-1]")
        return last_block
    raise RuntimeError(
        "Cannot auto-detect target layer.  "
        "Pass target_layer=<module> explicitly."
    )


# ═══════════════════════════════════════════════════════════════════════════
#  GRAD-CAM  (hook-based, no external library required)
# ═══════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for EfficientNet-B3.

    Algorithm
    ---------
    1. Forward pass to record feature maps A^k at the target layer.
    2. Backward pass of the target class score to record gradients ∂y^c/∂A^k.
    3. Global-average-pool the gradients → importance weights α_k^c.
    4. Weighted sum + ReLU:  L^c = ReLU( Σ_k α_k^c · A^k )
    5. Bilinear upsample to input resolution.

    Parameters
    ----------
    model        : EfficientNetB3Classifier
    target_layer : nn.Module | None
        Target layer for hook registration.  None → auto-detect (conv_head).
    use_gradcam_lib : bool
        If True and ``grad-cam`` package is installed, delegates to
        ``pytorch_grad_cam.GradCAMPlusPlus`` for improved results.
    """

    def __init__(
        self,
        model:           EfficientNetB3Classifier,
        target_layer:    Optional[nn.Module] = None,
        use_gradcam_lib: bool = True,
    ):
        self.model        = model
        self.target_layer = target_layer or _auto_target_layer(model)
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._hooks:       list = []
        self._lib_cams     = None   # pytorch-grad-cam backend if available

        # Try pytorch-grad-cam library
        if use_gradcam_lib:
            self._lib_cams = self._try_init_lib()

        if self._lib_cams is None:
            # Register hooks for manual Grad-CAM
            self._register_hooks()

        log.info(
            "GradCAM | backend=%s | target=%s",
            "pytorch-grad-cam" if self._lib_cams else "manual-hooks",
            type(self.target_layer).__name__,
        )

    # ─────────────────────────────────────────────────────────────────────
    #  LIBRARY BACKEND
    # ─────────────────────────────────────────────────────────────────────

    def _try_init_lib(self):
        try:
            from pytorch_grad_cam import GradCAMPlusPlus
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            log.info("pytorch-grad-cam library found — using GradCAM++.")
            cam = GradCAMPlusPlus(
                model=self.model,
                target_layers=[self.target_layer],
            )
            return cam
        except ImportError:
            log.info("pytorch-grad-cam not found — using manual hook implementation.")
            return None

    # ─────────────────────────────────────────────────────────────────────
    #  MANUAL HOOKS
    # ─────────────────────────────────────────────────────────────────────

    def _register_hooks(self):
        def _fwd_hook(module, input, output):
            self._activations = output.detach()

        def _bwd_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(_fwd_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(_bwd_hook)
        )

    def remove_hooks(self):
        """Remove all registered PyTorch hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()

    # ─────────────────────────────────────────────────────────────────────
    #  CORE GRAD-CAM LOGIC
    # ─────────────────────────────────────────────────────────────────────

    def _manual_gradcam(
        self,
        images:        torch.Tensor,
        class_indices: Optional[list[int]],
    ) -> np.ndarray:
        """
        Run full forward+backward pass and compute Grad-CAM for each image.

        Parameters
        ----------
        images        : (B, 3, H, W) on model device
        class_indices : list[int] of length B, or None (→ predicted class)

        Returns
        -------
        np.ndarray of shape (B, H, W)  in [0, 1]
        """
        B = images.shape[0]
        H, W = images.shape[2], images.shape[3]

        self.model.eval()
        images = images.requires_grad_(True)

        # Forward
        logits = self.model(images)   # (B, C)

        if class_indices is None:
            class_indices = logits.argmax(dim=-1).tolist()

        # Backward  — one class per image in the batch
        self.model.zero_grad()
        # Build target tensor: sum of target class logits across batch
        target_scores = torch.stack([
            logits[b, class_indices[b]]
            for b in range(B)
        ])
        target_scores.sum().backward()

        # Activations: (B, K, h, w)
        acts  = self._activations   # (B, K, h, w)
        grads = self._gradients     # (B, K, h, w)

        # α_k^c = GlobalAvgPool(grads)  →  (B, K)
        weights = grads.mean(dim=[2, 3])   # (B, K)

        # Weighted sum of activations → (B, h, w)
        cams = torch.einsum("bk,bkhw->bhw", weights, acts)
        cams = F.relu(cams)                # (B, h, w)

        # Normalise each map to [0, 1]
        cams_np = cams.cpu().numpy()
        for i in range(B):
            c = cams_np[i]
            mn, mx = c.min(), c.max()
            cams_np[i] = (c - mn) / (mx - mn + 1e-8)

        # Upsample to input resolution
        result = np.zeros((B, H, W), dtype=np.float32)
        for i in range(B):
            result[i] = cv2.resize(cams_np[i], (W, H),
                                   interpolation=cv2.INTER_LINEAR)
        return result   # (B, H, W) in [0, 1]

    def _lib_gradcam(
        self,
        images:        torch.Tensor,
        class_indices: Optional[list[int]],
    ) -> np.ndarray:
        """Delegate to pytorch-grad-cam library."""
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        B = images.shape[0]
        targets = None
        if class_indices:
            targets = [ClassifierOutputTarget(c) for c in class_indices]
        # pytorch-grad-cam returns (B, H, W) in [0, 1]
        grayscale_cam = self._lib_cams(
            input_tensor=images,
            targets=targets,
        )
        return grayscale_cam   # (B, H, W)

    # ─────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Inference only — returns (B, C) logits."""
        self.model.eval()
        return self.model(images)

    def generate(
        self,
        images:        torch.Tensor,
        class_indices: Optional[list[int]] = None,
        device:        Union[str, torch.device] = "cpu",
    ) -> tuple[np.ndarray, list[int], torch.Tensor]:
        """
        Generate Grad-CAM heatmaps for a batch of images.

        Parameters
        ----------
        images        : (B, 3, H, W) ImageNet-normalised tensor
        class_indices : target class per image (None → predicted class)
        device        : compute device

        Returns
        -------
        cams          : (B, H, W) float32 ∈ [0, 1]
        class_indices : list[int]          — final class used per image
        logits        : (B, C) float tensor
        """
        device = torch.device(device)
        self.model.to(device).eval()
        images = images.to(device)

        # Get logits for class selection (always via a clean forward)
        with torch.no_grad():
            logits = self.model(images)

        if class_indices is None:
            class_indices = logits.argmax(dim=-1).tolist()

        # Generate CAMs
        if self._lib_cams is not None:
            cams = self._lib_gradcam(images, class_indices)
        else:
            # Need gradients for manual method
            with torch.enable_grad():
                cams = self._manual_gradcam(images, class_indices)

        return cams, class_indices, logits.cpu()

    # ─────────────────────────────────────────────────────────────────────

    def overlay_batch(
        self,
        images_tensor: torch.Tensor,
        cams:          np.ndarray,
        class_indices: list[int],
        logits:        torch.Tensor,
        alpha:         float = 0.45,
    ) -> list[Image.Image]:
        """
        Produce an annotated composite image for each sample in the batch.

        Layout per image:
            [Original] | [Grad-CAM Overlay] | [Raw Heatmap]

        Parameters
        ----------
        images_tensor : (B, 3, H, W)
        cams          : (B, H, W)  float ∈ [0, 1]
        class_indices : predicted class per image
        logits        : (B, C) raw logits

        Returns
        -------
        list of PIL Images — one 3-panel composite per sample
        """
        probs = F.softmax(logits, dim=-1).numpy()
        results = []

        for i in range(len(cams)):
            img_np   = denormalize(images_tensor[i])     # (H, W, 3) uint8
            heat_rgb = heatmap_to_rgb(cams[i])            # (H, W, 3) uint8
            overlay  = overlay_heatmap(img_np, heat_rgb, alpha)

            H, W = img_np.shape[:2]

            # Create 3-panel figure
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
            fig.patch.set_facecolor("#0d1117")

            cls_code   = HAM_CLASSES[class_indices[i]]
            confidence = float(probs[i, class_indices[i]])

            for ax, panel, title in [
                (axes[0], img_np,   "Original"),
                (axes[1], overlay,  f"Grad-CAM  [{cls_code}]"),
                (axes[2], heat_rgb, "Heatmap"),
            ]:
                ax.imshow(panel)
                ax.set_title(title, color="white", fontsize=10, fontweight="bold")
                ax.axis("off")
                ax.set_facecolor("#161b22")

            # Confidence bar (top-3)
            top3_idx  = np.argsort(probs[i])[::-1][:3]
            bar_labels = [f"{HAM_CLASSES[j]} ({probs[i,j]*100:.1f}%)" for j in top3_idx]
            bar_vals   = [float(probs[i, j]) for j in top3_idx]
            bar_colors = ["#58a6ff", "#3fb950", "#d2a8ff"]

            ax_bar = fig.add_axes([0.05, -0.1, 0.9, 0.12])
            ax_bar.barh(bar_labels[::-1], bar_vals[::-1], color=bar_colors[::-1],
                        edgecolor="white", linewidth=0.3)
            ax_bar.set_xlim(0, 1)
            ax_bar.set_facecolor("#161b22")
            ax_bar.tick_params(colors="white", labelsize=8)
            ax_bar.set_xlabel("Confidence", color="white", fontsize=8)
            for spine in ax_bar.spines.values():
                spine.set_edgecolor("#21262d")

            fig.suptitle(
                f"Predicted: {HAM_FULL_NAMES[cls_code]}  "
                f"(confidence={confidence:.3f})",
                color="white", fontsize=11, fontweight="bold", y=1.01,
            )
            fig.tight_layout(pad=0.8)

            # Render to PIL
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            pil = Image.fromarray(buf)
            plt.close(fig)
            results.append(pil)

        return results

    def save_batch(
        self,
        composites:  list[Image.Image],
        out_dir:     Path,
        prefix:      str = "gradcam",
        indices:     Optional[list[int]] = None,
    ) -> list[Path]:
        """Save composite PIL images to disk. Returns list of saved paths."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, img in enumerate(composites):
            idx  = indices[i] if indices else i
            path = out_dir / f"{prefix}_{idx:05d}.png"
            img.save(path)
            paths.append(path)
        log.info("Saved %d Grad-CAM composites to %s", len(paths), out_dir)
        return paths
