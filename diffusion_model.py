"""
diffusion_model.py — HuggingFace Diffusers-Based Class-Conditional DDPM
========================================================================
Wraps diffusers UNet2DModel + DDPMScheduler into a clean HAM10000DiffusionModel
class that is compatible with both train_diffusion.py and generate.py.

Architecture:
    • UNet2DModel with num_class_embeds=7  (one per HAM10000 lesion class)
    • DDPMScheduler with 1000 linear-beta timesteps
    • Image resolution: 224 × 224 RGB
    • Mixed-precision ready (call .half() / autocast externally)

Dependencies:
    pip install diffusers accelerate transformers

Reference:
    Ho et al. "Denoising Diffusion Probabilistic Models" (2020)
    HuggingFace Diffusers: https://github.com/huggingface/diffusers
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# ── diffusers ──────────────────────────────────────────────────────────────
try:
    from diffusers import DDPMScheduler, UNet2DModel
    from diffusers.optimization import get_cosine_schedule_with_warmup
    _DIFFUSERS_OK = True
except ImportError as _e:
    raise ImportError(
        "diffusers is required.  Install with: pip install diffusers accelerate"
    ) from _e

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  HAM10000 CLASS MAP
# ─────────────────────────────────────────────────────────────────────────────

HAM_CLASSES   = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
HAM_LABEL_MAP = {c: i for i, c in enumerate(HAM_CLASSES)}

NUM_CLASSES  = len(HAM_CLASSES)    # 7
IMAGE_SIZE   = 224

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_UNET_CONFIG = dict(
    sample_size       = IMAGE_SIZE,
    in_channels       = 3,
    out_channels      = 3,
    layers_per_block  = 2,
    # 4 resolution levels: 224→112→56→28→14
    block_out_channels = (128, 256, 256, 512),
    down_block_types  = (
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types    = (
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    # Class conditioning: one embedding per HAM10000 class
    num_class_embeds  = NUM_CLASSES,
    dropout           = 0.1,
    norm_num_groups   = 32,
    attention_head_dim= 8,
)

DEFAULT_SCHEDULER_CONFIG = dict(
    num_train_timesteps = 1000,
    beta_start          = 1e-4,
    beta_end            = 2e-2,
    beta_schedule       = "linear",
    clip_sample         = True,
    clip_sample_range   = 1.0,
    prediction_type     = "epsilon",     # predict added noise ε
)


# ═══════════════════════════════════════════════════════════════════════════
#  HAM10000DiffusionModel
# ═══════════════════════════════════════════════════════════════════════════

class HAM10000DiffusionModel(nn.Module):
    """
    Class-Conditional DDPM for HAM10000 skin lesion synthesis.

    Wraps HuggingFace diffusers ``UNet2DModel`` + ``DDPMScheduler``.

    Parameters
    ----------
    unet_config : dict | None
        Override default UNet2DModel kwargs.  None → DEFAULT_UNET_CONFIG.
    scheduler_config : dict | None
        Override default DDPMScheduler kwargs.
    cfg_null_prob : float
        Probability of dropping the class label during training for
        classifier-free guidance.  0.0 disables CFG training.
    cfg_scale : float
        Guidance scale w at inference (1.0 = no CFG).
    """

    def __init__(
        self,
        unet_config:      Optional[dict] = None,
        scheduler_config: Optional[dict] = None,
        cfg_null_prob:    float = 0.1,
        cfg_scale:        float = 1.0,
    ):
        super().__init__()
        uc = {**DEFAULT_UNET_CONFIG,      **(unet_config      or {})}
        sc = {**DEFAULT_SCHEDULER_CONFIG, **(scheduler_config or {})}

        self.unet      = UNet2DModel(**uc)
        self.scheduler = DDPMScheduler(**sc)
        self.T         = sc["num_train_timesteps"]
        self.cfg_null_prob  = cfg_null_prob
        self.cfg_scale      = cfg_scale
        self.null_token     = NUM_CLASSES    # index beyond valid range → ignored embedding

        log.info(
            "HAM10000DiffusionModel: %s  |  scheduler T=%d  beta=[%.1e, %.1e]",
            self._model_summary(), self.T,
            sc["beta_start"], sc["beta_end"],
        )

    # ─────────────────────────────────────────────────────────────────────
    #  TRAINING STEP
    # ─────────────────────────────────────────────────────────────────────

    def train_step(
        self,
        clean_images: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the DDPM ε-prediction loss for one training batch.

        Parameters
        ----------
        clean_images : (B, 3, H, W)  in [-1, 1]
        class_labels : (B,)  int  class indices [0, 6]

        Returns
        -------
        loss : scalar tensor  MSE(ε̂, ε)
        """
        device = clean_images.device
        B      = clean_images.size(0)

        # Sample random timesteps
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)

        # Add noise
        noise    = torch.randn_like(clean_images)
        noisy_im = self.scheduler.add_noise(clean_images, noise, t)

        # Classifier-free guidance: randomly drop class label
        if self.cfg_null_prob > 0.0:
            drop_mask    = torch.rand(B, device=device) < self.cfg_null_prob
            class_labels = torch.where(
                drop_mask, torch.full_like(class_labels, self.null_token), class_labels
            )

        # Predict noise
        # UNet2DModel expects timestep as long tensor; class_labels as long
        model_out = self.unet(
            noisy_im,
            t,
            class_labels=class_labels,
            return_dict=False,
        )[0]    # (B, 3, H, W)

        return torch.nn.functional.mse_loss(model_out, noise)

    # ─────────────────────────────────────────────────────────────────────
    #  FULL DDPM REVERSE SAMPLING
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        class_label:   int | str,
        num_images:    int  = 1,
        device:        Union[str, torch.device] = "cuda",
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Run full DDPM reverse diffusion  x_T → x_0.

        Returns
        -------
        x_0 : (N, 3, H, W)  in [-1, 1]
        """
        device = _resolve_device(device)
        self.unet.to(device).eval()

        label_idx = _resolve_label(class_label)
        labels    = torch.full((num_images,), label_idx, dtype=torch.long, device=device)

        x = torch.randn(num_images, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        self.scheduler.set_timesteps(self.T)

        for i, t_val in enumerate(self.scheduler.timesteps):
            t_batch = torch.full((num_images,), t_val, dtype=torch.long, device=device)

            if self.cfg_scale > 1.0:
                null_labels = torch.full_like(labels, self.null_token)
                eps_cond   = self.unet(x, t_batch, class_labels=labels,      return_dict=False)[0]
                eps_uncond = self.unet(x, t_batch, class_labels=null_labels,  return_dict=False)[0]
                eps        = eps_uncond + self.cfg_scale * (eps_cond - eps_uncond)
            else:
                eps = self.unet(x, t_batch, class_labels=labels, return_dict=False)[0]

            x = self.scheduler.step(eps, t_val, x, return_dict=False)[0]

            if show_progress and (i % 100 == 0 or i == len(self.scheduler.timesteps) - 1):
                log.info("  Sampling  step %4d/%4d …", i, len(self.scheduler.timesteps))

        return x

    # ─────────────────────────────────────────────────────────────────────
    #  DDIM FAST SAMPLING  (via scheduler with fewer steps)
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def fast_sample(
        self,
        class_label: int | str,
        num_images:  int   = 1,
        num_steps:   int   = 50,
        device:      Union[str, torch.device] = "cuda",
    ) -> torch.Tensor:
        """
        DDIM-style fast sampling with ``num_steps`` function evaluations.
        Uses DDIMScheduler internally (falls back to subsampled DDPM steps).

        Returns (N, 3, H, W) in [-1, 1].
        """
        try:
            from diffusers import DDIMScheduler
            ddim_sched = DDIMScheduler(
                num_train_timesteps = self.T,
                beta_start          = self.scheduler.config.beta_start,
                beta_end            = self.scheduler.config.beta_end,
                beta_schedule       = self.scheduler.config.beta_schedule,
                clip_sample         = True,
            )
        except Exception:
            log.warning("DDIMScheduler unavailable — falling back to full DDPM.")
            return self.sample(class_label, num_images, device=device)

        device = _resolve_device(device)
        self.unet.to(device).eval()

        label_idx = _resolve_label(class_label)
        labels    = torch.full((num_images,), label_idx, dtype=torch.long, device=device)

        x = torch.randn(num_images, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        ddim_sched.set_timesteps(num_steps, device=device)

        for t_val in ddim_sched.timesteps:
            t_batch = torch.full((num_images,), t_val, dtype=torch.long, device=device)

            if self.cfg_scale > 1.0:
                null_labels = torch.full_like(labels, self.null_token)
                eps_cond   = self.unet(x, t_batch, class_labels=labels,     return_dict=False)[0]
                eps_uncond = self.unet(x, t_batch, class_labels=null_labels, return_dict=False)[0]
                eps        = eps_uncond + self.cfg_scale * (eps_cond - eps_uncond)
            else:
                eps = self.unet(x, t_batch, class_labels=labels, return_dict=False)[0]

            x = ddim_sched.step(eps, t_val, x, return_dict=False)[0]

        return x

    # ─────────────────────────────────────────────────────────────────────
    #  generate_samples — public PIL API
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_samples(
        self,
        class_label: int | str,
        num_images:  int  = 4,
        device:      Union[str, torch.device] = "cuda",
        use_ddim:    bool = True,
        ddim_steps:  int  = 50,
    ) -> list[Image.Image]:
        """
        Generate synthetic skin-lesion images.

        Parameters
        ----------
        class_label : int or str
            HAM10000 code ("nv" … "df") or integer index 0-6.
        num_images  : int
        device      : str | torch.device
        use_ddim    : bool   True = fast DDIM; False = full DDPM
        ddim_steps  : int    denoising steps when use_ddim=True

        Returns
        -------
        list[PIL.Image.Image]  – RGB 224×224 images
        """
        device = _resolve_device(device)
        cls_name = class_label if isinstance(class_label, str) \
                   else HAM_CLASSES[class_label]
        log.info(
            "Generating %d × '%s' using %s…",
            num_images, cls_name,
            f"DDIM({ddim_steps})" if use_ddim else "DDPM(full)",
        )

        if use_ddim:
            x = self.fast_sample(class_label, num_images, ddim_steps, device)
        else:
            x = self.sample(class_label, num_images, device)

        return _tensors_to_pil(x)

    # ─────────────────────────────────────────────────────────────────────
    #  CHECKPOINT I/O
    # ─────────────────────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        epoch:      int,
        optimizer:  torch.optim.Optimizer,
        loss:       float,
        ckpt_dir:   Union[str, Path] = "checkpoints_diffusers",
        prefix:     str = "ham_diffusion",
    ) -> Path:
        """Save model + scheduler + optimizer state."""
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out = ckpt_dir / f"{prefix}_epoch{epoch:04d}.pt"

        torch.save({
            "epoch":            epoch,
            "loss":             loss,
            "model_state":      self.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "unet_config":      dict(self.unet.config),
            "cfg_null_prob":    self.cfg_null_prob,
            "cfg_scale":        self.cfg_scale,
            "T":                self.T,
        }, out)
        log.info("Checkpoint → %s  (epoch=%d, loss=%.6f)", out, epoch, loss)
        return out

    @classmethod
    def load_checkpoint(
        cls,
        ckpt_path: Union[str, Path],
        device:    Union[str, torch.device] = "cpu",
    ) -> tuple["HAM10000DiffusionModel", dict]:
        """Load and return (model, checkpoint_metadata_dict)."""
        ckpt = torch.load(ckpt_path, map_location=device)
        model = cls(
            cfg_null_prob = ckpt.get("cfg_null_prob", 0.1),
            cfg_scale     = ckpt.get("cfg_scale",     1.0),
        )
        model.load_state_dict(ckpt["model_state"])
        model = model.to(torch.device(device))
        log.info(
            "Loaded checkpoint %s  (epoch=%d, loss=%.6f)",
            ckpt_path, ckpt.get("epoch", -1), ckpt.get("loss", float("nan")),
        )
        return model, ckpt

    # ─────────────────────────────────────────────────────────────────────

    def _model_summary(self) -> str:
        n = sum(p.numel() for p in self.unet.parameters()) / 1e6
        return f"UNet2DModel({n:.1f}M params, {IMAGE_SIZE}×{IMAGE_SIZE})"

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"HAM10000DiffusionModel(\n"
            f"  {self._model_summary()}\n"
            f"  scheduler = DDPMScheduler(T={self.T})\n"
            f"  cfg_scale = {self.cfg_scale}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA unavailable — using CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _resolve_label(class_label: int | str) -> int:
    if isinstance(class_label, str):
        if class_label not in HAM_LABEL_MAP:
            raise ValueError(f"Unknown class '{class_label}'. Valid: {HAM_CLASSES}")
        return HAM_LABEL_MAP[class_label]
    return int(class_label)


def _tensors_to_pil(x: torch.Tensor) -> list[Image.Image]:
    """Convert (N, 3, H, W) float tensor in [-1,1] to list of PIL RGB images."""
    x     = (x.clamp(-1.0, 1.0) + 1.0) / 2.0    # → [0, 1]
    x     = (x * 255.0).byte().cpu().numpy()      # → uint8
    x     = x.transpose(0, 2, 3, 1)               # (N,H,W,C)
    return [Image.fromarray(img, mode="RGB") for img in x]
