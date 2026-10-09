"""
DDPM Wrapper — Orchestrates U-Net + Noise Scheduler
=====================================================
Provides:
  • forward()           — training loss for a batch
  • sample()            — full DDPM reverse denoising (T → 0)
  • ddim_sample()       — fast DDIM sampler with configurable steps
  • generate_samples()  — user-facing inference API (returns PIL images)
  • save_checkpoint() / load_checkpoint()
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from PIL import Image
import numpy as np

from .noise_scheduler import LinearNoiseScheduler
from .unet import ClassConditionalUNet

log = logging.getLogger(__name__)

# HAM10000 class mapping (index → code)
HAM_CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
HAM_LABEL_MAP = {c: i for i, c in enumerate(HAM_CLASSES)}


class DDPM(nn.Module):
    """
    Class-Conditional DDPM (Ho et al. 2020) for HAM10000 skin-lesion synthesis.

    Parameters
    ----------
    unet : ClassConditionalUNet
        Noise-prediction network ε_θ.
    scheduler : LinearNoiseScheduler
        Noise schedule (β₁…β_T, precomputed √ᾱ_t, etc.).
    image_size : int
        Spatial resolution (default 224).
    num_classes : int
        Number of conditioning classes (7 for HAM10000).
    cfg_scale : float
        Classifier-free guidance scale (w).  w=1 means no guidance;
        w=7.5 is a reasonable default for strong conditioning.
    cfg_null_prob : float
        Probability of replacing the class label with the null token
        during training (enables CFG at inference).  Default 0.1.
    """

    def __init__(
        self,
        unet:          ClassConditionalUNet,
        scheduler:     LinearNoiseScheduler,
        image_size:    int   = 224,
        num_classes:   int   = 7,
        cfg_scale:     float = 1.0,
        cfg_null_prob: float = 0.1,
    ):
        super().__init__()
        self.unet          = unet
        self.scheduler     = scheduler
        self.image_size    = image_size
        self.num_classes   = num_classes
        self.cfg_scale     = cfg_scale
        self.cfg_null_prob = cfg_null_prob
        # null token index = num_classes (vocab size = num_classes + 1)
        self.null_token    = num_classes

    # ─────────────────────────────────────────────────────────────────────
    #  TRAINING
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x_0:   Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Compute the DDPM training loss for a batch.

        Steps
        -----
        1. Sample random timesteps t ~ Uniform(0, T-1).
        2. Sample noise ε ~ N(0, I).
        3. Compute x_t via closed-form forward process.
        4. Predict ε̂ = ε_θ(x_t, t, c).
        5. Optionally replace class labels with null token (CFG training).
        6. Return MSE(ε̂, ε).

        Parameters
        ----------
        x_0    : (B, 3, H, W)  clean images in [-1, 1]
        labels : (B,)  integer class labels

        Returns
        -------
        loss : scalar tensor
        """
        B      = x_0.size(0)
        device = x_0.device

        # Random timesteps
        t = torch.randint(0, self.scheduler.T, (B,), device=device, dtype=torch.long)

        # Classifier-free guidance: randomly drop class conditioning
        if self.cfg_null_prob > 0.0:
            mask   = torch.rand(B, device=device) < self.cfg_null_prob
            labels = torch.where(mask, torch.full_like(labels, self.null_token), labels)

        # Forward diffusion  q(x_t | x_0)
        x_t, noise = self.scheduler.q_sample(x_0, t)
        x_t        = x_t.to(dtype=x_0.dtype)

        # Predict noise
        eps_pred = self.unet(x_t, t, labels)

        # Loss
        return self.scheduler.training_loss(eps_pred, noise)

    # ─────────────────────────────────────────────────────────────────────
    #  INFERENCE — FULL DDPM REVERSE DIFFUSION
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        class_label:  int | Tensor,
        num_images:   int = 1,
        device:       str | torch.device = "cuda",
        show_progress: bool = True,
    ) -> Tensor:
        """
        Full DDPM reverse-diffusion denoising (T → 0 steps).

        Returns
        -------
        x_0 : (N, 3, H, W)  in [-1, 1]
        """
        device  = torch.device(device)
        unet    = self.unet.to(device).eval()
        sched   = self.scheduler

        # Class label tensor
        if isinstance(class_label, int):
            labels = torch.full((num_images,), class_label, dtype=torch.long, device=device)
        else:
            labels = class_label.to(device)
            if labels.ndim == 0:
                labels = labels.unsqueeze(0).expand(num_images)

        # Start from pure noise x_T ~ N(0, I)
        x = torch.randn(num_images, self.unet.in_channels,
                        self.image_size, self.image_size, device=device)

        # Reverse T → 0
        timesteps = list(range(self.scheduler.T - 1, -1, -1))
        for i, t_val in enumerate(timesteps):
            t_batch = torch.full((num_images,), t_val, dtype=torch.long, device=device)

            if self.cfg_scale > 1.0:
                # Classifier-Free Guidance:  ε̂ = ε_uncond + w·(ε_cond - ε_uncond)
                null_labels = torch.full_like(labels, self.null_token)
                eps_cond   = unet(x, t_batch, labels)
                eps_uncond = unet(x, t_batch, null_labels)
                eps        = eps_uncond + self.cfg_scale * (eps_cond - eps_uncond)
            else:
                eps = unet(x, t_batch, labels)

            x = sched.p_sample(eps, x, t_batch)

            if show_progress and (i % 100 == 0 or t_val == 0):
                log.info("  Sampling  t=%4d / %4d …", t_val, self.scheduler.T)

        return x

    # ─────────────────────────────────────────────────────────────────────
    #  INFERENCE — DDIM (FAST)
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        class_label:    int | Tensor,
        num_images:     int   = 1,
        num_steps:      int   = 50,
        eta:            float = 0.0,
        device:         str | torch.device = "cuda",
    ) -> Tensor:
        """
        DDIM sampler with ``num_steps`` denoising steps (much faster than DDPM).

        Parameters
        ----------
        num_steps : int   number of function evaluations (default 50).
        eta       : float stochasticity (0 = deterministic, 1 = full DDPM noise).

        Returns
        -------
        x_0 : (N, 3, H, W)  in [-1, 1]
        """
        device = torch.device(device)
        unet   = self.unet.to(device).eval()
        sched  = self.scheduler

        if isinstance(class_label, int):
            labels = torch.full((num_images,), class_label, dtype=torch.long, device=device)
        else:
            labels = class_label.to(device)
            if labels.ndim == 0:
                labels = labels.unsqueeze(0).expand(num_images)

        # Sub-sequence of timesteps (uniformly spaced in [0, T-1])
        step_size = sched.T // num_steps
        t_seq     = list(range(sched.T - 1, -1, -step_size))[:num_steps]
        t_seq_prev = t_seq[1:] + [-1]     # previous timestep (-1 → t=0 boundary)

        x = torch.randn(num_images, self.unet.in_channels,
                        self.image_size, self.image_size, device=device)

        for t_curr, t_prev in zip(t_seq, t_seq_prev):
            t_batch      = torch.full((num_images,), t_curr, dtype=torch.long, device=device)
            t_prev_batch = torch.full((num_images,), max(t_prev, 0), dtype=torch.long, device=device)

            if self.cfg_scale > 1.0:
                null_labels = torch.full_like(labels, self.null_token)
                eps_cond   = unet(x, t_batch, labels)
                eps_uncond = unet(x, t_batch, null_labels)
                eps        = eps_uncond + self.cfg_scale * (eps_cond - eps_uncond)
            else:
                eps = unet(x, t_batch, labels)

            x = sched.ddim_step(eps, x, t_batch, t_prev_batch, eta=eta)

        return x

    # ─────────────────────────────────────────────────────────────────────
    #  PUBLIC API — generate_samples
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_samples(
        self,
        class_label:  int | str,
        num_images:   int   = 4,
        device:       str | torch.device = "cuda",
        use_ddim:     bool  = True,
        ddim_steps:   int   = 50,
        eta:          float = 0.0,
    ) -> list[Image.Image]:
        """
        Generate synthetic skin-lesion images conditioned on a class label.

        Parameters
        ----------
        class_label : int or str
            Integer index (0-6) or string code ("nv", "mel", "bkl", …).
        num_images  : int
            Number of images to generate (default 4).
        device      : str | torch.device
            Target device (default "cuda"; falls back to "cpu" if unavailable).
        use_ddim    : bool
            Use fast DDIM sampling (True, default) or full DDPM reverse (False).
        ddim_steps  : int
            Number of DDIM denoising steps when ``use_ddim=True`` (default 50).
        eta         : float
            DDIM stochasticity (0 = deterministic, 1 = DDPM-equivalent).

        Returns
        -------
        List of PIL.Image objects (RGB, 224×224).

        Examples
        --------
        >>> pils = model.generate_samples("mel", num_images=8)
        >>> pils[0].save("melanoma_synth.png")
        """
        # Resolve device
        if isinstance(device, str) and device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA not available — falling back to CPU sampling.")
            device = "cpu"
        device = torch.device(device)

        # Resolve class label
        if isinstance(class_label, str):
            if class_label not in HAM_LABEL_MAP:
                raise ValueError(
                    f"Unknown class '{class_label}'. Choose from {HAM_CLASSES}."
                )
            class_label = HAM_LABEL_MAP[class_label]

        class_name = HAM_CLASSES[class_label] if class_label < len(HAM_CLASSES) else "unknown"
        log.info(
            "Generating %d image(s) for class '%s' (idx=%d) using %s …",
            num_images, class_name, class_label,
            f"DDIM({ddim_steps} steps)" if use_ddim else "DDPM(full)",
        )

        if use_ddim:
            x = self.ddim_sample(
                class_label=class_label,
                num_images=num_images,
                num_steps=ddim_steps,
                eta=eta,
                device=device,
            )
        else:
            x = self.sample(
                class_label=class_label,
                num_images=num_images,
                device=device,
            )

        # x in [-1, 1] → [0, 255] uint8
        x    = (x.clamp(-1.0, 1.0) + 1.0) / 2.0    # → [0, 1]
        x    = (x * 255.0).byte().cpu().numpy()      # → uint8 numpy
        # (N, C, H, W) → (N, H, W, C)
        x    = x.transpose(0, 2, 3, 1)

        pil_images = [Image.fromarray(img, mode="RGB") for img in x]
        log.info("Done. Returning %d PIL images.", len(pil_images))
        return pil_images

    # ─────────────────────────────────────────────────────────────────────
    #  CHECKPOINT I/O
    # ─────────────────────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        epoch:      int,
        optimizer:  torch.optim.Optimizer,
        loss:       float,
        ckpt_dir:   str | Path = "checkpoints",
        prefix:     str = "ddpm",
    ) -> Path:
        """
        Save model state to ``{ckpt_dir}/{prefix}_epoch{epoch:04d}.pt``.

        Returns the saved checkpoint path.
        """
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out = ckpt_dir / f"{prefix}_epoch{epoch:04d}.pt"

        torch.save(
            {
                "epoch":           epoch,
                "loss":            loss,
                "model_state":     self.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "num_classes":     self.num_classes,
                "image_size":      self.image_size,
                "cfg_scale":       self.cfg_scale,
                "scheduler_T":     self.scheduler.T,
            },
            out,
        )
        log.info("Checkpoint saved → %s  (epoch=%d, loss=%.6f)", out, epoch, loss)
        return out

    @classmethod
    def load_checkpoint(
        cls,
        ckpt_path: str | Path,
        device:    str | torch.device = "cpu",
        unet_kwargs: dict | None = None,
    ) -> tuple["DDPM", dict]:
        """
        Re-instantiate a DDPM model from a saved checkpoint.

        Parameters
        ----------
        ckpt_path    : path to `.pt` checkpoint file.
        device       : torch device.
        unet_kwargs  : extra kwargs passed to ClassConditionalUNet if provided;
                       otherwise the defaults are used.

        Returns
        -------
        (model, metadata_dict)
        """
        ckpt       = torch.load(ckpt_path, map_location=device)
        num_classes = ckpt.get("num_classes", 7)
        image_size  = ckpt.get("image_size",  224)
        T           = ckpt.get("scheduler_T", 1000)
        cfg_scale   = ckpt.get("cfg_scale",   1.0)

        unet_kw = unet_kwargs or {}
        unet    = ClassConditionalUNet(num_classes=num_classes, **unet_kw)
        sched   = LinearNoiseScheduler(num_timesteps=T, device=device)

        model = cls(
            unet=unet, scheduler=sched,
            image_size=image_size, num_classes=num_classes,
            cfg_scale=cfg_scale,
        )
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)

        log.info(
            "Loaded checkpoint %s  (epoch=%d, loss=%.6f)",
            ckpt_path, ckpt.get("epoch", -1), ckpt.get("loss", float("nan")),
        )
        return model, ckpt

    # ─────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"DDPM(\n"
            f"  unet      = {self.unet}\n"
            f"  scheduler = {self.scheduler}\n"
            f"  image_size= {self.image_size}  "
            f"  num_classes={self.num_classes}  "
            f"  cfg_scale={self.cfg_scale}\n"
            f")"
        )
