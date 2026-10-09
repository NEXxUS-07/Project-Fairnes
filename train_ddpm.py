"""
train_ddpm.py — Training Script for Class-Conditional DDPM on HAM10000
========================================================================
Usage:
    python train_ddpm.py \
        --train_csv  splits/train.csv \
        --image_dir  path/to/HAM10000_images \
        --epochs     200 \
        --batch_size 16 \
        --lr         2e-4 \
        --ckpt_dir   checkpoints/ \
        --log_dir    runs/ddpm \
        --save_every 10

Features:
  • Mixed-precision training (torch.cuda.amp) for 2-3× GPU speedup.
  • EMA (Exponential Moving Average) of model weights for better sample quality.
  • Gradient clipping  (max_norm=1.0) for stable training.
  • LR warm-up over 500 steps followed by cosine annealing.
  • Saves sample grids every ``--sample_every`` epochs to ``samples/`` dir.
  • Logs training loss + LR to TensorBoard (optional).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torchvision.utils import save_image

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

from ddpm import DDPM, LinearNoiseScheduler, ClassConditionalUNet
from dataset import HAMDataset, CLASS_TO_IDX, IDX_TO_CLASS

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  EMA HELPER
# ═══════════════════════════════════════════════════════════════════════════

class EMA:
    """
    Exponential Moving Average of model weights.
    Use ``ema.apply_shadow()`` before generation and
    ``ema.restore()`` to resume training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow = {k: v.clone().detach() for k, v in model.named_parameters()}
        self._backup: dict | None = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self._backup = {k: v.clone() for k, v in model.named_parameters()}
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        if self._backup is None:
            raise RuntimeError("No shadow backup. Call apply_shadow() first.")
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = None


# ═══════════════════════════════════════════════════════════════════════════
#  LEARNING-RATE SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════

def build_lr_schedule(
    optimizer:    torch.optim.Optimizer,
    warmup_steps: int,
    total_steps:  int,
) -> torch.optim.lr_scheduler.SequentialLR:
    """
    Linear warm-up for ``warmup_steps`` then cosine annealing to 1e-6.
    """
    warmup   = LambdaLR(optimizer, lambda s: min(1.0, s / max(warmup_steps, 1)))
    cosine   = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    schedule = SequentialLR(optimizer, schedulers=[warmup, cosine],
                            milestones=[warmup_steps])
    return schedule


# ═══════════════════════════════════════════════════════════════════════════
#  SAMPLE GRID  (for qualitative inspection during training)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def log_sample_grid(
    ddpm_model: DDPM,
    epoch:      int,
    device:     torch.device,
    save_dir:   Path,
    ddim_steps: int = 50,
    num_per_class: int = 2,
):
    """Generate 2 sample images per class and save as a grid PNG."""
    save_dir.mkdir(parents=True, exist_ok=True)
    all_tensors = []

    ddpm_model.unet.eval()
    for cls_idx in range(ddpm_model.num_classes):
        samples = ddpm_model.ddim_sample(
            class_label=cls_idx,
            num_images=num_per_class,
            num_steps=ddim_steps,
            device=device,
        )                                       # (N, 3, H, W) in [-1, 1]
        all_tensors.append(samples)

    grid_tensor = torch.cat(all_tensors, dim=0)   # (7*N, 3, H, W)
    out = save_dir / f"samples_epoch{epoch:04d}.png"
    save_image(
        (grid_tensor + 1.0) / 2.0,          # [-1,1] → [0,1]
        str(out),
        nrow=num_per_class * ddpm_model.num_classes // 7,
        padding=4,
        pad_value=1.0,
    )
    log.info("Sample grid saved → %s", out)
    ddpm_model.unet.train()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace):
    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps"  if torch.backends.mps.is_available() else "cpu")
    )
    log.info("Using device: %s", device)

    # ── Dataset ─────────────────────────────────────────────────────────
    train_ds = HAMDataset(
        image_dir=args.image_dir,
        metadata=args.train_csv,
        split="train",
        image_size=args.image_size,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    log.info(
        "Training set: %d images | %d batches/epoch | batch_size=%d",
        len(train_ds), len(train_loader), args.batch_size,
    )

    # ── Model ────────────────────────────────────────────────────────────
    scheduler = LinearNoiseScheduler(
        num_timesteps=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    unet = ClassConditionalUNet(
        in_channels=3,
        num_classes=args.num_classes,
        base_ch=args.base_channels,
        ch_mults=tuple(args.channel_mults),
        num_res=args.num_res_blocks,
        dropout=args.dropout,
    ).to(device)

    ddpm_model = DDPM(
        unet=unet,
        scheduler=scheduler,
        image_size=args.image_size,
        num_classes=args.num_classes,
        cfg_scale=args.cfg_scale,
        cfg_null_prob=args.cfg_null_prob,
    ).to(device)

    log.info("Model: %s", ddpm_model.unet)

    # Resume from checkpoint if provided
    start_epoch = 1
    if args.resume:
        ddpm_model, ckpt_meta = DDPM.load_checkpoint(args.resume, device=device)
        start_epoch = ckpt_meta.get("epoch", 0) + 1
        log.info("Resumed from epoch %d.", start_epoch - 1)

    # ── EMA ─────────────────────────────────────────────────────────────
    ema = EMA(ddpm_model, decay=args.ema_decay)

    # ── Optimizer + LR Schedule ──────────────────────────────────────────
    optimizer = torch.optim.Adam(
        ddpm_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = args.warmup_steps
    lr_sched     = build_lr_schedule(optimizer, warmup_steps, total_steps)

    # ── AMP scaler ──────────────────────────────────────────────────────
    use_amp  = (device.type == "cuda") and args.amp
    scaler   = GradScaler(enabled=use_amp)

    # ── TensorBoard ─────────────────────────────────────────────────────
    writer = None
    if _TB_AVAILABLE and args.log_dir:
        writer = SummaryWriter(log_dir=args.log_dir)
        log.info("TensorBoard → %s", args.log_dir)

    # ── Paths ────────────────────────────────────────────────────────────
    ckpt_dir   = Path(args.ckpt_dir)
    sample_dir = Path(args.sample_dir)

    # ────────────────────────────────────────────────────────────────────
    global_step = (start_epoch - 1) * len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        ddpm_model.train()
        epoch_loss = 0.0
        t0         = time.perf_counter()

        for batch_idx, (images, labels, _, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Scale images from [0,1] normalised to [-1, 1] for DDPM
            # (HAMDataset returns ImageNet-normalised tensors; undo that
            #  and re-scale to [-1, 1] for the diffusion process)
            _mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            _std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            images = images * _std + _mean          # → [0, 1]
            images = images * 2.0 - 1.0             # → [-1, 1]

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp, dtype=torch.float16):
                loss = ddpm_model(images, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(ddpm_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_sched.step()
            ema.update(ddpm_model)

            epoch_loss  += loss.item()
            global_step += 1

            if writer:
                writer.add_scalar("loss/step",  loss.item(), global_step)
                writer.add_scalar("lr/step", optimizer.param_groups[0]["lr"], global_step)

            if batch_idx % args.log_interval == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                log.info(
                    "Epoch %4d | step %4d/%4d | loss=%.6f | lr=%.2e",
                    epoch, batch_idx, len(train_loader),
                    loss.item(), current_lr,
                )

        # ── End of epoch ────────────────────────────────────────────────
        avg_loss   = epoch_loss / len(train_loader)
        epoch_time = time.perf_counter() - t0

        log.info(
            "═══ Epoch %4d/%4d  avg_loss=%.6f  time=%.1fs ═══",
            epoch, args.epochs, avg_loss, epoch_time,
        )

        if writer:
            writer.add_scalar("loss/epoch", avg_loss, epoch)

        # ── Checkpoint ──────────────────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            # Save EMA weights
            ema.apply_shadow(ddpm_model)
            ckpt_path = ddpm_model.save_checkpoint(
                epoch=epoch,
                optimizer=optimizer,
                loss=avg_loss,
                ckpt_dir=ckpt_dir,
                prefix="ddpm_ema",
            )
            ema.restore(ddpm_model)

            # Also save raw (non-EMA) weights for resuming training
            ddpm_model.save_checkpoint(
                epoch=epoch,
                optimizer=optimizer,
                loss=avg_loss,
                ckpt_dir=ckpt_dir,
                prefix="ddpm_raw",
            )

        # ── Sample grid ─────────────────────────────────────────────────
        if args.sample_every > 0 and epoch % args.sample_every == 0:
            ema.apply_shadow(ddpm_model)
            log_sample_grid(ddpm_model, epoch, device, sample_dir,
                            ddim_steps=args.ddim_steps)
            ema.restore(ddpm_model)

    if writer:
        writer.close()
    log.info("Training complete.")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Class-Conditional DDPM on HAM10000",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--train_csv",    required=True, help="Path to splits/train.csv")
    p.add_argument("--image_dir",    required=True, help="Root HAM10000 image directory")
    p.add_argument("--image_size",   type=int,   default=224)
    p.add_argument("--num_workers",  type=int,   default=4)

    # Diffusion schedule
    p.add_argument("--T",           type=int,   default=1000, help="Diffusion timesteps")
    p.add_argument("--beta_start",  type=float, default=1e-4)
    p.add_argument("--beta_end",    type=float, default=2e-2)

    # U-Net architecture
    p.add_argument("--num_classes",    type=int,   default=7)
    p.add_argument("--base_channels",  type=int,   default=128)
    p.add_argument("--channel_mults",  type=int,   nargs="+", default=[1, 2, 2, 4])
    p.add_argument("--num_res_blocks", type=int,   default=2)
    p.add_argument("--dropout",        type=float, default=0.1)

    # Training
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--weight_decay",  type=float, default=0.0)
    p.add_argument("--warmup_steps",  type=int,   default=500)
    p.add_argument("--ema_decay",     type=float, default=0.9999)
    p.add_argument("--amp",           action="store_true", default=True,
                   help="Mixed-precision training (CUDA only)")

    # CFG
    p.add_argument("--cfg_scale",     type=float, default=1.0,
                   help="CFG guidance scale at inference (1=off, 3-7=recommended)")
    p.add_argument("--cfg_null_prob", type=float, default=0.1,
                   help="Probability of null token during training")

    # Logging / saving
    p.add_argument("--ckpt_dir",    default="checkpoints")
    p.add_argument("--log_dir",     default="runs/ddpm")
    p.add_argument("--sample_dir",  default="samples")
    p.add_argument("--save_every",  type=int, default=10, help="Checkpoint every N epochs")
    p.add_argument("--sample_every",type=int, default=10, help="Generate samples every N epochs")
    p.add_argument("--ddim_steps",  type=int, default=50,  help="DDIM steps for sample grids")
    p.add_argument("--log_interval",type=int, default=50,  help="Log every N batches")
    p.add_argument("--resume",      type=str, default=None, help="Path to checkpoint to resume")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
