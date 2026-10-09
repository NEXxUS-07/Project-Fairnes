"""
train_diffusion.py — Training Script for HAM10000DiffusionModel
================================================================
Uses HuggingFace diffusers + Accelerate for multi-GPU / mixed-precision.

Usage:
    # Single GPU:
    python train_diffusion.py \
        --train_csv  splits/train.csv \
        --image_dir  path/to/HAM10000_images \
        --epochs     200 \
        --batch_size 16 \
        --lr         2e-4 \
        --ckpt_dir   checkpoints_diffusers/

    # Multi-GPU with Accelerate:
    accelerate launch --num_processes=2 train_diffusion.py \
        --train_csv splits/train.csv --image_dir ...

Features:
    • HuggingFace Accelerate (multi-GPU, BF16/FP16 AMP, gradient accumulation)
    • EMA of UNet weights for clean inference
    • Cosine LR schedule with warmup
    • Checkpoint every --save_every epochs
    • Sample grid saved every --sample_every epochs
    • TensorBoard + Wandb (both optional)
"""

from __future__ import annotations

import argparse
import logging
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# ── Accelerate ────────────────────────────────────────────────────────────
try:
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
    _ACCELERATE_OK = True
except ImportError:
    _ACCELERATE_OK = False

# ── TensorBoard (optional) ────────────────────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_OK = True
except ImportError:
    _TB_OK = False

from diffusion_model import HAM10000DiffusionModel, HAM_CLASSES, _tensors_to_pil
from dataset import HAMDataset

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  EMA HELPER
# ═══════════════════════════════════════════════════════════════════════════

class EMA:
    """Exponential Moving Average over model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow = {k: v.clone().detach() for k, v in model.named_parameters()}
        self._backup: dict | None = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model: torch.nn.Module):
        self._backup = {k: v.clone() for k, v in model.named_parameters()}
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        if self._backup is None:
            raise RuntimeError("No shadow backup. Call apply_shadow() first.")
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = None


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET WRAPPER  — undoes ImageNet normalisation → rescale to [-1, 1]
# ═══════════════════════════════════════════════════════════════════════════

class DDPMDataset(torch.utils.data.Dataset):
    """
    Thin wrapper around HAMDataset that re-scales images to [-1, 1]
    (HAMDataset returns ImageNet-normalised tensors; DDPM expects [-1, 1]).
    """

    def __init__(self, ham_dataset: HAMDataset):
        self.ds = ham_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, label, lesion_id, skin_tone = self.ds[idx]
        # Undo ImageNet normalisation: x = (x_norm * std + mean) * 2 - 1
        mean = IMAGENET_MEAN.squeeze()   # (3,)
        std  = IMAGENET_STD.squeeze()    # (3,)
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]   # → [0, 1]
        img = img * 2.0 - 1.0                      # → [-1, 1]
        return img, label


# ═══════════════════════════════════════════════════════════════════════════
#  SAMPLE GRID HELPER
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def save_sample_grid(
    model:      HAM10000DiffusionModel,
    epoch:      int,
    out_dir:    Path,
    device:     torch.device,
    num_per_cls: int = 2,
    ddim_steps:  int = 50,
):
    """Generate 2 images per class and save as one contact-sheet PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.unet.eval()
    all_x: list[torch.Tensor] = []

    for cls_idx in range(len(HAM_CLASSES)):
        x = model.fast_sample(
            class_label=cls_idx,
            num_images=num_per_cls,
            num_steps=ddim_steps,
            device=device,
        )
        all_x.append(x)

    grid = torch.cat(all_x, dim=0)             # (7*num_per_cls, 3, H, W)
    out  = out_dir / f"sample_epoch{epoch:04d}.png"
    save_image((grid + 1.0) / 2.0, str(out), nrow=num_per_cls * len(HAM_CLASSES) // 7,
               padding=4, pad_value=1.0)
    log.info("Sample grid → %s", out)
    model.unet.train()


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace):
    # ── Build model ──────────────────────────────────────────────────────
    model = HAM10000DiffusionModel(
        cfg_null_prob = args.cfg_null_prob,
        cfg_scale     = args.cfg_scale,
    )

    start_epoch = 1
    if args.resume:
        model, ckpt_meta = HAM10000DiffusionModel.load_checkpoint(args.resume)
        start_epoch = ckpt_meta.get("epoch", 0) + 1
        log.info("Resumed from epoch %d.", start_epoch - 1)

    log.info("Model: %s", model)

    # ── Dataset + DataLoader ─────────────────────────────────────────────
    ham_ds = HAMDataset(
        image_dir=args.image_dir,
        metadata=args.train_csv,
        split="train",
        image_size=args.image_size,
    )
    ddpm_ds = DDPMDataset(ham_ds)
    loader  = DataLoader(
        ddpm_ds,
        batch_size    = args.batch_size,
        shuffle       = True,
        num_workers   = args.num_workers,
        pin_memory    = True,
        drop_last     = True,
        persistent_workers = (args.num_workers > 0),
    )
    log.info("Training: %d images | %d batches/epoch", len(ham_ds), len(loader))

    # ── Optimizer + LR schedule ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr            = args.lr,
        betas         = (0.9, 0.999),
        weight_decay  = args.weight_decay,
    )
    total_steps = args.epochs * len(loader)

    try:
        from diffusers.optimization import get_cosine_schedule_with_warmup
        lr_sched = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps    = args.warmup_steps,
            num_training_steps  = total_steps,
        )
    except Exception:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        lr_sched = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # ── Accelerate setup ─────────────────────────────────────────────────
    if _ACCELERATE_OK:
        project_cfg = ProjectConfiguration(
            project_dir       = args.ckpt_dir,
            automatic_checkpoint_naming = True,
            total_limit       = 5,
        )
        accelerator = Accelerator(
            mixed_precision     = args.mixed_precision,
            gradient_accumulation_steps = args.grad_accum,
            log_with            = "tensorboard" if args.log_dir else None,
            project_config      = project_cfg,
        )
        device = accelerator.device
        model, optimizer, loader, lr_sched = accelerator.prepare(
            model, optimizer, loader, lr_sched
        )
        if args.log_dir:
            accelerator.init_trackers("ham10000_ddpm", config=vars(args))
        is_main = accelerator.is_main_process
    else:
        # Fallback: manual AMP
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = model.to(device)
        accelerator = None
        is_main     = True

    log.info("Device: %s", device)

    # ── Raw UNet reference for EMA (must unwrap from accelerator) ────────
    raw_unet = model.unet if not _ACCELERATE_OK else \
               getattr(model, "module", model).unet
    ema = EMA(raw_unet, decay=args.ema_decay)

    scaler = torch.cuda.amp.GradScaler(
        enabled=(str(device) == "cuda" and not _ACCELERATE_OK)
    )

    # ── TensorBoard (fallback when not using accelerate) ─────────────────
    writer = None
    if _TB_OK and args.log_dir and not _ACCELERATE_OK:
        writer = SummaryWriter(log_dir=args.log_dir)

    ckpt_dir   = Path(args.ckpt_dir)
    sample_dir = Path(args.sample_dir)
    global_step = (start_epoch - 1) * len(loader)

    # ════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for batch_idx, (images, labels) in enumerate(loader):
            if not _ACCELERATE_OK:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

            with (accelerator.accumulate(model) if _ACCELERATE_OK else
                  torch.cuda.amp.autocast(enabled=(str(device)=="cuda"))):
                loss = model.train_step(images, labels) \
                       if not _ACCELERATE_OK else \
                       model.module.train_step(images, labels) \
                       if hasattr(model, "module") else model.train_step(images, labels)

            if _ACCELERATE_OK:
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_sched.step()

            ema.update(raw_unet)
            epoch_loss  += loss.item()
            global_step += 1

            if writer:
                writer.add_scalar("loss/step", loss.item(), global_step)

            if is_main and batch_idx % args.log_interval == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                log.info(
                    "Epoch %4d | step %4d/%4d | loss=%.6f | lr=%.2e",
                    epoch, batch_idx, len(loader), loss.item(), cur_lr,
                )

        # ── End of epoch ─────────────────────────────────────────────────
        avg_loss = epoch_loss / len(loader)
        secs     = time.perf_counter() - t0
        if is_main:
            log.info(
                "═══ Epoch %4d/%4d  avg_loss=%.6f  %.1fs ═══",
                epoch, args.epochs, avg_loss, secs,
            )
            if writer:
                writer.add_scalar("loss/epoch", avg_loss, epoch)

        # ── Checkpoint ───────────────────────────────────────────────────
        if is_main and (epoch % args.save_every == 0 or epoch == args.epochs):
            # EMA weights
            raw_model = getattr(model, "module", model)
            ema.apply_shadow(raw_unet)
            raw_model.save_checkpoint(
                epoch=epoch, optimizer=optimizer,
                loss=avg_loss, ckpt_dir=ckpt_dir, prefix="ema",
            )
            ema.restore(raw_unet)
            # Raw weights
            raw_model.save_checkpoint(
                epoch=epoch, optimizer=optimizer,
                loss=avg_loss, ckpt_dir=ckpt_dir, prefix="raw",
            )

        # ── Sample grid ──────────────────────────────────────────────────
        if is_main and args.sample_every > 0 and epoch % args.sample_every == 0:
            raw_model = getattr(model, "module", model)
            ema.apply_shadow(raw_unet)
            save_sample_grid(raw_model, epoch, sample_dir, device,
                             ddim_steps=args.ddim_steps)
            ema.restore(raw_unet)

    if writer:
        writer.close()
    if _ACCELERATE_OK and is_main:
        accelerator.end_training()
    log.info("Training complete.")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train HAM10000DiffusionModel (HuggingFace diffusers)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--train_csv",   required=True)
    p.add_argument("--image_dir",   required=True)
    p.add_argument("--image_size",  type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    # Training
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int,   default=500)
    p.add_argument("--ema_decay",    type=float, default=0.9999)
    p.add_argument("--grad_accum",   type=int,   default=1)
    p.add_argument("--mixed_precision", default="fp16",
                   choices=["no", "fp16", "bf16"])
    # CFG
    p.add_argument("--cfg_scale",    type=float, default=1.0)
    p.add_argument("--cfg_null_prob",type=float, default=0.1)
    # Logging / saving
    p.add_argument("--ckpt_dir",     default="checkpoints_diffusers")
    p.add_argument("--log_dir",      default="runs/diffusers")
    p.add_argument("--sample_dir",   default="samples_diffusers")
    p.add_argument("--save_every",   type=int, default=10)
    p.add_argument("--sample_every", type=int, default=10)
    p.add_argument("--ddim_steps",   type=int, default=50)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--resume",       default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
