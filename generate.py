"""
generate.py — Unified Inference & Image Generation Script
==========================================================
Supports two model backends:
  1. HAM10000DiffusionModel  (HuggingFace diffusers — default)
  2. Custom DDPM (ddpm.DDPM from ddpm/model.py)

All generated images are saved to a class-subfolder structure:
    {output_dir}/{class_code}/{prefix}_{class_code}_{idx:04d}.png

Key public function:
    generate_samples(class_label, num_images, ...) → list[PIL.Image]

CLI usage:
    # Generate 8 melanoma images (diffusers model):
    python generate.py \\
        --checkpoint checkpoints_diffusers/ema_epoch0200.pt \\
        --class_label mel \\
        --num_images  8 \\
        --output_dir  generated/

    # All 7 classes, 4 each, CFG=5, diffusers backend:
    python generate.py \\
        --checkpoint checkpoints_diffusers/ema_epoch0200.pt \\
        --all_classes --num_images 4 --cfg_scale 5.0

    # Custom DDPM backend:
    python generate.py \\
        --checkpoint checkpoints/ddpm_ema_epoch0200.pt \\
        --backend custom \\
        --class_label bcc --num_images 4

Dependencies: torch, diffusers, pillow, torchvision
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

HAM_CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
HAM_FULL_NAMES = {
    "nv":    "Melanocytic Nevi",
    "mel":   "Melanoma",
    "bkl":   "Benign Keratosis",
    "bcc":   "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "vasc":  "Vascular Lesions",
    "df":    "Dermatofibroma",
}


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API — generate_samples
# ═══════════════════════════════════════════════════════════════════════════

def generate_samples(
    class_label:  Union[int, str],
    num_images:   int                       = 4,
    checkpoint:   Union[str, Path, None]    = None,
    model                                   = None,
    device:       Union[str, torch.device]  = "cuda",
    use_ddim:     bool                      = True,
    ddim_steps:   int                       = 50,
    cfg_scale:    Union[float, None]        = None,
    backend:      str                       = "diffusers",
) -> list[Image.Image]:
    """
    Generate synthetic HAM10000 skin-lesion images.

    Parameters
    ----------
    class_label : int | str
        HAM10000 class code ("nv", "mel", … "df") or integer index 0-6.
    num_images  : int
        Number of images to generate per call.
    checkpoint  : str | Path | None
        Path to a saved checkpoint (`.pt`).  Ignored if ``model`` is provided.
    model       : HAM10000DiffusionModel | DDPM | None
        Pre-loaded model.  Takes priority over ``checkpoint``.
    device      : str | torch.device
        Compute device.  Falls back to "cpu" if CUDA unavailable.
    use_ddim    : bool
        True = fast DDIM inference (default); False = full DDPM reverse loop.
    ddim_steps  : int
        DDIM denoising steps (ignored if use_ddim=False).
    cfg_scale   : float | None
        Override classifier-free guidance scale.  None → model default.
    backend     : {"diffusers", "custom"}
        Which model class to load when a checkpoint path is given.
        Ignored if ``model`` is provided directly.

    Returns
    -------
    list[PIL.Image.Image]  — RGB 224×224 images

    Examples
    --------
    >>> from generate import generate_samples
    >>> imgs = generate_samples("mel", num_images=4,
    ...     checkpoint="checkpoints_diffusers/ema_epoch0200.pt",
    ...     device="cuda", use_ddim=True, ddim_steps=50)
    >>> for i, img in enumerate(imgs):
    ...     img.save(f"melanoma_{i}.png")
    """
    # Resolve device
    device = _resolve_device(device)

    # Load model if not supplied
    if model is None:
        if checkpoint is None:
            raise ValueError("Provide either `model` or `checkpoint`.")
        model = _load_model(checkpoint, device, backend)

    # Override CFG scale
    if cfg_scale is not None:
        model.cfg_scale = cfg_scale

    return model.generate_samples(
        class_label = class_label,
        num_images  = num_images,
        device      = device,
        use_ddim    = use_ddim,
        ddim_steps  = ddim_steps,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SAVE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def save_images_to_class_dir(
    images:     list[Image.Image],
    output_dir: Union[str, Path],
    class_code: str,
    prefix:     str = "synth",
) -> list[Path]:
    """
    Save PIL images to:
        {output_dir}/{class_code}/{prefix}_{class_code}_{idx:04d}.png

    Returns the list of saved file paths.
    """
    output_dir  = Path(output_dir)
    class_dir   = output_dir / class_code
    class_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for idx, img in enumerate(images):
        out = class_dir / f"{prefix}_{class_code}_{idx:04d}.png"
        img.save(out)
        log.info("Saved → %s", out)
        saved_paths.append(out)
    return saved_paths


def save_contact_sheet(
    images:     list[Image.Image],
    output_dir: Union[str, Path],
    class_code: str,
    nrow:       int = 4,
) -> Path:
    """
    Save a contact-sheet grid of all generated images to:
        {output_dir}/grids/grid_{class_code}.png
    """
    tensors = torch.stack([TF.to_tensor(img) for img in images])
    grid    = make_grid(tensors, nrow=nrow, padding=4, pad_value=1.0)

    grid_dir = Path(output_dir) / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    out = grid_dir / f"grid_{class_code}.png"
    TF.to_pil_image(grid).save(str(out))
    log.info("Grid → %s", out)
    return out


def generate_all_classes(
    checkpoint:  Union[str, Path],
    num_images:  int  = 4,
    output_dir:  Union[str, Path] = "generated",
    device:      Union[str, torch.device] = "cuda",
    use_ddim:    bool = True,
    ddim_steps:  int  = 50,
    cfg_scale:   Union[float, None] = None,
    backend:     str  = "diffusers",
    save_grids:  bool = True,
) -> dict[str, list[Path]]:
    """
    Convenience function: generate ``num_images`` per class for all 7 classes.

    Returns
    -------
    dict mapping class_code → list of saved image paths.
    """
    device = _resolve_device(device)
    model  = _load_model(checkpoint, device, backend)
    if cfg_scale is not None:
        model.cfg_scale = cfg_scale

    all_paths: dict[str, list[Path]] = {}
    for cls_code in HAM_CLASSES:
        log.info("Generating %d × '%s' (%s)…",
                 num_images, cls_code, HAM_FULL_NAMES[cls_code])
        imgs  = model.generate_samples(cls_code, num_images, device, use_ddim, ddim_steps)
        paths = save_images_to_class_dir(imgs, output_dir, cls_code)
        if save_grids:
            save_contact_sheet(imgs, output_dir, cls_code, nrow=min(4, num_images))
        all_paths[cls_code] = paths

    log.info("All classes generated.  Root dir: %s", output_dir)
    return all_paths


# ═══════════════════════════════════════════════════════════════════════════
#  INTERNALS
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available — using CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _load_model(
    checkpoint: Union[str, Path],
    device:     torch.device,
    backend:    str,
):
    """Load model from checkpoint depending on backend."""
    if backend == "diffusers":
        from diffusion_model import HAM10000DiffusionModel
        model, _ = HAM10000DiffusionModel.load_checkpoint(checkpoint, device=device)
    elif backend == "custom":
        from ddpm import DDPM
        model, _ = DDPM.load_checkpoint(checkpoint, device=device)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'diffusers' or 'custom'.")
    return model.to(device).eval()


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic HAM10000 images from a trained DDPM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",  required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--class_label", type=str, default=None,
                   help="Class code (nv|mel|bkl|bcc|akiec|vasc|df) or int 0-6")
    p.add_argument("--all_classes", action="store_true",
                   help="Generate for ALL 7 classes")
    p.add_argument("--num_images",  type=int, default=4)
    p.add_argument("--output_dir",  default="generated")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--backend",     default="diffusers",
                   choices=["diffusers", "custom"])
    p.add_argument("--no_ddim",     action="store_true",
                   help="Use full DDPM reverse instead of DDIM")
    p.add_argument("--ddim_steps",  type=int, default=50)
    p.add_argument("--cfg_scale",   type=float, default=None)
    p.add_argument("--no_grid",     action="store_true",
                   help="Skip saving contact-sheet grids")
    return p.parse_args()


def main():
    args   = _parse_args()
    device = _resolve_device(args.device)

    if args.all_classes:
        generate_all_classes(
            checkpoint  = args.checkpoint,
            num_images  = args.num_images,
            output_dir  = args.output_dir,
            device      = device,
            use_ddim    = not args.no_ddim,
            ddim_steps  = args.ddim_steps,
            cfg_scale   = args.cfg_scale,
            backend     = args.backend,
            save_grids  = not args.no_grid,
        )
        return

    if args.class_label is None:
        raise SystemExit("Provide --class_label or --all_classes.")

    # Resolve label (int or string)
    try:
        cls = int(args.class_label)
        cls_code = HAM_CLASSES[cls]
    except ValueError:
        cls_code = args.class_label

    images = generate_samples(
        class_label = cls_code,
        num_images  = args.num_images,
        checkpoint  = args.checkpoint,
        device      = device,
        use_ddim    = not args.no_ddim,
        ddim_steps  = args.ddim_steps,
        cfg_scale   = args.cfg_scale,
        backend     = args.backend,
    )
    save_images_to_class_dir(images, args.output_dir, cls_code)
    if not args.no_grid:
        save_contact_sheet(images, args.output_dir, cls_code, nrow=min(4, len(images)))

    log.info("Done. Output: %s/%s/", args.output_dir, cls_code)


if __name__ == "__main__":
    main()
