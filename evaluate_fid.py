"""
evaluate_fid.py — FID, SSIM, PSNR Evaluation for Generated vs. Real Images
============================================================================
Computes:
  • FID   (Fréchet Inception Distance)  via pytorch-fid
  • SSIM  (Structural Similarity Index) via torchmetrics / skimage
  • PSNR  (Peak Signal-to-Noise Ratio)  via torchmetrics / manual

Per-class metrics are computed when ``--per_class`` is set.
All results are written to ``metrics.json``.

Usage:
    python evaluate_fid.py \\
        --real_dir      path/to/HAM10000_images \\
        --generated_dir generated/ \\
        --output_file   metrics.json \\
        --per_class \\
        --num_workers 4

    # Evaluate only FID (fastest):
    python evaluate_fid.py \\
        --real_dir HAM10000_images \\
        --generated_dir generated/ \\
        --metrics fid

Folder structure expected for ``--generated_dir``:
    generated/
        nv/    *.png
        mel/   *.png
        bkl/   *.png
        ...

For ``--real_dir``, images may be flat (scanned recursively) or
in class subfolders.  The class label is inferred from the metadata CSV
when ``--metadata`` is provided; otherwise from subfolder name.

Dependencies:
    pip install pytorch-fid torchmetrics scikit-image pillow torch torchvision
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

HAM_CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

# ─────────────────────────────────────────────────────────────────────────────
#  OPTIONAL DEPENDENCY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

def _try_import_fid():
    try:
        from pytorch_fid import fid_score
        return fid_score
    except ImportError:
        raise ImportError(
            "pytorch-fid not installed.  Run: pip install pytorch-fid"
        )


def _try_import_torchmetrics():
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM_tm
        from torchmetrics.image import PeakSignalNoiseRatio as PSNR_tm
        return SSIM_tm, PSNR_tm
    except ImportError:
        return None, None


def _try_import_skimage():
    try:
        from skimage.metrics import structural_similarity, peak_signal_noise_ratio
        return structural_similarity, peak_signal_noise_ratio
    except ImportError:
        return None, None


# ═══════════════════════════════════════════════════════════════════════════
#  IMAGE LOADING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _collect_image_paths(root: Path, extensions: set = {".jpg",".jpeg",".png",".bmp"}) -> list[Path]:
    """Recursively find all image files under ``root``."""
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in extensions])


def _copy_images_to_temp(paths: list[Path], dest: Path):
    """Copy images to a flat temp directory (required by pytorch-fid)."""
    dest.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(paths):
        dst = dest / f"img_{i:06d}{src.suffix}"
        shutil.copy2(src, dst)


def _load_image_tensor(path: Path, resize: Optional[tuple] = (224, 224)) -> torch.Tensor:
    """Load an image → float32 tensor (3, H, W) in [0, 1]."""
    img = Image.open(path).convert("RGB")
    if resize:
        img = img.resize(resize, Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _get_class_paths(
    image_dir:   Path,
    metadata_csv: Optional[Path],
    class_code:  str,
) -> list[Path]:
    """
    Return image paths for a given class.
    Priority: metadata CSV (maps image_id→dx) > subfolder named after class_code.
    """
    # Strategy 1: subfolder  generated/mel/*.png
    class_sub = image_dir / class_code
    if class_sub.exists():
        return _collect_image_paths(class_sub)

    # Strategy 2: metadata CSV for real images
    if metadata_csv and metadata_csv.exists():
        import pandas as pd
        df = pd.read_csv(metadata_csv)
        df.columns = [c.strip().lower() for c in df.columns]
        ids = set(df.loc[df["dx"].str.lower() == class_code, "image_id"])
        all_imgs = _collect_image_paths(image_dir)
        return [p for p in all_imgs if p.stem in ids]

    return []


# ═══════════════════════════════════════════════════════════════════════════
#  FID
# ═══════════════════════════════════════════════════════════════════════════

def compute_fid(
    real_paths: list[Path],
    gen_paths:  list[Path],
    batch_size: int = 32,
    device:     str = "cuda",
    dims:       int = 2048,
) -> float:
    """
    Compute Fréchet Inception Distance between real and generated images.

    Uses ``pytorch-fid`` (https://github.com/mseitzer/pytorch-fid).
    Images are copied to temp directories in the format pytorch-fid expects.

    Parameters
    ----------
    real_paths  : list of real image paths
    gen_paths   : list of generated image paths
    batch_size  : InceptionV3 feature extraction batch size
    device      : "cuda" or "cpu"
    dims        : InceptionV3 feature dimensionality (2048 default)

    Returns
    -------
    float  — FID score (lower is better)
    """
    fid_score = _try_import_fid()

    if len(real_paths) == 0 or len(gen_paths) == 0:
        log.warning("Empty image list — FID=NaN (real=%d, gen=%d)",
                    len(real_paths), len(gen_paths))
        return float("nan")

    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
        log.warning("CUDA unavailable — computing FID on CPU (slow).")

    with tempfile.TemporaryDirectory() as tmp:
        real_tmp = Path(tmp) / "real"
        gen_tmp  = Path(tmp) / "gen"
        _copy_images_to_temp(real_paths, real_tmp)
        _copy_images_to_temp(gen_paths,  gen_tmp)

        fid_val = fid_score.calculate_fid_given_paths(
            paths      = [str(real_tmp), str(gen_tmp)],
            batch_size = batch_size,
            device     = device,
            dims       = dims,
        )
    log.info("FID = %.4f  (real=%d, gen=%d)", fid_val, len(real_paths), len(gen_paths))
    return float(fid_val)


# ═══════════════════════════════════════════════════════════════════════════
#  SSIM & PSNR
# ═══════════════════════════════════════════════════════════════════════════

def _compute_ssim_psnr_skimage(
    real_paths: list[Path],
    gen_paths:  list[Path],
    n_pairs:    int = 500,
    resize:     tuple = (224, 224),
) -> tuple[float, float]:
    """
    Sample ``n_pairs`` random (real, gen) image pairs and compute mean SSIM/PSNR
    using scikit-image.  Used as fallback when torchmetrics is unavailable.
    """
    ssim_fn, psnr_fn = _try_import_skimage()
    if ssim_fn is None:
        return float("nan"), float("nan")

    rng    = np.random.default_rng(42)
    n      = min(n_pairs, len(real_paths), len(gen_paths))
    r_idx  = rng.choice(len(real_paths), n, replace=False)
    g_idx  = rng.choice(len(gen_paths),  n, replace=False)

    ssims, psnrs = [], []
    for ri, gi in zip(r_idx, g_idx):
        r_img = np.array(Image.open(real_paths[ri]).convert("RGB").resize(resize))
        g_img = np.array(Image.open(gen_paths[gi]).convert("RGB").resize(resize))
        ssims.append(ssim_fn(r_img, g_img, channel_axis=2, data_range=255))
        psnrs.append(psnr_fn(r_img, g_img, data_range=255))

    mean_ssim = float(np.mean(ssims))
    mean_psnr = float(np.mean(psnrs))
    log.info("SSIM = %.4f | PSNR = %.2f dB  (n_pairs=%d)", mean_ssim, mean_psnr, n)
    return mean_ssim, mean_psnr


def _compute_ssim_psnr_torchmetrics(
    real_paths: list[Path],
    gen_paths:  list[Path],
    n_pairs:    int  = 500,
    resize:     tuple = (224, 224),
    device:     str  = "cpu",
) -> tuple[float, float]:
    SSIM_cls, PSNR_cls = _try_import_torchmetrics()
    if SSIM_cls is None:
        return _compute_ssim_psnr_skimage(real_paths, gen_paths, n_pairs, resize)

    dev    = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ssim_m = SSIM_cls(data_range=1.0).to(dev)
    psnr_m = PSNR_cls(data_range=1.0).to(dev)

    rng   = np.random.default_rng(42)
    n     = min(n_pairs, len(real_paths), len(gen_paths))
    r_idx = rng.choice(len(real_paths), n, replace=False)
    g_idx = rng.choice(len(gen_paths),  n, replace=False)

    for ri, gi in zip(r_idx, g_idx):
        rt = _load_image_tensor(real_paths[ri], resize).unsqueeze(0).to(dev)
        gt = _load_image_tensor(gen_paths[gi],  resize).unsqueeze(0).to(dev)
        ssim_m.update(gt, rt)
        psnr_m.update(gt, rt)

    mean_ssim = float(ssim_m.compute())
    mean_psnr = float(psnr_m.compute())
    log.info("SSIM = %.4f | PSNR = %.2f dB  (n_pairs=%d)", mean_ssim, mean_psnr, n)
    return mean_ssim, mean_psnr


def compute_ssim_psnr(
    real_paths: list[Path],
    gen_paths:  list[Path],
    n_pairs:    int  = 500,
    resize:     tuple = (224, 224),
    device:     str  = "cpu",
) -> tuple[float, float]:
    """
    Compute mean SSIM and PSNR over ``n_pairs`` randomly sampled (real, gen) pairs.
    Tries torchmetrics first, falls back to scikit-image.

    Returns
    -------
    (ssim, psnr) — floats (SSIM ∈ [-1,1] higher better; PSNR dB higher better)
    """
    SSIM_cls, _ = _try_import_torchmetrics()
    if SSIM_cls is not None:
        return _compute_ssim_psnr_torchmetrics(real_paths, gen_paths, n_pairs, resize, device)
    return _compute_ssim_psnr_skimage(real_paths, gen_paths, n_pairs, resize)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN EVALUATION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(
    real_dir:      Path,
    generated_dir: Path,
    output_file:   Path                 = Path("metrics.json"),
    metadata_csv:  Optional[Path]       = None,
    per_class:     bool                 = True,
    metrics:       list[str]            = ["fid", "ssim", "psnr"],
    batch_size:    int                  = 32,
    n_pairs:       int                  = 500,
    device:        str                  = "cuda",
    num_workers:   int                  = 4,
) -> dict:
    """
    Full evaluation pipeline.

    Parameters
    ----------
    real_dir       : root directory of real HAM10000 images
    generated_dir  : root of synthetic images (class subfolders expected)
    output_file    : where to write metrics.json
    metadata_csv   : HAM10000 metadata CSV for class-based real image lookup
    per_class      : if True, compute metrics for each class separately
    metrics        : list of metric names to compute: "fid", "ssim", "psnr"
    batch_size     : InceptionV3 batch size for FID
    n_pairs        : number of (real, gen) pairs for SSIM/PSNR
    device         : "cuda" or "cpu"
    num_workers    : unused (reserved for future parallel loading)

    Returns
    -------
    dict with nested structure: {"global": {...}, "per_class": {cls: {...}}}
    Saved to ``output_file``.
    """
    real_dir      = Path(real_dir)
    generated_dir = Path(generated_dir)
    output_file   = Path(output_file)

    if not real_dir.exists():
        raise FileNotFoundError(f"Real image dir not found: {real_dir}")
    if not generated_dir.exists():
        raise FileNotFoundError(f"Generated image dir not found: {generated_dir}")

    results: dict = {"global": {}, "per_class": {}, "config": {
        "real_dir":      str(real_dir),
        "generated_dir": str(generated_dir),
        "metrics":       metrics,
        "n_pairs":       n_pairs,
        "device":        device,
    }}

    # ── Global metrics ───────────────────────────────────────────────────
    log.info("─── Global Evaluation ───────────────────────────────")
    all_real_paths  = _collect_image_paths(real_dir)
    all_gen_paths   = _collect_image_paths(generated_dir)
    log.info("Real: %d | Generated: %d", len(all_real_paths), len(all_gen_paths))

    if "fid" in metrics:
        results["global"]["fid"] = compute_fid(
            all_real_paths, all_gen_paths, batch_size=batch_size, device=device
        )
    if "ssim" in metrics or "psnr" in metrics:
        ssim, psnr = compute_ssim_psnr(
            all_real_paths, all_gen_paths, n_pairs=n_pairs, device=device
        )
        if "ssim" in metrics:
            results["global"]["ssim"] = round(ssim, 6)
        if "psnr" in metrics:
            results["global"]["psnr_db"] = round(psnr, 4)

    # ── Per-class metrics ────────────────────────────────────────────────
    if per_class:
        log.info("─── Per-Class Evaluation ────────────────────────────")
        for cls_code in HAM_CLASSES:
            log.info("  Class: %s", cls_code)
            real_cls = _get_class_paths(real_dir, metadata_csv, cls_code)
            gen_cls  = _get_class_paths(generated_dir, None, cls_code)

            if not gen_cls:
                log.warning("  No generated images found for class %s — skipping.", cls_code)
                results["per_class"][cls_code] = {"error": "no_generated_images",
                                                   "n_real": len(real_cls), "n_gen": 0}
                continue

            cls_result: dict = {
                "n_real": len(real_cls),
                "n_gen":  len(gen_cls),
            }

            if "fid" in metrics:
                cls_result["fid"] = compute_fid(
                    real_cls, gen_cls, batch_size=batch_size, device=device
                )
            if "ssim" in metrics or "psnr" in metrics:
                ssim, psnr = compute_ssim_psnr(
                    real_cls, gen_cls,
                    n_pairs=min(n_pairs, len(real_cls), len(gen_cls)),
                    device=device,
                )
                if "ssim" in metrics:
                    cls_result["ssim"] = round(ssim, 6)
                if "psnr" in metrics:
                    cls_result["psnr_db"] = round(psnr, 4)

            results["per_class"][cls_code] = cls_result
            log.info("  → %s", cls_result)

    # ── Save ─────────────────────────────────────────────────────────────
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("Metrics saved → %s", output_file)
    _print_summary(results)
    return results


def _print_summary(results: dict):
    """Print a nicely formatted summary table to stdout."""
    g = results.get("global", {})
    print("\n" + "═" * 55)
    print("  EVALUATION SUMMARY")
    print("═" * 55)
    print(f"  Global FID  : {g.get('fid',  'N/A')}")
    print(f"  Global SSIM : {g.get('ssim', 'N/A')}")
    print(f"  Global PSNR : {g.get('psnr_db', 'N/A')} dB")
    print("─" * 55)
    pc = results.get("per_class", {})
    if pc:
        print(f"  {'Class':<8}  {'FID':>8}  {'SSIM':>7}  {'PSNR(dB)':>9}")
        for cls, m in pc.items():
            if "error" in m:
                continue
            print(f"  {cls:<8}  {m.get('fid', 'N/A'):>8.3f}  "
                  f"{m.get('ssim', 'N/A'):>7.4f}  {m.get('psnr_db', 'N/A'):>9.2f}")
    print("═" * 55)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate FID / SSIM / PSNR between real and generated HAM10000 images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--real_dir",      required=True,
                   help="Root directory of real HAM10000 images")
    p.add_argument("--generated_dir", required=True,
                   help="Root directory of generated images (class subfolders)")
    p.add_argument("--metadata",      default=None,
                   help="HAM10000 metadata CSV for class-level real image lookup")
    p.add_argument("--output_file",   default="metrics.json")
    p.add_argument("--per_class",     action="store_true", default=True)
    p.add_argument("--metrics",       nargs="+", default=["fid", "ssim", "psnr"],
                   choices=["fid", "ssim", "psnr"])
    p.add_argument("--batch_size",    type=int, default=32,
                   help="InceptionV3 batch size (FID)")
    p.add_argument("--n_pairs",       type=int, default=500,
                   help="Random pairs for SSIM/PSNR")
    p.add_argument("--device",        default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num_workers",   type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        real_dir      = Path(args.real_dir),
        generated_dir = Path(args.generated_dir),
        output_file   = Path(args.output_file),
        metadata_csv  = Path(args.metadata) if args.metadata else None,
        per_class     = args.per_class,
        metrics       = args.metrics,
        batch_size    = args.batch_size,
        n_pairs       = args.n_pairs,
        device        = args.device,
        num_workers   = args.num_workers,
    )
