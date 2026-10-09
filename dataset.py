"""
HAM10000 — PyTorch Dataset and DataLoader Factory
==================================================

HAMDataset:
    - Accepts image_dir (str/Path) + metadata CSV (or pre-loaded DataFrame)
    - Returns (image_tensor, label_idx, lesion_id, skin_tone_label or None)
    - Applies heavy albumentations augmentations to minority classes (>1× ratio)
    - Applies only safe geometric/colour jitter for the majority class (nv)

Augmentation strategy:
    - Minority class (imbalance_ratio > 1): strong augmentation pipeline
    - Majority class (nv): light normalisation pipeline
    - Val / Test: resize + centre-crop + normalise only

Usage:
    from dataset import build_dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        image_dir   = "path/to/HAM10000_images",
        train_csv   = "splits/train.csv",
        val_csv     = "splits/val.csv",
        test_csv    = "splits/test.csv",
        image_size  = 224,
        batch_size  = 32,
        num_workers = 4,
    )

Dependencies:
    pip install torch torchvision pillow albumentations pandas
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── albumentations import with graceful fallback ──────────────────────────────
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALBUMENTATION_AVAILABLE = True
except ImportError:
    _ALBUMENTATION_AVAILABLE = False
    import torchvision.transforms as T  # fallback
    logging.getLogger(__name__).warning(
        "albumentations not found — falling back to torchvision transforms."
    )

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CLASS_ORDER: list[str] = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_ORDER)}
IDX_TO_CLASS: dict[int, str] = {i: c for i, c in enumerate(CLASS_ORDER)}

# ImageNet statistics for pre-trained backbone fine-tuning
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Classes that are minority (imbalance ratio > 1 relative to 'nv')
# nv has ~6705 samples; all others are minorities.
MINORITY_CLASSES: set[str] = {"mel", "bkl", "bcc", "akiec", "vasc", "df"}

# ─────────────────────────────────────────────────────────────────────────────
#  ALBUMENTATIONS PIPELINES
# ─────────────────────────────────────────────────────────────────────────────

def _build_minority_augment(image_size: int = 224) -> Any:
    """
    Aggressive augmentation pipeline for minority dermatology classes.

    Design rationale:
      - Skin lesion shape is diagnostically relevant → avoid excessive
        perspective distortion, but horizontal/vertical flips ± rotations are safe.
      - Colour matters (melanoma hue) → conservative HSV jitter only.
      - Dermatoscope artifacts → hair-like line dropout simulated via CoarseDropout.
    """
    if not _ALBUMENTATION_AVAILABLE:
        return None

    return A.Compose([
        A.Resize(height=image_size + 32, width=image_size + 32),

        # ── Geometric ──
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, interpolation=1, border_mode=0, p=0.7),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.10, rotate_limit=15,
            interpolation=1, border_mode=0, p=0.5,
        ),
        A.ElasticTransform(alpha=0.5, sigma=25, p=0.2),

        # ── Colour & brightness ──
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.6
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(var_limit=(5, 25), p=0.3),
        A.Blur(blur_limit=3, p=0.15),

        # ── Dermatoscope artifact simulation ──
        A.CoarseDropout(
            max_holes=8, max_height=int(image_size * 0.04),
            max_width=int(image_size * 0.04),
            min_holes=1, fill_value=0, p=0.3,
        ),

        # ── Normalise → Tensor ──
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _build_majority_augment(image_size: int = 224) -> Any:
    """
    Light augmentation pipeline for the majority class (nv).
    Only basic spatial flips + colour jitter to avoid over-training on nv.
    """
    if not _ALBUMENTATION_AVAILABLE:
        return None

    return A.Compose([
        A.Resize(height=image_size + 16, width=image_size + 16),
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _build_val_transform(image_size: int = 224) -> Any:
    """
    Deterministic transform for validation / test splits.
    Resize → Centre-Crop → Normalise → Tensor.
    """
    if not _ALBUMENTATION_AVAILABLE:
        return None

    return A.Compose([
        A.Resize(height=image_size + 32, width=image_size + 32),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# torchvision fallback transforms (used if albumentations not installed)
def _tv_minority(image_size: int = 224):
    return T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomRotation(30),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _tv_majority(image_size: int = 224):
    return T.Compose([
        T.Resize((image_size + 16, image_size + 16)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _tv_val(image_size: int = 224):
    return T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  HAMDATASET
# ─────────────────────────────────────────────────────────────────────────────

class HAMDataset(Dataset):
    """
    PyTorch Dataset for the HAM10000 skin lesion benchmark.

    Parameters
    ----------
    image_dir : str | Path
        Root directory (or set of directories) containing .jpg images.
        Searched **recursively** so both part1 / part2 sub-folders are found.
    metadata : str | Path | pd.DataFrame
        Either a path to a CSV file or a pre-loaded DataFrame.
        Required columns: ``image_id``, ``dx`` (lesion class code).
        Optional columns: ``lesion_id``, ``skin_tone`` / ``fitzpatrick`` /
        ``ita_angle`` (any fairness annotation column).
    split : {"train", "val", "test"}
        Controls which augmentation pipeline is used.
        - ``train`` applies class-aware augmentations.
        - ``val`` / ``test`` apply only deterministic resize + crop + normalise.
    image_size : int
        Spatial resolution to resize images to (default 224 for ImageNet models).
    transform : callable | None
        Override the default transform.  If provided, ``split`` is ignored.
    minority_classes : set[str] | None
        Subset of class codes to apply heavy augmentation.
        Defaults to all non-``nv`` classes if ``None``.

    Returns  (per __getitem__)
    -------
    image_tensor : torch.Tensor   shape (3, H, W), float32, normalised
    label        : int            class index (see CLASS_TO_IDX)
    lesion_id    : str            unique lesion identifier ("" if unavailable)
    skin_tone    : str | int      skin-tone label ("" if unavailable)
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        metadata:  Union[str, Path, pd.DataFrame],
        split:     str  = "train",
        image_size: int = 224,
        transform=None,
        minority_classes: Optional[set] = None,
    ):
        super().__init__()
        self.image_dir   = Path(image_dir)
        self.split       = split.lower()
        self.image_size  = image_size
        self.minority_classes = minority_classes or MINORITY_CLASSES

        # ── Load metadata ──────────────────────────────────────────────────
        if isinstance(metadata, pd.DataFrame):
            self.df = metadata.copy().reset_index(drop=True)
        else:
            self.df = pd.read_csv(metadata).reset_index(drop=True)

        self.df.columns = [c.strip().lower() for c in self.df.columns]

        # Normalise dx
        self.df["dx"] = self.df["dx"].str.strip().str.lower()

        # ── Build image lookup (stem → absolute path) ─────────────────────
        log.debug("Building image path lookup from %s …", self.image_dir)
        self._path_lookup: dict[str, Path] = {}
        for p in self.image_dir.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                self._path_lookup[p.stem] = p

        # Attach paths; drop rows with missing images
        self.df["_resolved_path"] = self.df["image_id"].map(
            lambda x: str(self._path_lookup[x]) if x in self._path_lookup else None
        )
        missing = self.df["_resolved_path"].isna().sum()
        if missing:
            log.warning(
                "HAMDataset [%s]: %d images not found on disk — dropping those rows.",
                split, missing,
            )
            self.df = self.df.dropna(subset=["_resolved_path"]).reset_index(drop=True)

        # ── Skin-tone column detection ─────────────────────────────────────
        tone_candidates = [
            c for c in self.df.columns
            if any(kw in c for kw in ("skin", "fitzpatrick", "tone", "ita"))
        ]
        self._skin_tone_col: Optional[str] = tone_candidates[0] if tone_candidates else None
        if self._skin_tone_col:
            log.info("HAMDataset: using '%s' as skin-tone label.", self._skin_tone_col)

        # ── Transforms ────────────────────────────────────────────────────
        # Named per-class transforms for training; single transform for val/test
        self._custom_transform = transform           # user override

        if transform is None:
            if _ALBUMENTATION_AVAILABLE:
                self._minority_tf = _build_minority_augment(image_size)
                self._majority_tf = _build_majority_augment(image_size)
                self._val_tf      = _build_val_transform(image_size)
            else:
                self._minority_tf = _tv_minority(image_size)
                self._majority_tf = _tv_majority(image_size)
                self._val_tf      = _tv_val(image_size)

        log.info(
            "HAMDataset [%s]: %d samples | skin_tone_col=%s",
            split, len(self.df), self._skin_tone_col,
        )

    # ─────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.df)

    # ─────────────────────────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]

        # ── Load image ──────────────────────────────────────────────────
        img_path = row["_resolved_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            log.error("Failed to open image %s: %s", img_path, exc)
            # Return a blank image so training doesn't crash
            image = Image.new("RGB", (self.image_size, self.image_size), 0)

        # ── Apply transform ─────────────────────────────────────────────
        image_tensor = self._apply_transform(image, row["dx"])

        # ── Labels ──────────────────────────────────────────────────────
        label        = int(CLASS_TO_IDX.get(row["dx"], -1))
        lesion_id    = str(row.get("lesion_id", ""))
        skin_tone    = (
            str(row[self._skin_tone_col])
            if self._skin_tone_col else ""
        )

        return image_tensor, label, lesion_id, skin_tone

    # ─────────────────────────────────────────────────────────────────────
    def _apply_transform(self, image: Image.Image, dx: str) -> torch.Tensor:
        """Select and apply the correct transform based on split and class."""
        if self._custom_transform is not None:
            # user-supplied transform
            if _ALBUMENTATION_AVAILABLE and hasattr(self._custom_transform, "__call__"):
                img_np = np.array(image)
                return self._custom_transform(image=img_np)["image"]
            else:
                return self._custom_transform(image)

        if self.split in {"val", "test"}:
            transform = self._val_tf
        elif dx in self.minority_classes:
            transform = self._minority_tf
        else:
            transform = self._majority_tf

        if _ALBUMENTATION_AVAILABLE:
            img_np = np.array(image)
            result = transform(image=img_np)
            return result["image"]
        else:
            return transform(image)

    # ─────────────────────────────────────────────────────────────────────
    @property
    def class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency class weights suitable for use with
        ``nn.CrossEntropyLoss(weight=dataset.class_weights)``.

        Returns
        -------
        torch.Tensor of shape (num_classes,)
        """
        counts = np.zeros(len(CLASS_ORDER), dtype=np.float32)
        for dx in self.df["dx"]:
            idx = CLASS_TO_IDX.get(dx, -1)
            if idx >= 0:
                counts[idx] += 1
        counts = np.where(counts == 0, 1, counts)   # avoid division by zero
        weights = 1.0 / counts
        weights = weights / weights.sum()            # normalise to sum=1
        return torch.tensor(weights, dtype=torch.float32)

    # ─────────────────────────────────────────────────────────────────────
    @property
    def sample_weights(self) -> list[float]:
        """
        Per-sample weights for ``WeightedRandomSampler`` to oversample
        minority classes at the batch level.

        Returns
        -------
        list of float, length == len(dataset)
        """
        cw = self.class_weights.numpy()
        weights = []
        for dx in self.df["dx"]:
            idx = CLASS_TO_IDX.get(dx, 0)
            weights.append(float(cw[idx]))
        return weights

    # ─────────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"HAMDataset(split='{self.split}', "
            f"n_samples={len(self)}, "
            f"image_size={self.image_size}, "
            f"skin_tone_col={self._skin_tone_col!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    image_dir:       Union[str, Path],
    train_csv:       Union[str, Path],
    val_csv:         Union[str, Path],
    test_csv:        Union[str, Path],
    image_size:      int  = 224,
    batch_size:      int  = 32,
    num_workers:     int  = 4,
    pin_memory:      bool = True,
    oversample_train: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct train / val / test DataLoaders for HAM10000.

    Parameters
    ----------
    image_dir : str | Path
        Root directory of HAM10000 images.
    train_csv / val_csv / test_csv : str | Path
        CSVs produced by ``preprocessing.py``'s ``save_splits()``.
    image_size : int
        Resize target (224 for ImageNet-pretrained backbones).
    batch_size : int
        Batch size for all loaders.
    num_workers : int
        Number of DataLoader worker processes.
    pin_memory : bool
        Enables pin_memory on DataLoaders for GPU transfers.
    oversample_train : bool
        If True, use ``WeightedRandomSampler`` to oversample minority classes
        in the training DataLoader.  Ignored for val / test.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    train_ds = HAMDataset(image_dir, train_csv, split="train", image_size=image_size)
    val_ds   = HAMDataset(image_dir, val_csv,   split="val",   image_size=image_size)
    test_ds  = HAMDataset(image_dir, test_csv,  split="test",  image_size=image_size)

    # ── Training sampler ──────────────────────────────────────────────────
    if oversample_train and len(train_ds) > 0:
        sample_w = train_ds.sample_weights
        sampler = WeightedRandomSampler(
            weights=sample_w,
            num_samples=len(sample_w),
            replacement=True,
        )
        train_shuffle = False   # sampler is mutually exclusive with shuffle=True
    else:
        sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,           # keep batch sizes uniform for BN layers
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    log.info(
        "DataLoaders ready:\n"
        "  train  : %d batches × %d  (oversampled=%s)\n"
        "  val    : %d batches × %d\n"
        "  test   : %d batches × %d",
        len(train_loader), batch_size, oversample_train,
        len(val_loader),   batch_size,
        len(test_loader),  batch_size,
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK SMOKE-TEST  (run: python dataset.py --image_dir X --train_csv Y ...)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    p = argparse.ArgumentParser(description="HAMDataset smoke-test")
    p.add_argument("--image_dir",  required=True)
    p.add_argument("--train_csv",  required=True)
    p.add_argument("--val_csv",    required=True)
    p.add_argument("--test_csv",   required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    train_loader, val_loader, test_loader = build_dataloaders(
        image_dir=args.image_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # Peek at first train batch
    imgs, labels, lesion_ids, skin_tones = next(iter(train_loader))
    print("\n── Smoke-test: first training batch ──")
    print(f"  image tensor : {imgs.shape}  dtype={imgs.dtype}")
    print(f"  labels       : {labels.tolist()}")
    print(f"  lesion_ids   : {list(lesion_ids)}")
    print(f"  skin_tones   : {list(skin_tones)}")
    print(f"  pixel  min/max: {imgs.min().item():.4f} / {imgs.max().item():.4f}")
    print("\n  Class weights:", train_loader.dataset.class_weights.numpy())
    print("\nAll DataLoaders constructed successfully ✓")
