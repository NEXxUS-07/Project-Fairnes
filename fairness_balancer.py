"""
fairness_balancer.py — Fairness-Aware Dataset Balancing with Synthetic Images
==============================================================================
Implements three class-balancing strategies using real HAM10000 images and
synthetically generated images from a diffusion model.

Strategies
----------
  1. equal_count   — All 7 classes → N samples each (N = max of target or majority)
                     Minority classes filled with synthetic images.
  2. proportional  — Reduce majority (subsample), fill minority with synthetic.
                     Target distribution determined by a user-specified ratio floor.
  3. hybrid        — Cap majority at 2× the count of the largest minority class;
                     fill remaining minority slots with synthetic images.

Metrics computed before and after balancing
-------------------------------------------
  • Class distribution (counts + percentages)
  • Demographic Parity Difference (DPD): max(P(Ŷ=1|A=a)) - min(P(Ŷ=1|A=a))
    computed over class groups (and over skin-tone groups if annotated).
  • Per-class recall from a pretrained ResNet50 classifier
    (loads a fine-tuned checkpoint if provided; otherwise uses ImageNet weights
     as a rough proxy — recall values will be near chance without fine-tuning).

Outputs
-------
  balanced_dataset.csv   — columns: image_path, label, source, split
  fairness_report.json   — before/after metrics, strategy details

Usage
-----
    python fairness_balancer.py \\
        --metadata       splits/train.csv \\
        --real_dir       path/to/HAM10000_images \\
        --synthetic_dir  generated/ \\
        --strategy       hybrid \\
        --output_csv     balanced_dataset.csv \\
        --report_json    fairness_report.json

    # With ResNet50 recall evaluation:
    python fairness_balancer.py \\
        --metadata splits/train.csv \\
        --real_dir HAM10000_images \\
        --synthetic_dir generated/ \\
        --strategy equal_count \\
        --classifier_ckpt checkpoints/resnet50_ham.pt \\
        --eval_recall

Dependencies:
    pip install pandas numpy scikit-learn torch torchvision tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

HAM_CLASSES   = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
HAM_LABEL_MAP = {c: i for i, c in enumerate(HAM_CLASSES)}
HAM_FULL_NAMES = {
    "nv":    "Melanocytic Nevi",
    "mel":   "Melanoma",
    "bkl":   "Benign Keratosis",
    "bcc":   "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "vasc":  "Vascular Lesions",
    "df":    "Dermatofibroma",
}

STRATEGIES = ["equal_count", "proportional", "hybrid"]


# ═══════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ClassMetrics:
    """Per-class fairness / distribution metrics."""
    cls:         str
    count:       int
    proportion:  float         # fraction of total
    recall:      float = -1.0  # -1 = not computed
    dpd_contrib: float = -1.0  # contribution to demographic parity


@dataclass
class FairnessMetrics:
    """Dataset-level fairness metrics (before or after balancing)."""
    total_samples:         int
    class_distribution:    dict         # cls → ClassMetrics
    demographic_parity_difference: float      # over class groups
    skin_tone_dpd:             float = -1.0   # over skin-tone groups (-1 = N/A)
    mean_recall:               float = -1.0
    min_recall:                float = -1.0
    max_recall:                float = -1.0
    minority_recall_gap:       float = -1.0   # max_recall - min_recall


# ═══════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_real_dataframe(
    metadata_csv: Union[str, Path],
    real_dir:     Union[str, Path],
) -> pd.DataFrame:
    """
    Load real HAM10000 image paths from a CSV + image directory.

    Required CSV columns: image_id, dx, split  (split = train/val/test).
    Returns DataFrame with columns:
        image_path, label (dx code), label_idx, source="real", split
    Missing images are dropped with a warning.
    """
    csv_path = Path(metadata_csv)
    img_root = Path(real_dir)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["dx"] = df["dx"].str.strip().str.lower()

    # Resolve image paths (recursive scan)
    log.info("Scanning %s for images…", img_root)
    lookup: dict[str, str] = {}
    for p in img_root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            lookup[p.stem] = str(p)

    df["image_path"] = df["image_id"].map(lookup)
    n_missing = df["image_path"].isna().sum()
    if n_missing:
        log.warning("Dropping %d rows with missing images.", n_missing)
        df = df.dropna(subset=["image_path"])

    df["label"]     = df["dx"]
    df["label_idx"] = df["dx"].map(HAM_LABEL_MAP)
    df["source"]    = "real"
    if "split" not in df.columns:
        df["split"] = "train"

    log.info("Loaded %d real images.", len(df))
    return df[["image_path", "label", "label_idx", "source", "split"] +
              [c for c in df.columns if c.startswith("skin") or "fitzpatrick" in c or "ita" in c]]


def load_synthetic_dataframe(
    synthetic_dir: Union[str, Path],
    split:         str = "train",
) -> pd.DataFrame:
    """
    Scan synthetic image directory (class subfolders) and build a DataFrame.

    Expected structure:
        synthetic_dir/
            nv/   *.png
            mel/  *.png
            ...

    Returns DataFrame with columns:
        image_path, label, label_idx, source="synthetic", split
    """
    synth_root = Path(synthetic_dir)
    rows = []

    for cls_code in HAM_CLASSES:
        cls_dir = synth_root / cls_code
        if not cls_dir.exists():
            log.warning("No synthetic subfolder for class '%s' at %s", cls_code, cls_dir)
            continue
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append({
                    "image_path": str(p),
                    "label":      cls_code,
                    "label_idx":  HAM_LABEL_MAP[cls_code],
                    "source":     "synthetic",
                    "split":      split,
                })

    df = pd.DataFrame(rows)
    log.info("Found %d synthetic images across %d classes.",
             len(df), df["label"].nunique() if len(df) else 0)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  2. BALANCING STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

class DatasetBalancer:
    """
    Applies one of three class-balancing strategies using real + synthetic images.

    Attributes
    ----------
    real_df      : DataFrame of real images (from ``load_real_dataframe()``)
    synthetic_df : DataFrame of synthetic images (from ``load_synthetic_dataframe()``)
    strategy     : one of "equal_count", "proportional", "hybrid"
    target_total : desired total dataset size (None = auto from majority class)
    hybrid_cap   : multiplier for majority cap in "hybrid" strategy (default 2)
    seed         : random seed for reproducibility
    """

    def __init__(
        self,
        real_df:       pd.DataFrame,
        synthetic_df:  pd.DataFrame,
        strategy:      str   = "hybrid",
        target_total:  Optional[int] = None,
        hybrid_cap:    float = 2.0,
        seed:          int   = 42,
    ):
        assert strategy in STRATEGIES, \
            f"strategy must be one of {STRATEGIES}, got '{strategy}'"

        self.real_df      = real_df
        self.synthetic_df = synthetic_df
        self.strategy     = strategy
        self.target_total = target_total
        self.hybrid_cap   = hybrid_cap
        self.rng          = random.Random(seed)
        self.np_rng       = np.random.default_rng(seed)

        # Counts
        self._real_counts = real_df["label"].value_counts().to_dict()
        self._synth_counts = (
            synthetic_df["label"].value_counts().to_dict()
            if len(synthetic_df) else {}
        )
        for c in HAM_CLASSES:
            self._real_counts.setdefault(c, 0)
            self._synth_counts.setdefault(c, 0)

        log.info(
            "Balancer | strategy=%s | real=%d | synthetic=%d",
            strategy, len(real_df), len(synthetic_df),
        )
        log.info("Real counts per class: %s", self._real_counts)
        log.info("Synthetic counts per class: %s", self._synth_counts)

    # ─────────────────────────────────────────────────────────────────────
    #  STRATEGY 1: EQUAL COUNT
    # ─────────────────────────────────────────────────────────────────────

    def _strategy_equal_count(self) -> pd.DataFrame:
        """
        All classes → exactly N images each.
        N = max(majority class count, target_total // num_classes).
        Minority classes are filled with synthetic images (with replacement
        if insufficient synthetic images exist).
        """
        majority_count = max(self._real_counts.values())
        if self.target_total:
            N = max(majority_count, self.target_total // len(HAM_CLASSES))
        else:
            N = majority_count

        log.info("[equal_count] Target per class: %d", N)
        parts = []

        for cls in HAM_CLASSES:
            real_cls   = self.real_df[self.real_df["label"] == cls]
            real_n     = len(real_cls)
            needed     = N - real_n

            # Take all real images for this class
            parts.append(real_cls)

            if needed > 0:
                synth_cls = self.synthetic_df[self.synthetic_df["label"] == cls]
                if len(synth_cls) == 0:
                    log.warning(
                        "[equal_count] No synthetic images for '%s', "
                        "cannot fill %d missing samples.", cls, needed
                    )
                else:
                    fill = self._sample_with_replacement(synth_cls, needed)
                    parts.append(fill)
                    log.info(
                        "  [%s] real=%d + synth=%d = %d (target %d)",
                        cls, real_n, len(fill), real_n + len(fill), N,
                    )
            elif needed < 0:
                # Subsample majority class
                parts[-1] = real_cls.sample(N, random_state=42)
                log.info("  [%s] subsampled real: %d → %d", cls, real_n, N)

        return pd.concat(parts, ignore_index=True)

    # ─────────────────────────────────────────────────────────────────────
    #  STRATEGY 2: PROPORTIONAL FAIRNESS
    # ─────────────────────────────────────────────────────────────────────

    def _strategy_proportional(
        self,
        minority_floor: float = 0.10,   # min fraction for any class
    ) -> pd.DataFrame:
        """
        Proportional fairness:
          • Majority classes are subsampled so no class exceeds
            ``1 / num_classes * (1 + slack_fraction)`` of the total.
          • Minority classes below ``minority_floor`` fraction are filled
            with synthetic images up to the floor count.
          • If target_total is set, proportional target counts are used.

        minority_floor : float
            Minimum fraction of total for any class (default 0.10 → 10%).
        """
        n_cls = len(HAM_CLASSES)

        # Determine total
        if self.target_total:
            total = self.target_total
        else:
            total = sum(self._real_counts.values())

        floor_count  = math.ceil(minority_floor * total)
        target_count = math.ceil(total / n_cls)    # ideal equal share

        log.info(
            "[proportional] total=%d | per-class-target=%d | floor=%d",
            total, target_count, floor_count,
        )

        parts = []
        for cls in HAM_CLASSES:
            real_cls = self.real_df[self.real_df["label"] == cls]
            real_n   = len(real_cls)
            goal     = max(target_count, floor_count)

            if real_n >= goal:
                # Subsample to target
                parts.append(real_cls.sample(goal, random_state=42))
                log.info("  [%s] subsampled: %d → %d", cls, real_n, goal)
            else:
                parts.append(real_cls)
                needed = goal - real_n
                synth_cls = self.synthetic_df[self.synthetic_df["label"] == cls]
                if len(synth_cls) == 0:
                    log.warning(
                        "  [%s] No synthetic images — class will be under-represented.", cls
                    )
                else:
                    fill = self._sample_with_replacement(synth_cls, needed)
                    parts.append(fill)
                    log.info(
                        "  [%s] real=%d + synth=%d = %d (goal %d)",
                        cls, real_n, len(fill), real_n + len(fill), goal,
                    )
        return pd.concat(parts, ignore_index=True)

    # ─────────────────────────────────────────────────────────────────────
    #  STRATEGY 3: HYBRID
    # ─────────────────────────────────────────────────────────────────────

    def _strategy_hybrid(self) -> pd.DataFrame:
        """
        Hybrid strategy:
          1. Identify the "reference minority count" = max count among
             all non-majority classes.
          2. Cap the majority class at ``hybrid_cap × reference_minority_count``.
          3. Fill every minority class up to the reference count with synthetic.

        Example with hybrid_cap=2:
            nv:    6705 real → capped at 2 × 1113 = 2226
            mel:   1113 real → target 1113 (already there)
            vasc:   142 real → fill with  971 synthetic → 1113 total
            df:     115 real → fill with  998 synthetic → 1113 total
        """
        # Majority class = nv (largest)
        sorted_counts = sorted(self._real_counts.items(), key=lambda x: -x[1])
        majority_cls  = sorted_counts[0][0]
        minority_counts = {c: n for c, n in sorted_counts[1:]}

        # Reference minority count (2nd largest = the largest non-majority class)
        ref_count = max(minority_counts.values()) if minority_counts else 1
        cap       = math.ceil(self.hybrid_cap * ref_count)

        log.info(
            "[hybrid] majority=%s(%d) → cap=%d  |  ref_minority=%d",
            majority_cls, sorted_counts[0][1], cap, ref_count,
        )

        parts = []
        for cls in HAM_CLASSES:
            real_cls = self.real_df[self.real_df["label"] == cls]
            real_n   = len(real_cls)

            if cls == majority_cls:
                # Cap majority
                take = min(real_n, cap)
                parts.append(real_cls.sample(take, random_state=42))
                log.info("  [%s] majority capped: %d → %d", cls, real_n, take)
            else:
                # Fill minority to ref_count
                parts.append(real_cls)
                needed = ref_count - real_n
                if needed > 0:
                    synth_cls = self.synthetic_df[self.synthetic_df["label"] == cls]
                    if len(synth_cls) == 0:
                        log.warning(
                            "  [%s] No synthetic images — target=%d, have=%d.",
                            cls, ref_count, real_n,
                        )
                    else:
                        fill = self._sample_with_replacement(synth_cls, needed)
                        parts.append(fill)
                        log.info(
                            "  [%s] real=%d + synth=%d = %d (target %d)",
                            cls, real_n, len(fill), real_n + len(fill), ref_count,
                        )

        return pd.concat(parts, ignore_index=True)

    # ─────────────────────────────────────────────────────────────────────
    #  ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────

    def balance(self) -> pd.DataFrame:
        """Run the configured strategy and return the balanced DataFrame."""
        if self.strategy == "equal_count":
            balanced = self._strategy_equal_count()
        elif self.strategy == "proportional":
            balanced = self._strategy_proportional()
        elif self.strategy == "hybrid":
            balanced = self._strategy_hybrid()
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")

        # Standardise output columns
        balanced = balanced.reset_index(drop=True)
        keep = ["image_path", "label", "label_idx", "source", "split"]
        extra = [c for c in balanced.columns if c not in keep]
        balanced = balanced[keep + extra]

        log.info(
            "Balanced dataset: %d samples | distribution: %s",
            len(balanced),
            balanced["label"].value_counts().to_dict(),
        )
        return balanced

    # ─────────────────────────────────────────────────────────────────────

    def _sample_with_replacement(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Sample ``n`` rows from df, allowing replacement if n > len(df)."""
        replace = n > len(df)
        return df.sample(n, replace=replace, random_state=42).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
#  3. FAIRNESS METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_class_distribution(df: pd.DataFrame) -> dict[str, dict]:
    """Compute class distribution metrics from a DataFrame."""
    total = len(df)
    dist  = {}
    for cls in HAM_CLASSES:
        n = int((df["label"] == cls).sum())
        dist[cls] = {
            "count":       n,
            "proportion":  round(n / total, 6) if total > 0 else 0.0,
            "full_name":   HAM_FULL_NAMES.get(cls, cls),
        }
    return dist


def compute_demographic_parity_difference(
    df: pd.DataFrame,
    group_col: str = "label",
) -> float:
    """
    Demographic Parity Difference (DPD):
        DPD = max_group(P(Y=1|G=g)) − min_group(P(Y=1|G=g))

    In the absence of a model, we use class proportion as the "positive rate"
    proxy (i.e., how over/under-represented each class is relative to equal
    representation).  A perfectly balanced dataset has DPD=0.

    When called on the raw dataset:   group_col="label"  (class-level parity)
    When called on skin-tone groups:  group_col="skin_tone" / "fitzpatrick"

    Returns
    -------
    float  — DPD ∈ [0, 1].  0 = perfect parity.
    """
    if group_col not in df.columns:
        return -1.0

    proportions = df[group_col].value_counts(normalize=True)
    if len(proportions) < 2:
        return 0.0

    dpd = float(proportions.max() - proportions.min())
    return round(dpd, 6)


def compute_per_class_recall(
    df:               pd.DataFrame,
    real_dir:         Union[str, Path],
    classifier_ckpt:  Optional[Union[str, Path]] = None,
    image_size:       int  = 224,
    batch_size:       int  = 32,
    device:           str  = "cpu",
) -> dict[str, float]:
    """
    Compute per-class recall using a ResNet50 classifier.

    If ``classifier_ckpt`` is provided (fine-tuned on HAM10000), those weights
    are loaded. Otherwise, an ImageNet-pretrained ResNet50 is used with a random
    final head — recall values will be near chance but the plumbing is validated.

    Parameters
    ----------
    df               : balanced dataset DataFrame (image_path, label)
    classifier_ckpt  : path to fine-tuned ResNet50 checkpoint (.pt)
    image_size       : resize target for inference
    batch_size       : inference batch size
    device           : "cuda" or "cpu"

    Returns
    -------
    dict[class_code → recall]  — per-class recall ∈ [0, 1]
    """
    try:
        import torch
        import torchvision.models as tv_models
        import torchvision.transforms as TF_tv
        from PIL import Image as PIL_Image
        from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDL
        from sklearn.metrics import recall_score
    except ImportError as e:
        log.warning("Recall computation skipped — missing dependency: %s", e)
        return {c: -1.0 for c in HAM_CLASSES}

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    # ── Build ResNet50 ────────────────────────────────────────────────────
    model_cls = tv_models.resnet50(
        weights=tv_models.ResNet50_Weights.IMAGENET1K_V2
    )
    model_cls.fc = torch.nn.Linear(model_cls.fc.in_features, len(HAM_CLASSES))

    if classifier_ckpt and Path(classifier_ckpt).exists():
        ckpt = torch.load(classifier_ckpt, map_location=dev)
        state = ckpt.get("model_state", ckpt)    # support both formats
        model_cls.load_state_dict(state, strict=False)
        log.info("Loaded classifier weights from %s", classifier_ckpt)
    else:
        log.warning(
            "No classifier checkpoint provided or found.  "
            "Using ImageNet-pretrained weights with random head "
            "— recall values will be ~random."
        )

    model_cls.to(dev).eval()

    # ── Inference dataset ─────────────────────────────────────────────────
    transform = TF_tv.Compose([
        TF_tv.Resize((image_size, image_size)),
        TF_tv.ToTensor(),
        TF_tv.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ])

    valid_rows = df[df["image_path"].apply(lambda p: Path(p).exists())]
    if len(valid_rows) < len(df):
        log.warning("Recall: %d image paths not found — skipping those.",
                    len(df) - len(valid_rows))

    class _InferDS(TorchDataset):
        def __init__(self, rows, tfm):
            self.rows, self.tfm = rows.reset_index(drop=True), tfm
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            row = self.rows.iloc[i]
            try:
                img = PIL_Image.open(row["image_path"]).convert("RGB")
            except Exception:
                img = PIL_Image.new("RGB", (image_size, image_size), 0)
            return self.tfm(img), HAM_LABEL_MAP.get(row["label"], 0)

    loader = TorchDL(
        _InferDS(valid_rows, transform),
        batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(dev.type == "cuda"),
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="ResNet50 inference", leave=False):
            imgs = imgs.to(dev)
            logits = model_cls(imgs)
            preds  = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.tolist())

    # ── Per-class recall ─────────────────────────────────────────────────
    recalls = recall_score(
        all_targets, all_preds,
        labels=list(range(len(HAM_CLASSES))),
        average=None,
        zero_division=0,
    )
    recall_dict = {HAM_CLASSES[i]: round(float(recalls[i]), 6)
                   for i in range(len(HAM_CLASSES))}
    log.info("Per-class recall: %s", recall_dict)
    return recall_dict


def compute_fairness_metrics(
    df:               pd.DataFrame,
    label:            str = "before",
    classifier_ckpt:  Optional[str] = None,
    eval_recall:      bool = False,
    real_dir:         Optional[str] = None,
    device:           str = "cpu",
) -> dict:
    """
    Compute the full set of fairness metrics for a dataset split.

    Returns a nested dict suitable for JSON serialisation.
    """
    dist = compute_class_distribution(df)
    dpd  = compute_demographic_parity_difference(df, group_col="label")

    # Skin-tone DPD (if column present)
    skin_cols = [c for c in df.columns
                 if any(kw in c for kw in ("skin", "fitzpatrick", "tone", "ita"))]
    skin_dpd  = (
        compute_demographic_parity_difference(df, group_col=skin_cols[0])
        if skin_cols else -1.0
    )

    recalls = {}
    if eval_recall and real_dir:
        recalls = compute_per_class_recall(
            df, real_dir, classifier_ckpt=classifier_ckpt, device=device
        )

    valid_recalls = [v for v in recalls.values() if v >= 0]
    metrics = {
        "label":           label,
        "total_samples":   int(len(df)),
        "n_real":          int((df["source"] == "real").sum()),
        "n_synthetic":     int((df["source"] == "synthetic").sum()),
        "class_distribution": dist,
        "demographic_parity_difference": dpd,
        "skin_tone_dpd":   skin_dpd,
        "per_class_recall": recalls,
        "mean_recall":     round(float(np.mean(valid_recalls)), 6) if valid_recalls else -1.0,
        "min_recall":      round(float(np.min(valid_recalls)),  6) if valid_recalls else -1.0,
        "max_recall":      round(float(np.max(valid_recalls)),  6) if valid_recalls else -1.0,
        "recall_gap":      round(float(np.max(valid_recalls) - np.min(valid_recalls)), 6)
                           if valid_recalls else -1.0,
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  4. REPORT + EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def export_balanced_csv(
    balanced_df:  pd.DataFrame,
    output_csv:   Union[str, Path],
):
    """
    Save the final balanced dataset to CSV.

    Ensures columns: image_path, label, source, split
    (label_idx and any extra columns are included but placed after).
    """
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Canonical column order
    priority = ["image_path", "label", "label_idx", "source", "split"]
    extras   = [c for c in balanced_df.columns if c not in priority]
    balanced_df = balanced_df[priority + extras]

    balanced_df.to_csv(out, index=False)
    log.info("Balanced dataset CSV → %s  (%d rows)", out, len(balanced_df))


def export_fairness_report(
    before_metrics: dict,
    after_metrics:  dict,
    strategy:       str,
    report_path:    Union[str, Path],
    extra_info:     Optional[dict] = None,
):
    """
    Write a comprehensive fairness report as JSON.

    Includes:
      - Strategy used
      - Before/after class distribution
      - Before/after DPD and skin-tone DPD
      - Before/after per-class recall (if computed)
      - Delta metrics (improvement/regression)
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute deltas
    def _delta(key):
        b = before_metrics.get(key, -1.0)
        a = after_metrics.get(key, -1.0)
        if b < 0 or a < 0:
            return "N/A"
        return round(a - b, 6)

    report = {
        "strategy": strategy,
        "summary": {
            "total_before": before_metrics["total_samples"],
            "total_after":  after_metrics["total_samples"],
            "n_real_after":      after_metrics["n_real"],
            "n_synthetic_after": after_metrics["n_synthetic"],
            "dpd_before":  before_metrics["demographic_parity_difference"],
            "dpd_after":   after_metrics["demographic_parity_difference"],
            "dpd_delta":   _delta("demographic_parity_difference"),
            "mean_recall_before": before_metrics["mean_recall"],
            "mean_recall_after":  after_metrics["mean_recall"],
            "recall_gap_before":  before_metrics["recall_gap"],
            "recall_gap_after":   after_metrics["recall_gap"],
        },
        "before": before_metrics,
        "after":  after_metrics,
    }
    if extra_info:
        report["extra"] = extra_info

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("Fairness report → %s", report_path)
    _print_fairness_summary(report)


def _print_fairness_summary(report: dict):
    """Print a concise summary of the fairness report."""
    s = report["summary"]
    print("\n" + "═" * 60)
    print("  FAIRNESS REPORT SUMMARY")
    print("═" * 60)
    print(f"  Strategy       : {report['strategy']}")
    print(f"  Total samples  : {s['total_before']} → {s['total_after']}")
    print(f"    Real         : {s['n_real_after']}")
    print(f"    Synthetic    : {s['n_synthetic_after']}")
    print(f"  DPD            : {s['dpd_before']:.4f} → {s['dpd_after']:.4f}  "
          f"(Δ = {s['dpd_delta']})")
    if s.get("mean_recall_before", -1) >= 0:
        print(f"  Mean recall    : {s['mean_recall_before']:.4f} → {s['mean_recall_after']:.4f}")
        print(f"  Recall gap     : {s['recall_gap_before']:.4f} → {s['recall_gap_after']:.4f}")
    print("─" * 60)
    after = report["after"]
    cd    = after["class_distribution"]
    print(f"  {'Class':<8}  {'Count':>6}  {'Prop%':>6}  {'Recall':>7}")
    for cls in HAM_CLASSES:
        m       = cd.get(cls, {})
        recall  = after["per_class_recall"].get(cls, "N/A")
        r_str   = f"{recall:.4f}" if isinstance(recall, float) and recall >= 0 else " N/A  "
        print(f"  {cls:<8}  {m.get('count',0):>6}  "
              f"{m.get('proportion',0)*100:>5.1f}%  {r_str:>7}")
    print("═" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  5. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_fairness_balancing(
    metadata_csv:     Union[str, Path],
    real_dir:         Union[str, Path],
    synthetic_dir:    Union[str, Path],
    strategy:         str  = "hybrid",
    output_csv:       Union[str, Path] = "balanced_dataset.csv",
    report_json:      Union[str, Path] = "fairness_report.json",
    target_total:     Optional[int]    = None,
    hybrid_cap:       float = 2.0,
    classifier_ckpt:  Optional[str]    = None,
    eval_recall:      bool  = False,
    device:           str   = "cpu",
    seed:             int   = 42,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    End-to-end fairness balancing pipeline.

    Returns
    -------
    (balanced_df, before_metrics, after_metrics)
    """
    # ── Load data ────────────────────────────────────────────────────────
    real_df   = load_real_dataframe(metadata_csv, real_dir)
    synth_df  = load_synthetic_dataframe(synthetic_dir)

    # ── Before metrics ───────────────────────────────────────────────────
    log.info("Computing BEFORE metrics…")
    before = compute_fairness_metrics(
        real_df, label="before",
        classifier_ckpt=classifier_ckpt,
        eval_recall=eval_recall,
        real_dir=str(real_dir),
        device=device,
    )

    # ── Balance ──────────────────────────────────────────────────────────
    balancer = DatasetBalancer(
        real_df=real_df,
        synthetic_df=synth_df,
        strategy=strategy,
        target_total=target_total,
        hybrid_cap=hybrid_cap,
        seed=seed,
    )
    balanced_df = balancer.balance()

    # ── After metrics ────────────────────────────────────────────────────
    log.info("Computing AFTER metrics…")
    after = compute_fairness_metrics(
        balanced_df, label="after",
        classifier_ckpt=classifier_ckpt,
        eval_recall=eval_recall,
        real_dir=str(real_dir),
        device=device,
    )

    # ── Export ───────────────────────────────────────────────────────────
    export_balanced_csv(balanced_df, output_csv)
    export_fairness_report(
        before_metrics=before,
        after_metrics=after,
        strategy=strategy,
        report_path=report_json,
        extra_info={
            "hybrid_cap":    hybrid_cap,
            "target_total":  target_total,
            "seed":          seed,
            "synthetic_dir": str(synthetic_dir),
            "metadata_csv":  str(metadata_csv),
        },
    )

    return balanced_df, before, after


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fairness-aware dataset balancing with synthetic images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metadata",       required=True,
                   help="Path to HAM10000 metadata or splits/train.csv")
    p.add_argument("--real_dir",       required=True,
                   help="Root HAM10000 image directory")
    p.add_argument("--synthetic_dir",  required=True,
                   help="Root of generated images (class subfolders)")
    p.add_argument("--strategy",       default="hybrid",
                   choices=STRATEGIES,
                   help="Balancing strategy")
    p.add_argument("--output_csv",     default="balanced_dataset.csv")
    p.add_argument("--report_json",    default="fairness_report.json")
    p.add_argument("--target_total",   type=int, default=None,
                   help="Total dataset size after balancing (None = auto)")
    p.add_argument("--hybrid_cap",     type=float, default=2.0,
                   help="Cap multiplier for hybrid strategy")
    p.add_argument("--classifier_ckpt",default=None,
                   help="Fine-tuned ResNet50 checkpoint for recall evaluation")
    p.add_argument("--eval_recall",    action="store_true",
                   help="Enable per-class recall computation (slow)")
    p.add_argument("--device",         default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_fairness_balancing(
        metadata_csv    = args.metadata,
        real_dir        = args.real_dir,
        synthetic_dir   = args.synthetic_dir,
        strategy        = args.strategy,
        output_csv      = args.output_csv,
        report_json     = args.report_json,
        target_total    = args.target_total,
        hybrid_cap      = args.hybrid_cap,
        classifier_ckpt = args.classifier_ckpt,
        eval_recall     = args.eval_recall,
        device          = args.device,
        seed            = args.seed,
    )
