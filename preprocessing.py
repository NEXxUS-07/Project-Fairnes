"""
HAM10000 Skin Lesion Dataset — Preprocessing Pipeline
======================================================
Senior ML Engineer pipeline for:
  - Loading metadata + validating image paths
  - EDA: class distribution + imbalance analysis
  - Generating bar charts saved to eda_plots/
  - Stratified train / val / test split (70 / 15 / 15)
  - Saving split CSVs to disk

Usage:
    python preprocessing.py \
        --metadata path/to/HAM10000_metadata.csv \
        --image_dir path/to/HAM10000_images \
        --output_dir ./splits

Dependencies:
    pip install pandas matplotlib scikit-learn seaborn
"""

import argparse
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CLASS_LABELS = {
    "nv":    "Melanocytic Nevi",
    "mel":   "Melanoma",
    "bkl":   "Benign Keratosis",
    "bcc":   "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "vasc":  "Vascular Lesions",
    "df":    "Dermatofibroma",
}

# Accessible colour palette (Wong 2011 colour-blind-safe)
PALETTE = {
    "nv":    "#0072B2",
    "mel":   "#D55E00",
    "bkl":   "#009E73",
    "bcc":   "#CC79A7",
    "akiec": "#E69F00",
    "vasc":  "#56B4E9",
    "df":    "#F0E442",
}

PLOT_DIR = Path("eda_plots")


# ═══════════════════════════════════════════════════════════════
#  1. DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_metadata(csv_path: str | Path, image_dir: str | Path) -> pd.DataFrame:
    """
    Load HAM10000 metadata CSV and attach verified image file paths.

    The dataset ships images in two folders (HAM10000_images_part1 /
    HAM10000_images_part2).  We handle both flat directories and any
    sub-folder layout by scanning recursively.

    Returns
    -------
    pd.DataFrame with extra column ``image_path`` (str) and ``label_idx`` (int).
    Rows whose image file cannot be found are **dropped** with a warning.
    """
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    log.info("Loading metadata from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Normalise column names (some releases use different capitalisation)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"image_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    log.info("Raw metadata: %d rows, %d columns", *df.shape)

    # ── Build image-id → path lookup (scan all .jpg / .jpeg / .png) ──
    log.info("Scanning image directory…")
    image_lookup: dict[str, Path] = {}
    for p in image_dir.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_lookup[p.stem] = p   # stem = filename without extension

    log.info("Found %d image files in directory tree.", len(image_lookup))

    df["image_path"] = df["image_id"].map(lambda x: str(image_lookup[x])
                                          if x in image_lookup else None)

    missing_images = df["image_path"].isna().sum()
    if missing_images:
        log.warning(
            "%d rows dropped — image files not found on disk.", missing_images
        )
        df = df.dropna(subset=["image_path"]).copy()

    # ── Encode labels ──
    class_order = list(CLASS_LABELS.keys())
    df["dx"] = df["dx"].str.strip().str.lower()
    unknown = set(df["dx"].unique()) - set(class_order)
    if unknown:
        log.warning("Unknown dx classes encountered and will be kept: %s", unknown)

    df["label_idx"] = df["dx"].map({c: i for i, c in enumerate(class_order)})

    log.info("Final dataset: %d samples across %d classes.", len(df), df["dx"].nunique())
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  2. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════

def compute_class_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-class sample counts, proportions, and imbalance ratio
    relative to the majority class.

    Returns
    -------
    pd.DataFrame indexed by dx (class code).
    """
    counts = df["dx"].value_counts().rename("count")
    stats = counts.to_frame()
    stats["proportion_%"] = (stats["count"] / stats["count"].sum() * 100).round(2)
    stats["label"] = stats.index.map(CLASS_LABELS)

    majority_count = stats["count"].max()
    stats["imbalance_ratio"] = (majority_count / stats["count"]).round(2)

    # Sort by count descending for readability
    stats = stats.sort_values("count", ascending=False)

    log.info("\n%s", stats[["label", "count", "proportion_%", "imbalance_ratio"]].to_string())
    return stats


def _style_axes(ax, title: str, xlabel: str, ylabel: str):
    """Apply consistent dark-theme styling to a matplotlib Axes."""
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=12)
    ax.set_xlabel(xlabel, fontsize=11, color="#a0a0c0")
    ax.set_ylabel(ylabel, fontsize=11, color="#a0a0c0")
    ax.tick_params(colors="#a0a0c0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="y", color="#333355", linewidth=0.6, linestyle="--", alpha=0.7)


def plot_class_distribution(stats: pd.DataFrame, save_dir: Path):
    """
    Bar chart — absolute sample counts per class.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0f0f23")

    classes = stats.index.tolist()
    colours = [PALETTE.get(c, "#888888") for c in classes]
    bars = ax.bar(classes, stats["count"], color=colours,
                  edgecolor="white", linewidth=0.5, width=0.65)

    # Annotate bars
    for bar, count in zip(bars, stats["count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + stats["count"].max() * 0.01,
            f"{count:,}",
            ha="center", va="bottom", fontsize=9,
            color="white", fontweight="bold",
        )

    # Add full class labels as secondary x-ticks
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(
        [f"{c}\n({CLASS_LABELS.get(c, c)})" for c in classes],
        fontsize=8, color="#c0c0e0",
    )

    _style_axes(ax, "HAM10000 — Class Sample Counts", "Lesion Class", "Number of Samples")
    fig.tight_layout(pad=2)
    out = save_dir / "class_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


def plot_imbalance_ratio(stats: pd.DataFrame, save_dir: Path):
    """
    Horizontal bar chart — imbalance ratio per class (majority / class_count).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f0f23")

    sorted_stats = stats.sort_values("imbalance_ratio", ascending=True)
    classes = sorted_stats.index.tolist()
    ratios  = sorted_stats["imbalance_ratio"].tolist()
    colours = [PALETTE.get(c, "#888888") for c in classes]

    bars = ax.barh(classes, ratios, color=colours,
                   edgecolor="white", linewidth=0.5, height=0.55)

    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_width() + max(ratios) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{ratio:.1f}×",
            ha="left", va="center", fontsize=9,
            color="white", fontweight="bold",
        )

    ax.set_facecolor("#1a1a2e")
    ax.set_title(
        "HAM10000 — Imbalance Ratio  (majority_count / class_count)",
        fontsize=13, fontweight="bold", color="white", pad=10,
    )
    ax.set_xlabel("Imbalance Ratio", fontsize=11, color="#a0a0c0")
    ax.tick_params(colors="#a0a0c0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(axis="x", color="#333355", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}×"))
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(
        [f"{c}  ({CLASS_LABELS.get(c, c)})" for c in classes],
        fontsize=8, color="#c0c0e0",
    )
    fig.patch.set_facecolor("#0f0f23")
    fig.tight_layout(pad=2)
    out = save_dir / "imbalance_ratio.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


def plot_proportion_pie(stats: pd.DataFrame, save_dir: Path):
    """
    Donut / pie chart — class proportion in percentage.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#0f0f23")

    labels  = [CLASS_LABELS.get(c, c) for c in stats.index]
    sizes   = stats["proportion_%"].tolist()
    colours = [PALETTE.get(c, "#888888") for c in stats.index]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colours,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops={"linewidth": 1.5, "edgecolor": "#0f0f23"},
    )
    for atext in autotexts:
        atext.set_color("white")
        atext.set_fontsize(8)

    # Draw inner circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.55, fc="#0f0f23")
    ax.add_patch(centre_circle)
    ax.axis("equal")

    ax.legend(
        wedges, [f"{c}  ({CLASS_LABELS.get(c, c)})" for c in stats.index],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=8,
        facecolor="#1a1a2e",
        edgecolor="#333355",
        labelcolor="#c0c0e0",
    )
    ax.set_title(
        "HAM10000 — Class Proportion (%)",
        fontsize=13, fontweight="bold", color="white", pad=16,
    )
    fig.tight_layout()
    out = save_dir / "class_proportion_pie.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


def plot_skin_tone_distribution(df: pd.DataFrame, save_dir: Path):
    """
    If a skin_tone / fitzpatrick column exists, plot distribution of
    skin tone labels across lesion classes.  This column is optional
    and anticipated from future fairness annotation.
    """
    candidate_cols = [c for c in df.columns
                      if any(kw in c for kw in ("skin", "fitzpatrick", "tone", "ita"))]
    if not candidate_cols:
        log.info("No skin-tone column found — skipping skin-tone plot.")
        return

    col = candidate_cols[0]
    log.info("Plotting skin-tone distribution using column '%s'.", col)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0f0f23")

    pivot = (
        df.groupby(["dx", col])
        .size()
        .unstack(fill_value=0)
    )
    pivot.plot(kind="bar", ax=ax, colormap="plasma", edgecolor="white", linewidth=0.4)

    _style_axes(ax, f"Class × Skin-Tone Distribution  [{col}]",
                "Lesion Class", "Sample Count")
    ax.legend(title=col, facecolor="#1a1a2e",
              edgecolor="#333355", labelcolor="#c0c0e0", fontsize=8)
    fig.tight_layout(pad=2)
    out = save_dir / "skin_tone_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ═══════════════════════════════════════════════════════════════
#  3. STRATIFIED TRAIN / VAL / TEST SPLIT
# ═══════════════════════════════════════════════════════════════

def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    test_ratio: float  = 0.15,
    random_state: int  = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a 70 / 15 / 15 stratified split by lesion class.

    Stratification is performed on ``dx`` (7-class code).  If a
    ``lesion_id`` column is present, we ensure that all images
    belonging to the same lesion stay in the same split (patient-level
    leakage prevention).

    Returns
    -------
    (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # ── Patient-level de-duplication (keeps one row per lesion_id) ──
    group_col = "lesion_id" if "lesion_id" in df.columns else None

    if group_col:
        log.info("Splitting at LESION level to prevent patient leakage.")
        unique_lesions = (
            df[[group_col, "dx"]]
            .drop_duplicates(subset=[group_col])
            .reset_index(drop=True)
        )
        strat_col = "dx"
        # train vs temp
        train_lesions, temp_lesions = train_test_split(
            unique_lesions,
            test_size=1 - train_ratio,
            stratify=unique_lesions[strat_col],
            random_state=random_state,
        )
        # val vs test
        relative_val = val_ratio / (val_ratio + test_ratio)
        val_lesions, test_lesions = train_test_split(
            temp_lesions,
            test_size=1 - relative_val,
            stratify=temp_lesions[strat_col],
            random_state=random_state,
        )
        train_ids = set(train_lesions[group_col])
        val_ids   = set(val_lesions[group_col])
        test_ids  = set(test_lesions[group_col])

        train_df = df[df[group_col].isin(train_ids)].copy()
        val_df   = df[df[group_col].isin(val_ids)].copy()
        test_df  = df[df[group_col].isin(test_ids)].copy()
    else:
        log.warning(
            "'lesion_id' column not found. Splitting at IMAGE level "
            "(patients may bleed across splits if same lesion has multiple images)."
        )
        train_df, temp_df = train_test_split(
            df,
            test_size=1 - train_ratio,
            stratify=df["dx"],
            random_state=random_state,
        )
        relative_val = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - relative_val,
            stratify=temp_df["dx"],
            random_state=random_state,
        )

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        log.info(
            "  %-5s  →  %5d samples  | class dist: %s",
            name, len(split),
            split["dx"].value_counts().to_dict(),
        )

    return train_df.reset_index(drop=True), \
           val_df.reset_index(drop=True), \
           test_df.reset_index(drop=True)


def save_splits(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    output_dir: str | Path,
):
    """
    Save train / val / test DataFrames to CSV files.
    A ``split`` column is added to each file for bookkeeping.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df = split_df.copy()
        split_df["split"] = name
        out = output_dir / f"{name}.csv"
        split_df.to_csv(out, index=False)
        log.info("Saved %s split → %s  (%d rows)", name, out, len(split_df))


# ═══════════════════════════════════════════════════════════════
#  4. NORMALISATION PARAMETERS
# ═══════════════════════════════════════════════════════════════

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_normalization_stats() -> dict:
    """
    Return ImageNet normalisation statistics used for fine-tuning
    pre-trained models on HAM10000.

    Note: these are ImageNet statistics.  Dataset-specific stats can
    be computed with ``compute_dataset_mean_std()`` if desired.
    """
    return {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}


def compute_dataset_mean_std(
    image_paths: list[str],
    max_samples: int = 2000,
    image_size:  int = 224,
) -> tuple[tuple, tuple]:
    """
    Compute per-channel mean and std across up to ``max_samples`` images
    so that dataset-specific normalisation stats can be used instead of
    ImageNet defaults.

    Requires PIL and torchvision installed.  Returns (mean, std) tuples.
    """
    try:
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF
    except ImportError:
        log.warning("torch / torchvision / PIL not installed. Returning ImageNet defaults.")
        return IMAGENET_MEAN, IMAGENET_STD

    import random
    sample_paths = random.sample(image_paths, min(max_samples, len(image_paths)))

    channel_sum  = torch.zeros(3)
    channel_sum2 = torch.zeros(3)
    n_pixels     = 0

    for path in sample_paths:
        try:
            img = Image.open(path).convert("RGB").resize((image_size, image_size))
            t   = TF.to_tensor(img)          # (3, H, W) in [0, 1]
            channel_sum  += t.sum(dim=[1, 2])
            channel_sum2 += (t ** 2).sum(dim=[1, 2])
            n_pixels     += image_size * image_size
        except Exception:
            pass   # skip corrupted images silently

    mean = (channel_sum / n_pixels).tolist()
    std  = ((channel_sum2 / n_pixels - torch.tensor(mean) ** 2) ** 0.5).tolist()

    log.info(
        "Dataset-specific stats (n=%d images):\n  mean=%.4f, %.4f, %.4f"
        "\n  std =%.4f, %.4f, %.4f",
        len(sample_paths), *mean, *std,
    )
    return tuple(mean), tuple(std)


# ═══════════════════════════════════════════════════════════════
#  5. ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="HAM10000 Preprocessing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metadata",   required=True,  help="Path to HAM10000_metadata.csv")
    p.add_argument("--image_dir",  required=True,  help="Root directory of HAM10000 images")
    p.add_argument("--output_dir", default="splits", help="Directory to save split CSVs")
    p.add_argument("--plot_dir",   default="eda_plots", help="Directory to save EDA plots")
    p.add_argument("--train",      type=float, default=0.70, help="Train split ratio")
    p.add_argument("--val",        type=float, default=0.15, help="Val split ratio")
    p.add_argument("--test",       type=float, default=0.15, help="Test split ratio")
    p.add_argument("--seed",       type=int,   default=42,   help="Random seed")
    return p.parse_args()


def run_pipeline(
    metadata_path: str | Path,
    image_dir:     str | Path,
    output_dir:    str | Path = "splits",
    plot_dir:      str | Path = "eda_plots",
    train_ratio:   float = 0.70,
    val_ratio:     float = 0.15,
    test_ratio:    float = 0.15,
    random_state:  int   = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Public API — call this from another script or notebook.
    """
    plot_dir = Path(plot_dir)

    # ── Load ──
    df = load_metadata(metadata_path, image_dir)

    # ── EDA ──
    stats = compute_class_stats(df)
    plot_class_distribution(stats, plot_dir)
    plot_imbalance_ratio(stats, plot_dir)
    plot_proportion_pie(stats, plot_dir)
    plot_skin_tone_distribution(df, plot_dir)

    # ── Split ──
    train_df, val_df, test_df = stratified_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    # ── Save ──
    save_splits(train_df, val_df, test_df, output_dir)

    return train_df, val_df, test_df


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        metadata_path=args.metadata,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        plot_dir=args.plot_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        random_state=args.seed,
    )
