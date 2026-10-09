"""
download_and_train.py
=====================
One-stop script that:
  1. Downloads HAM10000 images + metadata from the ISIC Archive (public API, no login)
  2. Splits data into train/val/test CSVs  (70/15/15, stratified by class)
  3. Applies class-weighted loss to handle imbalance
  4. Trains EfficientNet-B3 (pretrained ImageNet weights via timm)
  5. Saves best_model.pth → models/best_model.pth
  6. Prints final accuracy + per-class F1

Usage:
    python3 download_and_train.py                     # full run
    python3 download_and_train.py --skip_download      # if already downloaded
    python3 download_and_train.py --epochs 10 --batch_size 8  # quick test

Requirements (in ml_venv):
    torch torchvision timm scikit-learn pandas pillow tqdm requests
"""

from __future__ import annotations
import argparse, json, logging, math, os, random, sys, time
from pathlib import Path
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train")

# ─── constants ────────────────────────────────────────────────────────────────
HAM_CLASSES  = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
HAM_NAMES    = {
    "nv": "Melanocytic Nevi", "mel": "Melanoma",
    "bkl": "Benign Keratosis", "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis", "vasc": "Vascular Lesions",
    "df": "Dermatofibroma",
}
HAM_LABEL_MAP = {c: i for i, c in enumerate(HAM_CLASSES)}
IMAGE_SIZE   = 224          # EfficientNet-B3 native ≈300; 224 for CPU speed
ISIC_API     = "https://api.isic-archive.com/api/v2"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DOWNLOAD HAM10000 FROM ISIC ARCHIVE (public, no auth)
# ═══════════════════════════════════════════════════════════════════════════════

def download_ham10000(data_dir: Path, max_per_class: int = 500):
    """
    Download up to `max_per_class` images per HAM class via the ISIC public API.
    Total ≈ 3500 images (7 * 500) which is manageable on CPU.
    Full HAM10000 has 10,015 images — increase max_per_class for better accuracy.
    """
    import requests
    from PIL import Image
    import io

    data_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_dir / "metadata.csv"

    # HAM10000 diagnosis → ISIC diagnosis tag mapping
    DIAGNOSIS_MAP = {
        "nv":    "nevus",
        "mel":   "melanoma",
        "bkl":   "seborrheic keratosis",
        "bcc":   "basal cell carcinoma",
        "akiec": "actinic keratosis",
        "vasc":  "vascular lesion",
        "df":    "dermatofibroma",
    }

    all_meta = []
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    for cls, diagnosis in DIAGNOSIS_MAP.items():
        cls_dir = data_dir / cls
        cls_dir.mkdir(exist_ok=True)
        existing = list(cls_dir.glob("*.jpg"))
        if len(existing) >= max_per_class:
            log.info("  %s — already have %d images, skipping", cls, len(existing))
            for p in existing[:max_per_class]:
                all_meta.append({"image_path": str(p), "label": cls})
            continue

        log.info("  Fetching %s (%s) …", cls, diagnosis)
        collected = 0
        next_url = (
            f"{ISIC_API}/images/?"
            f"diagnosis={requests.utils.quote(diagnosis)}"
            f"&limit=50&offset=0"
        )

        while next_url and collected < max_per_class:
            try:
                r = session.get(next_url, timeout=30)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                log.warning("    API error: %s", e)
                break

            results = data.get("results", [])
            if not results:
                break

            for item in results:
                if collected >= max_per_class:
                    break
                isic_id = item.get("isic_id") or item.get("id")
                if not isic_id:
                    continue
                img_path = cls_dir / f"{isic_id}.jpg"
                if img_path.exists():
                    all_meta.append({"image_path": str(img_path), "label": cls})
                    collected += 1
                    continue

                # Download thumbnail (faster than full res)
                thumb_url = (
                    item.get("files", {}).get("thumbnail_256", {}).get("url")
                    or f"https://isic-archive.com/api/v1/image/{isic_id}/thumbnail?width=256"
                )
                try:
                    img_r = session.get(thumb_url, timeout=20)
                    img_r.raise_for_status()
                    img = Image.open(io.BytesIO(img_r.content)).convert("RGB")
                    img.save(img_path, "JPEG", quality=90)
                    all_meta.append({"image_path": str(img_path), "label": cls})
                    collected += 1
                    if collected % 50 == 0:
                        log.info("    %s: %d/%d", cls, collected, max_per_class)
                except Exception as e:
                    log.debug("    skip %s: %s", isic_id, e)

            next_url = data.get("next")
            if next_url and not next_url.startswith("http"):
                next_url = ISIC_API + next_url

        log.info("  %s — %d images downloaded", cls, collected)

    import pandas as pd
    df = pd.DataFrame(all_meta)
    df.to_csv(meta_path, index=False)
    log.info("Metadata saved → %s  (%d total images)", meta_path, len(df))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN/VAL/TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def make_splits(df, splits_dir: Path, seed: int = 42):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    splits_dir.mkdir(exist_ok=True)
    train_rows, valtest_rows = [], []

    for cls in HAM_CLASSES:
        cls_df = df[df["label"] == cls].copy()
        if len(cls_df) < 4:
            train_rows.append(cls_df)
            continue
        tr, vt = train_test_split(cls_df, test_size=0.30, random_state=seed)
        va, te = train_test_split(vt, test_size=0.50, random_state=seed)
        tr["split"] = "train"; va["split"] = "val"; te["split"] = "test"
        train_rows.append(tr); valtest_rows.extend([va, te])

    train_df = pd.concat([r for r in train_rows if isinstance(r, pd.DataFrame) and "split" in r.columns])
    valtest  = pd.concat(valtest_rows)
    val_df   = valtest[valtest["split"] == "val"]
    test_df  = valtest[valtest["split"] == "test"]

    train_df["label_idx"] = train_df["label"].map(HAM_LABEL_MAP)
    val_df  ["label_idx"] = val_df  ["label"].map(HAM_LABEL_MAP)
    test_df ["label_idx"] = test_df ["label"].map(HAM_LABEL_MAP)

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df  .to_csv(splits_dir / "val.csv",   index=False)
    test_df .to_csv(splits_dir / "test.csv",  index=False)

    log.info("Splits → train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class SkinDataset:
    def __init__(self, df, transform=None, image_size=IMAGE_SIZE):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        import torchvision.transforms.functional as F
        from PIL import Image
        row  = self.df.iloc[idx]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size))
        if self.transform:
            img = self.transform(img)
        return img, int(row["label_idx"])


def build_transforms(split: str, image_size: int):
    import torchvision.transforms as T
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)
    if split == "train":
        return T.Compose([
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomRotation(20),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    else:
        return T.Compose([
            T.Resize((image_size + 32, image_size + 32)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL — EfficientNet-B3 via timm
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int = 7, dropout: float = 0.3):
    import timm, torch.nn as nn
    model = timm.create_model(
        "efficientnet_b3",
        pretrained      = True,
        num_classes     = 0,       # remove original head
        drop_rate       = dropout,
    )
    in_features = model.num_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=dropout / 2),
        nn.Linear(512, num_classes),
    )
    return model


def count_params(model):
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(train_df, device):
    import torch
    counts = np.array([
        len(train_df[train_df["label"] == c]) for c in HAM_CLASSES
    ], dtype=np.float32)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(HAM_CLASSES)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def run_epoch(model, loader, loss_fn, optimizer, device, is_train: bool, scaler=None):
    import torch
    model.train(is_train)
    total_loss, all_preds, all_targets = 0.0, [], []

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            if scaler and is_train:
                from torch.amp import autocast
                with autocast("cpu"):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = loss_fn(logits, labels)
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_targets, all_preds)
    f1  = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    f1_per = f1_score(all_targets, all_preds, average=None,
                      labels=list(range(7)), zero_division=0)
    return total_loss / len(loader), acc, f1, f1_per


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import pandas as pd

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    ROOT     = Path(args.root)
    DATA_DIR = ROOT / "HAM10000_images"
    SPLIT_DIR= ROOT / "splits"
    OUT_DIR  = ROOT / "models"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")  # CPU-only system
    log.info("Device: CPU  (i5-1135G7, 8 threads)")
    torch.set_num_threads(args.threads)
    log.info("Using %d CPU threads", args.threads)

    # ── Step 1: Download ──────────────────────────────────────────────────
    meta_path = DATA_DIR / "metadata.csv"
    if not args.skip_download and not meta_path.exists():
        log.info("━━━ Downloading HAM10000 from ISIC Archive ━━━")
        df = download_ham10000(DATA_DIR, max_per_class=args.max_per_class)
    else:
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            log.info("Loaded existing metadata: %d images", len(df))
        else:
            log.error("No metadata found. Run without --skip_download first.")
            sys.exit(1)

    # ── Step 2: Split ─────────────────────────────────────────────────────
    train_csv = SPLIT_DIR / "train.csv"
    val_csv   = SPLIT_DIR / "val.csv"
    if not train_csv.exists():
        train_df, val_df, _ = make_splits(df, SPLIT_DIR)
    else:
        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)
        log.info("Loaded splits: train=%d  val=%d", len(train_df), len(val_df))

    # ── Step 3: Datasets + Loaders ────────────────────────────────────────
    train_ds = SkinDataset(train_df, build_transforms("train", args.image_size))
    val_ds   = SkinDataset(val_df,   build_transforms("val",   args.image_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=0)
    log.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # ── Step 4: Model ─────────────────────────────────────────────────────
    log.info("━━━ Building EfficientNet-B3 (pretrained ImageNet) ━━━")
    model = build_model(num_classes=7, dropout=args.dropout).to(device)
    total, trainable = count_params(model)
    log.info("Parameters: %.1fM total, %.1fM trainable", total/1e6, trainable/1e6)

    # ── Step 5: Loss with class weights (fights imbalance) ───────────────
    class_weights = compute_class_weights(train_df, device)
    log.info("Class weights: %s", {c: f"{w:.2f}" for c, w in zip(HAM_CLASSES, class_weights.tolist())})
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # ── Step 6: Optimizer — two param groups (backbone vs head) ──────────
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params     = [p for n, p in model.named_parameters() if "classifier"     in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},   # backbone: 10× slower
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    # Cosine annealing with linear warmup
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = 3 * len(train_loader)  # 3 warm-up epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Step 7: Training loop ─────────────────────────────────────────────
    log.info("━━━ Starting Training: %d epochs, batch=%d ━━━", args.epochs, args.batch_size)
    best_f1       = -1.0
    best_path     = OUT_DIR / "best_model.pth"
    patience_left = args.patience
    history       = []
    step          = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        # Train
        tr_loss, tr_acc, tr_f1, _ = run_epoch(
            model, train_loader, loss_fn, optimizer, device, is_train=True
        )
        # Update scheduler every batch
        for _ in range(len(train_loader)):
            scheduler.step()
            step += 1

        # Validate
        val_loss, val_acc, val_f1, val_f1_per = run_epoch(
            model, val_loader, loss_fn, optimizer, device, is_train=False
        )

        cur_lr  = optimizer.param_groups[1]["lr"]
        elapsed = time.perf_counter() - t0

        log.info(
            "Epoch %3d/%d  "
            "loss=%.4f/%.4f  acc=%.3f/%.3f  "
            "macro-F1=%.4f/%.4f  lr=%.2e  %.0fs",
            epoch, args.epochs,
            tr_loss, val_loss, tr_acc, val_acc,
            tr_f1, val_f1, cur_lr, elapsed,
        )

        per_cls_str = "  ".join(
            f"{c}={val_f1_per[i]:.3f}" for i, c in enumerate(HAM_CLASSES)
        )
        log.info("  Per-class val-F1: %s", per_cls_str)

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 5), "val_loss": round(val_loss, 5),
            "train_acc":  round(tr_acc,  4), "val_acc":  round(val_acc,  4),
            "train_f1":   round(tr_f1,   4), "val_f1":   round(val_f1,   4),
            "val_f1_per": {c: round(float(val_f1_per[i]), 4) for i, c in enumerate(HAM_CLASSES)},
            "lr": round(cur_lr, 8),
        })

        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_left = args.patience
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state":optimizer.state_dict(),
                "val_f1_macro":   val_f1,
                "val_accuracy":   val_acc,
                "val_f1_per_class": {c: float(val_f1_per[i]) for i,c in enumerate(HAM_CLASSES)},
                "ham_classes":    HAM_CLASSES,
                "image_size":     args.image_size,
                "backbone":       "efficientnet_b3",
            }, best_path)
            log.info("  ★  Saved best model → %s  (val_f1=%.4f)", best_path, best_f1)
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("Early stopping triggered at epoch %d (best val_f1=%.4f)", epoch, best_f1)
                break

    # ── Save history ──────────────────────────────────────────────────────
    hist_path = OUT_DIR / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump({"history": history, "best_val_f1": best_f1}, f, indent=2)
    log.info("Training history → %s", hist_path)
    log.info("━━━ Training complete. Best val macro-F1 = %.4f ━━━", best_f1)
    log.info("Model saved at: %s", best_path)
    log.info("To enable real predictions, set ML_DUMMY_MODE=false in backend/.env")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download HAM10000 + Train EfficientNet-B3")
    p.add_argument("--root",          default=".",        help="Project root dir")
    p.add_argument("--skip_download", action="store_true", help="Skip download if data exists")
    p.add_argument("--max_per_class", type=int, default=600,
                   help="Max images per class to download (600=~4200 total, fast)")
    p.add_argument("--image_size",    type=int, default=224)
    p.add_argument("--epochs",        type=int, default=30)
    p.add_argument("--batch_size",    type=int, default=16)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--patience",      type=int, default=8)
    p.add_argument("--threads",       type=int, default=6, help="CPU threads for PyTorch")
    args = p.parse_args()
    main(args)
