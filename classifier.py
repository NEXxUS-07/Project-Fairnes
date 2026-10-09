"""
classifier.py — EfficientNet-B3 Skin-Lesion Classifier for HAM10000
=====================================================================
Provides:
  • EfficientNetB3Classifier  – model class (timm backbone + classifier head)
  • build_class_weights()     – inverse-frequency class weights from a DataFrame
  • build_loss()              – WeightedCrossEntropyLoss factory
  • predict_batch()           – inference helper → (logits, probs, preds)
  • load_classifier()         – load model from checkpoint

Configuration defaults:
  backbone     =  tf_efficientnet_b3.ns  (NoisyStudent, best accuracy variant)
  pretrained   =  True  (ImageNet weights)
  num_classes  =  7
  dropout_rate =  0.3   (slightly higher than EfficientNet-B3 default for medical)
  image_size   =  300   (EfficientNet-B3 native resolution)

Dependencies:
  pip install timm
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False
    pd = None

try:
    import timm
    _TIMM_OK = True
except ImportError:
    raise ImportError("timm is required. Install with: pip install timm")

log = logging.getLogger(__name__)

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

NUM_CLASSES      = 7
EFFICIENTNET_MODEL = "efficientnet_b3"  # matches existing checkpoint
NATIVE_IMAGE_SIZE  = 300                        # EfficientNet-B3 native resolution


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════

class EfficientNetB3Classifier(nn.Module):
    """
    EfficientNet-B3 fine-tuned for HAM10000 7-class skin lesion classification.

    Architecture:
        • EfficientNet-B3 backbone (pretrained on ImageNet via timm)
        • Global Average Pooling (handled internally by timm)
        • Dropout → Linear(1536, num_classes)

    Conditioning modes during training:
        • Full fine-tuning (all layers trained from epoch 1)
        • Frozen backbone for first N epochs (set via freeze_backbone_epochs
          in train_classifier.py)

    Parameters
    ----------
    num_classes    : int    number of output classes (default 7)
    pretrained     : bool   load ImageNet weights from timm hub
    dropout_rate   : float  dropout before final classifier head
    backbone_name  : str    timm model identifier
    """

    def __init__(
        self,
        num_classes:    int   = NUM_CLASSES,
        pretrained:     bool  = True,
        dropout_rate:   float = 0.3,
        backbone_name:  str   = EFFICIENTNET_MODEL,
    ):
        super().__init__()
        self.num_classes   = num_classes
        self.backbone_name = backbone_name
        self.dropout_rate  = dropout_rate

        # ── Backbone (timm) ───────────────────────────────────────────────
        # num_classes=0 → removes the timm classifier head;
        # global_pool=""  → expose raw feature tensor (B, C, H, W)
        # global_pool="avg" → expose pooled (B, C) — we use avg
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,       # remove original head
            global_pool="avg",   # (B, num_features) after backbone
            drop_rate=dropout_rate,
        )
        num_features = self.backbone.num_features   # 1536 for EfficientNet-B3

        # ── Custom head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes),
        )
        self._init_head()

        log.info(
            "EfficientNetB3Classifier | backbone=%s | features=%d "
            "| dropout=%.2f | classes=%d | pretrained=%s",
            backbone_name, num_features, dropout_rate, num_classes, pretrained,
        )

    # ─────────────────────────────────────────────────────────────────────

    def _init_head(self):
        """Kaiming initialisation for the custom classifier head linear layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W)  ImageNet-normalised input

        Returns
        -------
        logits : (B, num_classes)  raw (un-softmaxed) logits
        """
        features = self.backbone(x)   # (B, 1536)
        return self.classifier(features)    # (B, num_classes)

    # ─────────────────────────────────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze all backbone parameters (train head only)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        log.info("Backbone frozen — training head only.")

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for full fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        log.info("Backbone unfrozen — full fine-tuning enabled.")

    def freeze_backbone_except_last(self, n_blocks: int = 3):
        """
        Partially freeze backbone — only unfreeze the last ``n_blocks``
        EfficientNet blocks.  Useful for gradual unfreezing.
        """
        # Freeze everything first
        self.freeze_backbone()
        # Unfreeze last n blocks via timm's named modules
        blocks = [m for name, m in self.backbone.named_modules()
                  if "blocks" in name and "." not in name.replace("blocks.", "").split(".")[0]]
        for blk in blocks[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        log.info("Unfroze last %d backbone blocks.", n_blocks)

    # ─────────────────────────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        total = self.num_parameters / 1e6
        train = self.num_trainable_parameters / 1e6
        return (
            f"EfficientNetB3Classifier(\n"
            f"  backbone    = {self.backbone_name}\n"
            f"  num_classes = {self.num_classes}\n"
            f"  dropout     = {self.dropout_rate}\n"
            f"  params      = {total:.2f}M total  |  {train:.2f}M trainable\n"
            f")"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  LOSS
# ═══════════════════════════════════════════════════════════════════════════

def build_class_weights(
    df:       Optional[pd.DataFrame] = None,
    label_col: str = "label",
    device:    Union[str, torch.device] = "cpu",
    smoothing: float = 0.05,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a dataset DataFrame.

    Formula:
        w_c = (1 / count_c) * (total / num_classes)

    Applied label smoothing to prevent extreme weights for very rare classes.
    ``smoothing`` shifts count_c ← count_c + smoothing * max_count.

    Parameters
    ----------
    df          : DataFrame with a ``label`` column containing class codes.
                  If None, returns uniform weights.
    label_col   : column name for class labels.
    device      : target device for the output tensor.
    smoothing   : small additive smoothing fraction.

    Returns
    -------
    torch.Tensor of shape (num_classes,) on ``device``.
    """
    if df is None or not _PANDAS_OK:
        weights = np.ones(NUM_CLASSES, dtype=np.float64)
        weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
        return weights_t

    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for cls, idx in HAM_LABEL_MAP.items():
        counts[idx] = (df[label_col] == cls).sum()

    # Optional: protect against 0-count classes
    max_cnt  = counts.max()
    counts   = counts + smoothing * max_cnt    # smooth

    weights  = 1.0 / counts
    weights  = weights / weights.sum() * NUM_CLASSES   # normalise so mean=1
    weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    log.info(
        "Class weights (smoothed): %s",
        {HAM_CLASSES[i]: round(float(weights_t[i]), 4) for i in range(NUM_CLASSES)},
    )
    return weights_t


def build_loss(
    weights: Optional[Tensor] = None,
    label_smoothing: float = 0.05,
) -> nn.CrossEntropyLoss:
    """
    Build a WeightedCrossEntropyLoss with optional label smoothing.

    Parameters
    ----------
    weights         : (num_classes,) class weight tensor.  None = unweighted.
    label_smoothing : epsilon for label smoothing regularisation.

    Returns
    -------
    nn.CrossEntropyLoss
    """
    return nn.CrossEntropyLoss(
        weight          = weights,
        label_smoothing = label_smoothing,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_batch(
    model:  EfficientNetB3Classifier,
    images: Tensor,
    device: Union[str, torch.device] = "cuda",
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Run inference on a batch of images.

    Parameters
    ----------
    model  : trained classifier
    images : (B, 3, H, W) ImageNet-normalised tensor

    Returns
    -------
    logits : (B, C)  raw logits
    probs  : (B, C)  class probabilities (softmax)
    preds  : (B,)    integer class predictions
    """
    device = torch.device(device)
    model  = model.to(device).eval()
    images = images.to(device)

    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        logits = model(images)

    probs = F.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    return logits.cpu(), probs.cpu(), preds.cpu()


# ═══════════════════════════════════════════════════════════════════════════
#  CHECKPOINT I/O
# ═══════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model:      EfficientNetB3Classifier,
    optimizer:  torch.optim.Optimizer,
    epoch:      int,
    metrics:    dict,
    path:       Union[str, Path] = "best_model.pth",
    scaler:     Optional[torch.cuda.amp.GradScaler] = None,
):
    """Save model, optimizer, scaler, epoch, and metric dict to a checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch":            epoch,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "metrics":          metrics,
        "backbone_name":    model.backbone_name,
        "num_classes":      model.num_classes,
        "dropout_rate":     model.dropout_rate,
    }
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()

    torch.save(payload, path)
    log.info(
        "Checkpoint saved → %s  (epoch=%d, val_f1=%.4f)",
        path, epoch, metrics.get("val_f1_macro", float("nan")),
    )


def load_classifier(
    ckpt_path: Union[str, Path],
    device:    Union[str, torch.device] = "cpu",
    strict:    bool = True,
) -> tuple[EfficientNetB3Classifier, dict]:
    """
    Load an EfficientNetB3Classifier from a checkpoint.

    Supports three checkpoint formats:
      1. classifier.py save_checkpoint() format  → has backbone_name, num_classes, dropout_rate
      2. train_fast.py / download_and_train.py    → has backbone, ham_classes, image_size
         (state dict keys may be bare conv_stem / blocks / classifier)
      3. Legacy raw timm model with classifier head assigned directly

    Returns
    -------
    (model, metadata_dict)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── Detect checkpoint format ─────────────────────────────────────────
    if "backbone_name" in ckpt:
        # Format from classifier.py save_checkpoint()
        backbone_name = ckpt.get("backbone_name", EFFICIENTNET_MODEL)
        num_classes   = ckpt.get("num_classes",   NUM_CLASSES)
        dropout_rate  = ckpt.get("dropout_rate",  0.3)
    elif "backbone" in ckpt:
        # Format from train_fast.py / download_and_train.py
        backbone_name = ckpt["backbone"]
        num_classes   = len(ckpt.get("ham_classes", HAM_CLASSES))
        dropout_rate  = 0.35
    else:
        backbone_name = EFFICIENTNET_MODEL
        num_classes   = NUM_CLASSES
        dropout_rate  = 0.3

    model = EfficientNetB3Classifier(
        num_classes   = num_classes,
        pretrained    = False,
        dropout_rate  = dropout_rate,
        backbone_name = backbone_name,
    )

    # ── Detect classifier head architecture from state dict ──────────────
    sd = ckpt["model_state"]
    classifier_keys = [k for k in sd.keys() if k.startswith("classifier.") or k.startswith("backbone.classifier.")]

    cls_indices = set()
    for k in classifier_keys:
        parts = k.split(".")
        if parts[0] == "backbone":
            parts = parts[1:]
        if len(parts) >= 2 and parts[0] == "classifier":
            try:
                cls_indices.add(int(parts[1]))
            except ValueError:
                pass

    has_second_bn = 4 in cls_indices

    if not has_second_bn and cls_indices == {0, 2, 5}:
        log.info("Detected old classifier head (no second BN), building compatible model")
        num_features = model.backbone.num_features
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(512, num_classes),
        )

    # ── Map state dict keys if needed ───────────────────────────────────
    new_sd = {}
    key_mismatch = False
    for k, v in sd.items():
        if k.startswith("backbone."):
            new_sd[k] = v
        elif k in ("conv_stem.weight", "bn1.weight", "bn1.bias",
                   "bn1.running_mean", "bn1.running_var", "bn1.num_batches_tracked",
                   "conv_head.weight", "bn2.weight", "bn2.bias",
                   "bn2.running_mean", "bn2.running_var", "bn2.num_batches_tracked"):
            new_sd[f"backbone.{k}"] = v
            key_mismatch = True
        elif k.startswith("blocks."):
            new_sd[f"backbone.{k}"] = v
            key_mismatch = True
        elif k.startswith("classifier."):
            new_sd[k] = v
        else:
            new_sd[k] = v

    if key_mismatch:
        log.info("Remapped bare timm backbone keys → backbone.* prefix")

    model.load_state_dict(new_sd, strict=strict)
    model = model.to(torch.device(device)).eval()
    log.info(
        "Loaded classifier from %s  (backbone=%s, classes=%d, epoch=%d)",
        ckpt_path, backbone_name, num_classes, ckpt.get("epoch", -1),
    )
    return model, ckpt
