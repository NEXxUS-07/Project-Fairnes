"""
load_trained_model.py
=====================
Adapter that loads a best_model.pth saved by download_and_train.py
and wires it into the running FastAPI backend (no restart required).

Usage:
    # After training finishes at models/best_model.pth:
    python3 load_trained_model.py          # validates the model
    python3 load_trained_model.py --test   # runs a test inference

The script also patches backend/.env to switch from dummy mode
to real mode automatically.
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "best_model.pth"
ENV_PATH   = ROOT / "backend" / ".env"

HAM_CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
HAM_NAMES   = {
    "nv": "Melanocytic Nevi", "mel": "Melanoma",
    "bkl": "Benign Keratosis", "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis", "vasc": "Vascular Lesions",
    "df": "Dermatofibroma",
}


def load_model(model_path: Path, device: str = "cpu"):
    """Load the trained EfficientNet-B3 checkpoint."""
    import torch
    import timm
    import torch.nn as nn

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    print(f"✓ Checkpoint loaded")
    print(f"  Epoch:       {ckpt.get('epoch', '?')}")
    print(f"  Val F1:      {ckpt.get('val_f1_macro', '?'):.4f}")
    print(f"  Val Acc:     {ckpt.get('val_accuracy', '?'):.4f}")
    print(f"  Backbone:    {ckpt.get('backbone', 'efficientnet_b3')}")
    f1_per = ckpt.get("val_f1_per_class", {})
    if f1_per:
        print("  Per-class F1:")
        for cls, val in f1_per.items():
            print(f"    {cls:8s}  {val:.4f}  ({HAM_NAMES.get(cls, cls)})")

    # Rebuild same architecture as download_and_train.py
    model = timm.create_model(
        ckpt.get("backbone", "efficientnet_b3"),
        pretrained=False,
        num_classes=0,
    )
    in_features = model.num_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=0.15),
        nn.Linear(512, 7),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✓ Model weights loaded successfully")
    return model, ckpt


def run_test_inference(model, image_size: int = 224):
    """Run a dummy inference to confirm the model works."""
    import torch
    import torch.nn.functional as F

    dummy = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        logits = model(dummy)
        probs  = F.softmax(logits, dim=1)[0]

    pred_idx   = probs.argmax().item()
    pred_class = HAM_CLASSES[pred_idx]
    print(f"\n✓ Test inference: predicted '{pred_class}' ({HAM_NAMES[pred_class]}) "
          f"with confidence {probs[pred_idx].item():.2%}")
    print("  All probabilities:")
    for i, cls in enumerate(HAM_CLASSES):
        bar = "█" * int(probs[i].item() * 30)
        print(f"    {cls:8s}  {probs[i].item():.4f}  {bar}")


def patch_env_for_real_mode():
    """Switch backend/.env from dummy mode to real mode."""
    if not ENV_PATH.exists():
        print(f"⚠  .env not found at {ENV_PATH}")
        return

    text = ENV_PATH.read_text()
    updates = {
        "ML_DUMMY_MODE":       "false",
        "DIFFUSION_DUMMY_MODE":"true",   # keep diffusion dummy (no diffusion model yet)
        "ML_CHECKPOINT":       str(MODEL_PATH),
        "ML_DEVICE":           "cpu",
    }
    lines = text.splitlines()
    new_lines = []
    updated = set()
    for line in lines:
        key = line.split("=")[0].strip()
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            updated.add(key)
        else:
            new_lines.append(line)

    for key, val in updates.items():
        if key not in updated:
            new_lines.append(f"{key}={val}")

    ENV_PATH.write_text("\n".join(new_lines) + "\n")
    print(f"✓ backend/.env patched for REAL model mode")
    print(f"  ML_DUMMY_MODE=false")
    print(f"  ML_CHECKPOINT={MODEL_PATH}")
    print(f"\n  → Restart backend to load the real model:")
    print(f"    pkill -f 'uvicorn main:app'")
    print(f"    cd backend && ML_DUMMY_MODE=false ML_CHECKPOINT={MODEL_PATH} \\")
    print(f"      .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test",       action="store_true", help="Run test inference")
    p.add_argument("--patch_env",  action="store_true", help="Patch .env to real mode")
    p.add_argument("--model_path", default=str(MODEL_PATH))
    args = p.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"✗ Model not found at {model_path}")
        print(f"  Run training first: ml_venv/bin/python download_and_train.py")
        sys.exit(1)

    model, ckpt = load_model(model_path)

    if args.test:
        run_test_inference(model, image_size=ckpt.get("image_size", 224))

    if args.patch_env:
        patch_env_for_real_mode()
