"""
explainer.py — Unified Skin Lesion Explainability Orchestrator
===============================================================
Runs Grad-CAM + SHAP together on one image or a batch.

Per-prediction output
---------------------
  • Original image, Grad-CAM overlay PNG, SHAP explanation PNG
  • Confidence scores per class
  • JSON summary: {predicted_class, confidence, top_3_classes,
                   explanation_path, gradcam_path, shap_path}

Batch report (validation set)
------------------------------
  • Per-class aggregate SHAP importance map (PNG)
  • JSON summary across all samples

Directory structure written to ``--output_dir``:
    output_dir/
        gradcam/
            gradcam_{idx}.png
        shap/
            shap_{idx}.png
        json/
            explanation_{idx}.json
        aggregate_shap_report.png
        batch_report.json

Usage:
    # Single image:
    python explainer.py \\
        --checkpoint results/best_model.pth \\
        --image      path/to/image.jpg \\
        --output_dir explanations/

    # Batch from CSV:
    python explainer.py \\
        --checkpoint  results/best_model.pth \\
        --test_csv    splits/test.csv \\
        --image_dir   HAM10000_images/ \\
        --output_dir  explanations/ \\
        --n_samples   50

    # Aggregate validation report:
    python explainer.py \\
        --checkpoint  results/best_model.pth \\
        --test_csv    splits/val.csv \\
        --image_dir   HAM10000_images/ \\
        --output_dir  explanations/ \\
        --aggregate   --n_background 100

Dependencies:
    pip install pytorch-grad-cam shap opencv-python
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifier import (
    EfficientNetB3Classifier,
    load_classifier,
    HAM_CLASSES, HAM_FULL_NAMES, NATIVE_IMAGE_SIZE,
)
from gradcam import GradCAM, denormalize, overlay_heatmap, heatmap_to_rgb
from shap_analysis import SHAPExplainer, _denorm

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
TEXT_COL = "#e6edf3"


# ═══════════════════════════════════════════════════════════════════════════
#  EXPLAINER ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class SkinLesionExplainer:
    """
    Unified explainability interface combining Grad-CAM and SHAP.

    Parameters
    ----------
    model       : loaded EfficientNetB3Classifier (eval mode)
    device      : torch device
    shap_loader : DataLoader for background samples (required for SHAP).
                  If None, SHAP is disabled.
    n_background: number of background samples for SHAP.
    use_gradcam_lib : pass to GradCAM — use pytorch-grad-cam if available.
    """

    def __init__(
        self,
        model:           EfficientNetB3Classifier,
        device:          Union[str, torch.device] = "cpu",
        shap_loader      = None,
        n_background:    int  = 100,
        use_gradcam_lib: bool = True,
    ):
        self.device = torch.device(device)
        self.model  = model.to(self.device).eval()

        # ── Grad-CAM ──────────────────────────────────────────────────────
        self.gradcam = GradCAM(
            model           = self.model,
            use_gradcam_lib = use_gradcam_lib,
        )
        log.info("Grad-CAM initialised.")

        # ── SHAP (optional) ───────────────────────────────────────────────
        self.shap_exp: Optional[SHAPExplainer] = None
        if shap_loader is not None:
            try:
                self.shap_exp = SHAPExplainer(
                    model             = self.model,
                    background_loader = shap_loader,
                    n_background      = n_background,
                    device            = self.device,
                )
                log.info("SHAP explainer initialised.")
            except Exception as e:
                log.warning("SHAP init failed (%s) — SHAP disabled.", e)

    # ─────────────────────────────────────────────────────────────────────
    #  SINGLE IMAGE EXPLANATION
    # ─────────────────────────────────────────────────────────────────────

    def explain_image(
        self,
        image_tensor:  torch.Tensor,
        class_index:   Optional[int] = None,
        sample_id:     int = 0,
        out_dir:       Path = Path("explanations"),
        return_pils:   bool = False,
    ) -> dict:
        """
        Explain a single image: Grad-CAM + SHAP + JSON summary.

        Parameters
        ----------
        image_tensor : (3, H, W) ImageNet-normalised tensor
        class_index  : target class (None → predicted)
        sample_id    : integer ID used for file naming
        out_dir      : root output directory
        return_pils  : if True, include PIL images in the returned dict

        Returns
        -------
        dict with keys:
            predicted_class, confidence, top_3_classes,
            gradcam_path, shap_path, explanation_path
        """
        out_dir = Path(out_dir)
        imgs    = image_tensor.unsqueeze(0).to(self.device)   # (1, 3, H, W)

        # ── Inference ─────────────────────────────────────────────────────
        with torch.no_grad():
            logits = self.model(imgs)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()[0]   # (C,)

        pred_cls = int(np.argmax(probs)) if class_index is None else class_index
        cls_code = HAM_CLASSES[pred_cls]

        top3_idx = np.argsort(probs)[::-1][:3]
        top3     = [
            {"class":      HAM_CLASSES[j],
             "full_name":  HAM_FULL_NAMES[HAM_CLASSES[j]],
             "confidence": round(float(probs[j]), 6)}
            for j in top3_idx
        ]
        confidence_all = {HAM_CLASSES[i]: round(float(probs[i]), 6)
                          for i in range(len(HAM_CLASSES))}

        summary: dict = {
            "sample_id":        sample_id,
            "predicted_class":  cls_code,
            "full_name":        HAM_FULL_NAMES[cls_code],
            "confidence":       round(float(probs[pred_cls]), 6),
            "confidence_all":   confidence_all,
            "top_3_classes":    top3,
        }

        # ── Grad-CAM ──────────────────────────────────────────────────────
        gc_dir = out_dir / "gradcam"
        gc_dir.mkdir(parents=True, exist_ok=True)

        cams, cls_list, logits_t = self.gradcam.generate(
            imgs, class_indices=[pred_cls], device=self.device
        )
        gc_composites = self.gradcam.overlay_batch(
            images_tensor=imgs.cpu(),
            cams=cams,
            class_indices=cls_list,
            logits=logits_t,
        )
        gc_path = gc_dir / f"gradcam_{sample_id:05d}.png"
        gc_composites[0].save(gc_path)
        summary["gradcam_path"] = str(gc_path)

        # ── SHAP ──────────────────────────────────────────────────────────
        shap_path_str = None
        if self.shap_exp is not None:
            shap_dir = out_dir / "shap"
            shap_dir.mkdir(parents=True, exist_ok=True)
            try:
                shap_maps, shap_summaries = self.shap_exp.explain(
                    imgs, class_indices=[pred_cls]
                )
                top3_regions = shap_summaries[0]["top_3_regions"]
                shap_pil = self.shap_exp.plot_shap_maps(
                    image_tensor=image_tensor,
                    shap_map=shap_maps[0],
                    summary=shap_summaries[0],
                    top3_regions=top3_regions,
                )
                shap_path = shap_dir / f"shap_{sample_id:05d}.png"
                shap_pil.save(shap_path)
                shap_path_str = str(shap_path)
                summary["top_3_regions"] = top3_regions
            except Exception as e:
                log.warning("SHAP failed for sample %d: %s", sample_id, e)

        summary["shap_path"] = shap_path_str

        # ── Overlay composite (Grad-CAM side-by-side with SHAP) ───────────
        combo_path = self._save_combined_composite(
            image_tensor   = image_tensor,
            cam_image      = gc_composites[0],
            shap_path      = Path(shap_path_str) if shap_path_str else None,
            summary        = summary,
            out_dir        = out_dir,
            sample_id      = sample_id,
        )
        summary["explanation_path"] = str(combo_path)

        # ── JSON ──────────────────────────────────────────────────────────
        json_dir = out_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / f"explanation_{sample_id:05d}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        summary["json_path"] = str(json_path)

        if return_pils:
            summary["_gradcam_pil"] = gc_composites[0]

        log.info(
            "[%d] %s (conf=%.3f)  → %s",
            sample_id, cls_code, summary["confidence"], combo_path,
        )
        return summary

    # ─────────────────────────────────────────────────────────────────────
    #  BATCH EXPLANATION
    # ─────────────────────────────────────────────────────────────────────

    def explain_batch(
        self,
        loader:        DataLoader,
        out_dir:       Path = Path("explanations"),
        n_samples:     int  = 50,
        class_filter:  Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Explain up to ``n_samples`` images from a DataLoader.

        Parameters
        ----------
        loader       : DataLoader yielding (imgs, labels, lesion_id, skin_tone)
        out_dir      : output root directory
        n_samples    : maximum number of images to explain
        class_filter : only explain images of these classes (None = all)

        Returns
        -------
        list of per-image summary dicts
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        all_summaries: list[dict] = []
        done = 0

        for batch in tqdm(loader, desc="Batch explanation"):
            if done >= n_samples:
                break
            images, labels = batch[0], batch[1]

            for b in range(len(images)):
                if done >= n_samples:
                    break
                label_idx = int(labels[b])
                cls_code  = HAM_CLASSES[label_idx]

                if class_filter and cls_code not in class_filter:
                    continue

                summary = self.explain_image(
                    image_tensor = images[b],
                    class_index  = None,    # use predicted
                    sample_id    = done,
                    out_dir      = out_dir,
                )
                summary["true_class"] = cls_code
                all_summaries.append(summary)
                done += 1

        # ── Batch JSON report ─────────────────────────────────────────────
        report_path = out_dir / "batch_report.json"
        with open(report_path, "w") as f:
            json.dump(
                {
                    "n_explained": done,
                    "summaries":   all_summaries,
                },
                f, indent=2, default=str,
            )
        log.info("Batch report → %s  (%d images)", report_path, done)
        self._print_batch_summary(all_summaries)
        return all_summaries

    # ─────────────────────────────────────────────────────────────────────
    #  AGGREGATE SHAP REPORT
    # ─────────────────────────────────────────────────────────────────────

    def aggregate_shap_report(
        self,
        loader:        DataLoader,
        out_dir:       Path = Path("explanations"),
        max_batches:   int  = 20,
    ) -> Optional[Path]:
        """
        Generate per-class aggregate SHAP importance map across the dataset.

        Calls SHAPExplainer.aggregate_importance() then saves the grid plot.

        Returns
        -------
        Path to the saved PNG report, or None if SHAP is disabled.
        """
        if self.shap_exp is None:
            log.warning("SHAP explainer not initialised — skipping aggregate report.")
            return None

        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("Generating aggregate SHAP importance (max %d batches)…", max_batches)

        mean_maps = self.shap_exp.aggregate_importance(
            loader=loader,
            max_batches=max_batches,
            device=self.device,
        )

        report_path = out_dir / "aggregate_shap_report.png"
        self.shap_exp.plot_aggregate_report(mean_maps, report_path)
        return report_path

    # ─────────────────────────────────────────────────────────────────────
    #  COMBINED COMPOSITE SAVER
    # ─────────────────────────────────────────────────────────────────────

    def _save_combined_composite(
        self,
        image_tensor: torch.Tensor,
        cam_image:    Image.Image,
        shap_path:    Optional[Path],
        summary:      dict,
        out_dir:      Path,
        sample_id:    int,
    ) -> Path:
        """
        Save a single-page composite:
            [Original] | [Grad-CAM] | [SHAP] (if available)
            + confidence table

        Returns saved path.
        """
        combo_dir = out_dir / "combined"
        combo_dir.mkdir(parents=True, exist_ok=True)

        img_np = denormalize(image_tensor)    # (H, W, 3) uint8
        n_panels = 3 if shap_path and shap_path.exists() else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 5.5, 5))
        fig.patch.set_facecolor(DARK_BG)
        if n_panels == 1:
            axes = [axes]

        # Panel 0: original
        axes[0].imshow(img_np)
        axes[0].set_title("Original", color=TEXT_COL, fontsize=10, fontweight="bold")
        axes[0].axis("off"); axes[0].set_facecolor(PANEL_BG)

        # Panel 1: Grad-CAM
        axes[1].imshow(cam_image)
        axes[1].set_title("Grad-CAM", color=TEXT_COL, fontsize=10, fontweight="bold")
        axes[1].axis("off"); axes[1].set_facecolor(PANEL_BG)

        # Panel 2: SHAP
        if n_panels == 3 and shap_path and shap_path.exists():
            axes[2].imshow(Image.open(shap_path))
            axes[2].set_title("SHAP", color=TEXT_COL, fontsize=10, fontweight="bold")
            axes[2].axis("off"); axes[2].set_facecolor(PANEL_BG)

        # Title with prediction info
        cls_code   = summary["predicted_class"]
        confidence = summary["confidence"]
        top3_str   = "  |  ".join(
            f"{t['class']}={t['confidence']:.3f}"
            for t in summary.get("top_3_classes", [])
        )
        fig.suptitle(
            f"Predicted: {HAM_FULL_NAMES[cls_code]} ({cls_code})  "
            f"conf={confidence:.3f}\n"
            f"Top-3: {top3_str}",
            color=TEXT_COL, fontsize=10, fontweight="bold", y=1.03,
        )
        fig.tight_layout(pad=0.5)

        out_path = combo_dir / f"explanation_{sample_id:05d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        return out_path

    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _print_batch_summary(summaries: list[dict]):
        """Print a quick accuracy and class distribution to console."""
        if not summaries:
            return
        total = len(summaries)
        correct = sum(
            1 for s in summaries
            if s.get("true_class") == s["predicted_class"]
        )
        print(f"\n{'═'*50}")
        print(f"  EXPLANATION BATCH SUMMARY  ({total} images)")
        print(f"{'═'*50}")
        acc_str = f"{100*correct/total:.1f}%" if "true_class" in summaries[0] else "N/A"
        print(f"  Accuracy:   {acc_str}")
        from collections import Counter
        pred_counts = Counter(s["predicted_class"] for s in summaries)
        print(f"  Predicted distribution:")
        for cls, cnt in sorted(pred_counts.items(), key=lambda x: -x[1]):
            bar = "█" * int(cnt / total * 30)
            print(f"    {cls:<8}  {bar:<30}  {cnt:>3} ({100*cnt/total:.0f}%)")
        print(f"{'═'*50}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  SINGLE IMAGE FROM DISK
# ═══════════════════════════════════════════════════════════════════════════

def explain_from_path(
    image_path:  Union[str, Path],
    checkpoint:  Union[str, Path],
    out_dir:     Union[str, Path] = "explanations",
    device:      str = "cpu",
    image_size:  int = NATIVE_IMAGE_SIZE,
    use_shap:    bool = False,
) -> dict:
    """
    Convenience wrapper: load a JPEG/PNG image from disk and explain it.

    Parameters
    ----------
    image_path  : path to input image
    checkpoint  : path to best_model.pth
    out_dir     : output root directory
    device      : "cuda" or "cpu"
    use_shap    : whether to run SHAP (requires a DataLoader for background)
    """
    import torchvision.transforms as T
    dev   = torch.device(device if torch.cuda.is_available()
                         or device == "cpu" else "cpu")
    model, _ = load_classifier(checkpoint, device=dev)

    tfm = T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = tfm(img)   # (3, H, W)

    explainer = SkinLesionExplainer(
        model=model, device=dev,
        shap_loader=None,     # no SHAP for single-image path mode
    )
    return explainer.explain_image(
        image_tensor=tensor,
        out_dir=Path(out_dir),
        sample_id=0,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Skin lesion explainability — Grad-CAM + SHAP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True,
                   help="Path to best_model.pth")
    p.add_argument("--output_dir",   default="explanations")
    p.add_argument("--device",       default="cpu", choices=["cpu", "cuda"])

    # Single image mode
    p.add_argument("--image",        default=None,
                   help="Path to a single image file")

    # Batch mode
    p.add_argument("--test_csv",     default=None)
    p.add_argument("--image_dir",    default=None)
    p.add_argument("--n_samples",    type=int, default=50)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--num_workers",  type=int, default=2)
    p.add_argument("--image_size",   type=int, default=NATIVE_IMAGE_SIZE)
    p.add_argument("--class_filter", nargs="*", default=None,
                   help="Only explain these class codes, e.g. mel bcc vasc")

    # Aggregate SHAP report
    p.add_argument("--aggregate",    action="store_true",
                   help="Generate aggregate per-class SHAP importance map")
    p.add_argument("--n_background", type=int, default=100)
    p.add_argument("--max_batches",  type=int, default=20)

    # Grad-CAM options
    p.add_argument("--no_gradcam_lib", action="store_true",
                   help="Force manual Grad-CAM hooks (skip pytorch-grad-cam)")
    return p.parse_args()


def main():
    args    = _parse_args()
    out_dir = Path(args.output_dir)
    dev     = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    # ── Load model ───────────────────────────────────────────────────────
    model, _ = load_classifier(args.checkpoint, device=dev)

    # ── Single image mode ─────────────────────────────────────────────────
    if args.image:
        result = explain_from_path(
            image_path = args.image,
            checkpoint = args.checkpoint,
            out_dir    = out_dir,
            device     = str(dev),
            image_size = args.image_size,
        )
        print(json.dumps(result, indent=2, default=str))
        return

    # ── Batch mode ────────────────────────────────────────────────────────
    if not args.test_csv:
        raise SystemExit("Provide --image for single-image mode or "
                         "--test_csv for batch mode.")

    from dataset import HAMDataset
    ds = HAMDataset(args.image_dir, args.test_csv,
                    split="test", image_size=args.image_size)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(dev.type == "cuda"),
    )

    # Background loader for SHAP — use same dataset
    bg_loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )

    explainer = SkinLesionExplainer(
        model           = model,
        device          = dev,
        shap_loader     = bg_loader if not args.aggregate else bg_loader,
        n_background    = args.n_background,
        use_gradcam_lib = not args.no_gradcam_lib,
    )

    # Explain individual samples
    summaries = explainer.explain_batch(
        loader        = loader,
        out_dir       = out_dir,
        n_samples     = args.n_samples,
        class_filter  = args.class_filter,
    )

    # Aggregate SHAP report
    if args.aggregate:
        agg_path = explainer.aggregate_shap_report(
            loader       = bg_loader,
            out_dir      = out_dir,
            max_batches  = args.max_batches,
        )
        if agg_path:
            log.info("Aggregate SHAP report → %s", agg_path)

    log.info("Done.  %d explanations in %s", len(summaries), out_dir)


if __name__ == "__main__":
    main()
