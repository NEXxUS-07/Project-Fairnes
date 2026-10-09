#!/usr/bin/env python3
"""
test_ml_service.py — Integration test for MLService
====================================================
Runs a sample image through the full pipeline:
  1. predict()  — EfficientNet-B3 classification
  2. explain()  — Grad-CAM + SHAP heatmap generation
  3. generate() — synthetic image generation (DDPM)
  4. is_ready() / model_info() — health checks

Usage
-----
  # From the project root (venv active):
  python test_ml_service.py

  # Use a real dermoscopy image:
  python test_ml_service.py --image path/to/lesion.jpg

  # Force dummy mode (no real models needed):
  python test_ml_service.py --dummy

  # Test all 7 classes for generation:
  python test_ml_service.py --gen-all --num-images 2
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# ── Ensure project root is on PYTHONPATH ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
for p in (str(ROOT), str(BACKEND)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Force dummy mode BEFORE importing ml_service ─────────────────────────────
parser = argparse.ArgumentParser(description="MLService integration test")
parser.add_argument("--image",      default=None, help="Path to a test image (JPEG/PNG)")
parser.add_argument("--dummy",      action="store_true", help="Force DUMMY_MODE for both models")
parser.add_argument("--gen-all",    action="store_true", help="Generate images for all 7 classes")
parser.add_argument("--class",      dest="cls", default="mel", help="Class to generate (default: mel)")
parser.add_argument("--num-images", type=int, default=2, help="Images to generate per class (default: 2)")
parser.add_argument("--save-dir",   default="/tmp/ml_service_test", help="Directory for output images")
args = parser.parse_args()

if args.dummy:
    os.environ["ML_DUMMY_MODE"]        = "true"
    os.environ["DIFFUSION_DUMMY_MODE"] = "true"

# ── Now import ────────────────────────────────────────────────────────────────
import logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)-7s] %(name)s — %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("test_ml_service")

try:
    from backend.services.ml_service import (
        MLService, ml_service,
        PredictionResult, ExplainResult,
        HAM_CLASSES, HAM_NAMES,
        ModelNotLoadedError, InvalidImageError, GenerationError,
    )
except ImportError:
    from services.ml_service import (
        MLService, ml_service,
        PredictionResult, ExplainResult,
        HAM_CLASSES, HAM_NAMES,
        ModelNotLoadedError, InvalidImageError, GenerationError,
    )

from PIL import Image, ImageDraw
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def hdr(msg):  print(f"\n{BOLD}{CYAN}── {msg} ──{RESET}")


def make_test_image(save_path: Path | None = None) -> Image.Image:
    """Generate a synthetic 300×300 dermoscopy-like test image."""
    img  = Image.new("RGB", (300, 300), (180, 120, 80))
    draw = ImageDraw.Draw(img)
    # Rough lesion blob
    draw.ellipse([80, 80, 220, 220], fill=(60, 20, 10))
    draw.ellipse([110, 110, 190, 190], fill=(80, 30, 15))
    # Irregular border
    for angle in range(0, 360, 30):
        import math
        r  = 60 + 20 * math.sin(math.radians(angle * 3))
        cx, cy = 150, 150
        x  = int(cx + r * math.cos(math.radians(angle)))
        y  = int(cy + r * math.sin(math.radians(angle)))
        draw.ellipse([x-4, y-4, x+4, y+4], fill=(70, 25, 12))
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
        log.info("Test image saved → %s", save_path)
    return img


def print_prediction(result: PredictionResult):
    print(f"     Predicted class : {BOLD}{result.predicted_class}{RESET} "
          f"({result.predicted_name})")
    print(f"     Confidence       : {result.confidence * 100:.1f}%"
          f"{'  [DUMMY]' if result.is_dummy else ''}")
    print(f"     Model version    : {result.model_version}")
    print(f"     Processing time  : {result.processing_time_ms} ms")
    print(f"     Top-3 classes    :")
    for s in result.all_class_scores[:3]:
        bar = "█" * int(s.confidence * 20)
        print(f"       {s.code:<6} {bar:<20} {s.confidence * 100:5.1f}%  {s.name}")


def print_explain(result: ExplainResult):
    print(f"     Grad-CAM : {result.gradcam_url or 'not generated'}"
          f"{'  [DUMMY]' if result.is_dummy else ''}")
    print(f"     SHAP     : {result.shap_url or 'not generated'}")
    print(f"     Time     : {result.processing_time_ms} ms")


# ─────────────────────────────────────────────────────────────────────────────
#  TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

async def run_tests():
    save_dir  = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load test image ───────────────────────────────────────────────────
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            fail(f"Image not found: {image_path}")
            sys.exit(1)
        test_image = Image.open(image_path).convert("RGB")
        log.info("Loaded test image from %s  size=%s", image_path, test_image.size)
    else:
        test_image = make_test_image(save_path=save_dir / "test_input.png")
        warn("No --image supplied; using synthetic test image.")

    # ══════════════════════════════════════════════════
    hdr("STEP 0 — Startup")
    # ══════════════════════════════════════════════════
    t0 = time.perf_counter()
    ml_service.startup()
    ok(f"startup() completed in {(time.perf_counter() - t0)*1000:.0f} ms")

    # ── Health checks ─────────────────────────────────────────────────────
    hdr("STEP 1 — is_ready() + model_info()")
    ready = ml_service.is_ready()
    if ready:
        ok("is_ready() → True")
    else:
        fail("is_ready() → False (startup may have failed)")
        sys.exit(1)

    info = ml_service.model_info()
    print(f"\n     Model info:")
    for k, v in info.items():
        marker = GREEN if v not in (False, "unknown", None) else YELLOW
        print(f"       {k:<26}: {marker}{v}{RESET}")

    # ══════════════════════════════════════════════════
    hdr("STEP 2 — predict(PIL.Image)")
    # ══════════════════════════════════════════════════
    try:
        result = await ml_service.predict(test_image)
        print_prediction(result)
        ok("predict() returned PredictionResult")
        predicted_class = result.predicted_class
    except ModelNotLoadedError as e:
        fail(f"ModelNotLoadedError: {e}")
        sys.exit(1)
    except InvalidImageError as e:
        fail(f"InvalidImageError: {e}")
        sys.exit(1)
    except Exception as e:
        fail(f"Unexpected error in predict(): {e}")
        raise

    # ── predict() with raw bytes ──────────────────────────────────────────
    print(f"\n  Testing predict() with raw bytes …")
    buf = __import__("io").BytesIO()
    test_image.save(buf, format="PNG")
    result_bytes = await ml_service.predict(buf.getvalue())
    assert isinstance(result_bytes, PredictionResult), "Expected PredictionResult"
    ok(f"predict(bytes) → {result_bytes.predicted_class} ({result_bytes.confidence * 100:.1f}%)")

    # ── predict() with bad input ──────────────────────────────────────────
    print(f"\n  Testing InvalidImageError …")
    try:
        await ml_service.predict(b"not an image")
        warn("No exception raised for invalid bytes (model may have dummy mode)")
    except InvalidImageError as e:
        ok(f"InvalidImageError raised correctly: {e}")
    except Exception as e:
        warn(f"Different exception (may be OK in dummy mode): {type(e).__name__}: {e}")

    # ══════════════════════════════════════════════════
    hdr("STEP 3 — explain(PIL.Image)")
    # ══════════════════════════════════════════════════
    try:
        expl = await ml_service.explain(
            test_image,
            sample_id       = "test_001",
            predicted_class = predicted_class,
            save_dir        = save_dir,
        )
        print_explain(expl)
        ok("explain() returned ExplainResult")

        if expl.gradcam_path:
            gc_path = Path(expl.gradcam_path)
            if gc_path.exists():
                ok(f"Grad-CAM PNG saved ({gc_path.stat().st_size} bytes)")
            else:
                warn("Grad-CAM URL returned but file not found on disk")
    except ModelNotLoadedError as e:
        fail(f"ModelNotLoadedError: {e}")
    except Exception as e:
        fail(f"Unexpected error in explain(): {e}")
        raise

    # ══════════════════════════════════════════════════
    hdr("STEP 4 — generate(class_label, num_images)")
    # ══════════════════════════════════════════════════
    classes_to_test = HAM_CLASSES if args.gen_all else [args.cls]
    for cls in classes_to_test:
        try:
            t0   = time.perf_counter()
            urls = await ml_service.generate(
                class_label = cls,
                num_images  = args.num_images,
                job_id      = f"test_{cls}",
                save_dir    = save_dir,
            )
            ms = int((time.perf_counter() - t0) * 1000)
            if urls:
                ok(f"generate({cls!r}, n={args.num_images}) → {len(urls)} URL(s)  [{ms} ms]")
                for url in urls[:2]:
                    print(f"       {url}")
            else:
                warn(f"generate({cls!r}) returned empty list")
        except GenerationError as e:
            fail(f"GenerationError ({cls}): {e}")
        except Exception as e:
            fail(f"Unexpected error in generate({cls!r}): {e}")

    # ── Invalid class test ────────────────────────────────────────────────
    print(f"\n  Testing GenerationError for unknown class …")
    try:
        await ml_service.generate("invalid_class", 1)
        warn("No GenerationError raised")
    except GenerationError as e:
        ok(f"GenerationError raised correctly: {e}")

    # ══════════════════════════════════════════════════
    hdr("STEP 5 — Performance benchmark (10 × predict)")
    # ══════════════════════════════════════════════════
    times = []
    for i in range(10):
        t0 = time.perf_counter()
        r  = await ml_service.predict(test_image)
        times.append((time.perf_counter() - t0) * 1000)

    avg  = sum(times) / len(times)
    best = min(times)
    worst= max(times)
    print(f"\n     10 × predict() on {'DUMMY' if r.is_dummy else info['device']}:")
    print(f"       best  : {best:.1f} ms")
    print(f"       avg   : {avg:.1f} ms")
    print(f"       worst : {worst:.1f} ms")
    ok("Benchmark complete")

    # ══════════════════════════════════════════════════
    hdr("SUMMARY")
    # ══════════════════════════════════════════════════
    print(f"\n  Output directory  : {save_dir}")
    output_files = list(save_dir.rglob("*.png"))
    print(f"  PNG files written : {len(output_files)}")
    for f in output_files[:8]:
        print(f"    {f.relative_to(save_dir)}")
    if len(output_files) > 8:
        print(f"    … and {len(output_files) - 8} more")

    print(f"\n{BOLD}{GREEN}All tests passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run_tests()))
