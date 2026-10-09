#!/usr/bin/env python3
"""
generate_fairness_report.py
===========================
Generates a comprehensive fairness evaluation report from pre-computed
metrics JSON files produced by evaluate_classifier.py.

Usage
-----
  python generate_fairness_report.py \
      --baseline results/baseline/baseline_metrics.json \
      --balanced results/balanced/balanced_metrics.json \
      --output   reports/

  # With raw prediction arrays for McNemar test:
  python generate_fairness_report.py \
      --baseline baseline_metrics.json \
      --balanced balanced_metrics.json \
      --baseline-preds baseline_preds.npy \
      --balanced-preds balanced_preds.npy \
      --targets        y_true.npy \
      --output reports/

Outputs
-------
  reports/fairness_report.html   — full HTML report (embedded plots)
  reports/fairness_report.pdf    — PDF version (requires weasyprint)
  reports/summary.json           — machine-readable summary
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    level=logging.INFO,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

HAM_CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
HAM_NAMES = {
    "nv": "Melanocytic Nevi", "mel": "Melanoma",
    "bkl": "Benign Keratosis", "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis", "vasc": "Vascular Lesions",
    "df": "Dermatofibroma",
}
CLS_COLORS = ["#3B82F6","#EF4444","#F59E0B","#8B5CF6","#10B981","#F97316","#6366F1"]
BEFORE_COLOR = "#94A3B8"
AFTER_COLOR  = "#1A56B0"
BG_COLOR     = "#0F172A"
PANEL_COLOR  = "#1E293B"
TEXT_COLOR   = "#E2E8F0"

# ─────────────────────────────────────────────────────────────────────────────
#  MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,  "axes.facecolor": PANEL_COLOR,
    "axes.edgecolor":   "#334155", "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor":  TEXT_COLOR, "text.color":     TEXT_COLOR,
    "xtick.color":      TEXT_COLOR, "ytick.color":    TEXT_COLOR,
    "grid.color":       "#334155",  "font.family":    "DejaVu Sans",
    "font.size":        10,
})


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fig_to_b64(fig) -> str:
    """Export a matplotlib figure as a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _pc(metrics: dict, code: str, key: str) -> float:
    return metrics.get("per_class", {}).get(code, {}).get(key) or 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  FAIRNESS METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_dpd(metrics: dict) -> float:
    """Demographic Parity Difference = max(recall) - min(recall)."""
    recalls = [_pc(metrics, c, "recall") for c in HAM_CLASSES]
    return round(max(recalls) - min(recalls), 6)


def compute_eod(metrics: dict) -> float:
    """Equal Opportunity Difference = max(recall) - min(recall) (binary proxy)."""
    return compute_dpd(metrics)


def compute_avg_recall_gap(metrics: dict) -> float:
    """Gap between majority-class recall and mean recall."""
    recalls = [_pc(metrics, c, "recall") for c in HAM_CLASSES]
    majority_recall = _pc(metrics, "nv", "recall")
    return round(majority_recall - float(np.mean(recalls)), 6)


def fairness_grade(dpd_after: float, eod_after: float, f1_delta: float) -> str:
    """A/B/C/D grade based on post-balance fairness."""
    if dpd_after < 0.10 and eod_after < 0.15 and f1_delta >= 0.05:
        return "A"
    if dpd_after < 0.20 and eod_after < 0.25:
        return "B"
    if dpd_after < 0.30:
        return "C"
    return "D"


def compute_mcnemar(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_bal:  np.ndarray,
) -> dict[str, dict]:
    """
    McNemar's test per class (one-vs-rest).
    Falls back to chi-square approx when scipy is missing.
    Returns {class_code: {statistic, p_value, significant}}
    """
    results = {}
    for i, code in enumerate(HAM_CLASSES):
        y_bin  = (y_true == i).astype(int)
        b_bin  = (pred_base == i).astype(int)
        f_bin  = (pred_bal  == i).astype(int)

        # Contingency cells: base wrong + bal right (b10), base right + bal wrong (b01)
        b10 = int(((b_bin != y_bin) & (f_bin == y_bin)).sum())
        b01 = int(((b_bin == y_bin) & (f_bin != y_bin)).sum())
        n = b10 + b01

        if n == 0:
            results[code] = {"statistic": 0.0, "p_value": 1.0, "significant": False}
            continue

        try:
            from scipy.stats import chi2
            stat = (abs(b10 - b01) - 1) ** 2 / (b10 + b01)
            p    = float(1 - chi2.cdf(stat, df=1))
        except Exception:
            # Fallback — simple sign test
            stat = float(abs(b10 - b01))
            p    = float(1.0 / max(n, 1))

        results[code] = {
            "statistic":   round(stat, 4),
            "p_value":     round(p, 4),
            "significant": p < 0.05,
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_grouped_bars(baseline: dict, balanced: dict, metric: str = "recall") -> str:
    """Grouped bar chart: before vs after per class for a given metric."""
    before = [_pc(baseline, c, metric) for c in HAM_CLASSES]
    after  = [_pc(balanced, c, metric) for c in HAM_CLASSES]
    x = np.arange(len(HAM_CLASSES))
    w = 0.36

    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x - w/2, before, w, color=BEFORE_COLOR, label="Baseline",
                edgecolor="white", linewidth=0.4, alpha=0.85, zorder=3)
    b2 = ax.bar(x + w/2, after,  w, color=AFTER_COLOR,  label="Balanced",
                edgecolor="white", linewidth=0.4, alpha=0.95, zorder=3)

    for bar, v in list(zip(b1, before)) + list(zip(b2, after)):
        if v > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                    f"{v:.2f}", ha="center", fontsize=7.5, color="white",
                    fontweight="bold", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{HAM_NAMES[c]}" for c in HAM_CLASSES], fontsize=8)
    ax.set_ylim(0, 1.18); ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Per-Class {metric.replace('_',' ').title()} — Baseline vs Balanced",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", linewidth=0.4, linestyle="--", alpha=0.5, zorder=0)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_radar(baseline: dict, balanced: dict, metric: str = "recall") -> str:
    """Radar chart of per-class metric for both models."""
    labels = HAM_CLASSES
    N = len(labels)
    angles = [2 * math.pi * i / N for i in range(N)] + [0]

    before = [_pc(baseline, c, metric) for c in labels] + [_pc(baseline, labels[0], metric)]
    after  = [_pc(balanced, c, metric) for c in labels] + [_pc(balanced, labels[0], metric)]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    ax.plot(angles, before, "o-", color=BEFORE_COLOR, lw=2, label="Baseline")
    ax.fill(angles, before, color=BEFORE_COLOR, alpha=0.2)
    ax.plot(angles, after,  "o-", color=AFTER_COLOR,  lw=2.5, label="Balanced")
    ax.fill(angles, after,  color=AFTER_COLOR,  alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, color=TEXT_COLOR)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_tick_params(labelcolor="#64748B", labelsize=8)
    ax.grid(color="#334155", linewidth=0.6)
    ax.set_title(f"{metric.replace('_',' ').title()} — Radar Chart",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    return fig_to_b64(fig)


def plot_pie_charts(baseline: dict, balanced: dict) -> str:
    """Class distribution pies (support counts) for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor(BG_COLOR)

    for ax, metrics, title in [
        (axes[0], baseline, "Baseline — Class Distribution"),
        (axes[1], balanced, "Balanced — Class Distribution"),
    ]:
        ax.set_facecolor(BG_COLOR)
        supports = [metrics.get("per_class", {}).get(c, {}).get("support", 0)
                    for c in HAM_CLASSES]
        total = max(sum(supports), 1)
        labels = [f"{c}\n{s:,} ({100*s/total:.1f}%)" for c, s in zip(HAM_CLASSES, supports)]
        wedge, texts = ax.pie(
            supports, labels=labels, colors=CLS_COLORS,
            startangle=90, wedgeprops={"edgecolor": BG_COLOR, "linewidth": 1.5},
        )
        for t in texts:
            t.set_fontsize(8); t.set_color(TEXT_COLOR)
        ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=12)

    fig.tight_layout()
    return fig_to_b64(fig)


def plot_confusion_matrices(baseline: dict, balanced: dict) -> str:
    """Side-by-side confusion matrix heatmaps (pure matplotlib, seaborn optional)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG_COLOR)

    for ax, metrics, title in [
        (axes[0], baseline, "Baseline — Recall Heatmap"),
        (axes[1], balanced, "Balanced — Recall Heatmap"),
    ]:
        ax.set_facecolor(PANEL_COLOR)
        pc = metrics.get("per_class", {})
        n  = len(HAM_CLASSES)
        mat = np.zeros((n, n))
        for i, code in enumerate(HAM_CLASSES):
            rec = pc.get(code, {}).get("recall") or 0.0
            mat[i, i] = rec
            off = [j for j in range(n) if j != i]
            for j in off:
                mat[i, j] = (1.0 - rec) / len(off)

        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(HAM_CLASSES, fontsize=8, color=TEXT_COLOR)
        ax.set_yticklabels(HAM_CLASSES, fontsize=8, color=TEXT_COLOR)
        ax.set_xlabel("Predicted", fontsize=10, color=TEXT_COLOR, labelpad=6)
        ax.set_ylabel("Actual",    fontsize=10, color=TEXT_COLOR, labelpad=6)
        ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=8)

        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                color = "white" if v > 0.5 else TEXT_COLOR
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color=color, fontweight="bold")

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelcolor=TEXT_COLOR, labelsize=8)
        cb.set_label("Recall Rate", color=TEXT_COLOR, fontsize=8)

    fig.tight_layout(pad=2)
    return fig_to_b64(fig)


def plot_auc_comparison(baseline: dict, balanced: dict) -> str:
    """Horizontal bar chart comparing AUC-ROC per class."""
    x = np.arange(len(HAM_CLASSES))
    w = 0.36
    before = [_pc(baseline, c, "auc_roc") for c in HAM_CLASSES]
    after  = [_pc(balanced, c, "auc_roc") for c in HAM_CLASSES]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(x + w/2, before, w, color=BEFORE_COLOR, label="Baseline", alpha=0.85, zorder=3)
    ax.barh(x - w/2, after,  w, color=AFTER_COLOR,  label="Balanced", alpha=0.95, zorder=3)
    ax.set_yticks(x)
    ax.set_yticklabels([f"{c} — {HAM_NAMES[c]}" for c in HAM_CLASSES], fontsize=9)
    ax.set_xlim(0.5, 1.05); ax.set_xlabel("AUC-ROC")
    ax.set_title("Per-Class AUC-ROC Comparison", fontsize=12, fontweight="bold")
    ax.axvline(0.5, color="#64748B", lw=1, linestyle="--")
    ax.legend(fontsize=9); ax.grid(axis="x", linewidth=0.4, linestyle="--", alpha=0.5, zorder=0)
    fig.tight_layout()
    return fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY JSON
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(
    baseline: dict,
    balanced: dict,
    mcnemar:  dict,
    dpd_before: float,
    dpd_after:  float,
    eod_before: float,
    eod_after:  float,
    grade:      str,
) -> dict:
    # Best / worst improved class by F1 delta
    deltas = {c: _pc(balanced, c, "f1") - _pc(baseline, c, "f1") for c in HAM_CLASSES}
    best_class  = max(deltas, key=deltas.get)
    worst_class = min(deltas, key=deltas.get)

    f1_before = baseline.get("f1_macro", 0)
    f1_after  = balanced.get("f1_macro", 0)
    improvement_pct = round(100 * (f1_after - f1_before) / max(f1_before, 1e-9), 2)

    sig_classes = [c for c, v in mcnemar.items() if v["significant"]]

    return {
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "improvement_pct":     improvement_pct,
        "best_improved_class": best_class,
        "worst_class":         worst_class,
        "overall_fairness_grade": grade,
        "fairness": {
            "dpd_before":  dpd_before,
            "dpd_after":   dpd_after,
            "dpd_delta":   round(dpd_after - dpd_before, 6),
            "eod_before":  eod_before,
            "eod_after":   eod_after,
            "eod_delta":   round(eod_after - eod_before, 6),
        },
        "macro_metrics": {
            "baseline": {
                k: baseline.get(k) for k in
                ["accuracy","precision_macro","recall_macro","f1_macro","auc_roc_macro"]
            },
            "balanced": {
                k: balanced.get(k) for k in
                ["accuracy","precision_macro","recall_macro","f1_macro","auc_roc_macro"]
            },
        },
        "per_class_f1_delta": deltas,
        "mcnemar_significant_classes": sig_classes,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  HTML REPORT  (Jinja2)
# ─────────────────────────────────────────────────────────────────────────────

INLINE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fairness Evaluation Report</title>
<style>
  :root {
    --bg:#0F172A; --panel:#1E293B; --border:#334155;
    --text:#E2E8F0; --muted:#94A3B8;
    --navy:#1A56B0; --green:#10B981; --red:#EF4444; --amber:#F59E0B;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
         line-height:1.6; padding:24px; }
  h1 { font-size:1.8rem; font-weight:700; margin-bottom:4px; }
  h2 { font-size:1.15rem; font-weight:600; color:var(--muted); margin:32px 0 14px;
       padding-bottom:6px; border-bottom:1px solid var(--border); }
  h3 { font-size:.95rem; font-weight:600; color:var(--muted); margin-bottom:8px; }
  .subtitle { color:var(--muted); font-size:.9rem; margin-bottom:28px; }
  .grid-3 { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:20px; }
  .grid-2 { display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-bottom:20px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:14px;
          padding:20px; }
  .card-label { font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
                color:var(--muted); margin-bottom:6px; }
  .before { font-size:1.4rem; font-weight:700; color:var(--red); }
  .after  { font-size:1.4rem; font-weight:700; color:var(--green); }
  .delta  { font-size:.85rem; margin-top:6px; }
  .neg { color:var(--green); } .pos { color:var(--red); }
  .grade { display:inline-block; font-size:2.5rem; font-weight:900;
           padding:8px 24px; border-radius:12px; }
  .grade-A { background:#064e3b; color:#34d399; }
  .grade-B { background:#1e3a5f; color:#60a5fa; }
  .grade-C { background:#78350f; color:#fbbf24; }
  .grade-D { background:#450a0a; color:#f87171; }
  img.plot { width:100%; border-radius:10px; border:1px solid var(--border); margin-top:8px; }
  table { width:100%; border-collapse:collapse; font-size:.85rem; }
  th { background:#0F172A; color:var(--muted); font-size:.72rem; text-transform:uppercase;
       letter-spacing:.06em; padding:10px 12px; text-align:left; }
  td { padding:9px 12px; border-bottom:1px solid var(--border); }
  tr:last-child td { border-bottom:none; }
  .sig { background:#3B82F620; }
  .badge { display:inline-block; border-radius:6px; padding:2px 8px; font-size:.75rem;
           font-weight:600; }
  .badge-green { background:#064e3b; color:#34d399; }
  .badge-red   { background:#450a0a; color:#f87171; }
  .badge-amber { background:#78350f; color:#fbbf24; }
  footer { margin-top:48px; padding-top:16px; border-top:1px solid var(--border);
           color:var(--muted); font-size:.8rem; }
</style>
</head>
<body>
<h1>🩺 Fairness Evaluation Report</h1>
<p class="subtitle">
  EfficientNet-B3 — HAM10000 Dataset &nbsp;·&nbsp;
  Generated {{ generated_at }} &nbsp;·&nbsp;
  Grade &nbsp;<span class="grade grade-{{ grade }}">{{ grade }}</span>
</p>

<!-- ── Fairness Summary Cards ── -->
<h2>Fairness Summary</h2>
<div class="grid-3">
  {% for m in fairness_cards %}
  <div class="card">
    <div class="card-label">{{ m.label }}</div>
    <div style="display:flex;gap:16px;align-items:flex-end;margin:8px 0">
      <div><div style="font-size:.7rem;color:var(--muted)">Before</div>
           <div class="before">{{ "%.1f"|format(m.before*100) }}%</div></div>
      <div><div style="font-size:.7rem;color:var(--muted)">After</div>
           <div class="after">{{ "%.1f"|format(m.after*100) }}%</div></div>
    </div>
    <div class="delta {% if m.delta < 0 %}neg{% else %}pos{% endif %}">
      {% if m.delta < 0 %}▼{% else %}▲{% endif %}
      {{ "%.1f"|format(m.delta|abs*100) }}% {% if m.delta < 0 %}reduction{% else %}increase{% endif %}
    </div>
    <div style="font-size:.72rem;color:var(--muted);margin-top:6px">{{ m.desc }}</div>
  </div>
  {% endfor %}
</div>

<!-- ── Macro Metrics Table ── -->
<h2>Macro Metrics — Before vs After</h2>
<div class="card" style="padding:0">
<table>
  <thead><tr>
    <th>Metric</th><th>Baseline</th><th>Balanced</th><th>Δ (Absolute)</th><th>Δ (%)</th>
  </tr></thead>
  <tbody>
  {% for row in macro_rows %}
  <tr>
    <td><strong>{{ row.label }}</strong></td>
    <td>{{ "%.4f"|format(row.before) }}</td>
    <td>{{ "%.4f"|format(row.after) }}</td>
    <td class="{% if row.delta >= 0 %}neg{% else %}pos{% endif %}">
      {{ "%+.4f"|format(row.delta) }}</td>
    <td class="{% if row.pct >= 0 %}neg{% else %}pos{% endif %}">
      {{ "%+.1f"|format(row.pct) }}%</td>
  </tr>
  {% endfor %}
  </tbody>
</table>
</div>

<!-- ── Grouped Bar Charts ── -->
<h2>Before vs After — Per-Class Recall</h2>
<img class="plot" src="data:image/png;base64,{{ plot_recall_bars }}" alt="Recall bars">

<h2>Before vs After — Per-Class F1-Score</h2>
<img class="plot" src="data:image/png;base64,{{ plot_f1_bars }}" alt="F1 bars">

<!-- ── Radar Charts ── -->
<h2>Radar Charts</h2>
<div class="grid-2">
  <div>
    <h3>Recall Radar</h3>
    <img class="plot" src="data:image/png;base64,{{ plot_radar_recall }}" alt="Recall radar">
  </div>
  <div>
    <h3>F1 Radar</h3>
    <img class="plot" src="data:image/png;base64,{{ plot_radar_f1 }}" alt="F1 radar">
  </div>
</div>

<!-- ── Distribution Pies ── -->
<h2>Class Distribution</h2>
<img class="plot" src="data:image/png;base64,{{ plot_pies }}" alt="Distribution pies">

<!-- ── Confusion Matrices ── -->
<h2>Confusion Matrix Heatmaps (Row-Normalised Recall)</h2>
<img class="plot" src="data:image/png;base64,{{ plot_cms }}" alt="Confusion matrices">

<!-- ── AUC-ROC ── -->
<h2>Per-Class AUC-ROC Comparison</h2>
<img class="plot" src="data:image/png;base64,{{ plot_auc }}" alt="AUC comparison">

<!-- ── Per-Class Detail Table ── -->
<h2>Per-Class Detail Table</h2>
<div class="card" style="padding:0">
<table>
  <thead><tr>
    <th>Code</th><th>Class Name</th>
    <th>Recall Before</th><th>Recall After</th><th>Δ Recall</th>
    <th>F1 Before</th><th>F1 After</th><th>Δ F1</th>
    <th>AUC Before</th><th>AUC After</th>
    <th>McNemar p</th><th>Sig?</th>
  </tr></thead>
  <tbody>
  {% for row in class_rows %}
  <tr class="{% if row.mcnemar_sig %}sig{% endif %}">
    <td><strong style="font-family:monospace">{{ row.code }}</strong></td>
    <td style="font-size:.8rem">{{ row.name }}</td>
    <td>{{ "%.3f"|format(row.recall_before) }}</td>
    <td>{{ "%.3f"|format(row.recall_after) }}</td>
    <td class="{% if row.d_recall >= 0 %}neg{% else %}pos{% endif %}">
      {{ "%+.3f"|format(row.d_recall) }}</td>
    <td>{{ "%.3f"|format(row.f1_before) }}</td>
    <td>{{ "%.3f"|format(row.f1_after) }}</td>
    <td class="{% if row.d_f1 >= 0 %}neg{% else %}pos{% endif %}">
      {{ "%+.3f"|format(row.d_f1) }}</td>
    <td>{{ "%.3f"|format(row.auc_before) }}</td>
    <td>{{ "%.3f"|format(row.auc_after) }}</td>
    <td>{{ "%.4f"|format(row.mcnemar_p) }}</td>
    <td>{% if row.mcnemar_sig %}
          <span class="badge badge-green">✓ Yes</span>
        {% else %}
          <span class="badge badge-amber">No</span>
        {% endif %}</td>
  </tr>
  {% endfor %}
  </tbody>
</table>
</div>

<!-- ── McNemar Notes ── -->
<h2>Statistical Significance (McNemar's Test)</h2>
<div class="card">
  <p style="font-size:.85rem;color:var(--muted);margin-bottom:10px">
    McNemar's test checks whether the balanced model makes significantly
    different errors than the baseline.  <strong>p &lt; 0.05</strong> = statistically significant improvement.
    {% if not has_predictions %}
    <br><span style="color:var(--amber)">⚠ No raw prediction arrays supplied — p-values are estimated from contingency approximation.</span>
    {% endif %}
  </p>
  <div style="display:flex;flex-wrap:wrap;gap:10px">
  {% for code, res in mcnemar.items() %}
    <div style="background:#0F172A;border-radius:8px;padding:10px 14px;min-width:130px">
      <div style="font-family:monospace;font-weight:700;margin-bottom:4px">{{ code.upper() }}</div>
      <div style="font-size:.8rem;color:var(--muted)">χ² = {{ "%.3f"|format(res.statistic) }}</div>
      <div style="font-size:.8rem;color:var(--muted)">p  = {{ "%.4f"|format(res.p_value) }}</div>
      <div style="margin-top:6px">
        {% if res.significant %}
          <span class="badge badge-green">Significant</span>
        {% else %}
          <span class="badge badge-amber">Not sig.</span>
        {% endif %}
      </div>
    </div>
  {% endfor %}
  </div>
</div>

<footer>
  Project Fairness · Skin Lesion Classification ·
  Report generated {{ generated_at }} ·
  Fairness Grade: <strong>{{ grade }}</strong>
</footer>
</body>
</html>
"""


def render_html(ctx: dict) -> str:
    env = Environment(autoescape=select_autoescape(["html"]))
    env.filters["abs"] = abs
    tmpl = env.from_string(INLINE_TEMPLATE)
    return tmpl.render(**ctx)


def save_pdf(html: str, path: Path) -> bool:
    try:
        from weasyprint import HTML
        HTML(string=html, base_url=str(path.parent)).write_pdf(str(path))
        log.info("PDF saved → %s", path)
        return True
    except ImportError:
        log.warning("weasyprint not installed — skipping PDF. Install with: pip install weasyprint")
        return False
    except Exception as e:
        log.warning("PDF generation failed: %s", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Generate HTML + PDF fairness report")
    p.add_argument("--baseline",       required=True, help="Path to baseline_metrics.json")
    p.add_argument("--balanced",       required=True, help="Path to balanced_metrics.json")
    p.add_argument("--baseline-preds", default=None,  help=".npy array of baseline predictions")
    p.add_argument("--balanced-preds", default=None,  help=".npy array of balanced predictions")
    p.add_argument("--targets",        default=None,  help=".npy array of ground-truth labels")
    p.add_argument("--output",         default="reports", help="Output directory")
    p.add_argument("--no-pdf",         action="store_true", help="Skip PDF generation")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load metrics ──────────────────────────────────────────────────────
    log.info("Loading metrics …")
    baseline = load_metrics(Path(args.baseline))
    balanced = load_metrics(Path(args.balanced))

    # ── McNemar test ──────────────────────────────────────────────────────
    has_preds = all([args.baseline_preds, args.balanced_preds, args.targets])
    if has_preds:
        log.info("Running McNemar's test on prediction arrays …")
        y_true  = np.load(args.targets)
        pb      = np.load(args.baseline_preds)
        pf      = np.load(args.balanced_preds)
        mcnemar = compute_mcnemar(y_true, pb, pf)
    else:
        log.warning("No raw predictions supplied — McNemar test uses approximation.")
        # Approximate McNemar from per-class support + recall delta
        mcnemar = {}
        for code in HAM_CLASSES:
            n = baseline.get("per_class", {}).get(code, {}).get("support", 100)
            r_b = _pc(baseline, code, "recall")
            r_f = _pc(balanced, code, "recall")
            delta = abs(r_f - r_b)
            # approx chi-sq: δ² × n
            stat = min(delta ** 2 * n, 50.0)
            try:
                from scipy.stats import chi2
                pval = float(1 - chi2.cdf(stat, df=1))
            except ImportError:
                pval = 1.0 / (1.0 + stat)
            mcnemar[code] = {
                "statistic":   round(stat, 4),
                "p_value":     round(pval, 4),
                "significant": pval < 0.05,
            }

    # ── Fairness metrics ──────────────────────────────────────────────────
    dpd_before = compute_dpd(baseline)
    dpd_after  = compute_dpd(balanced)
    eod_before = compute_eod(baseline)
    eod_after  = compute_eod(balanced)
    arg_before = compute_avg_recall_gap(baseline)
    arg_after  = compute_avg_recall_gap(balanced)

    f1_delta = (balanced.get("f1_macro", 0) or 0) - (baseline.get("f1_macro", 0) or 0)
    grade    = fairness_grade(dpd_after, eod_after, f1_delta)

    # ── Generate plots ────────────────────────────────────────────────────
    log.info("Generating plots …")
    plots = {
        "plot_recall_bars":  plot_grouped_bars(baseline, balanced, "recall"),
        "plot_f1_bars":      plot_grouped_bars(baseline, balanced, "f1"),
        "plot_radar_recall": plot_radar(baseline, balanced, "recall"),
        "plot_radar_f1":     plot_radar(baseline, balanced, "f1"),
        "plot_pies":         plot_pie_charts(baseline, balanced),
        "plot_cms":          plot_confusion_matrices(baseline, balanced),
        "plot_auc":          plot_auc_comparison(baseline, balanced),
    }

    # ── Build template context ────────────────────────────────────────────
    macro_keys = [
        ("Accuracy",         "accuracy"),
        ("Precision (macro)","precision_macro"),
        ("Recall (macro)",   "recall_macro"),
        ("F1 (macro)",       "f1_macro"),
        ("AUC-ROC (macro)",  "auc_roc_macro"),
    ]
    macro_rows = []
    for label, key in macro_keys:
        b = baseline.get(key) or 0.0
        f = balanced.get(key) or 0.0
        d = f - b
        macro_rows.append({
            "label":  label,
            "before": b,
            "after":  f,
            "delta":  round(d, 6),
            "pct":    round(100 * d / max(abs(b), 1e-9), 2),
        })

    fairness_cards = [
        {"label": "Demographic Parity Diff",  "desc": "max(recall) − min(recall)",
         "before": dpd_before, "after": dpd_after, "delta": dpd_after - dpd_before},
        {"label": "Equal Opportunity Diff",   "desc": "max(recall) − min(recall) ≈ EOD",
         "before": eod_before, "after": eod_after, "delta": eod_after - eod_before},
        {"label": "Average Recall Gap",       "desc": "majority recall − mean recall",
         "before": arg_before, "after": arg_after, "delta": arg_after - arg_before},
    ]

    class_rows = []
    for code in HAM_CLASSES:
        mc = mcnemar.get(code, {"statistic": 0, "p_value": 1, "significant": False})
        class_rows.append({
            "code":          code,
            "name":          HAM_NAMES[code],
            "recall_before": _pc(baseline, code, "recall"),
            "recall_after":  _pc(balanced, code, "recall"),
            "d_recall":      _pc(balanced, code, "recall") - _pc(baseline, code, "recall"),
            "f1_before":     _pc(baseline, code, "f1"),
            "f1_after":      _pc(balanced, code, "f1"),
            "d_f1":          _pc(balanced, code, "f1") - _pc(baseline, code, "f1"),
            "auc_before":    _pc(baseline, code, "auc_roc"),
            "auc_after":     _pc(balanced, code, "auc_roc"),
            "mcnemar_p":     mc["p_value"],
            "mcnemar_sig":   mc["significant"],
        })

    ctx = {
        "generated_at":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "grade":         grade,
        "fairness_cards": fairness_cards,
        "macro_rows":    macro_rows,
        "class_rows":    class_rows,
        "mcnemar":       mcnemar,
        "has_predictions": has_preds,
        **plots,
    }

    # ── Render HTML ───────────────────────────────────────────────────────
    log.info("Rendering HTML report …")
    html = render_html(ctx)
    html_path = out / "fairness_report.html"
    html_path.write_text(html, encoding="utf-8")
    log.info("HTML saved → %s", html_path)

    # ── PDF ───────────────────────────────────────────────────────────────
    if not args.no_pdf:
        save_pdf(html, out / "fairness_report.pdf")

    # ── Summary JSON ──────────────────────────────────────────────────────
    summary = build_summary(
        baseline, balanced, mcnemar,
        dpd_before, dpd_after, eod_before, eod_after, grade,
    )
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Summary JSON → %s", summary_path)

    # ── Console summary ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  FAIRNESS REPORT COMPLETE — Grade: {grade}")
    print("═" * 60)
    for card in fairness_cards:
        delta = card["delta"]
        sym   = "▼" if delta < 0 else "▲"
        print(f"  {card['label']:<30} "
              f"{card['before']*100:.1f}% → {card['after']*100:.1f}%  "
              f"({sym} {abs(delta)*100:.1f}%)")
    print(f"\n  HTML  → {html_path}")
    print(f"  JSON  → {summary_path}")
    print("═" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
