"""Generate the cross-experiment summary figure for the final report.

Two panels, kept separate so the comparison is honest:
  A) Full 7,532-doc test set  -> Q1b/c/d, Q2b, Q2c, Q3, Q-bonus(full)
  B) 200-doc stratified subset -> Q4 zero/k1/k3, Q-bonus(subset)

Reads the canonical numbers from presentation/data/dashboard.json so the
figure can never drift from the rest of the deliverables.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
d = json.load(open(ROOT / "presentation/data/dashboard.json"))

# Colorblind-safe (Okabe-Ito) palette.
ACC = "#0072B2"   # blue
F1 = "#E69F00"    # orange

full = [
    ("Q1b\nw2v mean", 0.6321, 0.6098),
    ("Q1c\nTF-IDF", 0.6964, 0.6895),
    ("Q1d\nw2v m+max", 0.6202, 0.6004),
    ("Q2b\nBERT", 0.708046, 0.688844),
    ("Q2c\nBERT*", 0.718667, 0.703872),
    ("Q3\nMLM+FT", 0.716, 0.6981),
    ("Q-bonus\nQLoRA", 0.7573021773765268, 0.7477034793625819),
]
sub = [
    ("Q4\nzero-shot", 0.155, 0.13165794969741568),
    ("Q4\nk=1/cls", 0.24, 0.2459423377360881),
    ("Q4\nk=3/cls", 0.365, 0.3505633546345621),
    ("Q-bonus\nQLoRA", 0.805, 0.7985354427881686),
]


def panel(ax, rows, title):
    labels = [r[0] for r in rows]
    acc = [r[1] for r in rows]
    f1 = [r[2] for r in rows]
    x = np.arange(len(rows))
    w = 0.4
    b1 = ax.bar(x - w / 2, acc, w, label="Accuracy", color=ACC)
    b2 = ax.bar(x + w / 2, f1, w, label="Macro-F1", color=F1)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [7, 4]}
)
panel(ax1, full, "(a) Full 7,532-doc test set")
panel(ax2, sub, "(b) 200-doc stratified subset")
ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)
fig.tight_layout()
out = ROOT / "figures/cross_question_summary.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
