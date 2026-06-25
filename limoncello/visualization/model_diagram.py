"""
ROI-validator architecture diagram 🍋🧠
=======================================

Draws a clean, dependency-free (matplotlib only) layer diagram of the ROI cilia
validator CNNs defined in :mod:`limoncello.ml.roi_validator` — the ``big`` (GPU)
and ``tiny`` (CPU) variants. Used by the data app's Screening → Train tab.

The layer list mirrors ``ROINetBig`` / ``TinyROINet``; keep it in sync if those
change. Parameter counts are the exact totals of those modules.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

_KIND_COLOR = {
    "io":   "#2C3E50",   # input
    "conv": "#27AE60",   # convolution block
    "pool": "#E67E22",   # pooling
    "reg":  "#95A5A6",   # flatten / dropout
    "fc":   "#2980B9",   # dense
    "out":  "#C0392B",   # softmax / output
}
_PARAMS = {"big": "≈ 1.2 M", "tiny": "≈ 31 k"}
_DEFAULT_SIZE = {"big": 96, "tiny": 64}


def _stages(arch: str, s: int) -> list[tuple[str, str, str]]:
    """``[(kind, title, output_shape), …]`` top (input) → bottom (output)."""
    if arch == "big":
        s1, s2, s3, s4 = s // 2, s // 4, s // 8, s // 16
        return [
            ("io",   "Input image",                          f"3×{s}×{s}"),
            ("conv", "Conv block 1  (2× Conv3×3+BN+ReLU, ↓2)", f"32×{s1}×{s1}"),
            ("conv", "Conv block 2  (32→64, ↓2)",            f"64×{s2}×{s2}"),
            ("conv", "Conv block 3  (64→128, ↓2)",           f"128×{s3}×{s3}"),
            ("conv", "Conv block 4  (128→256, ↓2)",          f"256×{s4}×{s4}"),
            ("pool", "Global average pool",                  "256×1×1"),
            ("reg",  "Flatten + Dropout 0.4",                "256"),
            ("fc",   "Dense 256→128 + ReLU",                 "128"),
            ("reg",  "Dropout 0.3",                          "128"),
            ("fc",   "Dense 128→2",                          "2 logits"),
            ("out",  "Softmax",                              "P(real cilium)"),
        ]
    return [
        ("io",   "Input image",                       f"3×{s}×{s}"),
        ("conv", "Conv 3→16 + ReLU, MaxPool ↓2",      f"16×{s // 2}×{s // 2}"),
        ("conv", "Conv 16→32 + ReLU, MaxPool ↓2",     f"32×{s // 4}×{s // 4}"),
        ("conv", "Conv 32→32 + ReLU, AvgPool→4",      "32×4×4"),
        ("reg",  "Flatten + Dropout 0.3",             "512"),
        ("fc",   "Dense 512→32 + ReLU",               "32"),
        ("fc",   "Dense 32→2",                        "2 logits"),
        ("out",  "Softmax",                           "P(real cilium)"),
    ]


def model_architecture_figure(arch: str = "big", size: int | None = None):
    """Return a matplotlib ``Figure`` of the chosen architecture."""
    arch = "big" if arch == "big" else "tiny"
    size = int(size or _DEFAULT_SIZE[arch])
    stages = _stages(arch, size)
    n = len(stages)

    fig, ax = plt.subplots(figsize=(7.6, 1.4 + 0.82 * n))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n + 1.4)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    ax.text(5, n + 0.95, f"ROI validator — {arch.upper()} CNN",
            ha="center", fontsize=13, fontweight="bold", color="#2C3E50")
    ax.text(5, n + 0.5,
            f"input 3×{size}×{size}  ·  {_PARAMS[arch]} parameters  ·  "
            "2-class (keep / reject)",
            ha="center", fontsize=8.5, color="#666")

    box_w, cx = 5.8, 4.2
    for i, (kind, title, shape) in enumerate(stages):
        y = n - i
        rect = mpatches.FancyBboxPatch(
            (cx - box_w / 2, y - 0.34), box_w, 0.68,
            boxstyle="round,pad=0.04", facecolor=_KIND_COLOR[kind],
            edgecolor="white", linewidth=1.4, alpha=0.93, zorder=2)
        ax.add_patch(rect)
        ax.text(cx, y, title, ha="center", va="center", fontsize=8.4,
                color="white", fontweight="bold", zorder=3)
        ax.text(cx + box_w / 2 + 0.25, y, shape, ha="left", va="center",
                fontsize=8, color="#333", family="monospace")
        if i < n - 1:
            ax.annotate("", xy=(cx, y - 0.34), xytext=(cx, y - 0.66),
                        arrowprops=dict(arrowstyle="-|>", color="#777", lw=1.2,
                                        mutation_scale=10), zorder=1)

    _legend = [("Conv", "conv"), ("Pool", "pool"), ("Dense", "fc"),
               ("Reshape/Dropout", "reg"), ("I/O", "io")]
    for li, (lab, k) in enumerate(_legend):
        ax.text(0.1 + li * 1.95, 0.18, lab, fontsize=7,
                color=_KIND_COLOR[k], fontweight="bold")

    plt.tight_layout(pad=0.4)
    return fig
