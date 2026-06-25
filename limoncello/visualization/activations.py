"""
ROI-validator activation maps 🍋🔬
==================================

Matplotlib-only renderers for *seeing* what the cilia validator CNN responds to,
fed by :func:`limoncello.ml.roi_validator.collect_activations` and
:func:`limoncello.ml.roi_validator.grad_cam`:

  * :func:`layer_overview_figure` – one tile per conv layer, the mean activation
    across its channels (where each block "fires" as the image flows deeper).
  * :func:`feature_maps_figure` – a grid of individual channel maps for a single
    chosen layer (the per-filter detail).
  * :func:`gradcam_figure` – the Grad-CAM heatmap overlaid on the ROI (the region
    that drove the keep / reject decision).

Everything is pure NumPy + matplotlib so it imports without torch.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _norm(a: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1] for display (flat map → all zeros)."""
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def layer_overview_figure(result: dict, cmap: str = "inferno"):
    """Mean activation map per conv layer + the input image. ``result`` is the
    dict from ``collect_activations``."""
    layers = result.get("layers", [])
    inp = result.get("input")
    n = len(layers) + 1
    ncol = min(n, 5)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()

    axes[0].imshow(np.clip(inp, 0, 1))
    axes[0].set_title(f"input\nP(keep)={result.get('prob', float('nan')):.2f}",
                      fontsize=9)
    for ax, layer in zip(axes[1:], layers):
        m = _norm(layer["act"].mean(axis=0))               # mean over channels
        ax.imshow(m, cmap=cmap)
        ax.set_title(f"{layer['name']}\n{layer['n_channels']} ch · "
                     f"{m.shape[0]}×{m.shape[1]}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Mean activation per conv layer", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def feature_maps_figure(layer: dict, max_channels: int = 32,
                        cmap: str = "inferno"):
    """Grid of individual channel activation maps for one conv layer. ``layer``
    is one entry of ``result["layers"]``."""
    act = np.asarray(layer["act"], dtype=np.float32)        # (C, H, W)
    c = min(act.shape[0], int(max_channels))
    ncol = min(8, c)
    nrow = int(np.ceil(c / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.5 * ncol, 1.55 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(axes)):
        if i < c:
            axes[i].imshow(_norm(act[i]), cmap=cmap)
            axes[i].set_title(f"#{i}", fontsize=7)
        axes[i].axis("off")
    fig.suptitle(f"{layer['name']} — first {c} of {act.shape[0]} channels",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def gradcam_figure(result: dict, cmap: str = "jet", alpha: float = 0.5):
    """Input, Grad-CAM heatmap, and the overlay side-by-side. ``result`` is the
    dict from ``grad_cam``."""
    inp = np.clip(np.asarray(result["input"], dtype=np.float32), 0, 1)
    cam = np.asarray(result["cam"], dtype=np.float32)
    if cam.shape != inp.shape[:2]:                          # upsample CAM to ROI
        from PIL import Image
        cam = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (inp.shape[1], inp.shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
    axes[0].imshow(inp)
    axes[0].set_title("ROI", fontsize=10)
    axes[1].imshow(cam, cmap=cmap)
    axes[1].set_title("Grad-CAM", fontsize=10)
    axes[2].imshow(inp)
    axes[2].imshow(cam, cmap=cmap, alpha=alpha)
    axes[2].set_title("overlay", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"What drove '{result.get('target', 'keep')}'  ·  P(keep)="
        f"{result.get('prob', float('nan')):.2f}",
        fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig
