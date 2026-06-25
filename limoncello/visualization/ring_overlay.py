"""
Validation ring overlays 🍋🟢🔴
================================

Draw colour-coded rings on a top-down (XY max-projection) overview of a sample,
one ring per cilium at its (y, x) position:

  * **green**  – kept / AI-validated,
  * **red**    – rejected,
  * **yellow** – undecided / unknown.

Used both by the pipeline (to bake an ``*_ai_validation.png`` overview when AI
validation runs in a batch) and by the data app (to overlay rings live on the
saved MIPs). Pure NumPy + matplotlib so it imports without torch / napari.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def _norm(a):
    """Plain linear min-max to [0, 1] — no percentile clipping. The saved MIPs
    are already display-scaled, so this just maps them into range without
    altering relative contrast."""
    a = np.asarray(a, dtype=np.float32)
    if a.size == 0:
        return a
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _window(a, clim):
    """Linear min-max to [0, 1] (no percentile), then apply a user display window
    ``clim = (lo, hi)`` on that normalised range."""
    n = _norm(np.asarray(a, dtype=np.float32))
    lo, hi = float(clim[0]), float(clim[1])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((n - lo) / (hi - lo), 0.0, 1.0)


def mip_rgb(cilia=None, neurite=None, nuclei=None, bb=None, *,
            cilia_clim=(0.0, 1.0), bb_clim=(0.0, 1.0),
            neurite_clim=(0.0, 1.0), nuclei_clim=(0.0, 1.0),
            ctx_weight: float = 0.16) -> np.ndarray | None:
    """Compose an RGB background from per-channel XY MIPs (each 2-D, any may be
    None): cilia→green, basal body→magenta, neurite→dim cyan, nuclei→dim blue.

    Each channel takes a display window ``*_clim = (lo, hi)`` in [0, 1] (applied
    after a plain min-max, no percentile) so the caller can set min/max per
    channel. Context channels (neurite, nuclei) are scaled by ``ctx_weight`` and
    gamma-crushed; objects (cilia, BB) are gamma-lifted so they stay readable.
    """
    ref = next((x for x in (cilia, bb, neurite, nuclei) if x is not None), None)
    if ref is None:
        return None
    h, w = np.asarray(ref).shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    # Context first (dim), objects on top (bright) so they dominate.
    if neurite is not None:
        n = np.power(_window(neurite, neurite_clim), 1.5) * ctx_weight
        rgb[..., 1] += n; rgb[..., 2] += n                 # faint cyan context
    if nuclei is not None:
        rgb[..., 2] += np.power(_window(nuclei, nuclei_clim), 1.5) * ctx_weight
    if cilia is not None:
        rgb[..., 1] += np.power(_window(cilia, cilia_clim), 0.55)   # green
    if bb is not None:
        b = np.power(_window(bb, bb_clim), 0.55)
        rgb[..., 0] += b; rgb[..., 2] += b                 # magenta
    return np.clip(rgb, 0.0, 1.0)


def ring_overlay_figure(background, coords_yx, keep, scores=None, *,
                        ids=None, title: str | None = None, radius: float = 14.0,
                        show_scores: bool = True, show_ids: bool = True,
                        max_width: float = 7.5):
    """Return a matplotlib ``Figure`` of ``background`` (2-D grayscale or RGB)
    with a ring per cilium.

    Parameters
    ----------
    coords_yx : sequence of (y, x)
        Pixel positions in the projection plane.
    keep : sequence of {True, False, None}
        Per-cilium decision → green / red / yellow.
    scores : sequence of float or None
        Optional per-cilium score, drawn above each ring when finite.
    ids : sequence or None
        Optional per-cilium id, drawn below each ring (so the ROI gallery can be
        matched to its position in the overview).
    """
    bg = np.asarray(background, dtype=np.float32)
    h, w = bg.shape[:2]
    fw = min(max_width, max(3.0, w / 130.0))
    fh = fw * (h / max(w, 1))
    fig, ax = plt.subplots(figsize=(fw, fh))
    if bg.ndim == 2:
        ax.imshow(_norm(bg), cmap="gray")
    else:
        ax.imshow(np.clip(bg, 0, 1))

    _COL = {True: "lime", False: "red", None: "yellow"}
    n_keep = n_rej = 0
    for i, (y, x) in enumerate(coords_yx):
        k = keep[i] if i < len(keep) else None
        col = _COL.get(k, "yellow")
        n_keep += int(k is True)
        n_rej += int(k is False)
        ax.add_patch(Circle((float(x), float(y)), radius, fill=False,
                            edgecolor=col, linewidth=1.4, alpha=0.95))
        if show_scores and scores is not None and i < len(scores):
            s = scores[i]
            if s is not None and np.isfinite(s):
                ax.text(float(x), float(y) - radius - 1, f"{s:.2f}", color=col,
                        fontsize=6, ha="center", va="bottom")
        if show_ids and ids is not None and i < len(ids):
            ax.text(float(x), float(y) + radius + 1, f"{ids[i]}", color=col,
                    fontsize=6, ha="center", va="top", fontweight="bold")

    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.axis("off")
    if title:
        ax.set_title(f"{title}   {n_keep} kept · {n_rej} rejected", fontsize=10)
    fig.tight_layout(pad=0.2)
    return fig
