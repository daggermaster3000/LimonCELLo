"""
Fast per-cilium ROI export 🍋📸  (no napari, no GUI)
====================================================

Replaces the slow napari per-cilium 3-D screenshot loop. For each cilium it:

  * crops the raw volume to the cilium's bounding box (+ its paired basal body),
  * writes a compact **2-D MIP thumbnail** PNG for the human screening gallery
    (top-down XY view, plus a side XZ view when the stack is 3-D), and
  * optionally writes the **raw intensity crop** as a compressed ``.npz`` so a
    classifier can later train on the actual data instead of a screenshot.

Everything is plain NumPy + Pillow, so it runs in milliseconds per cilium on the
CPU — orders of magnitude faster than rendering each ROI in napari.

The PNG keeps the same ``{stem}_cilia{cid}.png`` name the data app already looks
for, so the Screening tab needs no changes.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Thumbnail shows ONLY the cilia + basal-body channels (the neurite/nuclei
# channels just wash out the small, bright structures we care about).
#   cilia → green,  basal body → magenta
_DISPLAY_GAMMA = 0.7    # < 1 lifts dim signal so faint cilia stay visible
_BG_PCTL = 50.0         # background level (mapped to black)
_OBJ_PCTL = 90.0        # object brightness reference (mapped to full)


def _clamp_ch(n_ch: int, idx) -> int:
    try:
        return int(min(max(int(idx), 0), n_ch - 1))
    except (TypeError, ValueError):
        return 0


def _norm_obj(a: np.ndarray, obj_mask: np.ndarray | None) -> np.ndarray:
    """Contrast stretch for display: background → black, the *labelled object*
    sets the bright point. This keeps tiny/sparse structures (a basal body is
    only tens of voxels) visible instead of being lost under a percentile that
    falls in the noise."""
    a = np.asarray(a, dtype=np.float32)
    if a.size == 0:
        return a
    # No labelled object in this crop → nothing to show (avoid stretching noise).
    if obj_mask is None or not obj_mask.any():
        return np.zeros_like(a)
    lo = float(np.percentile(a, _BG_PCTL))
    hi = float(np.percentile(a[obj_mask], _OBJ_PCTL))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _norm_raw(a: np.ndarray) -> np.ndarray:
    """Plain linear min-max of the crop to [0, 1] — no object-aware contrast, no
    gamma. Used when display correction is switched off (raw intensities)."""
    a = np.asarray(a, dtype=np.float32)
    if a.size == 0:
        return a
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _to_rgb(cilia: np.ndarray, bb: np.ndarray,
            gamma: float = _DISPLAY_GAMMA) -> np.ndarray:
    """Cilia (green) + basal body (magenta) → RGB. ``gamma`` < 1 brightens; pass
    ``gamma = 1.0`` for a raw (linear) display."""
    cilia = np.power(cilia, gamma)
    bb = np.power(bb, gamma)
    rgb = np.zeros((*cilia.shape, 3), dtype=np.float32)
    rgb[..., 1] = cilia               # green
    rgb[..., 0] = bb                  # magenta = red + blue
    rgb[..., 2] = bb
    return np.clip(rgb, 0.0, 1.0)


def _xy_thumb(cilia: np.ndarray, bb: np.ndarray, px: int,
              gamma: float = _DISPLAY_GAMMA) -> np.ndarray:
    """Top-down XY max-projection (the view the model trains on), padded to a
    centred square and resized to ``px`` so every thumbnail is the same size."""
    rgb = _to_rgb(cilia.max(axis=0), bb.max(axis=0), gamma)        # (Y, X, 3)
    h, w = rgb.shape[:2]
    s = max(h, w)
    sq = np.zeros((s, s, 3), dtype=np.float32)
    y0, x0 = (s - h) // 2, (s - w) // 2
    sq[y0:y0 + h, x0:x0 + w] = rgb
    return (sq * 255).astype(np.uint8)


def _norm_pctl(a: np.ndarray, lo: float = 50.0, hi: float = 99.7) -> np.ndarray:
    """Robust percentile contrast stretch (for dense channels without a mask)."""
    a = np.asarray(a, dtype=np.float32)
    if a.size == 0:
        return a
    p_lo, p_hi = np.percentile(a, [lo, hi])
    if p_hi <= p_lo:
        p_hi = p_lo + 1e-6
    return np.clip((a - p_lo) / (p_hi - p_lo), 0.0, 1.0)


def render_npz_rgb(npz_path, px: int = 240, gamma: float | None = None):
    """Render a saved ``.npz`` crop as an RGB XY max-projection showing **all
    four channels**: cilia = green, basal body = magenta, neurite = cyan,
    nuclei = blue. Returns a ``(px, px, 3)`` uint8 array, or None on failure.

    Used by the interactive scatter's ROI preview (the screening thumbnail keeps
    only cilia + basal body)."""
    g = _DISPLAY_GAMMA if gamma is None else gamma
    try:
        d = np.load(npz_path)
    except Exception:                                     # noqa: BLE001
        return None
    keys = set(getattr(d, "files", []))

    # Objects of interest: object-aware contrast + brightening, full intensity.
    def _obj(name, mask):
        if name not in keys:
            return None
        return np.power(_norm_obj(d[name], mask).max(axis=0), g)

    # Context channels (neurite/nuclei) are large & dense, so they easily swamp
    # the small cilia/BB. Crush their background hard (high low-percentile), do
    # NOT brighten, and keep them as a faint wash so cilia/BB stay readable.
    def _ctx(name):
        if name not in keys:
            return None
        return np.power(_norm_pctl(d[name], 88.0, 99.8).max(axis=0), 1.4)

    _CTX_W = 0.22                                         # faint context wash
    cil = _obj("cilia", d["mask"] if "mask" in keys else None)
    # No paired BB mask → object-aware norm has nothing to scale to and would
    # render black; fall back to a percentile stretch so real BB signal shows.
    if "bb" in keys and "bb_mask" not in keys:
        bb = np.power(_norm_pctl(d["bb"], 50.0, 99.5).max(axis=0), g)
    else:
        bb = _obj("bb", d["bb_mask"] if "bb_mask" in keys else None)
    neu = _ctx("neurite")
    nuc = _ctx("nuclei")
    ref = next((x for x in (cil, bb, neu, nuc) if x is not None), None)
    if ref is None:
        return None
    h, w = ref.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    # Context first (underneath), objects added on top so they dominate.
    if neu is not None:
        rgb[..., 1] += _CTX_W * neu;  rgb[..., 2] += _CTX_W * neu   # cyan
    if nuc is not None:
        rgb[..., 2] += _CTX_W * nuc                       # blue
    if cil is not None:
        rgb[..., 1] += cil                                # green (full)
    if bb is not None:
        rgb[..., 0] += bb;  rgb[..., 2] += bb             # magenta (full)
    rgb = np.clip(rgb, 0.0, 1.0)

    s = max(h, w)                                          # centre-pad to square
    sq = np.zeros((s, s, 3), dtype=np.float32)
    y0, x0 = (s - h) // 2, (s - w) // 2
    sq[y0:y0 + h, x0:x0 + w] = rgb
    img = (sq * 255).astype(np.uint8)
    if px and s != px:
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize((px, px), Image.BILINEAR))
    return img


def render_npz_thumb(npz_path, px: int = 192,
                     correct_display: bool = True) -> np.ndarray | None:
    """Re-render the cilia+basal-body **training thumbnail** from a saved ``.npz``
    crop (same look as ``save_cilia_rois``), with display correction on/off.

    on  → object-aware contrast + gamma (BB percentile fallback when unpaired).
    off → raw intensities (plain min-max, no gamma).

    Returns a ``(px, px, 3)`` uint8 array (cilia green, BB magenta), or None.
    """
    try:
        d = np.load(npz_path)
    except Exception:                                     # noqa: BLE001
        return None
    keys = set(getattr(d, "files", []))
    if "cilia" not in keys or "bb" not in keys:
        return None
    cil, bb = d["cilia"], d["bb"]
    if correct_display:
        cmask = d["mask"] if "mask" in keys else None
        bmask = d["bb_mask"] if "bb_mask" in keys else None
        cdisp = _norm_obj(cil, cmask)
        bbdisp = (_norm_obj(bb, bmask) if bmask is not None
                  else _norm_pctl(bb, 50.0, 99.5))
        gamma = _DISPLAY_GAMMA
    else:
        cdisp, bbdisp, gamma = _norm_raw(cil), _norm_raw(bb), 1.0
    from PIL import Image
    img = _xy_thumb(cdisp, bbdisp, px, gamma)
    return np.asarray(Image.fromarray(img).resize((px, px), Image.BILINEAR))


def save_cilia_rois(raw, cilia_labels, bb_labels, cilia_df, channels,
                    voxel_size, roi_dir, stem, *, margin: int = 22,
                    save_crops: bool = True, thumb_px: int = 192,
                    correct_display: bool = True) -> int:
    """Write a MIP thumbnail (+ optional raw ``.npz`` crop) per cilium.

    Parameters mirror the data already available in the pipeline / napari batch
    callback. Returns the number of cilia successfully exported.
    """
    from PIL import Image

    roi_dir = Path(roi_dir)
    roi_dir.mkdir(parents=True, exist_ok=True)

    raw = np.asarray(raw)                       # (C, Z, Y, X)
    cilia_labels = np.asarray(cilia_labels)
    bb_labels = np.asarray(bb_labels)
    n_ch = raw.shape[0]
    ci = _clamp_ch(n_ch, channels.get("ch_cilia", 0))
    bi = _clamp_ch(n_ch, channels.get("ch_bb", 0))
    ni = _clamp_ch(n_ch, channels.get("ch_neurites", 0))
    di = _clamp_ch(n_ch, channels.get("ch_nuclei", 0))
    shp = cilia_labels.shape

    saved = 0
    for _, row in cilia_df.iterrows():
        try:
            cid = int(row["cilia_id"])
            mask = cilia_labels == cid
            pid = row.get("paired_id")
            if pid is not None and not (isinstance(pid, float) and np.isnan(pid)):
                mask = mask | (bb_labels == int(pid))
            pts = np.argwhere(mask)
            if pts.size == 0:
                continue
            lo = np.maximum(pts.min(0) - margin, 0)
            hi = np.minimum(pts.max(0) + margin + 1, shp)
            sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))

            # Thumbnail: cilia + basal-body channels only, each scaled to its own
            # labelled object so both stay clearly visible. Uniform square size.
            _cmask = cilia_labels[sl] == cid
            _bmask = (bb_labels[sl] == int(pid)) if (
                pid is not None and not (isinstance(pid, float) and np.isnan(pid))
            ) else None
            if correct_display:
                # Object-aware contrast + gamma (the default "corrected" view).
                # When a cilium has no paired basal body (e.g. BB-distance
                # filtering off), there's no BB label to scale against — fall back
                # to a robust percentile stretch so the BB channel still shows real
                # signal instead of rendering black.
                _cdisp = _norm_obj(raw[ci][sl], _cmask)
                _bb_disp = (_norm_obj(raw[bi][sl], _bmask) if _bmask is not None
                            else _norm_pctl(raw[bi][sl], 50.0, 99.5))
                _gamma = _DISPLAY_GAMMA
            else:
                # Raw intensities: plain linear min-max per channel, no gamma.
                _cdisp = _norm_raw(raw[ci][sl])
                _bb_disp = _norm_raw(raw[bi][sl])
                _gamma = 1.0
            img = Image.fromarray(
                _xy_thumb(_cdisp, _bb_disp, thumb_px, _gamma))
            img = img.resize((thumb_px, thumb_px), Image.BILINEAR)
            img.save(roi_dir / f"{stem}_cilia{cid}.png")

            if save_crops:
                _npz = dict(
                    cilia=raw[ci][sl].astype(np.float16),
                    bb=raw[bi][sl].astype(np.float16),
                    neurite=raw[ni][sl].astype(np.float16),
                    nuclei=raw[di][sl].astype(np.float16),
                    mask=(cilia_labels[sl] == cid),
                    voxel_size=np.asarray(voxel_size, dtype=np.float32),
                    cilia_id=cid,
                )
                if _bmask is not None:
                    _npz["bb_mask"] = _bmask
                np.savez_compressed(roi_dir / f"{stem}_cilia{cid}.npz", **_npz)
            saved += 1
        except Exception as exc:                            # noqa: BLE001
            print(f"[LC] ROI export failed for cilium "
                  f"{row.get('cilia_id')}: {exc}")
    return saved
