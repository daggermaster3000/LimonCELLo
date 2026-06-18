"""
synthetic_data.py — neuron-like validation images for the LimonCELLo pipeline.

Generates 4-channel 3-D images (matching the napari app's default channel order
``[Cilia, Neurites, Basal Bodies, Nuclei]``) with **known** cilia placement, so
the pipeline's soma/axon classification can be validated against ground truth.

Scenarios (cilia ≈ 10–20 per image):
  A  all_soma   — every cilium sits next to a nucleus (→ should classify "soma")
  B  all_axon   — every cilium sits on a neurite, away from soma (→ "axon")
  C  mix_20/50/80 — given fraction of cilia on axons, the rest on somas
  D  random     — cilia placed uniformly at random (no structure association)

Each cilium gets a basal body stamped at its base. For every image we also write
its ground-truth (intended class + position) to ``ground_truth.csv`` and an APOC
sparse-label TIFF (``<stem>_labels.tif``: cilia=2, background=1) so you can train
a classifier on the synthetic set via the napari "Train classifier" tab.

Output is real Imaris ``.ims`` (HDF5) so the existing app/batch read it directly.

Usage:
    python synthetic_data.py --out synthetic_runs            # default dataset
    python synthetic_data.py --out d --reps 3 --shape 16 384 384 --no-labels
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# Channel order — matches napari_app._CH_NAMES defaults (Ch: Cilia=0 … Nuclei=3).
CILIA, NEURITES, BB, NUCLEI = 0, 1, 2, 3
CHANNEL_NAMES = ["Cilia", "Neurites", "Basal Bodies", "Nuclei"]
N_CHANNELS = 4


# ─────────────────────────────────────────────────────────────────────────────
# Imaris .ims writer  (HDF5; attributes are S1 char-arrays, the Imaris convention)
# ─────────────────────────────────────────────────────────────────────────────
def _ims_attr(group, name: str, value) -> None:
    """Store an attribute as a 1-D array of single bytes, which the reader decodes
    via the numpy buffer protocol (``str(arr, 'ascii')``)."""
    group.attrs.create(name, np.array(list(str(value)), dtype="S1"))


def write_ims(path, vol_czyx, voxel_size=(0.4, 0.15, 0.15), dtype=np.uint16) -> None:
    """Write a (C, Z, Y, X) volume as an Imaris .ims file the pipeline can read."""
    import h5py  # lazy: only needed to actually write .ims

    vol = np.asarray(vol_czyx)
    C, Z, Y, X = vol.shape
    vz, vy, vx = (float(v) for v in voxel_size)
    info_max = float(np.iinfo(dtype).max)

    with h5py.File(str(path), "w") as hf:
        tp = hf.create_group("DataSet/ResolutionLevel 0/TimePoint 0")
        for c in range(C):
            data = np.clip(vol[c], 0, info_max).astype(dtype)
            ch = tp.create_group(f"Channel {c}")
            ch.create_dataset(
                "Data", data=data,
                chunks=(1, min(Y, 256), min(X, 256)), compression="gzip",
            )
            _ims_attr(ch, "ImageSizeX", X)
            _ims_attr(ch, "ImageSizeY", Y)
            _ims_attr(ch, "ImageSizeZ", Z)
            _ims_attr(ch, "HistogramMin", 0)
            _ims_attr(ch, "HistogramMax", int(data.max()) if data.size else 1)

        img = hf.create_group("DataSetInfo/Image")
        _ims_attr(img, "X", X); _ims_attr(img, "Y", Y); _ims_attr(img, "Z", Z)
        # Physical extent → reader computes resolution = (ExtMax - ExtMin) / N.
        _ims_attr(img, "ExtMin0", 0.0); _ims_attr(img, "ExtMax0", round(X * vx, 4))
        _ims_attr(img, "ExtMin1", 0.0); _ims_attr(img, "ExtMax1", round(Y * vy, 4))
        _ims_attr(img, "ExtMin2", 0.0); _ims_attr(img, "ExtMax2", round(Z * vz, 4))


# ─────────────────────────────────────────────────────────────────────────────
# Geometry primitives
# ─────────────────────────────────────────────────────────────────────────────
def _draw_ball(arr, center, radii, val=1.0) -> None:
    """Paint a (possibly anisotropic) solid ellipsoid into ``arr`` (max-combine)."""
    z0, y0, x0 = center
    rz, ry, rx = radii
    Z, Y, X = arr.shape
    zlo, zhi = max(0, int(z0 - rz)), min(Z, int(z0 + rz) + 1)
    ylo, yhi = max(0, int(y0 - ry)), min(Y, int(y0 + ry) + 1)
    xlo, xhi = max(0, int(x0 - rx)), min(X, int(x0 + rx) + 1)
    if zlo >= zhi or ylo >= yhi or xlo >= xhi:
        return
    zz, yy, xx = np.ogrid[zlo:zhi, ylo:yhi, xlo:xhi]
    m = (((zz - z0) / max(rz, 1e-6)) ** 2
         + ((yy - y0) / max(ry, 1e-6)) ** 2
         + ((xx - x0) / max(rx, 1e-6)) ** 2) <= 1.0
    sub = arr[zlo:zhi, ylo:yhi, xlo:xhi]
    sub[m] = np.maximum(sub[m], val)


def _draw_tube(arr, p0, p1, radius=(1, 2, 2), val=1.0) -> None:
    """Paint a thick line between two (z, y, x) points by stamping balls along it."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    dist = float(np.linalg.norm(p1 - p0))
    n = max(2, int(dist * 2))
    for t in np.linspace(0, 1, n):
        _draw_ball(arr, p0 + (p1 - p0) * t, radius, val)


def _walk_path(rng, start, shape, n_steps=14, step=18.0, z_wander=0.6):
    """A gently meandering 3-D poly-line from ``start`` toward a random direction
    (mostly in-plane, since stacks are thin in Z). Returns a list of (z, y, x)."""
    Z, Y, X = shape
    ang = rng.uniform(0, 2 * np.pi)
    pts = [np.asarray(start, float)]
    for _ in range(n_steps):
        ang += rng.normal(0, 0.35)                      # slight curvature
        dz = rng.normal(0, z_wander)
        dy = step * np.sin(ang)
        dx = step * np.cos(ang)
        nxt = pts[-1] + np.array([dz, dy, dx])
        nxt[0] = np.clip(nxt[0], 1, Z - 2)
        nxt[1] = np.clip(nxt[1], 1, Y - 2)
        nxt[2] = np.clip(nxt[2], 1, X - 2)
        pts.append(nxt)
    return pts


def _point_along(path, rng, lo=0.35, hi=1.0):
    """A random point on a poly-line, restricted to the [lo, hi] arc-fraction
    (so axonal cilia don't land on the soma at the path start)."""
    seg = rng.integers(max(1, int(lo * (len(path) - 1))), len(path) - 1 + 1)
    seg = int(np.clip(seg, 1, len(path) - 1))
    a, b = path[seg - 1], path[seg]
    return a + (b - a) * rng.uniform(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Image generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_image(rng, shape=(16, 384, 384), n_neurons=4, n_cilia=15,
                   axon_fraction=0.0, scenario="custom", voxel=(0.4, 0.15, 0.15)):
    """Build one neuron-like 4-channel image.

    Returns ``(vol_czyx_uint16, ground_truth_records, cilia_mask_bool)``.
    ``axon_fraction`` is the share of cilia placed on neurites; the rest go on
    somas. ``scenario == 'random'`` overrides placement to uniform-random.
    """
    from scipy.ndimage import gaussian_filter

    Z, Y, X = shape
    zc = Z // 2
    f_nuc = np.zeros(shape, np.float32)
    f_neu = np.zeros(shape, np.float32)
    f_cil = np.zeros(shape, np.float32)
    f_bb = np.zeros(shape, np.float32)
    cilia_mask = np.zeros(shape, bool)

    # ── Somas (nuclei) + neurites emanating from each ────────────────────────
    nuc_r = (max(1, Z // 6), 22, 22)                    # nucleus radii (z, y, x)
    centers, paths = [], []
    for _ in range(n_neurons):
        cy = rng.uniform(0.15, 0.85) * Y
        cx = rng.uniform(0.15, 0.85) * X
        c = np.array([zc + rng.normal(0, 0.5), cy, cx])
        centers.append(c)
        _draw_ball(f_nuc, c, nuc_r, val=1.0)
        for _ in range(int(rng.integers(2, 4))):        # 2–3 processes per soma
            path = _walk_path(rng, c, shape)
            paths.append(path)
            for a, b in zip(path[:-1], path[1:]):
                _draw_tube(f_neu, a, b, radius=(1, 2.5, 2.5), val=1.0)

    # ── Cilia placement ──────────────────────────────────────────────────────
    records = []
    cil_r = (max(1, Z // 8), 2.2, 2.2)
    for cid in range(1, n_cilia + 1):
        if scenario == "random":
            pos = np.array([rng.uniform(1, Z - 2),
                            rng.uniform(8, Y - 8), rng.uniform(8, X - 8)])
            cls = "random"
        elif rng.uniform() < axon_fraction and paths:
            # On a neurite, nudged just off the centre-line.
            path = paths[int(rng.integers(0, len(paths)))]
            base = _point_along(path, rng)
            off = rng.normal(0, 1.5, 3); off[0] *= 0.2
            pos = base + off
            cls = "axon"
        else:
            # Just outside a nucleus surface (on the soma, away from neurites).
            c = centers[int(rng.integers(0, len(centers)))]
            ang = rng.uniform(0, 2 * np.pi)
            rad = nuc_r[1] + rng.uniform(3, 8)
            pos = c + np.array([rng.normal(0, 0.5),
                                rad * np.sin(ang), rad * np.cos(ang)])
            cls = "soma"

        pos[0] = np.clip(pos[0], 0, Z - 1)
        pos[1] = np.clip(pos[1], 0, Y - 1)
        pos[2] = np.clip(pos[2], 0, X - 1)
        _draw_ball(f_cil, pos, cil_r, val=1.0)
        _draw_ball(cilia_mask.view(np.uint8), pos, cil_r, val=1)
        # Basal body: a tiny punctum a couple of pixels from the cilium centre.
        bbpos = pos + np.array([0.0, rng.normal(0, 2), rng.normal(0, 2)])
        _draw_ball(f_bb, bbpos, (max(1, Z // 10), 1.2, 1.2), val=1.0)
        records.append({"cilia_id": cid,
                        "z": float(pos[0]), "y": float(pos[1]), "x": float(pos[2]),
                        "intended_class": cls})

    # ── Blur + noise → realistic intensities, cast to uint16 ─────────────────
    def _finish(f, blur, peak, bg):
        f = gaussian_filter(f, sigma=blur)
        if f.max() > 0:
            f = f / f.max() * peak
        f = f + rng.poisson(bg, size=f.shape).astype(np.float32)
        f = f + rng.normal(0, bg * 0.3, size=f.shape)
        return np.clip(f, 0, 65535)

    vol = np.stack([
        _finish(f_cil, (0.7, 1.0, 1.0), 9000, 120),       # Cilia
        _finish(f_neu, (0.8, 1.2, 1.2), 6000, 150),       # Neurites
        _finish(f_bb,  (0.6, 0.8, 0.8), 12000, 100),      # Basal bodies
        _finish(f_nuc, (1.0, 2.0, 2.0), 8000, 200),       # Nuclei
    ]).astype(np.uint16)
    return vol, records, cilia_mask


def sparse_labels(cilia_mask, rng, n_bg=4000):
    """APOC sparse ground truth for the cilia channel: object=2, background=1,
    unannotated=0 (matches the napari label-maker / trainer convention)."""
    lab = np.zeros(cilia_mask.shape, np.uint16)
    lab[cilia_mask] = 2
    bg = np.argwhere(~cilia_mask)
    if len(bg):
        pick = bg[rng.choice(len(bg), size=min(n_bg, len(bg)), replace=False)]
        lab[pick[:, 0], pick[:, 1], pick[:, 2]] = 1
    return lab


# ─────────────────────────────────────────────────────────────────────────────
# Dataset driver
# ─────────────────────────────────────────────────────────────────────────────
# (scenario_name, axon_fraction)  — 'random' handled specially.
_SCENARIOS = [
    ("A_all_soma",   0.0),
    ("B_all_axon",   1.0),
    ("C_mix_axon20", 0.2),
    ("C_mix_axon50", 0.5),
    ("C_mix_axon80", 0.8),
    ("D_random",     None),
]


def build_dataset(out_dir, reps=3, shape=(16, 384, 384), voxel=(0.4, 0.15, 0.15),
                  n_neurons=4, cilia_range=(10, 20), seed=0, write_labels=True):
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    gt_rows = []

    for name, frac in _SCENARIOS:
        for rep in range(reps):
            n_cil = int(rng.integers(cilia_range[0], cilia_range[1] + 1))
            scen = "random" if name.startswith("D_") else "custom"
            vol, recs, cmask = generate_image(
                rng, shape=shape, n_neurons=n_neurons, n_cilia=n_cil,
                axon_fraction=(0.0 if frac is None else frac),
                scenario=scen, voxel=voxel,
            )
            stem = f"{name}_rep{rep + 1}"
            write_ims(out / f"{stem}.ims", vol, voxel_size=voxel)
            if write_labels:
                import tifffile
                tifffile.imwrite(str(out / f"{stem}_labels.tif"),
                                 sparse_labels(cmask, rng))
            for r in recs:
                gt_rows.append({"filename": f"{stem}.ims", "scenario": name,
                                "axon_fraction": (np.nan if frac is None else frac),
                                **r})
            n_ax = sum(r["intended_class"] == "axon" for r in recs)
            n_so = sum(r["intended_class"] == "soma" for r in recs)
            print(f"  ✓ {stem}: {len(recs)} cilia (axon={n_ax}, soma={n_so})")

    gt = pd.DataFrame(gt_rows)
    gt.to_csv(out / "ground_truth.csv", index=False)
    print(f"\nWrote {gt['filename'].nunique()} images + ground_truth.csv → {out}")
    print(f"Channels: 0={CHANNEL_NAMES[0]} 1={CHANNEL_NAMES[1]} "
          f"2={CHANNEL_NAMES[2]} 3={CHANNEL_NAMES[3]}  (set these in the app)")
    return gt


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic neuron images for "
                                             "pipeline validation.")
    ap.add_argument("--out", default="synthetic_runs", help="output folder")
    ap.add_argument("--reps", type=int, default=3, help="images per scenario")
    ap.add_argument("--shape", type=int, nargs=3, default=(16, 384, 384),
                    metavar=("Z", "Y", "X"))
    ap.add_argument("--voxel", type=float, nargs=3, default=(0.4, 0.15, 0.15),
                    metavar=("VZ", "VY", "VX"))
    ap.add_argument("--neurons", type=int, default=4)
    ap.add_argument("--cilia", type=int, nargs=2, default=(10, 20),
                    metavar=("MIN", "MAX"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-labels", action="store_true",
                    help="skip writing <stem>_labels.tif training annotations")
    args = ap.parse_args()
    build_dataset(args.out, reps=args.reps, shape=tuple(args.shape),
                  voxel=tuple(args.voxel), n_neurons=args.neurons,
                  cilia_range=tuple(args.cilia), seed=args.seed,
                  write_labels=not args.no_labels)


if __name__ == "__main__":
    main()
