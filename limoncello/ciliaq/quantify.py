"""Per-cilium 3-D quantification (CiliaQ-style).

Given a labelled cilia volume and one or more intensity channels, compute a
table of per-cilium measurements mirroring CiliaQ's output:

morphology   : volume (voxels + µm³), surface (µm²), sphere radius, max span,
               shape complexity index
skeleton     : cilium length (longest geodesic path, µm), # branches, # endpoints,
               bending index (path length / end-to-end straight distance)
intensity    : mean / min / max / SD / integrated intensity per channel,
               plus a base→tip intensity profile along the cilium
colocalisation: colocalised volume and % with a second channel
orientation  : principal-axis orientation vector (z, y, x)

All physical units use an anisotropic ``voxel_size = (vz, vy, vx)`` in µm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize


# ── geometry helpers ───────────────────────────────────────────────────────────
def _surface_area(mask: np.ndarray, vs) -> float:
    """Exposed-face surface area of a 3-D binary mask with anisotropic voxels."""
    vz, vy, vx = (float(s) for s in vs)
    p = np.pad(mask.astype(bool), 1)
    c = p[1:-1, 1:-1, 1:-1]
    a = 0.0
    a += np.count_nonzero(c & ~p[:-2, 1:-1, 1:-1]) * (vy * vx)
    a += np.count_nonzero(c & ~p[2:, 1:-1, 1:-1]) * (vy * vx)
    a += np.count_nonzero(c & ~p[1:-1, :-2, 1:-1]) * (vz * vx)
    a += np.count_nonzero(c & ~p[1:-1, 2:, 1:-1]) * (vz * vx)
    a += np.count_nonzero(c & ~p[1:-1, 1:-1, :-2]) * (vz * vy)
    a += np.count_nonzero(c & ~p[1:-1, 1:-1, 2:]) * (vz * vy)
    return float(a)


def _max_span(coords_um: np.ndarray) -> float:
    """Maximum distance between any two object voxels (calibrated)."""
    if len(coords_um) < 2:
        return 0.0
    if len(coords_um) > 2000:
        # subsample for speed; the extremes dominate the span
        idx = np.random.default_rng(0).choice(len(coords_um), 2000, replace=False)
        coords_um = coords_um[idx]
    # pairwise via broadcasting on a modest set
    d = np.sqrt(((coords_um[:, None, :] - coords_um[None, :, :]) ** 2).sum(-1))
    return float(d.max())


def _orientation(coords_um: np.ndarray):
    """Principal-axis unit vector (z, y, x) from PCA of the voxel cloud."""
    if len(coords_um) < 2:
        return (np.nan, np.nan, np.nan)
    c = coords_um - coords_um.mean(0)
    cov = np.cov(c.T)
    w, v = np.linalg.eigh(cov)
    axis = v[:, int(np.argmax(w))]
    return tuple(float(x) for x in axis)


# ── skeleton analysis ──────────────────────────────────────────────────────────
_NB = np.array([(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1)
                for dx in (-1, 0, 1) if not (dz == dy == dx == 0)])


def _skeleton_metrics(mask: np.ndarray, vs):
    """Skeletonise a binary mask and return
    (length_um, n_branches, n_endpoints, path_voxels) where ``path_voxels`` is
    the ordered list of (z,y,x) along the longest geodesic path (base→tip)."""
    skel = skeletonize(mask)
    pts = np.argwhere(skel)
    if len(pts) == 0:
        return 0.0, 0, 0, np.empty((0, 3), int)
    if len(pts) == 1:
        return 0.0, 0, 1, pts

    index = {tuple(p): i for i, p in enumerate(pts)}
    vs = np.asarray(vs, float)
    rows, cols, data = [], [], []
    degree = np.zeros(len(pts), int)
    pset = set(index)
    for i, p in enumerate(pts):
        for d in _NB:
            q = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
            if q in pset:
                j = index[q]
                degree[i] += 1
                if j > i:
                    w = float(np.sqrt(((d * vs) ** 2).sum()))
                    rows += [i, j]; cols += [j, i]; data += [w, w]
    n_endpoints = int(np.count_nonzero(degree == 1))
    n_branches = int(np.count_nonzero(degree > 2))

    graph = csr_matrix((data, (rows, cols)), shape=(len(pts), len(pts)))
    # longest geodesic: two Dijkstra passes (double-sweep)
    seeds = np.where(degree == 1)[0]
    src = int(seeds[0]) if len(seeds) else 0
    d0 = dijkstra(graph, indices=src)
    d0[~np.isfinite(d0)] = -1
    a = int(np.argmax(d0))
    da = dijkstra(graph, indices=a)
    da[~np.isfinite(da)] = -1
    b = int(np.argmax(da))
    length = float(da[b])

    # reconstruct the a→b path via a predecessor pass
    _, preds = dijkstra(graph, indices=a, return_predecessors=True)
    path = []
    node = b
    while node != -9999 and node >= 0:
        path.append(node)
        if node == a:
            break
        node = preds[node]
    path_voxels = pts[path[::-1]] if path else np.empty((0, 3), int)
    return length, n_branches, n_endpoints, path_voxels


def _intensity_stats(values: np.ndarray, prefix: str) -> dict:
    if values.size == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_min": np.nan,
                f"{prefix}_max": np.nan, f"{prefix}_sd": np.nan,
                f"{prefix}_integrated": np.nan}
    v = values.astype(float)
    return {
        f"{prefix}_mean": float(v.mean()),
        f"{prefix}_min": float(v.min()),
        f"{prefix}_max": float(v.max()),
        f"{prefix}_sd": float(v.std()),
        f"{prefix}_integrated": float(v.sum()),
    }


# ── main entry point ────────────────────────────────────────────────────────────
def quantify_cilia(
    labels: np.ndarray,
    voxel_size=(1.0, 1.0, 1.0),
    intensity_images: dict | None = None,
    coloc_image: np.ndarray | None = None,
    coloc_threshold: float | None = None,
    base_ref_points: np.ndarray | None = None,
    min_voxels: int = 1,
    file: str = "",
    profiles_out: dict | None = None,
) -> pd.DataFrame:
    """Quantify every labelled cilium.

    Parameters
    ----------
    labels : (Z,Y,X) int label image (0 = background).
    voxel_size : (vz, vy, vx) in µm.
    intensity_images : {name: (Z,Y,X) array}. The first entry is treated as the
        primary (reconstruction) channel; all are measured per cilium.
    coloc_image : optional (Z,Y,X) channel for colocalisation.
    coloc_threshold : threshold for ``coloc_image``; if None uses its mean+2·SD.
    base_ref_points : optional (N,3) array of reference points (e.g. basal-body
        centroids, in voxel coords) used to orient each cilium base→tip.
    min_voxels : ignore cilia smaller than this.
    file : filename stored in the output rows.
    profiles_out : if a dict is supplied, the base→tip intensity profile of each
        cilium (per channel) is stored under ``profiles_out[cilia_id]``.

    Returns
    -------
    pandas.DataFrame, one row per cilium.
    """
    labels = np.asarray(labels)
    vs = tuple(float(s) for s in voxel_size)
    vox_um3 = float(np.prod(vs))
    intensity_images = intensity_images or {}
    chan_names = list(intensity_images)
    primary = chan_names[0] if chan_names else None

    if coloc_image is not None and coloc_threshold is None:
        _ci = np.asarray(coloc_image, float)
        coloc_threshold = float(_ci.mean() + 2.0 * _ci.std())

    objects = ndi.find_objects(labels)
    rows = []
    for lab_id, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        sub = labels[sl] == lab_id
        n_vox = int(sub.sum())
        if n_vox < min_voxels:
            continue

        # coordinates (global) and calibrated cloud
        local = np.argwhere(sub)
        offset = np.array([s.start for s in sl])
        gcoords = local + offset
        coords_um = gcoords * np.array(vs)
        centroid = gcoords.mean(0)

        # morphology
        volume_um3 = n_vox * vox_um3
        surface_um2 = _surface_area(sub, vs)
        sphere_r = float((3.0 * volume_um3 / (4.0 * np.pi)) ** (1.0 / 3.0))
        sphere_surface = 4.0 * np.pi * sphere_r ** 2
        shape_cx = float(surface_um2 / sphere_surface) if sphere_surface > 0 else np.nan
        max_span = _max_span(coords_um)
        orient = _orientation(coords_um)

        # skeleton / length / bending
        length_um, n_branches, n_endpoints, path_vox = _skeleton_metrics(sub, vs)
        if len(path_vox) >= 2:
            ends_um = path_vox[[0, -1]] * np.array(vs) + 0  # local µm
            straight = float(np.sqrt(((ends_um[0] - ends_um[1]) ** 2).sum()))
            bending = float(length_um / straight) if straight > 1e-9 else np.nan
        else:
            bending = np.nan

        row = {
            "filename": file,
            "cilia_id": lab_id,
            "coords": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
            "volume_voxels": n_vox,
            "volume_um3": volume_um3,
            "surface_um2": surface_um2,
            "sphere_radius_um": sphere_r,
            "shape_complexity": shape_cx,
            "max_span_um": max_span,
            "length_um": length_um,
            "n_branches": n_branches,
            "n_endpoints": n_endpoints,
            "bending_index": bending,
            "orientation_z": orient[0],
            "orientation_y": orient[1],
            "orientation_x": orient[2],
        }

        # intensity per channel
        for name, img in intensity_images.items():
            vals = np.asarray(img)[sl][sub]
            row.update(_intensity_stats(vals, name))

        # base→tip intensity profile along the longest path
        if len(path_vox) >= 2:
            gpath = path_vox + offset
            # orient base→tip using the nearest reference point if available
            if base_ref_points is not None and len(base_ref_points):
                d_start = np.min(((gpath[0] * vs - base_ref_points * vs) ** 2).sum(1))
                d_end = np.min(((gpath[-1] * vs - base_ref_points * vs) ** 2).sum(1))
                if d_end < d_start:
                    gpath = gpath[::-1]
            if profiles_out is not None and primary is not None:
                prof = {}
                for name, img in intensity_images.items():
                    arr = np.asarray(img)
                    prof[name] = [float(arr[z, y, x]) for z, y, x in gpath]
                profiles_out[lab_id] = prof

        # colocalisation
        if coloc_image is not None:
            cv = np.asarray(coloc_image)[sl][sub]
            coloc_vox = int(np.count_nonzero(cv >= coloc_threshold))
            row["coloc_volume_um3"] = coloc_vox * vox_um3
            row["coloc_pct"] = float(100.0 * coloc_vox / n_vox) if n_vox else np.nan

        rows.append(row)

    return pd.DataFrame(rows)
