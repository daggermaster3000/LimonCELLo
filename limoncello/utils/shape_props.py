"""Per-cilium 3-D shape descriptors (anisotropic-voxel aware).

Returns, per label id, a dict of morphology features for downstream analysis:
volume (voxels + µm³), surface area (µm²), sphericity, major/minor axis length,
elongation, solidity, extent and equivalent diameter.
"""
from __future__ import annotations

import numpy as np

# Columns produced by :func:`cilia_shape_props` (stable order for dataframes).
SHAPE_COLS = [
    "volume_voxels", "volume_um3", "surface_area_um2", "sphericity",
    "length_um", "width_um", "elongation", "solidity", "extent",
    "equivalent_diameter_um",
]


def _voxel_face_area(mask: np.ndarray, vs) -> float:
    """Exposed-face (staircase) surface area of a 3-D binary mask. Robust but
    overestimates curved surfaces (~0.66 sphericity for a digital ball)."""
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


def _surface_area(mask: np.ndarray, vs) -> float:
    """Surface area (µm²). Prefer a smooth marching-cubes mesh (accurate
    sphericity); fall back to the voxel-face staircase if that fails (e.g. flat
    or 1-voxel-thick objects)."""
    try:
        from skimage.measure import marching_cubes, mesh_surface_area
        padded = np.pad(mask.astype(bool), 1)            # close boundary faces
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=tuple(vs))
        area = float(mesh_surface_area(verts, faces))
        if np.isfinite(area) and area > 0:
            return area
    except Exception:                                     # noqa: BLE001
        pass
    return _voxel_face_area(mask, vs)


def cilia_shape_props(labels, voxel_size) -> dict[int, dict]:
    """Map ``{label_id: {feature: value, …}}`` for a 3-D label volume.

    Sphericity is the isoperimetric ratio ``π^(1/3) · (6V)^(2/3) / A`` (1.0 for a
    perfect sphere). All physical quantities use the anisotropic ``voxel_size``
    ``(vz, vy, vx)`` in µm.
    """
    from skimage.measure import regionprops

    lab = np.asarray(labels).astype(np.int32)
    out: dict[int, dict] = {}
    if lab.size == 0 or lab.max() == 0:
        return out
    spacing = tuple(float(s) for s in voxel_size)
    vox_um3 = float(np.prod(spacing))

    def _safe(region, attr):
        try:
            return float(getattr(region, attr))
        except Exception:                                 # noqa: BLE001
            return float("nan")

    try:
        regions = regionprops(lab, spacing=spacing)
    except Exception as exc:                              # noqa: BLE001
        print(f"[LC] regionprops failed ({exc}); voxel counts only.")
        counts = np.bincount(lab.ravel())
        for lid in range(1, len(counts)):
            if counts[lid]:
                out[int(lid)] = {c: float("nan") for c in SHAPE_COLS}
                out[int(lid)]["volume_voxels"] = int(counts[lid])
                out[int(lid)]["volume_um3"] = counts[lid] * vox_um3
        return out

    for r in regions:
        n = int(r.num_pixels)
        vol_um3 = n * vox_um3
        try:
            area = _surface_area(r.image, spacing)
        except Exception:                                 # noqa: BLE001
            area = float("nan")
        sphericity = (
            float(np.pi ** (1 / 3) * (6.0 * vol_um3) ** (2 / 3) / area)
            if np.isfinite(area) and area > 0 else float("nan")
        )
        major = _safe(r, "axis_major_length")
        minor = _safe(r, "axis_minor_length")
        out[int(r.label)] = {
            "volume_voxels": n,
            "volume_um3": vol_um3,
            "surface_area_um2": area,
            "sphericity": sphericity,
            "length_um": major,
            "width_um": minor,
            "elongation": (major / minor if np.isfinite(minor) and minor > 0 else float("nan")),
            "solidity": _safe(r, "solidity"),
            "extent": _safe(r, "extent"),
            "equivalent_diameter_um": _safe(r, "equivalent_diameter_area"),
        }
    return out
