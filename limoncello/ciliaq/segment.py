"""CiliaQ-style intensity-threshold segmentation of a cilia channel.

Workflow (mirrors CiliaQ_Preparator):
    optional background subtraction (white top-hat)
    → Gaussian blur
    → automatic intensity threshold (Otsu / Li / Triangle / Yen / Mean) or
      hysteresis thresholding
    → 3-D connected-component labelling (26-connectivity)
    → size filtering
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology

_THRESHOLD_FUNCS = {
    "otsu": filters.threshold_otsu,
    "li": filters.threshold_li,
    "triangle": filters.threshold_triangle,
    "yen": filters.threshold_yen,
    "mean": filters.threshold_mean,
    "isodata": filters.threshold_isodata,
}

THRESHOLD_METHODS = list(_THRESHOLD_FUNCS) + ["hysteresis"]


def auto_threshold_value(image, method: str = "otsu") -> float:
    """Return the automatic threshold value for an image using ``method``."""
    fn = _THRESHOLD_FUNCS.get(method.lower())
    if fn is None:
        fn = filters.threshold_otsu
    return float(fn(np.asarray(image)))


def segment_cilia_threshold(
    volume,
    method: str = "otsu",
    gaussian_sigma=(1.0, 1.0, 1.0),
    background_radius: float = 0.0,
    threshold_factor: float = 1.0,
    hysteresis_low_factor: float = 0.5,
    min_voxels: int = 10,
    max_voxels: int = 0,
    connectivity: int = 3,
):
    """Segment cilia by intensity thresholding + 3-D connected components.

    Parameters
    ----------
    volume : (Z, Y, X) image.
    method : threshold method (see ``THRESHOLD_METHODS``).
    gaussian_sigma : (sz, sy, sx) blur applied before thresholding. (0,0,0)=off.
    background_radius : white-top-hat radius (voxels) for background subtraction.
        0 disables it.
    threshold_factor : multiply the automatic threshold (>1 stricter).
    hysteresis_low_factor : for ``method='hysteresis'``, low threshold as a
        fraction of the high (Otsu) threshold.
    min_voxels / max_voxels : drop objects outside this voxel-count range
        (max_voxels = 0 disables the upper bound).
    connectivity : 1 (faces), 2 (edges) or 3 (corners, 26-conn) for labelling.

    Returns
    -------
    labels : (Z, Y, X) int32 label image.
    """
    vol = np.asarray(volume).astype(np.float32)

    if background_radius and background_radius > 0:
        footprint = morphology.ball(int(background_radius))
        vol = morphology.white_tophat(vol, footprint)

    if any(s > 0 for s in gaussian_sigma):
        vol = ndi.gaussian_filter(
            vol, sigma=(gaussian_sigma[0], gaussian_sigma[1], gaussian_sigma[2])
        )

    if method.lower() == "hysteresis":
        high = auto_threshold_value(vol, "otsu") * threshold_factor
        low = high * float(hysteresis_low_factor)
        mask = filters.apply_hysteresis_threshold(vol, low, high)
    else:
        thr = auto_threshold_value(vol, method) * float(threshold_factor)
        mask = vol > thr

    structure = ndi.generate_binary_structure(3, connectivity)
    labels, _ = ndi.label(mask, structure=structure)

    labels = _size_filter(labels, min_voxels, max_voxels)
    return labels.astype(np.int32)


def _size_filter(labels, min_voxels, max_voxels):
    if labels.max() == 0:
        return labels
    counts = np.bincount(labels.ravel())
    remove = np.zeros(counts.shape[0], dtype=bool)
    if min_voxels > 0:
        remove |= counts < min_voxels
    if max_voxels and max_voxels > 0:
        remove |= counts > max_voxels
    remove[0] = False
    if remove.any():
        labels = labels.copy()
        labels[remove[labels]] = 0
        # relabel to keep ids contiguous
        labels, _ = ndi.label(labels > 0,
                              structure=ndi.generate_binary_structure(3, 3))
    return labels
