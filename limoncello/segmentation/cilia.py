from __future__ import annotations

import numpy as np
import pyclesperanto_prototype as cle
from apoc import ObjectSegmenter
from ..utils.gpu import to_gpu


def _size_filter(labels_gpu, min_size: int, max_size: int):
    """Remove labels smaller than min_size or larger than max_size (0 = disabled)."""
    if min_size > 0:
        labels_gpu = cle.exclude_small_labels(labels_gpu, maximum_size=min_size)
    if max_size > 0:
        arr = np.asarray(labels_gpu).astype(np.int32)
        if arr.max() > 0:
            counts = np.bincount(arr.ravel())
            too_large = np.zeros(len(counts), dtype=bool)
            too_large[1:] = counts[1:] > max_size
            arr[too_large[arr]] = 0
            labels_gpu = cle.push(arr)
    return labels_gpu


def segment_cilia_ml(
    volume,
    classifier_path: str,
    min_size: int = 20,
    max_size: int = 0,
    gaussian_sigma=(1.0, 1.0, 1.0),
    log_transform: bool = False,
    **kwargs,
):
    """
    Segment cilia using a trained APOC classifier.

    Parameters
    ----------
    volume : np.ndarray
        3D image (Z, Y, X) or 2D image (Y, X).
        If Z == 1, squeezed to 2-D before prediction
        (APOC requires at least 2 Z-slices for 3-D feature extraction).
    classifier_path : str
        Path to APOC classifier.
    min_size : int
        Remove objects smaller than this (voxels). 0 = disabled.
    max_size : int
        Remove objects larger than this (voxels). 0 = disabled.
    gaussian_sigma : tuple
        Gaussian blur sigma (sz, sy, sx) applied before APOC prediction.
        Set to (0, 0, 0) to disable.
    log_transform : bool
        If True, apply a log1p intensity transform before blur/prediction.
        Note: APOC classifiers are trained on specific intensities — enabling
        this changes the feature space and may degrade a model trained on
        raw data. Default False.

    Returns
    -------
    labels_gpu : cle.Image
        Labeled cilia image (GPU), same dimensionality as input.
    """
    _vol_in = np.asarray(volume)
    if log_transform:
        _vol_in = np.log1p(_vol_in.astype(np.float32))
    volume_gpu = to_gpu(_vol_in)

    if any(s > 0 for s in gaussian_sigma):
        volume_gpu = cle.gaussian_blur(
            volume_gpu,
            sigma_x=gaussian_sigma[2],
            sigma_y=gaussian_sigma[1],
            sigma_z=gaussian_sigma[0],
        )

    vol = np.asarray(volume_gpu)

    # APOC 3-D feature extraction needs >= 2 Z-slices; squeeze single-Z to 2-D
    single_z = vol.ndim == 3 and vol.shape[0] == 1
    if single_z:
        vol = vol[0]  # (Y, X)

    segmenter = ObjectSegmenter(opencl_filename=classifier_path)
    labels = segmenter.predict(image=vol)
    labels_gpu = _size_filter(to_gpu(labels), min_size, max_size)

    if single_z:
        labels_gpu = cle.push(np.asarray(labels_gpu)[np.newaxis])  # (1, Y, X)

    return labels_gpu
