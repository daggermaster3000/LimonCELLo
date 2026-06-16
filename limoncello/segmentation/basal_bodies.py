from __future__ import annotations

import numpy as np
import pyclesperanto_prototype as cle
from ..utils.gpu import to_gpu
from .cilia import _size_filter


def segment_basal_bodies(
    volume,
    spot_sigma: float = 2.0,
    outline_sigma: float = 2.0,
    gaussian_sigma=(1.0, 1.0, 0.0),
    min_size: int = 5,
    max_size: int = 0,
    log_transform: bool = False,
    **kwargs,
):
    """
    Segment basal bodies using Gaussian blur + Voronoi-Otsu.

    Parameters
    ----------
    volume : np.ndarray
        3D image (Z, Y, X)
    spot_sigma : float
        Object separation parameter.
    outline_sigma : float
        Boundary precision parameter.
    gaussian_sigma : tuple
        Gaussian blur sigma (sz, sy, sx) applied before Voronoi-Otsu.
    min_size : int
        Remove objects smaller than this (voxels). 0 = disabled.
    max_size : int
        Remove objects larger than this (voxels). 0 = disabled.
    log_transform : bool
        If True, apply a log1p intensity transform before all other
        processing (compresses dynamic range). Default False.

    Returns
    -------
    labels_gpu : cle.Image
        Labeled basal bodies (GPU).
    """
    vol = np.asarray(volume)
    if log_transform:
        vol = np.log1p(vol.astype(np.float32))
    volume_gpu = to_gpu(vol)

    if any(s > 0 for s in gaussian_sigma):
        blurred_gpu = cle.gaussian_blur(
            volume_gpu,
            sigma_x=gaussian_sigma[2],
            sigma_y=gaussian_sigma[1],
            sigma_z=gaussian_sigma[0],
        )
    else:
        blurred_gpu = volume_gpu

    labels_gpu = cle.voronoi_otsu_labeling(
        blurred_gpu,
        spot_sigma=spot_sigma,
        outline_sigma=outline_sigma,
        **kwargs,
    )

    return _size_filter(labels_gpu, min_size, max_size)
