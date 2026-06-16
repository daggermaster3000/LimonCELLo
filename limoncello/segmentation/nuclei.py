from __future__ import annotations

import numpy as np
import pyclesperanto_prototype as cle
from ..utils.gpu import to_gpu, to_cpu


def segment_nuclei(
    volume,
    tophat_radius=(2.0, 2.0, 2.0),
    spot_sigma: float = 5.0,
    outline_sigma: float = 1.0,
    gaussian_sigma=(0.0, 0.0, 0.0),
    log_transform: bool = False,
    **kwargs,
):
    """
    Segment nuclei using top-hat filtering + Voronoi-Otsu.

    Parameters
    ----------
    volume : np.ndarray
        3D image (Z, Y, X)
    tophat_radius : tuple
        Radius for top-hat filtering (rx, ry, rz)
    spot_sigma : float
        Object separation parameter
    outline_sigma : float
        Boundary precision parameter
    gaussian_sigma : tuple
        Gaussian blur sigma (sz, sy, sx) applied before processing.
        (0, 0, 0) = disabled.
    log_transform : bool
        If True, apply a log1p intensity transform before all other
        processing (compresses dynamic range). Default False.
    **kwargs :
        Additional arguments passed to cle.voronoi_otsu_labeling

    Returns
    -------
    labels : np.ndarray
        Labeled nuclei (CPU)
    """
    vol = np.asarray(volume)
    if log_transform:
        vol = np.log1p(vol.astype(np.float32))
    volume_gpu = to_gpu(vol)

    if any(s > 0 for s in gaussian_sigma):
        volume_gpu = cle.gaussian_blur(
            volume_gpu,
            sigma_x=gaussian_sigma[2],
            sigma_y=gaussian_sigma[1],
            sigma_z=gaussian_sigma[0],
        )

    th_gpu = cle.top_hat_sphere(
        volume_gpu,
        radius_x=tophat_radius[2],
        radius_y=tophat_radius[1],
        radius_z=tophat_radius[0],
    )

    labels_gpu = cle.voronoi_otsu_labeling(
        th_gpu,
        spot_sigma=spot_sigma,
        outline_sigma=outline_sigma,
        **kwargs,
    )

    return to_cpu(labels_gpu)