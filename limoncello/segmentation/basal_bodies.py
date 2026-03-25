from __future__ import annotations

import pyclesperanto_prototype as cle
from ..utils.gpu import to_gpu


def segment_basal_bodies(
    volume,
    spot_sigma: float = 2.0,
    outline_sigma: float = 2.0,
    gaussian_sigma=(1.0, 1.0, 0.0),
    **kwargs,
):
    """
    Segment basal bodies using Gaussian blur + Voronoi-Otsu.

    Parameters
    ----------
    volume : np.ndarray
        3D image (Z, Y, X)
    spot_sigma : float
        Object separation parameter
    outline_sigma : float
        Boundary precision parameter
    gaussian_sigma : tuple
        Gaussian blur sigma (sx, sy, sz)
    **kwargs :
        Additional arguments passed to cle.voronoi_otsu_labeling

    Returns
    -------
    labels_gpu : cle.Image
        Labeled basal bodies (GPU)
    """
    volume_gpu = to_gpu(volume)

    blurred_gpu = cle.gaussian_blur(
        volume_gpu,
        sigma_x=gaussian_sigma[2],
        sigma_y=gaussian_sigma[1],
        sigma_z=gaussian_sigma[0],
    )

    labels_gpu = cle.voronoi_otsu_labeling(
        blurred_gpu,
        spot_sigma=spot_sigma,
        outline_sigma=outline_sigma,
        **kwargs,
    )

    return labels_gpu