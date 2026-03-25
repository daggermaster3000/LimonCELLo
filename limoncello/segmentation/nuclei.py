from __future__ import annotations

import pyclesperanto_prototype as cle
from ..utils.gpu import to_gpu, to_cpu


def segment_nuclei(
    volume,
    tophat_radius=(2.0, 2.0, 2.0),
    spot_sigma: float = 5.0,
    outline_sigma: float = 1.0,
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
    **kwargs :
        Additional arguments passed to cle.voronoi_otsu_labeling

    Returns
    -------
    labels : np.ndarray
        Labeled nuclei (CPU)
    """
    volume_gpu = to_gpu(volume)

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