from __future__ import annotations

import numpy as np
import pyclesperanto_prototype as cle
from skimage.morphology import skeletonize
from ..utils.gpu import to_gpu, to_cpu


def segment_neurites(
    volume,
    min_size: int = 50,
    spot_sigma: float = 10.0,
    outline_sigma: float = 1.0,
    gaussian_sigma=(0.0, 0.0, 0.0),
    log_transform: bool = False,
    **kwargs,
):
    """
    Segment neurites and compute skeleton.

    Parameters
    ----------
    volume : np.ndarray
        3D image (Z, Y, X)
    min_size : int
        Minimum object size
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
    skeleton : np.ndarray
        Binary skeleton (CPU)
    labels_gpu : cle.Image
        Labeled neurites (GPU)
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

    # labels_gpu = cle.voronoi_otsu_labeling(
    #     volume_gpu,
    #     spot_sigma=spot_sigma,
    #     outline_sigma=outline_sigma,
    #     **kwargs,
    # )
    labels_gpu = cle.gauss_otsu_labeling(
                volume_gpu,
        outline_sigma=outline_sigma,
        **kwargs,
    )
    labels_gpu = cle.exclude_small_labels(
        labels_gpu,
        maximum_size=min_size,
    )

    binary = to_cpu(labels_gpu) > 0
    skeleton = skeletonize(binary)

    return skeleton, labels_gpu