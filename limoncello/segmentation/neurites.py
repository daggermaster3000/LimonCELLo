from __future__ import annotations

import pyclesperanto_prototype as cle
from skimage.morphology import skeletonize
from ..utils.gpu import to_gpu, to_cpu


def segment_neurites(
    volume,
    min_size: int = 50,
    spot_sigma: float = 10.0,
    outline_sigma: float = 1.0,
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
    **kwargs :
        Additional arguments passed to cle.voronoi_otsu_labeling

    Returns
    -------
    skeleton : np.ndarray
        Binary skeleton (CPU)
    labels_gpu : cle.Image
        Labeled neurites (GPU)
    """
    volume_gpu = to_gpu(volume)

    labels_gpu = cle.voronoi_otsu_labeling(
        volume_gpu,
        spot_sigma=spot_sigma,
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