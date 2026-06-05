from __future__ import annotations

import numpy as np
import pyclesperanto_prototype as cle
from apoc import ObjectSegmenter
from ..utils.gpu import to_gpu


def segment_cilia_ml(
    volume,
    classifier_path: str,
    min_size: int = 20,
    **kwargs,
):
    """
    Segment cilia using a trained APOC classifier.

    Parameters
    ----------
    volume : np.ndarray
        3D image (Z, Y, X) or 2D image (Y, X).
        If Z == 1, the volume is squeezed to 2-D before prediction
        (APOC requires at least 2 Z-slices for 3-D feature extraction).
    classifier_path : str
        Path to APOC classifier
    min_size : int
        Minimum object size

    Returns
    -------
    labels_gpu : cle.Image
        Labeled cilia image (GPU), same dimensionality as input.
    """
    vol = np.asarray(volume)

    # APOC 3-D feature extraction needs >= 2 Z-slices; squeeze single-Z to 2-D
    single_z = vol.ndim == 3 and vol.shape[0] == 1
    if single_z:
        vol = vol[0]  # (Y, X)

    segmenter = ObjectSegmenter(opencl_filename=classifier_path)
    labels = segmenter.predict(image=vol)
    labels_gpu = to_gpu(labels)
    labels_gpu = cle.exclude_small_labels(labels_gpu, maximum_size=min_size)

    if single_z:
        # Restore Z dimension so downstream 3-D code works unchanged
        labels_gpu = cle.push(np.asarray(labels_gpu)[np.newaxis])  # (1, Y, X)

    return labels_gpu
