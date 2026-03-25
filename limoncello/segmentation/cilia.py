from __future__ import annotations

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
        3D image (Z, Y, X)
    classifier_path : str
        Path to APOC classifier
    min_size : int
        Minimum object size
    **kwargs :
        Currently unused (reserved for future extensions)

    Returns
    -------
    labels_gpu : cle.Image
        Labeled cilia image (GPU)
    """
    segmenter = ObjectSegmenter(opencl_filename=classifier_path)

    labels = segmenter.predict(image=volume)
    labels_gpu = to_gpu(labels)

    labels_gpu = cle.exclude_small_labels(
        labels_gpu,
        maximum_size=min_size,
    )

    return labels_gpu