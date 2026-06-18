"""
Train an APOC ``ObjectSegmenter`` from sparse annotations across several images.

Design notes
------------
* Training runs on **raw** intensities (no percentile normalisation), matching
  how :func:`limoncello.segmentation.cilia.segment_cilia_ml` and
  :func:`limoncello.segmentation.basal_bodies.segment_basal_bodies_ml` *predict*
  (they read ``state["raw"]``). Normalising here would shift the feature space
  away from prediction.
* Ground truth is the APOC sparse-label convention: a label image where
  annotated background = 1, annotated object = 2 (``positive_class``), and
  unannotated voxels = 0 (ignored by the trainer).
* Dimensionality follows each image: a single-Z stack ``(1, Y, X)`` is squeezed
  to 2-D before training, exactly like the prediction path. All images in one
  training run must share the same dimensionality (you cannot mix 2-D and 3-D
  into a single classifier).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# Feature specification copied from the shipped ``segmenters/cilia-segmenter.cl``
# so classifiers trained here use the same 41-feature space as the existing ones.
DEFAULT_FEATURES = (
    "gaussian_blur=1 difference_of_gaussian=1 laplace_box_of_gaussian_blur=1 "
    "sobel_of_gaussian_blur=1 gaussian_blur=0.3 gaussian_blur=0.5 "
    "difference_of_gaussian=0.3 difference_of_gaussian=0.5 "
    "laplace_box_of_gaussian_blur=0.3 laplace_box_of_gaussian_blur=0.5 "
    "sobel_of_gaussian_blur=0.3 sobel_of_gaussian_blur=0.5 gaussian_blur=2 "
    "difference_of_gaussian=2 gaussian_blur=3 gaussian_blur=4 gaussian_blur=5 "
    "difference_of_gaussian=5 difference_of_gaussian=4 difference_of_gaussian=3 "
    "laplace_box_of_gaussian_blur=3 laplace_box_of_gaussian_blur=2 "
    "sobel_of_gaussian_blur=2 sobel_of_gaussian_blur=3 "
    "laplace_box_of_gaussian_blur=4 sobel_of_gaussian_blur=4 "
    "laplace_box_of_gaussian_blur=5 sobel_of_gaussian_blur=5 "
    "difference_of_gaussian=10 gaussian_blur=10 laplace_box_of_gaussian_blur=10 "
    "sobel_of_gaussian_blur=10 sobel_of_gaussian_blur=15 "
    "laplace_box_of_gaussian_blur=15 difference_of_gaussian=15 gaussian_blur=15 "
    "original difference_of_gaussian=25 gaussian_blur=25 "
    "laplace_box_of_gaussian_blur=25 sobel_of_gaussian_blur=25"
)


def _read_labels(path) -> np.ndarray:
    """Read a label/annotation image, preferring tifffile, falling back to skimage."""
    try:
        import tifffile
        return np.asarray(tifffile.imread(str(path)))
    except Exception:
        from skimage.io import imread
        return np.asarray(imread(str(path)))


def find_labels_file(labels_dir, stem: str, suffix: str = "_labels"):
    """Locate ``<stem><suffix>.tif`` in ``labels_dir`` (then ``.tiff``, then any
    tif whose name contains ``stem``). Returns a Path or None."""
    d = Path(labels_dir)
    for ext in (".tif", ".tiff"):
        cand = d / f"{stem}{suffix}{ext}"
        if cand.exists():
            return cand
    for ext in (".tif", ".tiff"):
        for f in sorted(d.glob(f"*{ext}")):
            if stem in f.stem:
                return f
    return None


def _squeeze_single_z(arr: np.ndarray) -> np.ndarray:
    """(1, Y, X) → (Y, X); leave everything else untouched (matches prediction)."""
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return arr


def load_training_pairs(image_paths, labels_dir, channel: int,
                        labels_suffix: str = "_labels"):
    """
    Build ``[(raw_channel_image, ground_truth), …]`` for training.

    For each ``.ims`` path, loads the requested raw ``channel`` (no
    normalisation) and the matching ``<stem><labels_suffix>.tif``. Single-Z
    stacks are squeezed to 2-D. Pairs whose label file is missing or whose
    shape disagrees with the image are skipped.

    Returns
    -------
    (pairs, skipped) : (list[tuple[np.ndarray, np.ndarray]], list[tuple[str, str]])
        ``skipped`` holds ``(stem, reason)`` for anything left out.
    """
    from ..utils.reader import load_image

    pairs, skipped = [], []
    for p in image_paths:
        stem = Path(p).stem
        lp = find_labels_file(labels_dir, stem, labels_suffix)
        if lp is None:
            skipped.append((stem, "no matching labels .tif"))
            continue

        img, _ = load_image(p)
        n_ch = img.shape[1]
        ch = int(np.clip(channel, 0, n_ch - 1))
        raw = _squeeze_single_z(np.asarray(img[0, ch]))
        gt = _squeeze_single_z(_read_labels(lp))

        if gt.shape != raw.shape:
            skipped.append((stem, f"labels shape {gt.shape} != image {raw.shape}"))
            continue
        pairs.append((raw, gt))

    return pairs, skipped


def train_object_segmenter(
    pairs,
    output_cl,
    *,
    features: str | None = None,
    positive_class: int = 2,
    max_depth: int = 2,
    num_trees: int = 100,
    gpu_device: str | None = None,
    progress_callback=None,
) -> str:
    """
    Train one APOC ``ObjectSegmenter`` across many ``(image, ground_truth)`` pairs.

    Each pair is trained incrementally (``continue_training``) into a single
    ``.cl`` so the classifier sees every annotated image. Images are used as-is
    (raw intensities); the caller is responsible for not normalising them.

    Parameters
    ----------
    pairs : list of (np.ndarray, np.ndarray)
        (raw image, sparse ground truth) pairs, all the same dimensionality.
    output_cl : str or Path
        Destination ``.cl`` file (overwritten if it exists).
    features : str, optional
        APOC feature specification. Defaults to :data:`DEFAULT_FEATURES`.
    positive_class : int
        Label value marking the object (foreground) in the ground truth.
    max_depth, num_trees : int
        Random-forest depth and ensemble size.
    gpu_device : str, optional
        cle device name to select before training.

    Returns
    -------
    str
        The path to the written classifier.
    """
    from apoc import ObjectSegmenter

    if not pairs:
        raise ValueError("No (image, ground_truth) pairs to train on.")

    dims = {img.ndim for img, _ in pairs}
    if len(dims) > 1:
        raise ValueError(
            f"Mixed image dimensionalities {sorted(dims)} in one training run. "
            "Train 2-D and 3-D classifiers separately (e.g. enable MIP for every "
            "image, or use only multi-Z stacks)."
        )

    if gpu_device:
        import pyclesperanto_prototype as cle
        cle.select_device(gpu_device)

    feats = features or DEFAULT_FEATURES
    out = Path(output_cl)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()                                  # start fresh, don't append to an old model

    segmenter = ObjectSegmenter(
        opencl_filename=str(out),
        positive_class_identifier=positive_class,
        max_depth=max_depth,
        num_ensembles=num_trees,
    )

    n = len(pairs)
    for i, (image, ground_truth) in enumerate(pairs):
        segmenter.train(
            feats,
            np.asarray(ground_truth),
            np.asarray(image),
            continue_training=(i > 0),
        )
        if progress_callback is not None:
            progress_callback(i + 1, n)

    return str(out)
