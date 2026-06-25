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


# APOC feature operations selectable for training (besides the raw "original").
FEATURE_OPERATIONS = [
    "gaussian_blur",
    "difference_of_gaussian",
    "laplace_box_of_gaussian_blur",
    "sobel_of_gaussian_blur",
]


def build_feature_spec(operations, sigmas, include_original: bool = True) -> str:
    """Build an APOC ``feature_specification`` string from selected operations and
    sigma scales, e.g. ``"gaussian_blur=1 difference_of_gaussian=1 … original"``.

    Falls back to :data:`DEFAULT_FEATURES` if nothing usable is selected.
    """
    ops = [o for o in operations if o in FEATURE_OPERATIONS]
    parts = [f"{op}={s:g}" for s in sigmas for op in ops]
    if include_original:
        parts.append("original")
    return " ".join(parts) if (ops and parts) else DEFAULT_FEATURES


def feature_spec_from_pairs(pairs, include_original: bool = True) -> str:
    """Build an APOC ``feature_specification`` from explicit ``(operation, sigma)``
    pairs (e.g. from a selection grid), e.g. ``[("gaussian_blur", 1), …]`` →
    ``"gaussian_blur=1 …"``. Falls back to :data:`DEFAULT_FEATURES` if empty."""
    parts = [f"{op}={float(s):g}" for op, s in pairs if op in FEATURE_OPERATIONS]
    if include_original:
        parts.append("original")
    # "original" alone isn't a usable feature set — require at least one filter.
    return " ".join(parts) if any(op in FEATURE_OPERATIONS for op, _ in pairs) else DEFAULT_FEATURES


def read_feature_importances(cl_path) -> list[tuple[str, float]]:
    """Parse a trained ``.cl`` header into ``[(feature, importance), …]`` sorted by
    importance (descending). Returns ``[]`` if the fields are missing."""
    spec = imps = None
    try:
        with open(cl_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("feature_specification"):
                    spec = line.split("=", 1)[1].split()
                elif line.startswith("feature_importances"):
                    imps = [float(x) for x in line.split("=", 1)[1].split(",") if x.strip()]
                if spec is not None and imps is not None:
                    break
    except Exception:                                     # noqa: BLE001
        return []
    if not spec or not imps:
        return []
    n = min(len(spec), len(imps))
    rows = list(zip(spec[:n], imps[:n]))
    rows.sort(key=lambda t: t[1], reverse=True)
    return rows


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


def _preprocess_geometry(raw_zyx, voxel_size, *, use_mip: bool, make_iso: bool):
    """Apply the same geometry transforms the GUI uses at load time so the raw
    training image matches the resolution the annotation was painted on.

    Mirrors ``napari_app.step_load``: MIP collapses Z; otherwise, anisotropic
    voxels are resampled to isotropic (coarsest spacing). Intensities are *not*
    normalised (training runs on raw values)."""
    raw_zyx = np.asarray(raw_zyx)
    if use_mip:
        return np.max(raw_zyx, axis=0, keepdims=True)
    if make_iso and voxel_size is not None:
        iso = max(voxel_size)
        if not all(abs(v - iso) < 1e-6 for v in voxel_size):
            from ..preprocessing.preprocessing import make_isotropic
            return make_isotropic(raw_zyx, voxel_size, iso)[0]
    return raw_zyx


def load_training_pairs(image_paths, labels_dir, channel: int,
                        labels_suffix: str = "_labels",
                        *, use_mip: bool = False, make_iso: bool = False):
    """
    Build ``[(raw_channel_image, ground_truth), …]`` for training.

    For each ``.ims`` path, loads the requested raw ``channel`` (no
    normalisation), applies the same MIP / isotropic-resampling the GUI used
    when the annotation was painted (so shapes match), and pairs it with the
    matching ``<stem><labels_suffix>.tif``. Single-Z stacks are squeezed to
    2-D. Pairs whose label file is missing or whose shape disagrees with the
    (preprocessed) image are skipped.

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

        img, meta = load_image(p)
        n_ch = img.shape[1]
        ch = int(np.clip(channel, 0, n_ch - 1))
        voxel_size = (meta or {}).get("voxel_size") or None
        raw_zyx = _preprocess_geometry(
            np.asarray(img[0, ch]), voxel_size, use_mip=use_mip, make_iso=make_iso)
        raw = _squeeze_single_z(raw_zyx)
        gt = _squeeze_single_z(_read_labels(lp))

        if gt.shape != raw.shape:
            skipped.append((stem, f"labels shape {gt.shape} != image {raw.shape} "
                                  "(check MIP / 'make isotropic' match how you "
                                  "annotated)"))
            continue
        pairs.append((raw, gt))

    return pairs, skipped


def train_segmenter_single(
    image,
    ground_truth,
    output_cl,
    *,
    features: str | None = None,
    continue_training: bool = False,
    positive_class: int = 2,
    max_depth: int = 2,
    num_trees: int = 100,
    gpu_device: str | None = None,
) -> str:
    """Train (or extend) an APOC ``ObjectSegmenter`` from a single in-viewer
    annotation, the way the napari-apoc plugin does.

    With ``continue_training=False`` a fresh classifier is written to
    ``output_cl`` (any existing file is replaced). With ``continue_training=True``
    the annotation is *added* to the existing classifier at ``output_cl`` so you
    can paint on several images in turn and accumulate one model.
    """
    from apoc import ObjectSegmenter

    if gpu_device:
        import pyclesperanto_prototype as cle
        cle.select_device(gpu_device)

    feats = features or DEFAULT_FEATURES
    out = Path(output_cl)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not continue_training:
        out.unlink()                                  # start fresh

    image = _squeeze_single_z(np.asarray(image))
    ground_truth = _squeeze_single_z(np.asarray(ground_truth))
    if ground_truth.shape != image.shape:
        raise ValueError(
            f"Annotation shape {ground_truth.shape} != image shape {image.shape}.")
    if int(np.max(ground_truth)) < positive_class:
        raise ValueError(
            f"Annotation has no '{positive_class}' (object) voxels — paint "
            f"background=1 and object={positive_class} before training.")

    segmenter = ObjectSegmenter(
        opencl_filename=str(out),
        positive_class_identifier=positive_class,
        max_depth=max_depth,
        num_ensembles=num_trees,
    )
    segmenter.train(feats, ground_truth, image,
                    continue_training=bool(continue_training) and out.exists())
    return str(out)


def predict_segmenter(image, cl_path, *, gpu_device: str | None = None) -> np.ndarray:
    """Run a trained ``ObjectSegmenter`` (``.cl``) on ``image`` and return the
    raw object-label result — the napari-apoc 'preview' (no size filtering)."""
    from apoc import ObjectSegmenter

    if gpu_device:
        import pyclesperanto_prototype as cle
        cle.select_device(gpu_device)
    seg = ObjectSegmenter(opencl_filename=str(cl_path))
    image = _squeeze_single_z(np.asarray(image))
    return np.asarray(seg.predict(image=image))


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
