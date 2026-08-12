"""
LimonCELLo — napari pipeline app.

Workflow:
  1. Open a folder of .ims files and step through them one at a time.
  2. Run each pipeline step individually (load → segment → distances →
     classify) and verify the result in the viewer before moving on.
  3. Once the parameters look right, run the whole folder in batch mode.

Run with:  python napari_app.py
"""

from __future__ import annotations

import os
import json
import hashlib
import tempfile
import threading
from pathlib import Path

import numpy as np
import napari
from magicgui.widgets import (
    Container, PushButton, ComboBox, FileEdit, Label, CheckBox,
    SpinBox, FloatSpinBox, Select, LineEdit, Table,
)
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel, QFrame,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QProgressBar,
    QApplication, QPushButton, QShortcut,
)
from qtpy.QtGui import QPixmap, QColor, QKeySequence
from qtpy.QtCore import Qt, QTimer, QItemSelectionModel, QObject, Signal
from superqt import QCollapsible
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning
from scipy.ndimage import center_of_mass
from limoncello.utils.gpu_distance import distance_transform_edt
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:                                              # older matplotlib
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from limoncello.utils.reader import load_image
from limoncello.preprocessing.preprocessing import percentile_minmax_normalize, make_isotropic
from limoncello.segmentation.cilia import segment_cilia_ml
from limoncello.segmentation.nuclei import segment_nuclei
from limoncello.segmentation.neurites import segment_neurites
from limoncello.segmentation.basal_bodies import segment_basal_bodies, segment_basal_bodies_ml
from limoncello.segmentation.train import (
    load_training_pairs, train_object_segmenter, find_labels_file,
    train_segmenter_single, predict_segmenter,
    build_feature_spec, feature_spec_from_pairs,
    read_feature_importances, FEATURE_OPERATIONS,
)
from limoncello.utils.assign_label_features import (
    assign_label_features, use_basal_body_ratio)
from limoncello.utils.shape_props import cilia_shape_props, SHAPE_COLS
from limoncello.analysis.pair_cilia_to_bb import (
    pair_from_mixed_df, pair_from_mixed_df_directional)
from limoncello.analysis.pipeline import run_pipeline3, run_roi_only_batch


_DEFAULT_CLASSIFIER = str(
    (Path(__file__).parent / "segmenters" / "Cilia-d38-3D.cl").resolve()
)
_DEFAULT_BB_CLASSIFIER = str(
    (Path(__file__).parent / "segmenters" / "BB-d38-3D.cl").resolve()
)
_DEFAULT_NUCLEI_CLASSIFIER = str(
    (Path(__file__).parent / "segmenters" / "Nuclei-d38-3D.cl").resolve()
)

# Parameter state auto-saved on quit and reloaded on startup.
_STATE_PATH = Path(__file__).parent / ".napari_app_state.json"

# ── Under-the-hood cache for the (slow) isotropic-resampled load ─────────────
# The load step normalises then resamples every channel to isotropic voxels,
# which is the expensive part. We cache its output on disk keyed on the file +
# the load parameters, so re-running load with the same image/normalisation (but
# different *segmentation* parameters) is instant. Non-resampled loads are cheap
# and are not cached.
_LOAD_CACHE_DIR = Path(tempfile.gettempdir()) / "lc_iso_cache"
_LOAD_CACHE_MAX = 40                       # keep at most N cached loads

# Per-cilium XY bounding-box export — same schema/sheet the annotator app and the
# data app's manual validation read, so a table exported here reloads there.
_BOX_SHEET = "cilia_boxes"
_BOX_COLS = [
    "filename", "box_id", "class",
    "y_min", "x_min", "y_max", "x_max",
    "img_height", "img_width",
    "voxel_z", "voxel_y", "voxel_x",
    "mip", "make_isotropic",
]
_BOX_XY_COLS = ["y_min", "x_min", "y_max", "x_max"]   # shown in the property table


def _cilia_bboxes(labels) -> dict:
    """Map ``{label_id: (y_min, x_min, y_max, x_max)}`` — the XY bounding box of
    each cilium projected onto the (max-)MIP grid (inclusive max, like the
    annotator app). Works for 2-D (y,x) or 3-D (z,y,x) label volumes."""
    from skimage.measure import regionprops
    lab = np.asarray(labels)
    if lab.size == 0 or lab.max() == 0:
        return {}
    if lab.dtype != np.int32:
        lab = lab.astype(np.int32)
    out = {}
    for r in regionprops(lab):
        bb = r.bbox
        if lab.ndim == 3:
            _, y0, x0, _, y1, x1 = bb
        else:
            y0, x0, y1, x1 = bb
        out[int(r.label)] = (int(y0), int(x0), int(y1) - 1, int(x1) - 1)
    return out


def _load_cache_key(p: dict, mtime: float) -> str:
    payload = repr({
        "path": p["ims_path"], "mtime": round(mtime, 3),
        "p_low": p["p_low"], "p_high": p["p_high"],
        "ch_norm": sorted((int(k), tuple(v))
                          for k, v in (p.get("ch_norm") or {}).items()),
        "use_mip": bool(p["use_mip"]),
        "make_isotropic": bool(p.get("make_isotropic")),
    })
    return hashlib.md5(payload.encode()).hexdigest()


def _load_cache_read(key: str) -> dict | None:
    path = _LOAD_CACHE_DIR / f"{key}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return dict(
                raw=z["raw"], norm=z["norm"],
                voxel_size=tuple(float(v) for v in z["voxel_size"]),
                n_ch=int(z["n_ch"]),
            )
    except Exception as exc:                                     # noqa: BLE001
        print(f"[LC] Ignoring unreadable load cache: {exc}")
        return None


def _load_cache_write(key: str, result: dict) -> None:
    try:
        _LOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _LOAD_CACHE_DIR / f"{key}.tmp.npz"
        np.savez(
            tmp, raw=result["raw"], norm=result["norm"],
            voxel_size=np.asarray(result["voxel_size"], dtype=float),
            n_ch=np.asarray(result["n_ch"], dtype=int),
        )
        tmp.replace(_LOAD_CACHE_DIR / f"{key}.npz")
        _load_cache_prune()
    except Exception as exc:                                     # noqa: BLE001
        print(f"[LC] Could not write load cache: {exc}")


def _load_cache_prune() -> None:
    """Evict the oldest cached loads once the cap is exceeded."""
    try:
        files = sorted(_LOAD_CACHE_DIR.glob("*.npz"),
                       key=lambda f: f.stat().st_mtime)
        for f in files[:-_LOAD_CACHE_MAX]:
            f.unlink(missing_ok=True)
    except Exception:                                            # noqa: BLE001
        pass

_CH_NAMES     = ["Cilia", "Neurites", "Basal Bodies", "Nuclei"]
_CH_COLORMAPS = ["green", "cyan", "magenta", "blue"]


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE STEPS (pure compute, no Qt) — each reads/writes a shared ``state`` dict
# ──────────────────────────────────────────────────────────────────────────────

_GPU_KEYWORDS = ("nvidia", "rtx", "geforce", "quadro", "tesla", "cuda")


def _detect_gpus() -> list[tuple[str, str | None]]:
    """Return [(display_label, cle_device_name), …]; CPU fallback appended last."""
    devices: list[tuple[str, str | None]] = []
    try:
        import pyclesperanto_prototype as cle
        for name in cle.available_device_names(dev_type="gpu"):
            if name:
                devices.append((name, name))
    except Exception as exc:
        print(f"[LC] GPU enumeration failed: {exc}")
    devices.append(("CPU (no explicit device)", None))
    return devices


def _default_gpu(devices: list[tuple[str, str | None]]) -> str | None:
    """Prefer an NVIDIA device, else the first real GPU, else CPU."""
    for _label, val in devices:
        if val and any(k in val.lower() for k in _GPU_KEYWORDS):
            return val
    for _label, val in devices:
        if val:
            return val
    return None


def _select_device(name: str | None) -> None:
    if not name:
        return
    try:
        import pyclesperanto_prototype as cle
        cle.select_device(name)
    except Exception as exc:
        print(f"[LC] Could not select device '{name}': {exc}")


def _clamp_ch(n_ch: int, idx: int, label: str) -> int:
    """Clamp a channel index into [0, n_ch-1], warning if it was out of range."""
    if idx < 0 or idx >= n_ch:
        clamped = max(0, min(idx, n_ch - 1))
        print(f"[LC] {label} channel {idx} out of range "
              f"(image has {n_ch} channel(s)); using {clamped}.")
        return clamped
    return idx


def step_load(state: dict, p: dict) -> dict:
    """Load the .ims file, build raw + percentile-normalised channel stacks."""
    # The resample path is the slow one — serve it from disk when the file and
    # load parameters are unchanged (segmentation params don't affect this step).
    _do_cache = bool(p.get("make_isotropic")) and not p["use_mip"]
    _key = None
    if _do_cache and os.path.exists(p["ims_path"]):
        _key = _load_cache_key(p, os.path.getmtime(p["ims_path"]))
        cached = _load_cache_read(_key)
        if cached is not None:
            print(f"[LC] Loaded isotropic data from cache ({p['ims_path']}).")
            return cached

    print(f"[LC] Loading {p['ims_path']} …")
    img, meta = load_image(p["ims_path"])
    voxel_size = meta["voxel_size"] or (1.0, 1.0, 1.0)
    n_ch = img.shape[1]

    raw = np.stack([np.asarray(img[0, c]) for c in range(n_ch)])      # (C, Z, Y, X)
    # Per-channel percentile normalisation: use the channel's override if set,
    # else the global default (2–98).
    ch_norm = p.get("ch_norm") or {}
    norm = np.stack([
        percentile_minmax_normalize(
            raw[c], *ch_norm.get(c, (p["p_low"], p["p_high"]))
        )
        for c in range(n_ch)
    ])

    if p["use_mip"]:
        print("[LC] Applying MIP projection …")
        raw  = np.max(raw,  axis=1, keepdims=True)
        norm = np.max(norm, axis=1, keepdims=True)
    elif p.get("make_isotropic"):
        iso = max(voxel_size)
        if not all(abs(v - iso) < 1e-6 for v in voxel_size):
            print(f"[LC] Anisotropic voxels {tuple(round(v, 3) for v in voxel_size)} — "
                  f"resampling to isotropic {iso:.3f} µm …")
            raw  = np.stack([make_isotropic(raw[c],  voxel_size, iso)[0] for c in range(n_ch)])
            norm = np.stack([make_isotropic(norm[c], voxel_size, iso)[0] for c in range(n_ch)])
            voxel_size = (iso, iso, iso)

    result = dict(raw=raw, norm=norm, voxel_size=voxel_size, n_ch=n_ch)
    if _key is not None:
        _load_cache_write(_key, result)
    return result


def step_cilia(state: dict, p: dict) -> dict:
    print("[LC] Segmenting cilia …")
    ch = _clamp_ch(state["raw"].shape[0], p["ch_cilia"], "Cilia")
    cilia_labels = np.asarray(segment_cilia_ml(
        state["raw"][ch],
        classifier_path=p["classifier_path"],
        gaussian_sigma=(p["cilia_gauss_z"], p["cilia_gauss_y"], p["cilia_gauss_x"]),
        log_transform=p["cilia_log"],
        min_size=p["cilia_min_size"],
        max_size=p["cilia_max_size"],
    )).astype(np.int32)
    return dict(cilia_labels=cilia_labels)


def step_nuclei(state: dict, p: dict) -> dict:
    gauss = (p["nuclei_gauss_z"], p["nuclei_gauss_y"], p["nuclei_gauss_x"])
    if p.get("nuclei_method") == "APOC classifier":
        print("[LC] Segmenting nuclei (APOC classifier) …")
        ch = _clamp_ch(state["raw"].shape[0], p["ch_nuclei"], "Nuclei")
        nuclei_labels = np.asarray(segment_cilia_ml(
            state["raw"][ch],
            classifier_path=p["nuclei_classifier_path"],
            gaussian_sigma=gauss,
            log_transform=p["nuclei_log"],
            min_size=p["nuclei_min_size"],
            max_size=p["nuclei_max_size"],
        )).astype(np.int32)
    else:
        print("[LC] Segmenting nuclei (Voronoi-Otsu) …")
        tr = p["tophat_radius"]
        ch = _clamp_ch(state["norm"].shape[0], p["ch_nuclei"], "Nuclei")
        nuclei_labels = segment_nuclei(
            state["norm"][ch],
            tophat_radius=(tr, tr, tr),
            spot_sigma=p["nuclei_sigma"],
            outline_sigma=p["nuclei_outline_sigma"],
            gaussian_sigma=gauss,
            log_transform=p["nuclei_log"],
        ).astype(np.int32)
    return dict(nuclei_labels=nuclei_labels)


def step_neurites(state: dict, p: dict) -> dict:
    print("[LC] Segmenting neurites …")
    ch = _clamp_ch(state["norm"].shape[0], p["ch_neurites"], "Neurites")
    neurite_img = state["norm"][ch]
    if p.get("merge_nuclei_neurite"):
        nch = _clamp_ch(state["norm"].shape[0], p["ch_nuclei"], "Nuclei")
        print("[LC] Merging nuclei into neurite channel before skeletonising …")
        neurite_img = np.maximum(neurite_img, state["norm"][nch])
    skeleton, neurites_gpu = segment_neurites(
        neurite_img,
        spot_sigma=p["neurite_sigma"],
        gaussian_sigma=(p["neurite_gauss_z"], p["neurite_gauss_y"], p["neurite_gauss_x"]),
        log_transform=p["neurite_log"],
    )
    return dict(
        neurite_labels=np.asarray(neurites_gpu).astype(np.int32),
        skeleton_mask=np.asarray(skeleton) > 0,
    )


def step_basal_bodies(state: dict, p: dict) -> dict:
    ch = _clamp_ch(state["raw"].shape[0], p["ch_bb"], "Basal Bodies")
    gauss = (p["bb_gauss_z"], p["bb_gauss_y"], p["bb_gauss_x"])
    if p.get("bb_method") == "APOC classifier":
        print("[LC] Segmenting basal bodies (APOC classifier) …")
        bb_labels = np.asarray(segment_basal_bodies_ml(
            state["raw"][ch],
            classifier_path=p["bb_classifier_path"],
            gaussian_sigma=gauss,
            log_transform=p["bb_log"],
            min_size=p["bb_min_size"],
            max_size=p["bb_max_size"],
        )).astype(np.int32)
    else:
        print("[LC] Segmenting basal bodies (Voronoi-Otsu) …")
        bb_labels = np.asarray(segment_basal_bodies(
            state["raw"][ch],
            spot_sigma=p["bb_spot_sigma"],
            outline_sigma=p["bb_outline_sigma"],
            gaussian_sigma=gauss,
            log_transform=p["bb_log"],
            min_size=p["bb_min_size"],
            max_size=p["bb_max_size"],
        )).astype(np.int32)
    return dict(bb_labels=bb_labels)


def step_distances(state: dict, p: dict) -> dict:
    """Distance maps + regularised ratio map. Needs neurites + nuclei."""
    print("[LC] Computing distance maps …")
    vs           = state["voxel_size"]
    neurite_mask = state["neurite_labels"] > 0
    eps          = p["ratio_epsilon"]

    dt_nuclei  = distance_transform_edt(~(state["nuclei_labels"] > 0), sampling=vs)
    dt_neurite = distance_transform_edt(neurite_mask, sampling=vs)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_map = np.where(
            neurite_mask,
            (dt_nuclei + eps) / (dt_neurite + eps),
            np.nan,
        )
    ratio_map[~np.isfinite(ratio_map)] = np.nan
    log_ratio_map = np.where(np.isfinite(ratio_map), np.log(ratio_map), np.nan)

    dist_to_neurite = distance_transform_edt(~neurite_mask, sampling=vs)
    _, nearest_skel_idx = distance_transform_edt(
        ~state["skeleton_mask"], return_indices=True, sampling=vs
    )

    return dict(
        dt_nuclei=dt_nuclei, dt_neurite=dt_neurite,
        ratio_map=ratio_map, log_ratio_map=log_ratio_map,
        dist_to_neurite=dist_to_neurite, nearest_skel_idx=nearest_skel_idx,
    )


def step_assign(state: dict, p: dict) -> dict:
    """Assign distance features to cilia + basal bodies and classify."""
    print("[LC] Assigning features & classifying …")
    import pandas as pd

    fname = Path(p["ims_path"]).name
    eps   = p["ratio_epsilon"]

    def _features(labels, dist_cutoff, obj_type):
        ids = np.unique(labels)
        ids = ids[ids != 0]
        if ids.size == 0:
            return pd.DataFrame()
        centroids = center_of_mass(labels > 0, labels=labels, index=ids)
        return assign_label_features(
            labels, centroids, ids,
            state["dist_to_neurite"], state["nearest_skel_idx"],
            state["dt_neurite"], state["dt_nuclei"], state["ratio_map"],
            dist_cutoff, fname, object_type=obj_type, ratio_epsilon=eps,
        )

    cilia_df = _features(state["cilia_labels"], p["max_cilia_dist_um"], "cilia")
    bb_df    = _features(state["bb_labels"],    p["max_basal_dist_um"], "basal_body")

    for df in (cilia_df, bb_df):
        if not df.empty:
            df["class"] = "ambiguous"
            df.loc[df["log_ratio"] > p["neurite_threshold"], "class"] = "neurite"
            df.loc[df["log_ratio"] < p["soma_threshold"], "class"] = "soma"

    # ── Per-cilium 3-D shape descriptors (volume, sphericity, …) ────────────────
    if not cilia_df.empty:
        cprops = cilia_shape_props(state["cilia_labels"], state["voxel_size"])
        for _col in SHAPE_COLS:
            cilia_df[_col] = [cprops.get(int(i), {}).get(_col, np.nan)
                              for i in cilia_df["cilia_id"]]

    # ── Pair cilia ↔ basal bodies (closest, strict 1:1, µm cutoff) ──────────────
    # Keep only validated cilia (paired to a BB within the cutoff) and the BBs
    # they pair with, so the table + overlays match the batch outputs.
    if not cilia_df.empty or not bb_df.empty:
        combined = pd.concat([cilia_df, bb_df], ignore_index=True)
        if p.get("directional_pairing"):
            # Search along each cilium's major axis first, fall back to nearest.
            paired = pair_from_mixed_df_directional(
                combined, state["cilia_labels"], voxel_size=state["voxel_size"],
                max_pair_distance_um=p["max_basal_dist_um"],
                search_dist_um=p.get("pair_search_dist_um"),
                cone_angle_deg=p.get("pair_cone_angle_deg", 45.0),
            )
        else:
            paired = pair_from_mixed_df(
                combined, voxel_size=state["voxel_size"],
                max_pair_distance_um=p["max_basal_dist_um"],
            )
        if p.get("require_basal_body", True):
            cilia_df = paired[(paired["object_type"] == "cilia")
                              & paired["validated"]].reset_index(drop=True)
            bb_df    = paired[(paired["object_type"] == "basal_body")
                              & paired["validated"]].reset_index(drop=True)
        else:
            # BB distance filtering off — keep ALL cilia/BBs (pairing info kept).
            cilia_df = paired[paired["object_type"] == "cilia"].reset_index(drop=True)
            bb_df    = paired[paired["object_type"] == "basal_body"].reset_index(drop=True)

    # Optionally take the cilium ratio from its paired basal body, then
    # re-classify so `class` reflects the basal-body ratio.
    if p.get("ratio_from_basal_body", True) and not cilia_df.empty:
        cilia_df = use_basal_body_ratio(cilia_df, bb_df)
        cilia_df["class"] = "ambiguous"
        cilia_df.loc[cilia_df["log_ratio"] > p["neurite_threshold"], "class"] = "neurite"
        cilia_df.loc[cilia_df["log_ratio"] < p["soma_threshold"], "class"] = "soma"

    print(f"[LC] Done — {len(cilia_df)} cilia, {len(bb_df)} basal bodies "
          f"({'paired-only' if p.get('require_basal_body', True) else 'all kept'}).")
    return dict(cilia_df=cilia_df, bb_df=bb_df)


# ──────────────────────────────────────────────────────────────────────────────
# LAYER UPDATERS — refresh only the layers a given step produced
# ──────────────────────────────────────────────────────────────────────────────

def _remove(viewer: napari.Viewer, *names: str) -> None:
    for n in names:
        if n in viewer.layers:
            viewer.layers.remove(n)


def _add_image_safe(viewer, data, **kwargs):
    """Add an Image layer with guaranteed-finite data and a non-degenerate
    contrast window.

    A constant image (min == max) or one containing NaN/inf makes napari pick
    degenerate ``contrast_limits``, which crashes vispy's Image visual with an
    OpenGL access violation in ``glDrawArrays``. We sanitise non-finite values
    and always pass a contrast window with ``vmax > vmin``.
    """
    arr = np.asarray(data)
    finite = np.isfinite(arr)
    if arr.size and not finite.all():
        fill = float(arr[finite].min()) if finite.any() else 0.0
        arr = np.where(finite, arr, fill)
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32, copy=False)
    if arr.size and finite.any():
        vmin = float(arr[finite].min()) if not finite.all() else float(arr.min())
        vmax = float(arr[finite].max()) if not finite.all() else float(arr.max())
    else:
        vmin, vmax = 0.0, 1.0
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        vmax = vmin + 1.0
    kwargs.setdefault("contrast_limits", (vmin, vmax))
    return viewer.add_image(arr, **kwargs)


def show_channels(viewer, state) -> None:
    vs = state["voxel_size"]
    n_ch = state["raw"].shape[0]
    ch_keys = {"Cilia": "ch_cilia", "Neurites": "ch_neurites",
               "Basal Bodies": "ch_bb", "Nuclei": "ch_nuclei"}
    for name, cmap in zip(_CH_NAMES, _CH_COLORMAPS):
        ch = _clamp_ch(n_ch, state["channels"][ch_keys[name]], name)
        lname = f"LC: Raw {name}"
        _remove(viewer, lname)
        _add_image_safe(
            viewer, state["raw"][ch], name=lname, scale=vs,
            colormap=cmap, blending="additive", visible=(name == "Cilia"),
        )


def show_labels(viewer, state, key, lname, scale_from="voxel_size") -> None:
    _remove(viewer, lname)
    viewer.add_labels(state[key], name=lname, scale=state[scale_from])


def show_skeleton(viewer, state) -> None:
    """Overlay the neurite skeleton as a thin bright layer (for inspection)."""
    if state.get("skeleton_mask") is None:
        return
    _remove(viewer, "LC: Neurite Skeleton")
    _add_image_safe(
        viewer, np.asarray(state["skeleton_mask"]).astype(np.float32),
        name="LC: Neurite Skeleton", scale=state["voxel_size"],
        colormap="red", blending="additive", opacity=0.9,
    )


def show_ratio(viewer, state) -> None:
    _remove(viewer, "LC: Log Ratio")
    lr = state["log_ratio_map"]
    finite = np.isfinite(lr)
    if finite.any():
        fill = float(np.nanmin(lr[finite]))
        _add_image_safe(
            viewer, np.where(finite, lr, fill), name="LC: Log Ratio",
            scale=state["voxel_size"], colormap="bwr",
            opacity=0.65, blending="translucent", visible=False,
        )


def show_centroids(viewer, state):
    """Add cilia + BB centroid layers. Returns the cilia Points layer (or None)."""
    vs = state["voxel_size"]
    _remove(viewer, "LC: Cilia Centroids", "LC: BB Centroids")

    cilia_pts = None
    cdf = state.get("cilia_df")
    if cdf is not None and not cdf.empty:
        # NaN/inf in marker positions or in the colour-mapped property make
        # vispy's Markers visual crash (access violation in glDrawArrays), so
        # everything fed to add_points below must be finite.
        coords  = np.nan_to_num(np.asarray(cdf["coords"].tolist(), dtype=float))
        lr_vals = cdf["log_ratio"].to_numpy(dtype=float)

        # Contrast limits from the finite log-ratios only; fall back to a unit
        # window when they're absent or degenerate (vmin == vmax → NaN colours).
        finite = lr_vals[np.isfinite(lr_vals)]
        if finite.size > 1 and np.ptp(finite) > 0:
            vmin = float(np.percentile(finite, 5))
            vmax = float(np.percentile(finite, 95))
        elif finite.size:
            vmin, vmax = float(finite.min()) - 1.0, float(finite.max()) + 1.0
        else:
            vmin, vmax = -1.0, 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0

        # Colour unassigned cilia (NaN log-ratio) neutrally at the mid-point
        # instead of letting NaN reach the colormap.
        lr_color = np.where(np.isfinite(lr_vals), lr_vals, (vmin + vmax) / 2.0)

        cilia_pts = viewer.add_points(
            coords, name="LC: Cilia Centroids", scale=vs, size=5,
            properties={"log_ratio": lr_color},
            face_color="log_ratio", face_colormap="bwr",
        )
        cilia_pts.face_contrast_limits = (vmin, vmax)

    bdf = state.get("bb_df")
    if bdf is not None and not bdf.empty:
        viewer.add_points(
            np.nan_to_num(np.asarray(bdf["coords"].tolist(), dtype=float)),
            name="LC: BB Centroids", scale=vs, size=4, face_color="yellow",
        )
    return cilia_pts


def show_associations(viewer, state) -> None:
    """Draw two line overlays:
      • each cilium → its paired basal body (yellow, labelled with µm distance)
      • each cilium → its nearest neurite skeleton voxel (cyan, labelled µm)
    """
    vs = np.asarray(state["voxel_size"], dtype=float)
    _remove(viewer, "LC: Cilia–BB links", "LC: Cilia→Neurite")

    cdf = state.get("cilia_df")
    bdf = state.get("bb_df")
    if cdf is None or cdf.empty:
        return

    # ── Cilia ↔ basal body pairing lines ────────────────────────────────────────
    if bdf is not None and not bdf.empty and "paired_id" in cdf.columns:
        bb_coord_by_id = {
            int(r["cilia_id"]): np.asarray(r["coords"], dtype=float)
            for _, r in bdf.iterrows()
        }
        pair_lines, pair_dist = [], []
        for _, r in cdf.iterrows():
            if r.get("pairing_status") != "paired" or pd_isna(r.get("paired_id")):
                continue
            bb_c = bb_coord_by_id.get(int(r["paired_id"]))
            if bb_c is None:
                continue
            c_c = np.asarray(r["coords"], dtype=float)
            pair_lines.append(np.stack([c_c, bb_c]))
            pair_dist.append(float(np.linalg.norm((c_c - bb_c) * vs)))
        if pair_lines:
            viewer.add_shapes(
                pair_lines, shape_type="line", name="LC: Cilia–BB links",
                scale=vs, edge_color="yellow", edge_width=1.5,
                features={"d": np.array(pair_dist)},
                text={"string": "{d:.1f} µm", "size": 7,
                      "color": "yellow", "anchor": "center"},
            )

    # ── Cilium → nearest neurite skeleton voxel ─────────────────────────────────
    nsi = state.get("nearest_skel_idx")
    if nsi is not None:
        shp = state["cilia_labels"].shape
        neur_lines, neur_dist = [], []
        for _, r in cdf.iterrows():
            c_c = np.asarray(r["coords"], dtype=float)
            z, y, x = (int(np.clip(round(c_c[i]), 0, shp[i] - 1)) for i in range(3))
            sk = np.array([nsi[0, z, y, x], nsi[1, z, y, x], nsi[2, z, y, x]], dtype=float)
            neur_lines.append(np.stack([c_c, sk]))
            neur_dist.append(float(r.get("distance_to_neurite_um", np.nan)))
        if neur_lines:
            viewer.add_shapes(
                neur_lines, shape_type="line", name="LC: Cilia→Neurite",
                scale=vs, edge_color="cyan", edge_width=1.0,
                features={"d": np.array(neur_dist)},
                text={"string": "{d:.2f} µm", "size": 7,
                      "color": "cyan", "anchor": "center"},
            )


def pd_isna(v) -> bool:
    try:
        import pandas as pd
        return bool(pd.isna(v))
    except Exception:
        return v is None


# ──────────────────────────────────────────────────────────────────────────────
# CONTROLLER WIDGET
# ──────────────────────────────────────────────────────────────────────────────

class _ProgressBridge(QObject):
    """Marshals batch progress from the worker thread to the GUI thread."""
    progressed = Signal(int, int, str)   # (file_idx, n_files, filename)


class _BatchVizBridge(QObject):
    """Marshals a per-file visualisation request from the batch worker thread to
    the GUI thread. The worker emits (payload, event) and blocks on the event;
    the GUI slot draws + screenshots the layers, then sets the event."""
    visualize = Signal(object, object)   # (payload: dict, done: threading.Event)


class LimoncelloApp:
    """Builds the dock widget and owns the per-image pipeline state."""

    # Cilia property columns shown in the browse table (in display order)
    _TABLE_COLS = [
        "cilia_id", "class", "ai_score", "ai_validated", "log_ratio", "ratio",
        "distance_to_neurite_um", "dt_neurite", "dt_nuclei", "volume_um3",
        "length_um", "sphericity",
        "y_min", "x_min", "y_max", "x_max",
        "paired_id", "pair_distance_um", "pairing_status",
    ]

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.state: dict = {}          # intermediate arrays for current image
        self.files: list[str] = []     # .ims filenames in the folder
        self._busy = False
        self.cilia_pts = None          # current cilia Points layer
        self._del_cb = None            # active "delete cilium" mouse callback (or None)
        self._cilia_df_view = None     # df backing the table (row order == points)
        self._populating = False       # guards table edits fired during (re)populate
        self._syncing = False          # guards two-way selection sync
        self._ch_norm: dict[int, tuple[int, int]] = {}   # per-channel (p_low, p_high) overrides
        self._loading_norm = False     # guards programmatic norm spin-box writes
        self._bridge = _ProgressBridge()
        self._bridge.progressed.connect(self._on_batch_progress)
        # Per-file batch visualisation (queued connection → runs on GUI thread)
        self._viz_bridge = _BatchVizBridge()
        self._viz_bridge.visualize.connect(self._on_batch_visualize)
        self.widget = self._build()
        self._build_table()
        # Reload parameters from the previous session, then persist on quit.
        self._load_state()
        _app = QApplication.instance()
        if _app is not None:
            _app.aboutToQuit.connect(self._save_state)

    # ── cilia property table ────────────────────────────────────────────────────
    def _build_table(self):
        self.table = QTableWidget()
        self.table.setColumnCount(len(self._TABLE_COLS))
        self.table.setHorizontalHeaderLabels(self._TABLE_COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # Cells are read-only except the ``ai_validated`` checkbox (handled per item).
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self._table_cols: list[str] = list(self._TABLE_COLS)

        # Ctrl+C copies the selection (or the whole table) as Excel-pasteable TSV.
        sc = QShortcut(QKeySequence.Copy, self.table)
        sc.activated.connect(self._copy_table_tsv)

        if self.viewer is not None:
            wrap = QWidget()
            lay = QVBoxLayout(wrap)
            lay.setContentsMargins(2, 2, 2, 2)
            bar = QHBoxLayout()
            copy_btn = QPushButton("📋 Copy for Excel")
            copy_btn.setToolTip("Copy the whole table to the clipboard (paste into Excel)")
            copy_btn.clicked.connect(lambda: self._copy_table_tsv(whole=True))
            boxes_btn = QPushButton("📦 Copy boxes")
            boxes_btn.setToolTip("Copy per-cilium bounding boxes in the annotator "
                                 "'cilia_boxes' schema (reloads in the annotator / data app)")
            boxes_btn.clicked.connect(self._copy_boxes_tsv)
            export_btn = QPushButton("💾 Export .xlsx")
            export_btn.setToolTip("Save a cilia_table sheet + a cilia_boxes sheet")
            export_btn.clicked.connect(self._export_table_xlsx)
            bar.addWidget(copy_btn)
            bar.addWidget(boxes_btn)
            bar.addWidget(export_btn)
            bar.addStretch(1)
            lay.addLayout(bar)
            lay.addWidget(self.table)
            self.viewer.window.add_dock_widget(
                wrap, area="bottom", name="Cilia properties",
            )

    def _populate_table(self, cdf):
        self._cilia_df_view = cdf.reset_index(drop=True) if cdf is not None else None
        self._populating = True
        try:
            self.table.setRowCount(0)
            if cdf is None or cdf.empty:
                self._table_cols = []
                return
            cols = [c for c in self._TABLE_COLS if c in cdf.columns]
            self._table_cols = cols
            self.table.setColumnCount(len(cols))
            self.table.setHorizontalHeaderLabels(cols)
            self.table.setRowCount(len(cdf))
            for r in range(len(cdf)):
                for c, col in enumerate(cols):
                    v = cdf.iloc[r][col]
                    item = QTableWidgetItem()
                    if col == "ai_validated":
                        # editable checkbox — the user can flip the AI decision
                        item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled
                                      | Qt.ItemIsSelectable)
                        item.setCheckState(Qt.Checked if bool(v) else Qt.Unchecked)
                        item.setText("keep" if bool(v) else "reject")
                    else:
                        if isinstance(v, float):
                            txt = "—" if v != v else f"{v:.3g}"   # v!=v → NaN
                        else:
                            txt = str(v)
                        item.setText(txt)
                    self.table.setItem(r, c, item)
            self.table.resizeColumnsToContents()
        finally:
            self._populating = False

    def _on_table_item_changed(self, item):
        """Flip the AI decision when the user toggles the ``ai_validated`` box."""
        if self._populating or self._cilia_df_view is None:
            return
        col = item.column()
        if col >= len(self._table_cols) or self._table_cols[col] != "ai_validated":
            return
        keep = item.checkState() == Qt.Checked
        row = item.row()
        self._populating = True                 # avoid re-entrancy from setText
        try:
            item.setText("keep" if keep else "reject")
        finally:
            self._populating = False
        if 0 <= row < len(self._cilia_df_view):
            self._cilia_df_view.at[row, "ai_validated"] = bool(keep)
            self._refresh_ai_overlay()

    def _refresh_ai_overlay(self):
        """Redraw the AI-validated / AI-rejected point overlays from the (edited)
        table so the viewer matches the user's manual decisions."""
        cdf = self._cilia_df_view
        if cdf is None or "ai_validated" not in cdf.columns or "coords" not in cdf.columns:
            return
        _remove(self.viewer, "LC: AI-validated", "LC: AI-rejected")
        vs = self.state.get("voxel_size", (1, 1, 1))
        scored = cdf[cdf["ai_score"].notna()] if "ai_score" in cdf.columns else cdf
        for keep, name, color in ((True, "LC: AI-validated", "lime"),
                                  (False, "LC: AI-rejected", "red")):
            sub = scored[scored["ai_validated"] == keep]
            if len(sub):
                pts = np.asarray([np.asarray(c, float) for c in sub["coords"]])
                self.viewer.add_points(pts, name=name, size=6, face_color=color,
                                       scale=vs, opacity=0.9)

    def _table_tsv(self, whole: bool) -> str:
        """The table as tab-separated text (header + rows). ``whole`` ignores the
        selection and copies every row."""
        cols = self.table.columnCount()
        rows = self.table.rowCount()
        headers = [self.table.horizontalHeaderItem(c).text() if self.table.horizontalHeaderItem(c)
                   else "" for c in range(cols)]
        sel_rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        want = range(rows) if (whole or not sel_rows) else sel_rows
        out = ["\t".join(headers)]
        for r in want:
            cells = []
            for c in range(cols):
                it = self.table.item(r, c)
                cells.append(it.text() if it is not None else "")
            out.append("\t".join(cells))
        return "\n".join(out)

    def _copy_table_tsv(self, whole: bool = False):
        if self.table.rowCount() == 0:
            return
        QApplication.clipboard().setText(self._table_tsv(whole))
        n = self.table.rowCount() if (whole or not self.table.selectedIndexes()) else \
            len({idx.row() for idx in self.table.selectedIndexes()})
        self._set_status(f"📋 Copied {n} rows — paste into Excel.")

    def _attach_boxes(self, state):
        """Add XY bounding-box columns (y_min/x_min/y_max/x_max) to the cilia
        dataframe so they show in the table, get copied, and get exported."""
        cdf = state.get("cilia_df")
        lab = state.get("cilia_labels")
        if cdf is None or getattr(cdf, "empty", True) or lab is None:
            return
        boxes = _cilia_bboxes(lab)
        for col in _BOX_XY_COLS:
            cdf[col] = np.nan
        ids = cdf["cilia_id"].astype(int)
        for pos, cid in zip(cdf.index, ids):
            bb = boxes.get(int(cid))
            if bb is not None:
                cdf.loc[pos, _BOX_XY_COLS] = bb
        state["cilia_df"] = cdf

    def _cilia_boxes_df(self, state):
        """The cilia as a DataFrame in the annotator/data-app ``cilia_boxes``
        schema (one box per cilium), so an export/copy reloads there."""
        import pandas as pd
        cdf = self._cilia_df_view if self._cilia_df_view is not None else state.get("cilia_df")
        lab = state.get("cilia_labels")
        if cdf is None or cdf.empty or lab is None:
            return None
        boxes = _cilia_bboxes(lab)
        lab = np.asarray(lab)
        H, W = int(lab.shape[-2]), int(lab.shape[-1])
        vz, vy, vx = (float(s) for s in state.get("voxel_size", (1, 1, 1)))
        p = self._params()
        fname = Path(str(p.get("ims_path") or "")).stem
        mip = 1 if lab.ndim == 2 else 0
        miso = 1 if p.get("make_isotropic") else 0
        rows = []
        for _, row in cdf.iterrows():
            cid = int(row["cilia_id"])
            bb = boxes.get(cid)
            if bb is None:
                continue
            y0, x0, y1, x1 = bb
            cls = str(row.get("class", "uncertain"))
            cls = "uncertain" if cls == "ambiguous" else cls   # annotator vocab
            rows.append({
                "filename": fname, "box_id": cid, "class": cls,
                "y_min": y0, "x_min": x0, "y_max": y1, "x_max": x1,
                "img_height": H, "img_width": W,
                "voxel_z": vz, "voxel_y": vy, "voxel_x": vx,
                "mip": mip, "make_isotropic": miso,
            })
        return pd.DataFrame(rows, columns=_BOX_COLS)

    def _copy_boxes_tsv(self):
        """Copy the boxes (annotator schema) as TSV → paste into a sheet named
        'cilia_boxes' to reload in the annotator or the data app."""
        df = self._cilia_boxes_df(self.state)
        if df is None or df.empty:
            show_warning("No cilia boxes — run “Assign & classify” first.")
            return
        tsv = "\t".join(_BOX_COLS) + "\n" + "\n".join(
            "\t".join(str(v) for v in r) for r in df.itertuples(index=False, name=None))
        QApplication.clipboard().setText(tsv)
        self._set_status(f"📦 Copied {len(df)} boxes — paste into a 'cilia_boxes' sheet.")

    def _export_table_xlsx(self):
        """Save the table (with manual AI edits) to .xlsx: a ``cilia_table`` sheet
        plus a ``cilia_boxes`` sheet that reloads in the annotator / data app."""
        cdf = self._cilia_df_view
        if cdf is None or cdf.empty:
            show_warning("No cilia table to export — run “Assign & classify” first.")
            return
        from qtpy.QtWidgets import QFileDialog
        default = str(Path(str(self.output.value or ".")) / "cilia_table.xlsx")
        path, _ = QFileDialog.getSaveFileName(
            self.table, "Export cilia table", default, "Excel (*.xlsx)")
        if not path:
            return
        cols = [c for c in self._table_cols if c in cdf.columns]
        try:
            import pandas as pd
            bdf = self._cilia_boxes_df(self.state)
            with pd.ExcelWriter(path) as xw:
                cdf[cols].to_excel(xw, sheet_name="cilia_table", index=False)
                if bdf is not None and not bdf.empty:
                    bdf.to_excel(xw, sheet_name=_BOX_SHEET, index=False)
            extra = f" (+ {len(bdf)} boxes)" if bdf is not None and not bdf.empty else ""
            self._set_status(f"💾 Exported {len(cdf)} cilia{extra} → {Path(path).name}")
            show_info(f"Saved cilia table + boxes → {path}")
        except Exception as e:                  # noqa: BLE001
            show_warning(f"Export failed: {e}")

    def _on_cilia_selection(self, event=None):
        """napari point selection → highlight matching table rows."""
        if self._syncing or self.cilia_pts is None:
            return
        sel = sorted(int(i) for i in self.cilia_pts.selected_data)
        self._syncing = True
        try:
            sm = self.table.selectionModel()
            sm.clearSelection()
            for i in sel:
                if 0 <= i < self.table.rowCount():
                    idx = self.table.model().index(i, 0)
                    sm.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
            if sel and 0 <= sel[0] < self.table.rowCount():
                self.table.scrollToItem(self.table.item(sel[0], 0))
        finally:
            self._syncing = False

    def _on_table_selection(self):
        """Table row selection → select matching cilia points + centre camera."""
        if self._syncing or self.cilia_pts is None:
            return
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        self._syncing = True
        try:
            self.cilia_pts.selected_data = set(rows)
            self.cilia_pts.refresh()
            if rows and self._cilia_df_view is not None:
                coords = np.asarray(self._cilia_df_view.iloc[rows[0]]["coords"], dtype=float)
                self.viewer.camera.center = tuple(coords * np.asarray(self.state["voxel_size"]))
        finally:
            self._syncing = False

    # ── parameter collection ──────────────────────────────────────────────────
    def _params(self) -> dict:
        """Flatten all parameter widgets into the dict the step functions read."""
        p = dict(
            gpu_device=self.gpu_combo.value,
            classifier_path=str(self.classifier.value),
            ch_cilia=self.ch_cilia.value, ch_neurites=self.ch_neurites.value,
            ch_bb=self.ch_bb.value, ch_nuclei=self.ch_nuclei.value,
            use_mip=self.use_mip.value,
            make_isotropic=self.make_iso.value,
            p_low=self.p_low.value, p_high=self.p_high.value,
            ch_norm=dict(self._ch_norm),
            nuclei_method=self.nuclei_method.value,
            nuclei_classifier_path=str(self.nuclei_classifier.value),
            nuclei_min_size=self.nuclei_min_size.value,
            nuclei_max_size=self.nuclei_max_size.value,
            nuclei_sigma=self.nuclei_sigma.value,
            tophat_radius=self.tophat_radius.value,
            nuclei_outline_sigma=self.nuclei_outline_sigma.value,
            nuclei_log=self.nuclei_log.value,
            neurite_sigma=self.neurite_sigma.value,
            neurite_log=self.neurite_log.value,
            merge_nuclei_neurite=self.neurite_merge_nuclei.value,
            cilia_log=self.cilia_log.value,
            cilia_min_size=self.cilia_min_size.value,
            cilia_max_size=self.cilia_max_size.value,
            bb_method=self.bb_method.value,
            bb_classifier_path=str(self.bb_classifier.value),
            bb_spot_sigma=self.bb_spot_sigma.value,
            bb_outline_sigma=self.bb_outline_sigma.value,
            bb_log=self.bb_log.value,
            bb_min_size=self.bb_min_size.value,
            bb_max_size=self.bb_max_size.value,
            max_cilia_dist_um=self.max_cilia_dist.value,
            max_basal_dist_um=self.max_basal_dist.value,
            require_basal_body=self.require_bb.value,
            ratio_from_basal_body=self.ratio_from_bb.value,
            ratio_epsilon=self.ratio_epsilon.value,
            neurite_threshold=self.neurite_threshold.value,
            soma_threshold=self.soma_threshold.value,
            directional_pairing=self.directional_pairing.value,
            pair_search_dist_um=self.pair_search_dist.value,
            pair_cone_angle_deg=self.pair_cone_angle.value,
        )
        # Gaussian pre-blur was removed from the UI; segmentation runs unblurred.
        for pre in ("cilia", "nuclei", "neurite", "bb"):
            for ax in ("z", "y", "x"):
                p[f"{pre}_gauss_{ax}"] = 0.0
        # current image path
        if self.file_combo.value and self.folder.value:
            p["ims_path"] = str(Path(str(self.folder.value)) / self.file_combo.value)
        else:
            p["ims_path"] = ""
        return p

    # ── persistent parameter state (auto save/reload) ───────────────────────────
    def _state_widgets(self) -> dict:
        """(key → widget) whose ``.value`` is saved on quit and reloaded on start.

        Widgets with data-dependent choices (hist/AI/train dropdowns, feature
        grid) are excluded — they're rebuilt from the loaded image. ``file_combo``
        is restored last, after the folder is re-scanned so its choices exist.
        """
        return {
            "folder": self.folder, "output": self.output,
            "classifier": self.classifier, "gpu_combo": self.gpu_combo,
            "ch_cilia": self.ch_cilia, "ch_neurites": self.ch_neurites,
            "ch_bb": self.ch_bb, "ch_nuclei": self.ch_nuclei,
            "use_mip": self.use_mip, "make_iso": self.make_iso,
            "p_low": self.p_low, "p_high": self.p_high,
            "nuclei_method": self.nuclei_method,
            "nuclei_classifier": self.nuclei_classifier,
            "nuclei_min_size": self.nuclei_min_size,
            "nuclei_max_size": self.nuclei_max_size,
            "nuclei_sigma": self.nuclei_sigma, "tophat_radius": self.tophat_radius,
            "nuclei_outline_sigma": self.nuclei_outline_sigma,
            "nuclei_log": self.nuclei_log,
            "neurite_sigma": self.neurite_sigma, "neurite_log": self.neurite_log,
            "neurite_merge_nuclei": self.neurite_merge_nuclei,
            "cilia_log": self.cilia_log, "cilia_min_size": self.cilia_min_size,
            "cilia_max_size": self.cilia_max_size,
            "bb_method": self.bb_method, "bb_classifier": self.bb_classifier,
            "bb_spot_sigma": self.bb_spot_sigma,
            "bb_outline_sigma": self.bb_outline_sigma, "bb_log": self.bb_log,
            "bb_min_size": self.bb_min_size, "bb_max_size": self.bb_max_size,
            "max_cilia_dist": self.max_cilia_dist,
            "max_basal_dist": self.max_basal_dist, "require_bb": self.require_bb,
            "directional_pairing": self.directional_pairing,
            "pair_search_dist": self.pair_search_dist,
            "pair_cone_angle": self.pair_cone_angle,
            "ratio_from_bb": self.ratio_from_bb, "ratio_epsilon": self.ratio_epsilon,
            "neurite_threshold": self.neurite_threshold,
            "soma_threshold": self.soma_threshold,
            "train_feat_sigmas": self.train_feat_sigmas,
            "train_feat_original": self.train_feat_original,
            "train_channel": self.train_channel,
            "train_positive": self.train_positive,
            "train_max_depth": self.train_max_depth,
            "train_num_trees": self.train_num_trees,
            "train_output": self.train_output, "train_continue": self.train_continue,
            "train_labels_dir": self.train_labels_dir,
            "train_all_labeled": self.train_all_labeled,
            "batch_capture": self.batch_capture, "batch_rois": self.batch_rois,
            "roi_correct": self.roi_correct, "batch_ai": self.batch_ai,
            "batch_roi_only": self.batch_roi_only, "batch_xy_um": self.batch_xy_um,
            "batch_roi_pct": self.batch_roi_pct, "batch_roi_tile": self.batch_roi_tile,
            "ai_keep_thr": self.ai_keep_thr,
            "file_combo": self.file_combo,      # restored last (needs scan first)
        }

    def _save_state(self, *_):
        data = {}
        for key, w in self._state_widgets().items():
            try:
                v = w.value
                data[key] = str(v) if isinstance(v, Path) else v
            except Exception:
                pass
        # Per-channel norm overrides: int keys → JSON-safe [ch, lo, hi] rows.
        data["_ch_norm"] = [[c, lo, hi] for c, (lo, hi) in self._ch_norm.items()]
        try:
            _STATE_PATH.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            print(f"[LC] Could not save parameter state: {exc}")

    def _load_state(self):
        if not _STATE_PATH.exists():
            return
        try:
            data = json.loads(_STATE_PATH.read_text())
        except Exception as exc:
            print(f"[LC] Could not read parameter state: {exc}")
            return
        self._ch_norm = {int(c): (int(lo), int(hi))
                         for c, lo, hi in data.get("_ch_norm", [])}
        widgets = self._state_widgets()
        file_val = data.get("file_combo")
        for key, w in widgets.items():
            if key == "file_combo" or key not in data:
                continue
            v = data[key]
            try:
                if isinstance(w, FileEdit) and isinstance(v, str):
                    v = Path(v)
                w.value = v
            except Exception as exc:
                print(f"[LC] Skipped restoring '{key}': {exc}")
        # Re-scan the saved folder so file_combo has its choices, then reselect.
        if str(self.folder.value):
            try:
                self._scan_folder()
            except Exception:
                pass
        if file_val and file_val in getattr(self.file_combo, "choices", ()):
            self.file_combo.value = file_val

    # ── folder / navigation ────────────────────────────────────────────────────
    def _scan_folder(self):
        folder = Path(str(self.folder.value))
        if not folder.is_dir():
            show_warning("Select a valid input folder first.")
            return
        self.files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".ims"))
        if not self.files:
            show_warning(f"No .ims files found in {folder}")
            self.file_combo.choices = ()
            self.train_images.choices = ()
            return
        self.file_combo.choices = self.files
        self.file_combo.value = self.files[0]
        self.train_images.choices = self.files      # subset picker for training
        show_info(f"Found {len(self.files)} .ims file(s).")

    def _step_file(self, delta: int):
        if not self.files:
            return
        cur = self.file_combo.value
        idx = self.files.index(cur) if cur in self.files else 0
        self.file_combo.value = self.files[(idx + delta) % len(self.files)]

    def _on_file_changed(self, *_):
        # New image selected → drop stale intermediate results.
        self.state = {}
        self.cilia_pts = None
        if getattr(self, "table", None) is not None:
            self.table.setRowCount(0)
            self._cilia_df_view = None
        self._set_status(f"Loaded image selection: {self.file_combo.value or '—'}")

    # ── step execution (threaded) ───────────────────────────────────────────────
    def _set_status(self, msg: str):
        # Surface progress as a transient napari notification popup (the status
        # text no longer lives at the bottom of the settings panel).
        self.status.value = msg
        show_info(msg)

    def _progress_busy(self, text: str = ""):
        """Indeterminate 'working…' pulse (unknown duration)."""
        self.progress.setRange(0, 0)
        self.progress.setFormat(text)
        self.progress.setVisible(True)

    def _progress_set(self, cur: int, total: int, text: str = ""):
        """Determinate progress: cur / total."""
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(cur)
        self.progress.setFormat(text or "%p%")
        self.progress.setVisible(True)

    def _set_busy(self, busy: bool):
        self._busy = busy
        for b in self._step_buttons + [self.run_all_btn, self.batch_btn,
                                       self.super_batch_btn, self.train_btn]:
            b.enabled = not busy
        if busy:
            self._progress_busy()
        else:
            self.progress.setVisible(False)
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.progress.setFormat("")

    def _on_batch_progress(self, file_idx: int, n_files: int, filename: str):
        """GUI-thread slot fed by the batch worker via the signal bridge."""
        self._progress_set(file_idx, n_files, f"%p%  ({file_idx + 1}/{n_files})")
        self._set_status(f"⚙️ Batch [{file_idx + 1}/{n_files}] — {filename}")

    def _require(self, *keys) -> bool:
        missing = [k for k in keys if k not in self.state]
        if missing:
            show_warning(f"Run the earlier steps first (missing: {', '.join(missing)}).")
            return False
        return True

    def _run_step(self, fn, prereq, after, label):
        """Run ``fn(state, params)`` in a worker, merge result, call ``after``."""
        if self._busy:
            return
        p = self._params()
        if not p["ims_path"]:
            show_warning("Open a folder and select an image first.")
            return
        if prereq and not self._require(*prereq):
            return
        # Channel assignment is needed by show_channels regardless of which step.
        self.state["channels"] = {
            "ch_cilia": p["ch_cilia"], "ch_neurites": p["ch_neurites"],
            "ch_bb": p["ch_bb"], "ch_nuclei": p["ch_nuclei"],
        }
        self._set_busy(True)
        self._progress_busy(f"{label} …")
        self._set_status(f"⚙️ {label} …")

        @thread_worker
        def _work():
            _select_device(p["gpu_device"])
            return fn(self.state, p)

        def _done(updates: dict):
            self.state.update(updates)
            after(self.viewer, self.state)
            self._set_busy(False)
            self._set_status(f"✅ {label} done.")

        def _err(e):
            self._set_busy(False)
            show_warning(f"❌ {label} failed: {e}")

        w = _work()
        w.returned.connect(_done)
        w.errored.connect(_err)
        w.start()

    def _flash_complete(self):
        """Briefly (0.3 s) flash the 'analysis complete' asset over the viewer."""
        path = Path(__file__).parent / "assets" / "analysis-complete.png"
        if not path.exists():
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            return
        try:
            parent = self.viewer.window._qt_window
        except Exception:
            parent = None

        lbl = QLabel(parent)
        lbl.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint
        )
        lbl.setAttribute(Qt.WA_TranslucentBackground)
        lbl.setAttribute(Qt.WA_ShowWithoutActivating)
        pix = pix.scaledToWidth(360, Qt.SmoothTransformation)
        lbl.setPixmap(pix)
        lbl.resize(pix.size())
        lbl.setWindowOpacity(0.9)
        if parent is not None:
            c = parent.geometry().center()
            lbl.move(c.x() - pix.width() // 2, c.y() - pix.height() // 2)
        lbl.show()
        # Keep a reference so it isn't garbage-collected before the timer fires.
        self._flash_lbl = lbl
        QTimer.singleShot(300, lbl.close)

    def _after_load(self, viewer, state):
        """Constrain channel spin-boxes to the image's real channel count, then
        display the raw channels and refresh the intensity histogram."""
        n = int(state.get("n_ch", state["raw"].shape[0]))
        for sb in (self.ch_cilia, self.ch_neurites, self.ch_bb, self.ch_nuclei):
            sb.max = max(0, n - 1)          # setter clamps the current value too
        show_channels(viewer, state)
        self._apply_norm_to_layers()

        # Populate the histogram channel dropdown (keeps current selection if valid)
        cur = self.hist_channel.value
        self.hist_channel.choices = [(f"Channel {i}", i) for i in range(n)]
        self.hist_channel.value = cur if isinstance(cur, int) and cur < n else 0
        self._draw_histogram()

    # ── histogram & per-channel normalisation ───────────────────────────────────
    def _current_hist_ch(self):
        v = self.hist_channel.value
        return int(v) if v is not None else None

    def _effective_norm(self, ch: int) -> tuple[int, int]:
        return self._ch_norm.get(ch, (self.p_low.value, self.p_high.value))

    def _channel_sample(self, ch: int) -> np.ndarray:
        arr = np.asarray(self.state["raw"][ch]).ravel()
        if arr.size > 2_000_000:                      # subsample for snappy redraws
            arr = arr[:: max(1, arr.size // 2_000_000)]
        return arr

    def _draw_histogram(self, *_):
        if "raw" not in self.state:
            return
        ch = self._current_hist_ch()
        if ch is None or ch >= self.state["raw"].shape[0]:
            return
        arr = self._channel_sample(ch)
        lo_p, hi_p = self._effective_norm(ch)
        v_lo, v_hi = np.percentile(arr, [lo_p, hi_p])
        self.hist_ax.clear()
        self.hist_ax.hist(arr, bins=128, color="#4C9BE0", log=True)
        self.hist_ax.axvline(v_lo, color="#e74c3c", lw=1.2, label=f"{lo_p}% = {v_lo:.0f}")
        self.hist_ax.axvline(v_hi, color="#2ecc71", lw=1.2, label=f"{hi_p}% = {v_hi:.0f}")
        over = " (override)" if ch in self._ch_norm else " (default)"
        self.hist_ax.set_title(f"Channel {ch} intensity{over}", fontsize=8)
        self.hist_ax.tick_params(labelsize=6)
        self.hist_ax.legend(fontsize=6, loc="upper right")
        self.hist_canvas.draw_idle()

    def _apply_norm_to_layers(self, *_):
        """Live-preview the current normalisation by mapping each channel's
        percentile range onto the displayed raw layer's contrast limits.

        ``percentile_minmax_normalize`` linearly maps [p_low, p_high] → [0, 1]
        with clipping, which is visually identical to setting napari
        ``contrast_limits`` to those percentile intensities — so this previews
        normalisation without recomputing the whole stack.
        """
        if "raw" not in self.state or "channels" not in self.state:
            return
        n_ch = self.state["raw"].shape[0]
        ch_keys = {"Cilia": "ch_cilia", "Neurites": "ch_neurites",
                   "Basal Bodies": "ch_bb", "Nuclei": "ch_nuclei"}
        for name in _CH_NAMES:
            lname = f"LC: Raw {name}"
            if lname not in self.viewer.layers:
                continue
            ch = _clamp_ch(n_ch, self.state["channels"][ch_keys[name]], name)
            lo_p, hi_p = self._effective_norm(ch)
            v_lo, v_hi = np.percentile(self._channel_sample(ch), [lo_p, hi_p])
            if v_hi <= v_lo:
                v_hi = v_lo + 1
            self.viewer.layers[lname].contrast_limits = (float(v_lo), float(v_hi))

    def _refresh_norm(self, *_):
        """Redraw the histogram and live-update the displayed channels."""
        self._draw_histogram()
        self._apply_norm_to_layers()

    def _on_hist_channel_changed(self, *_):
        ch = self._current_hist_ch()
        if ch is None:
            return
        lo, hi = self._effective_norm(ch)
        self._loading_norm = True                     # avoid creating a spurious override
        try:
            self.ch_p_low.value, self.ch_p_high.value = lo, hi
        finally:
            self._loading_norm = False
        self._draw_histogram()

    def _on_ch_norm_changed(self, *_):
        if self._loading_norm:
            return
        ch = self._current_hist_ch()
        if ch is None:
            return
        self._ch_norm[ch] = (self.ch_p_low.value, self.ch_p_high.value)
        self._refresh_norm()

    def _apply_norm_all(self, *_):
        n = self.state.get("n_ch", len(self.hist_channel.choices))
        lo, hi = self.ch_p_low.value, self.ch_p_high.value
        for c in range(int(n)):
            self._ch_norm[c] = (lo, hi)
        show_info(f"Applied p_low={lo}, p_high={hi} to all {int(n)} channel(s).")
        self._refresh_norm()

    def _reset_norm(self, *_):
        ch = self._current_hist_ch()
        if ch is None:
            return
        self._ch_norm.pop(ch, None)
        self._on_hist_channel_changed()               # reload default values + redraw
        self._apply_norm_to_layers()

    def _on_bb_method_changed(self, *_):
        """Show only the controls relevant to the selected basal-body method:
        the .cl classifier for APOC, spot/outline σ for Voronoi-Otsu."""
        apoc = self.bb_method.value == "APOC classifier"
        self.bb_classifier.visible = apoc
        self.bb_spot_sigma.visible = not apoc
        self.bb_outline_sigma.visible = not apoc

    def _on_nuclei_method_changed(self, *_):
        """Show only the controls relevant to the selected nuclei method:
        the .cl classifier + size gates for APOC, spot/tophat/outline σ for
        Voronoi-Otsu."""
        apoc = self.nuclei_method.value == "APOC classifier"
        self.nuclei_classifier.visible = apoc
        self.nuclei_min_size.visible = apoc
        self.nuclei_max_size.visible = apoc
        self.nuclei_sigma.visible = not apoc
        self.tophat_radius.visible = not apoc
        self.nuclei_outline_sigma.visible = not apoc

    def _on_directional_pairing_changed(self, *_):
        """Show the axis-search controls only when directional pairing is on."""
        on = self.directional_pairing.value
        self.pair_search_dist.visible = on
        self.pair_cone_angle.visible = on

    def _after_assign(self, viewer, state):
        """Draw centroids + association overlays, fill the property table, wire
        the selection sync, then flash the completion image."""
        self.cilia_pts = show_centroids(viewer, state)
        show_associations(viewer, state)
        self._attach_boxes(state)
        self._populate_table(state.get("cilia_df"))
        if self.cilia_pts is not None:
            # napari point selection → table highlight
            try:
                self.cilia_pts.selected_data.events.changed.connect(self._on_cilia_selection)
            except Exception:
                self.cilia_pts.events.highlight.connect(self._on_cilia_selection)
        self._flash_complete()

    # individual step handlers ----------------------------------------------------
    def _do_load(self):
        self._run_step(step_load, None, self._after_load, "Load & normalise")

    def _do_cilia(self):
        self._run_step(step_cilia, ("raw",),
                       lambda v, s: show_labels(v, s, "cilia_labels", "LC: Cilia Labels"),
                       "Segment cilia")

    # ── manual cilia editing (add / delete between steps) ────────────────────────
    _CILIA_LAYER = "LC: Cilia Labels"

    def _cilia_layer(self):
        """The editable cilia Labels layer, or None (with a warning)."""
        if self._CILIA_LAYER in self.viewer.layers:
            return self.viewer.layers[self._CILIA_LAYER]
        show_warning("Run step 2 (Segment cilia) first.")
        return None

    def _do_add_cilium(self):
        """Paint a brand-new cilium: select the cilia layer, pick an unused label
        id and switch to a 3-D paint brush. Click ✅ Apply edits when done."""
        lyr = self._cilia_layer()
        if lyr is None:
            return
        data = np.asarray(lyr.data)
        new_id = int(data.max()) + 1 if data.size else 1
        self.viewer.layers.selection.active = lyr
        lyr.selected_label = new_id
        try:                                    # make the brush fill across Z too
            lyr.n_edit_dimensions = data.ndim
        except Exception:                       # noqa: BLE001 (older napari)
            pass
        lyr.mode = "paint"
        self._set_status(
            f"✏️ Paint the new cilium (id {new_id}); brush paints in 3-D. "
            "Click ✅ Apply edits to commit.")

    def _toggle_delete_cilium(self):
        """Toggle a click-to-delete mode: clicking a cilium zeroes that label."""
        lyr = self._cilia_layer()
        if lyr is None:
            return
        if self._del_cb is not None:            # turn OFF
            try:
                lyr.mouse_drag_callbacks.remove(self._del_cb)
            except ValueError:
                pass
            self._del_cb = None
            self.del_cilium_btn.text = "🗑️ Delete cilium (click)"
            self._set_status("Delete mode off.")
            return

        self.viewer.layers.selection.active = lyr
        lyr.mode = "pan_zoom"                    # so a click reads, doesn't paint

        def _cb(layer, event):
            data = np.asarray(layer.data)
            try:
                idx = tuple(int(round(c)) for c in layer.world_to_data(event.position))
            except Exception:                   # noqa: BLE001
                return
            if len(idx) != data.ndim or not all(0 <= i < s for i, s in zip(idx, data.shape)):
                return
            val = int(data[idx])
            if val:
                data[data == val] = 0
                layer.data = data
                layer.refresh()
                self._set_status(
                    f"🗑️ Deleted cilium #{val}. Click ✅ Apply edits to commit.")

        self._del_cb = _cb
        lyr.mouse_drag_callbacks.append(_cb)
        self.del_cilium_btn.text = "🗑️ Delete: ON — click cilia"
        self._set_status("Delete mode ON — click cilia to remove; toggle off when done.")

    def _do_apply_cilia_edits(self):
        """Commit the layer's manual edits back into the pipeline state.

        Re-derives one label per connected component (so painted blobs become
        distinct cilia and ids stay contiguous), then refreshes the layer. Re-run
        step 7 (Assign & classify) to fold the edits into the results."""
        lyr = self._cilia_layer()
        if lyr is None:
            return
        if self._del_cb is not None:            # leave delete mode first
            self._toggle_delete_cilium()
        from skimage.measure import label as _cc
        data = np.asarray(lyr.data)
        relabeled = _cc(data > 0).astype(np.int32)
        self.state["cilia_labels"] = relabeled
        lyr.mode = "pan_zoom"
        show_labels(self.viewer, self.state, "cilia_labels", self._CILIA_LAYER)
        n = int(relabeled.max())
        self._set_status(f"✅ Applied cilia edits — {n} cilia. Re-run step 7 (Assign & classify).")
        show_info(f"{n} cilia after manual edits. Re-run “7 · Assign & classify”.")

    def _do_nuclei(self):
        self._run_step(step_nuclei, ("norm",),
                       lambda v, s: show_labels(v, s, "nuclei_labels", "LC: Nuclei Labels"),
                       "Segment nuclei")

    def _do_neurites(self):
        def _after(v, s):
            show_labels(v, s, "neurite_labels", "LC: Neurite Labels")
            show_skeleton(v, s)
        self._run_step(step_neurites, ("norm",), _after, "Segment neurites")

    def _do_bb(self):
        self._run_step(step_basal_bodies, ("raw",),
                       lambda v, s: show_labels(v, s, "bb_labels", "LC: Basal Body Labels"),
                       "Segment basal bodies")

    def _do_distances(self):
        self._run_step(step_distances, ("neurite_labels", "nuclei_labels", "skeleton_mask"),
                       lambda v, s: show_ratio(v, s), "Distance maps")

    def _do_assign(self):
        self._run_step(
            step_assign,
            ("cilia_labels", "bb_labels", "dist_to_neurite", "nearest_skel_idx"),
            self._after_assign, "Assign & classify",
        )

    def _do_run_all(self):
        """Chain every step for the current image (each runs in its own worker)."""
        if self._busy:
            return
        p = self._params()
        if not p["ims_path"]:
            show_warning("Open a folder and select an image first.")
            return
        self.state["channels"] = {
            "ch_cilia": p["ch_cilia"], "ch_neurites": p["ch_neurites"],
            "ch_bb": p["ch_bb"], "ch_nuclei": p["ch_nuclei"],
        }
        sequence = [
            (step_load,          self._after_load,                                       "Load & normalise"),
            (step_cilia,         lambda v, s: show_labels(v, s, "cilia_labels",   "LC: Cilia Labels"),      "Segment cilia"),
            (step_nuclei,        lambda v, s: show_labels(v, s, "nuclei_labels",  "LC: Nuclei Labels"),     "Segment nuclei"),
            (step_neurites,      lambda v, s: (show_labels(v, s, "neurite_labels", "LC: Neurite Labels"), show_skeleton(v, s)),    "Segment neurites"),
            (step_basal_bodies,  lambda v, s: show_labels(v, s, "bb_labels",      "LC: Basal Body Labels"), "Segment basal bodies"),
            (step_distances,     show_ratio,                                             "Distance maps"),
            (step_assign,        self._after_assign,                                     "Assign & classify"),
        ]
        self._set_busy(True)

        def _run_idx(i: int):
            if i >= len(sequence):
                self._set_busy(False)
                self._set_status("✅ All steps complete.")
                show_info("Pipeline complete for this image.")
                return  # the assign step's callback already flashed completion
            fn, after, label = sequence[i]
            self._set_status(f"⚙️ [{i + 1}/{len(sequence)}] {label} …")
            self._progress_set(i, len(sequence), f"{i + 1}/{len(sequence)}  {label} …")

            @thread_worker
            def _work():
                _select_device(p["gpu_device"])
                return fn(self.state, p)

            def _done(updates):
                self.state.update(updates)
                after(self.viewer, self.state)
                _run_idx(i + 1)

            def _err(e):
                self._set_busy(False)
                show_warning(f"❌ {label} failed: {e}")

            w = _work()
            w.returned.connect(_done)
            w.errored.connect(_err)
            w.start()

        _run_idx(0)

    # ── batch ───────────────────────────────────────────────────────────────────
    def _batch_kwargs(self, p: dict) -> tuple[dict, dict]:
        """Shared kwargs for the full pipeline and the ROI-only fast batch, minus
        ``input_path`` / ``output_path`` (the callers fill those in per run so the
        same parameter set drives both single-folder and super-batch)."""
        kwargs = dict(
            gpu_device=p["gpu_device"],
            cilia_classifier_path=p["classifier_path"],
            cilia_channel=p["ch_cilia"], neurites_channel=p["ch_neurites"],
            basal_bodies_channel=p["ch_bb"], nuclei_channel=p["ch_nuclei"],
            use_mip=p["use_mip"], make_isotropic=p["make_isotropic"],
            p_low=p["p_low"], p_high=p["p_high"],
            per_channel_norm=p["ch_norm"] or None,
            nuclei_spot_sigma=p["nuclei_sigma"], tophat_radius=p["tophat_radius"],
            outline_sigma=p["nuclei_outline_sigma"], nuclei_log=p["nuclei_log"],
            nuclei_gaussian_sigma=(p["nuclei_gauss_z"], p["nuclei_gauss_y"], p["nuclei_gauss_x"]),
            neurite_spot_sigma=p["neurite_sigma"], neurite_log=p["neurite_log"],
            merge_nuclei_neurite=p["merge_nuclei_neurite"],
            neurite_gaussian_sigma=(p["neurite_gauss_z"], p["neurite_gauss_y"], p["neurite_gauss_x"]),
            cilia_log=p["cilia_log"],
            cilia_gaussian_sigma=(p["cilia_gauss_z"], p["cilia_gauss_y"], p["cilia_gauss_x"]),
            cilia_min_size=p["cilia_min_size"], cilia_max_size=p["cilia_max_size"],
            bb_method=p["bb_method"], bb_classifier_path=p["bb_classifier_path"],
            bb_spot_sigma=p["bb_spot_sigma"], bb_outline_sigma=p["bb_outline_sigma"],
            bb_log=p["bb_log"],
            bb_gaussian_sigma=(p["bb_gauss_z"], p["bb_gauss_y"], p["bb_gauss_x"]),
            bb_min_size=p["bb_min_size"], bb_max_size=p["bb_max_size"],
            max_cilia_dist_cutoff_um=p["max_cilia_dist_um"],
            max_basal_body_cutoff_um=p["max_basal_dist_um"],
            require_basal_body=p["require_basal_body"],
            ratio_from_basal_body=p["ratio_from_basal_body"],
            ratio_epsilon=p["ratio_epsilon"],
            neurite_threshold=p["neurite_threshold"], soma_threshold=p["soma_threshold"],
            # Per-cilium ROIs are now exported inside the pipeline (fast, no GUI),
            # so they're produced whether or not live capture is on.
            save_rois=self.batch_rois.value or self.batch_ai.value,
            roi_correct_display=self.roi_correct.value,
            # Optional AI validation → writes csv/human_validation.csv (uses the
            # model + threshold chosen in the 🤖 AI cilia validation section).
            batch_ai_model=(
                os.path.join(self._roi_models_dir(), str(self.ai_model_combo.value))
                if self.batch_ai.value
                and str(self.ai_model_combo.value).endswith(".pt") else None),
            batch_ai_threshold=float(self.ai_keep_thr.value),
        )
        roi_kwargs = dict(
            gpu_device=p["gpu_device"],
            cilia_classifier_path=p["classifier_path"],
            cilia_channel=p["ch_cilia"], neurites_channel=p["ch_neurites"],
            basal_bodies_channel=p["ch_bb"], nuclei_channel=p["ch_nuclei"],
            use_mip=p["use_mip"], make_isotropic=p["make_isotropic"],
            p_low=p["p_low"], p_high=p["p_high"],
            per_channel_norm=p["ch_norm"] or None,
            cilia_log=p["cilia_log"],
            cilia_gaussian_sigma=(p["cilia_gauss_z"], p["cilia_gauss_y"], p["cilia_gauss_x"]),
            cilia_min_size=p["cilia_min_size"], cilia_max_size=p["cilia_max_size"],
            bb_method=p["bb_method"], bb_classifier_path=p["bb_classifier_path"],
            bb_spot_sigma=p["bb_spot_sigma"], bb_outline_sigma=p["bb_outline_sigma"],
            bb_log=p["bb_log"],
            bb_gaussian_sigma=(p["bb_gauss_z"], p["bb_gauss_y"], p["bb_gauss_x"]),
            bb_min_size=p["bb_min_size"], bb_max_size=p["bb_max_size"],
            roi_correct_display=self.roi_correct.value,
            expected_xy_um=(self.batch_xy_um.value or None),
            roi_sample_frac=float(self.batch_roi_pct.value) / 100.0,
            roi_tile_px=int(self.batch_roi_tile.value),
        )
        return kwargs, roi_kwargs

    @staticmethod
    def _discover_ims_folders(root: Path) -> list[Path]:
        """Every folder at/under ``root`` that *directly* holds ≥1 .ims file,
        sorted. Used by super-batch to walk a tree of sample folders."""
        hits = []
        for dp, _dn, fns in os.walk(root):
            if any(f.lower().endswith(".ims") for f in fns):
                hits.append(Path(dp))
        return sorted(hits)

    def _do_super_batch(self):
        """Super batch: process every subfolder (recursively) that contains .ims
        files as its own run, mirroring each input folder's path under the output
        folder so the naming convention is preserved. Reuses every batch option
        (full vs ROI-only, live capture, AI validation)."""
        if self._busy:
            return
        root = Path(str(self.folder.value))
        out_root = Path(str(self.output.value))
        if not root.is_dir():
            show_warning("Super batch needs an input FOLDER (not a .txt manifest).")
            return
        if not str(self.output.value):
            show_warning("Select an output folder first.")
            return
        folders = self._discover_ims_folders(root)
        if not folders:
            show_warning(f"No .ims files found in {root} or any subfolder.")
            return

        p = self._params()
        kwargs, roi_kwargs = self._batch_kwargs(p)
        bridge     = self._bridge
        roi_only   = self.batch_roi_only.value
        capture    = self.batch_capture.value and not roi_only
        viz_bridge = self._viz_bridge
        n_fold     = len(folders)

        def _per_file(payload):
            payload["capture_rois"] = False   # ROIs handled by the pipeline now
            done = threading.Event()
            viz_bridge.visualize.emit(payload, done)
            done.wait()

        self._set_busy(True)
        self._set_status(f"⚙️ Super batch — {n_fold} folder(s) …")

        @thread_worker
        def _work():
            for fi, fdir in enumerate(folders):
                rel = fdir.relative_to(root)
                # Mirror the input tree under the output folder, keeping names.
                # The root itself (rel == '.') maps to a folder named after root.
                sub = str(rel) if str(rel) != "." else root.name
                out_sub = out_root / sub
                out_sub.mkdir(parents=True, exist_ok=True)

                def _cb(i, n, f, _fi=fi, _sub=sub):
                    bridge.progressed.emit(i, n, f"[{_fi + 1}/{n_fold}] {_sub} — {f}")

                if roi_only:
                    run_roi_only_batch(
                        **{**roi_kwargs, "input_path": str(fdir),
                           "output_path": str(out_sub)},
                        progress_callback=_cb,
                    )
                else:
                    run_pipeline3(
                        **{**kwargs, "input_path": str(fdir),
                           "output_path": str(out_sub)},
                        progress_callback=_cb,
                        per_file_callback=_per_file if capture else None,
                    )

        def _done(_):
            self._set_busy(False)
            self._set_status(f"✅ Super batch complete — {n_fold} folder(s). See output.")
            show_info("Super batch processing complete.")
            self._flash_complete()

        def _err(e):
            self._set_busy(False)
            show_warning(f"❌ Super batch failed: {e}")

        w = _work()
        w.returned.connect(_done)
        w.errored.connect(_err)
        w.start()

    def _do_batch(self):
        if self._busy:
            return
        folder = Path(str(self.folder.value))
        out    = Path(str(self.output.value))
        # Input may be a folder of .ims OR a .txt manifest (one .ims path per
        # line) — type/paste the manifest path into the input folder field.
        _is_manifest = folder.is_file() and folder.suffix.lower() == ".txt"
        if not (folder.is_dir() or _is_manifest):
            show_warning("Select an input folder, or a .txt manifest of .ims paths.")
            return
        if not str(out):
            show_warning("Select an output folder first.")
            return
        out.mkdir(parents=True, exist_ok=True)
        p = self._params()
        self._set_busy(True)
        self._set_status("⚙️ Batch running over folder …")

        kwargs, roi_kwargs = self._batch_kwargs(p)
        kwargs.update(input_path=str(folder), output_path=str(out))
        roi_kwargs.update(input_path=str(folder), output_path=str(out))

        bridge = self._bridge   # emit progress from the worker thread → GUI thread
        roi_only = self.batch_roi_only.value

        # Per-file live visualisation (optional). The callback runs in the worker
        # thread: it hands the file's arrays to the GUI thread and blocks until
        # the screenshots are taken, so napari is only ever touched on the GUI thread.
        capture     = self.batch_capture.value and not roi_only
        viz_bridge  = self._viz_bridge

        def _per_file(payload):
            payload["capture_rois"] = False   # ROIs handled by the pipeline now
            done = threading.Event()
            viz_bridge.visualize.emit(payload, done)
            done.wait()

        @thread_worker
        def _work():
            if roi_only:
                run_roi_only_batch(
                    **roi_kwargs,
                    progress_callback=lambda i, n, f: bridge.progressed.emit(i, n, f),
                )
            else:
                run_pipeline3(
                    **kwargs,
                    progress_callback=lambda i, n, f: bridge.progressed.emit(i, n, f),
                    per_file_callback=_per_file if capture else None,
                )

        def _done(_):
            self._set_busy(False)
            self._set_status("✅ Batch complete — see output folder.")
            show_info("Batch processing complete.")
            self._flash_complete()

        def _err(e):
            self._set_busy(False)
            show_warning(f"❌ Batch failed: {e}")

        w = _work()
        w.returned.connect(_done)
        w.errored.connect(_err)
        w.start()

    # ── batch live visualisation (GUI thread) ────────────────────────────────────
    def _screenshot(self, path):
        try:
            # Let the event loop process pending repaints/renders so the window
            # stays responsive and the canvas is fully drawn before capture.
            QApplication.processEvents()
            self.viewer.screenshot(str(path), canvas_only=True, flash=False)
            QApplication.processEvents()
        except Exception as exc:
            print(f"[LC] screenshot failed ({path}): {exc}")

    def _clear_named_layers(self, prefix: str):
        for lyr in list(self.viewer.layers):
            if lyr.name.startswith(prefix):
                self.viewer.layers.remove(lyr)

    def _show_only(self, visible_names: set[str]):
        """Toggle visibility of the ``LC:`` layers so only ``visible_names`` show."""
        for lyr in self.viewer.layers:
            if lyr.name.startswith("LC:"):
                lyr.visible = lyr.name in visible_names

    def _capture_progress(self, text: str):
        """Show fine-grained capture progress on the progress bar (no popup)."""
        self.progress.setFormat(text)
        QApplication.processEvents()

    def _on_batch_visualize(self, payload: dict, done: threading.Event):
        """Display one batch file's layers, screenshot the overlay (+ optional
        per-cilium 3-D ROIs), then clear layers and release the worker.

        Must never raise: the worker is always unblocked via ``done.set()``.
        """
        try:
            state = {
                "raw":              payload["raw"],
                "voxel_size":       payload["voxel_size"],
                "channels":         payload["channels"],
                "cilia_labels":     payload["cilia_labels"],
                "bb_labels":        payload["bb_labels"],
                "nuclei_labels":    payload["nuclei_labels"],
                "neurite_labels":   payload["neurite_labels"],
                "skeleton_mask":    payload["skeleton_mask"],
                "nearest_skel_idx": payload["nearest_skel_idx"],
                "log_ratio_map":    payload.get("log_ratio_map"),
                "cilia_df":         payload["cilia_df"],
                "bb_df":            payload["bb_df"],
            }
            self._clear_named_layers("LC:")
            try:
                self.viewer.dims.ndisplay = 3
            except Exception:
                pass

            # Build every layer once, then shoot each overview by toggling which
            # layers are visible.
            show_channels(self.viewer, state)
            show_labels(self.viewer, state, "cilia_labels",   "LC: Cilia Labels")
            show_labels(self.viewer, state, "nuclei_labels",  "LC: Nuclei Labels")
            show_labels(self.viewer, state, "neurite_labels", "LC: Neurite Labels")
            show_labels(self.viewer, state, "bb_labels",      "LC: Basal Body Labels")
            if state.get("log_ratio_map") is not None:
                show_ratio(self.viewer, state)
            show_centroids(self.viewer, state)
            show_associations(self.viewer, state)
            self.viewer.reset_view()
            QApplication.processEvents()        # repaint after the heavy layer build
            self._set_status(f"📸 {payload['stem']}")

            sdir = Path(payload["screenshots_dir"])
            stem = payload["stem"]
            dots = "LC: Cilia Centroids"
            overviews = [
                ("logratio", {"LC: Log Ratio", dots}),
                ("channels", {"LC: Raw Cilia", "LC: Raw Neurites",
                              "LC: Raw Basal Bodies", "LC: Raw Nuclei", dots}),
                ("labels",   {"LC: Cilia Labels", "LC: Basal Body Labels", dots}),
            ]
            for i, (tag, visible) in enumerate(overviews, 1):
                self._show_only(visible)
                self._capture_progress(f"📸 {stem} — overview {i}/{len(overviews)} ({tag})")
                self._screenshot(sdir / f"{stem}_{tag}.png")

            if payload.get("capture_rois"):
                self._capture_cilia_rois(state, Path(payload["rois_dir"]), payload["stem"])
        except Exception as exc:
            print(f"[LC] batch visualisation error: {exc}")
        finally:
            self._clear_named_layers("ROI:")
            self._clear_named_layers("LC:")
            done.set()

    def _capture_cilia_rois(self, state: dict, roi_dir: Path, stem: str, margin: int = 22):
        """Export one fast 2-D MIP thumbnail (+ raw .npz crop) per cilium.

        Pure NumPy/Pillow — no napari rendering — so it's orders of magnitude
        faster than the old per-cilium 3-D screenshot loop while still giving the
        human a clear close-up (top-down XY + side XZ) in the Screening gallery.
        The raw crops let a classifier later train on the actual data.
        """
        cdf = state.get("cilia_df")
        if cdf is None or cdf.empty:
            return
        from limoncello.visualization.cilia_rois import save_cilia_rois
        self._capture_progress(f"📸 {stem} — exporting {len(cdf)} cilia ROIs…")
        try:
            saved = save_cilia_rois(
                state["raw"], state["cilia_labels"], state["bb_labels"],
                cdf, state["channels"], state["voxel_size"],
                Path(roi_dir), stem, margin=margin,
                correct_display=self.roi_correct.value,
            )
            print(f"[LC] saved {saved}/{len(cdf)} cilia ROIs for {stem} → {roi_dir}")
        except Exception as exc:                          # never abort the batch
            print(f"[LC] ROI export failed for {stem}: {exc}")

    # ── AI cilia validation (run the ROI-validator CNN on this image) ─────────────
    def _roi_models_dir(self) -> str:
        """Codebase ``models/`` dir (next to this file), where ROI validators live."""
        try:
            base = os.path.dirname(os.path.abspath(__file__))
        except NameError:                                  # __file__ may be unset
            base = os.path.abspath(".")
        return os.path.join(base, "models")

    def _refresh_ai_models(self):
        """Repopulate the AI-model dropdown from the codebase ``models/`` dir."""
        d = self._roi_models_dir()
        found = [f for f in sorted(os.listdir(d))
                 if f.endswith(".pt")] if os.path.isdir(d) else []
        self.ai_model_combo.choices = found or ["(no models found)"]

    def _do_ai_validate(self):
        """Score this image's cilia ROIs with the chosen validator CNN and add a
        Labels layer showing only the AI-kept (score ≥ threshold) cilia."""
        if self._busy:
            return
        state = self.state
        cdf = state.get("cilia_df")
        if cdf is None or cdf.empty or "cilia_labels" not in state \
                or "cilia_id" not in getattr(cdf, "columns", []):
            show_warning("Run cilia segmentation + assign on an image first.")
            return
        sel = str(self.ai_model_combo.value or "")
        mpath = os.path.join(self._roi_models_dir(), sel)
        if not sel.endswith(".pt") or not os.path.isfile(mpath):
            show_warning("Pick a trained model (.pt) in the AI validation section.")
            return
        thr = float(self.ai_keep_thr.value)
        self._set_busy(True)
        self._set_status("🤖 AI-validating cilia …")

        @thread_worker
        def _work():
            import tempfile, shutil
            from limoncello.visualization.cilia_rois import save_cilia_rois
            from limoncello.ml.roi_validator import load_bundle, predict_proba
            tmp = Path(tempfile.mkdtemp(prefix="lc_ai_"))
            try:
                # Render the SAME thumbnails the model trained on (no raw crops).
                save_cilia_rois(
                    state["raw"], state["cilia_labels"], state["bb_labels"],
                    cdf, state["channels"], state["voxel_size"],
                    tmp, "ai", save_crops=False)
                model, meta = load_bundle(mpath)
                ids = [int(c) for c in cdf["cilia_id"].tolist()]
                paths = [str(tmp / f"ai_cilia{c}.png") for c in ids]
                probs = predict_proba(model, paths, size=int(meta.get("size", 64)),
                                       normalize=meta.get("normalize"))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            return ids, np.asarray(probs, dtype=float)

        def _done(res):
            ids, probs = res
            keep_ids = [i for i, p in zip(ids, probs)
                        if np.isfinite(p) and p >= thr]
            keep_set = set(keep_ids)
            state["ai_keep_ids"] = keep_ids

            # Write scores back into the cilia table + refresh the displayed grid.
            _score = {int(i): float(p) for i, p in zip(ids, probs)
                      if np.isfinite(p)}
            cdf2 = state.get("cilia_df")
            if cdf2 is not None and not cdf2.empty:
                cdf2 = cdf2.copy()
                cdf2["ai_score"] = [_score.get(int(c), np.nan)
                                    for c in cdf2["cilia_id"]]
                cdf2["ai_validated"] = [bool(int(c) in keep_set)
                                        for c in cdf2["cilia_id"]]
                state["cilia_df"] = cdf2
                self._populate_table(cdf2)

            # Mark cilia with Points layers at their centroids (scaled to physical
            # µm like the other centroid layers): green rings = AI-kept,
            # red rings = AI-rejected. Each ring is labelled with its score.
            _remove(self.viewer, "LC: AI-validated", "LC: AI-rejected",
                    "LC: AI-kept Cilia")

            def _ai_points(sub, name, color):
                if sub.empty:
                    return
                _coords = np.nan_to_num(
                    np.asarray(sub["coords"].tolist(), dtype=float))
                _sc = sub["ai_score"].to_numpy(dtype=float)
                self.viewer.add_points(
                    _coords, name=name, scale=state["voxel_size"], size=10,
                    symbol="ring", face_color=color,
                    properties={"ai_score": _sc},
                    text={"string": "{ai_score:.2f}", "size": 8,
                          "color": color, "translation": [0, -6, 0]},
                )

            if cdf2 is not None and not cdf2.empty:
                _scored = cdf2[cdf2["ai_score"].notna()]
                _ai_points(_scored[_scored["ai_validated"]],
                           "LC: AI-validated", "lime")
                _ai_points(_scored[~_scored["ai_validated"]],
                           "LC: AI-rejected", "red")

            self._set_busy(False)
            n_eval = int(np.isfinite(probs).sum())
            self._set_status(
                f"🤖 AI kept {len(keep_ids)}/{n_eval} cilia (score ≥ {thr:.2f}).")
            show_info(f"AI validated {len(keep_ids)} of {n_eval} cilia.")

        def _err(e):
            self._set_busy(False)
            show_warning(f"❌ AI validation failed: {e}")

        w = _work()
        w.returned.connect(_done)
        w.errored.connect(_err)
        w.start()

    # ── classifier training ──────────────────────────────────────────────────────
    def _resolve_train_output(self, ch: int) -> str:
        """Return the output ``.cl`` path: the field value if set, else a default
        ``segmenter_ch{ch}.cl`` in the labels/input folder. Always ends in .cl
        and is written back to the field."""
        out = str(self.train_output.value or "").strip()
        folder = Path(str(self.folder.value)) if self.folder.value else Path(".")
        labels_dir = str(self.train_labels_dir.value or "")
        if not out or out in (".", str(folder)):
            base = Path(labels_dir) if (labels_dir and Path(labels_dir).is_dir()) else folder
            out = str(base / f"segmenter_ch{ch}.cl")
        if not out.lower().endswith(".cl"):
            out += ".cl"
        self.train_output.value = out
        return out

    # ── feature-selection grid (operation × sigma checkboxes) ────────────────
    _OP_SHORT = {"gaussian_blur": "Gauss", "difference_of_gaussian": "DoG",
                 "laplace_box_of_gaussian_blur": "LoG", "sobel_of_gaussian_blur": "Sobel"}

    def _parse_sigmas(self):
        """Sigma column values from the σ field, sorted & de-duplicated."""
        out = []
        for tok in str(self.train_feat_sigmas.value).replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(float(tok))
            except ValueError:
                continue
        return sorted(dict.fromkeys(out)) or [1.0, 2.0, 3.0, 5.0, 10.0]

    def _build_feature_grid(self):
        """A checkbox grid: rows = filter operations, columns = sigma scales.
        Ticked cells become ``operation=sigma`` features (napari-apoc style)."""
        self._feat_sigmas = self._parse_sigmas()
        t = QTableWidget()
        t.setRowCount(len(FEATURE_OPERATIONS))
        t.setVerticalHeaderLabels([self._OP_SHORT.get(o, o) for o in FEATURE_OPERATIONS])
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionMode(QAbstractItemView.NoSelection)
        # show every row, no inner scrolling (user-friendly full-size grid)
        t.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        t.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        t.horizontalHeader().setStretchLastSection(True)
        self.feat_table = t
        self._populate_feature_columns(default_checked=True)
        # double-click header to toggle a whole column would be nice, but keep simple
        t.cellDoubleClicked.connect(self._toggle_feature_cell)

    def _populate_feature_columns(self, default_checked=True, preserve=True):
        """(Re)create grid columns from the σ list, keeping prior checks if asked."""
        prev = {}
        if preserve and self.feat_table.columnCount():
            for r, op in enumerate(FEATURE_OPERATIONS):
                for c, s in enumerate(getattr(self, "_feat_sigmas_shown", [])):
                    it = self.feat_table.item(r, c)
                    if it is not None:
                        prev[(op, s)] = it.checkState() == Qt.Checked
        sig = self._feat_sigmas
        self.feat_table.setColumnCount(len(sig))
        self.feat_table.setHorizontalHeaderLabels([f"σ{s:g}" for s in sig])
        for r, op in enumerate(FEATURE_OPERATIONS):
            for c, s in enumerate(sig):
                it = QTableWidgetItem()
                it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                checked = prev.get((op, s), default_checked)
                it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                it.setToolTip(f"{op}={s:g}")
                self.feat_table.setItem(r, c, it)
        self._feat_sigmas_shown = list(sig)
        self.feat_table.resizeColumnsToContents()
        self._fit_feature_grid_height()

    def _fit_feature_grid_height(self):
        """Size the grid to show all rows so it never scrolls vertically."""
        t = self.feat_table
        h = t.horizontalHeader().height() + 2 * t.frameWidth()
        for r in range(t.rowCount()):
            h += t.rowHeight(r)
        t.setFixedHeight(h)

    def _rebuild_feature_columns(self):
        self._feat_sigmas = self._parse_sigmas()
        self._populate_feature_columns(default_checked=True, preserve=True)

    def _toggle_feature_cell(self, r, c):
        it = self.feat_table.item(r, c)
        if it is not None:
            it.setCheckState(Qt.Unchecked if it.checkState() == Qt.Checked else Qt.Checked)

    def _feature_spec_from_grid(self) -> str:
        """Build the APOC feature_specification from the ticked grid cells."""
        pairs = []
        for r, op in enumerate(FEATURE_OPERATIONS):
            for c, s in enumerate(self._feat_sigmas_shown):
                it = self.feat_table.item(r, c)
                if it is not None and it.checkState() == Qt.Checked:
                    pairs.append((op, s))
        return feature_spec_from_pairs(pairs, bool(self.train_feat_original.value))

    # ── feature-importance table (filled after training) ─────────────────────
    def _build_importance_table(self):
        t = QTableWidget()
        t.setColumnCount(4)
        t.setHorizontalHeaderLabels(["feature", "import.", "share", "weight"])
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setMinimumHeight(200)
        t.horizontalHeader().setStretchLastSection(True)
        self.imp_table = t

    def _set_importance_table(self, rows):
        """Fill the importance table (rows: ``[(feature, importance), …]`` desc)
        with share %% + weight bars, and tint the grid cells accordingly."""
        self.imp_table.setRowCount(0)
        if not rows:
            return
        mx = max((imp for _, imp in rows), default=0.0) or 1.0
        tot = sum(imp for _, imp in rows) or 1.0
        self.imp_table.setRowCount(len(rows))
        for i, (feat, imp) in enumerate(rows):
            self.imp_table.setItem(i, 0, QTableWidgetItem(str(feat)))
            self.imp_table.setItem(i, 1, QTableWidgetItem(f"{imp:.4f}"))
            self.imp_table.setItem(i, 2, QTableWidgetItem(f"{100 * imp / tot:.1f}%"))
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(round(100 * imp / mx)))
            bar.setTextVisible(False)
            bar.setMaximumHeight(14)
            self.imp_table.setCellWidget(i, 3, bar)
        self.imp_table.resizeColumnsToContents()
        self._tint_feature_grid({str(f): imp for f, imp in rows}, mx)

    def _tint_feature_grid(self, imp_by_feature, mx):
        """Shade grid cells green by how important that feature turned out."""
        for r, op in enumerate(FEATURE_OPERATIONS):
            for c, s in enumerate(self._feat_sigmas_shown):
                it = self.feat_table.item(r, c)
                if it is None:
                    continue
                imp = imp_by_feature.get(f"{op}={s:g}")
                if imp is None:
                    it.setBackground(QColor(0, 0, 0, 0))
                    it.setToolTip(f"{op}={s:g}")
                else:
                    frac = max(0.0, min(1.0, imp / mx))
                    it.setBackground(QColor(46, 204, 113, int(40 + 180 * frac)))
                    it.setToolTip(f"{op}={s:g} · importance {imp:.4f}")

    def _refresh_train_annotation(self, *_):
        """Sync the annotation-layer dropdown with the Labels layers in the viewer."""
        names = [lyr.name for lyr in self.viewer.layers
                 if isinstance(lyr, napari.layers.Labels)]
        cur = self.train_annotation.value
        self.train_annotation.choices = names
        if names:
            self.train_annotation.value = cur if cur in names else names[-1]

    def _do_train_current(self):
        """APOC-plugin style: train (or extend) a classifier on the currently
        loaded image channel + a painted annotation layer, then preview the
        prediction as a new Labels layer in the viewer."""
        if self._busy:
            return
        if not self.state or "raw" not in self.state:
            show_warning("Load an image first (step 1 · Load & normalise).")
            return
        ann_name = self.train_annotation.value
        layer = next((lyr for lyr in self.viewer.layers
                      if lyr.name == ann_name and isinstance(lyr, napari.layers.Labels)),
                     None)
        if layer is None:
            show_warning("Pick an annotation (Labels) layer. Use '🏷️ New empty "
                         "label layer', then paint background=1 and object=2.")
            return
        ch  = self.train_channel.value
        chc = _clamp_ch(self.state["raw"].shape[0], ch, "train")
        image = np.asarray(self.state["raw"][chc])
        gt    = np.asarray(layer.data)
        pos   = self.train_positive.value
        if gt.shape != image.shape:
            show_warning(f"Annotation shape {gt.shape} ≠ image channel shape "
                         f"{image.shape}. Create the label layer while this image "
                         "is loaded.")
            return
        if int(gt.max()) < pos:
            show_warning(f"Paint background=1 and object={pos} before training.")
            return

        out = self._resolve_train_output(chc)
        features = self._feature_spec_from_grid()
        cont = bool(self.train_continue.value)
        gpu  = self.gpu_combo.value
        vs   = self.state["voxel_size"]
        img_shape = image.shape
        params = dict(positive_class=pos, max_depth=self.train_max_depth.value,
                      num_trees=self.train_num_trees.value)

        self._set_busy(True)
        self._progress_busy("Training on current image …")
        self._set_status(f"🎓 {'Extending' if cont else 'Training'} classifier on "
                         f"current image (ch {chc}) …")

        @thread_worker
        def _work():
            _select_device(gpu)
            train_segmenter_single(image, gt, out, features=features,
                                   continue_training=cont, gpu_device=None, **params)
            pred = predict_segmenter(image, out, gpu_device=None)
            return dict(out=out, pred=pred, channel=chc)

        def _done(res):
            self._set_busy(False)
            _remove(self.viewer, "LC: Prediction (train)")
            pred = np.asarray(res["pred"])
            if pred.shape != img_shape and pred.size == int(np.prod(img_shape)):
                pred = pred.reshape(img_shape)
            self.viewer.add_labels(pred, name="LC: Prediction (train)", scale=vs)
            if res["channel"] == self.ch_cilia.value:
                self.classifier.value = res["out"]
            elif res["channel"] == self.ch_bb.value:
                self.bb_classifier.value = res["out"]
            try:
                self._set_importance_table(read_feature_importances(res["out"]))
            except Exception as exc:                      # noqa: BLE001
                print(f"[LC] could not read feature importances: {exc}")
            msg = (f"✅ {'Extended' if cont else 'Trained'} → {Path(res['out']).name}"
                   "  · preview layer added")
            self._set_status(msg)
            show_info(msg)

        def _err(e):
            self._set_busy(False)
            show_warning(f"❌ Training failed: {e}")

        w = _work()
        w.returned.connect(_done)
        w.errored.connect(_err)
        w.start()

    def _do_predict_current(self):
        """Apply the classifier in the output field to the current image channel
        and show the raw prediction — no retraining."""
        if self._busy:
            return
        if not self.state or "raw" not in self.state:
            show_warning("Load an image first (step 1 · Load & normalise).")
            return
        cl = str(self.train_output.value or "").strip()
        if not cl or not Path(cl).is_file():
            show_warning("No classifier .cl to apply — train one first or pick a file.")
            return
        ch  = self.train_channel.value
        chc = _clamp_ch(self.state["raw"].shape[0], ch, "predict")
        image = np.asarray(self.state["raw"][chc])
        gpu = self.gpu_combo.value
        vs  = self.state["voxel_size"]
        img_shape = image.shape

        self._set_busy(True)
        self._progress_busy("Predicting …")

        @thread_worker
        def _work():
            _select_device(gpu)
            return predict_segmenter(image, cl, gpu_device=None)

        def _done(pred):
            self._set_busy(False)
            _remove(self.viewer, "LC: Prediction (train)")
            pred = np.asarray(pred)
            if pred.shape != img_shape and pred.size == int(np.prod(img_shape)):
                pred = pred.reshape(img_shape)
            self.viewer.add_labels(pred, name="LC: Prediction (train)", scale=vs)
            self._set_status(f"👁 Applied {Path(cl).name} to current image (ch {chc}).")

        def _err(e):
            self._set_busy(False)
            show_warning(f"❌ Prediction failed: {e}")

        w = _work()
        w.returned.connect(_done)
        w.errored.connect(_err)
        w.start()

    def _do_train(self):
        """Train one APOC ObjectSegmenter on a chosen subset of images using their
        raw (un-normalised) channel + matching ``<stem>_labels.tif`` annotations."""
        if self._busy:
            return
        folder = Path(str(self.folder.value))
        if not folder.is_dir():
            show_warning("Scan a valid input folder first.")
            return
        labels_dir = str(self.train_labels_dir.value)
        if not labels_dir or not Path(labels_dir).is_dir():
            show_warning("Select a valid labels folder (with <stem>_labels.tif files).")
            return

        # Training set: either every scanned image that has a matching labels file
        # (default — decoupled from the click-to-view selection), or just the
        # images explicitly selected in the subset list.
        if self.train_all_labeled.value:
            selected = [f for f in (self.files or [])
                        if find_labels_file(labels_dir, Path(f).stem) is not None]
            if not selected:
                show_warning(f"No <stem>_labels.tif found in {labels_dir} for any "
                             "scanned image. Save some annotations first.")
                return
        else:
            selected = list(self.train_images.value or [])
            if not selected:
                show_warning("Select one or more images (subset) to train on, "
                             "or tick 'Train on ALL images with labels'.")
                return

        ch = self.train_channel.value
        out = self._resolve_train_output(ch)             # default name, always .cl
        features = self._feature_spec_from_grid()        # ticked op×σ grid cells

        image_paths = [str(folder / name) for name in selected]
        gpu         = self.gpu_combo.value
        # Match the geometry the annotations were painted on (MIP / isotropic).
        use_mip  = bool(self.use_mip.value)
        make_iso = bool(self.make_iso.value)
        params = dict(
            positive_class=self.train_positive.value,
            max_depth=self.train_max_depth.value,
            num_trees=self.train_num_trees.value,
        )

        self._set_busy(True)
        self._progress_busy("Training classifier …")
        self._set_status(f"🎓 Training on {len(image_paths)} image(s) (raw, no normalisation) …")

        @thread_worker
        def _work():
            _select_device(gpu)
            pairs, skipped = load_training_pairs(image_paths, labels_dir, ch,
                                                 use_mip=use_mip, make_iso=make_iso)
            if not pairs:
                reasons = "; ".join(f"{s}: {r}" for s, r in skipped) or "no usable pairs"
                raise RuntimeError(f"No (image, labels) pairs to train on — {reasons}")
            skip_stems = {s for s, _ in skipped}
            used = [Path(p).stem for p in image_paths if Path(p).stem not in skip_stems]
            print(f"[LC] training on {len(pairs)} image(s): {', '.join(used)}")
            train_object_segmenter(pairs, out, features=features, gpu_device=None, **params)
            return dict(n=len(pairs), skipped=skipped, out=out, channel=ch, used=used)

        def _done(res):
            self._set_busy(False)
            # Offer the freshly trained classifier to the matching segmenter field.
            if res["channel"] == self.ch_cilia.value:
                self.classifier.value = res["out"]
            elif res["channel"] == self.ch_bb.value:
                self.bb_classifier.value = res["out"]
            # Feature-importance statistics table
            try:
                self._set_importance_table(read_feature_importances(res["out"]))
            except Exception as exc:                      # noqa: BLE001
                print(f"[LC] could not read feature importances: {exc}")
            msg = (f"✅ Trained on {res['n']} image(s) → {Path(res['out']).name}  "
                   f"[{', '.join(res['used'])}]")
            if res["skipped"]:
                msg += f"  ({len(res['skipped'])} skipped: " \
                       + ", ".join(s for s, _ in res["skipped"]) + ")"
                for s, r in res["skipped"]:
                    print(f"[LC] train skipped {s}: {r}")
            self._set_status(msg)
            show_info(msg)

        def _err(e):
            self._set_busy(False)
            show_warning(f"❌ Training failed: {e}")

        w = _work()
        w.returned.connect(_done)
        w.errored.connect(_err)
        w.start()

    def _on_train_image_clicked(self, *_):
        """Selecting an image in the training subset loads it into the viewer with
        the current channel / normalisation / MIP / isotropic settings."""
        if self._busy:
            return
        cur = set(self.train_images.value or [])
        added = cur - self._train_sel_prev
        self._train_sel_prev = cur
        if not added:
            return
        name = sorted(added)[0]
        if name in (self.file_combo.choices or ()):
            self.file_combo.value = name          # triggers _on_file_changed (clears state)
            self._do_load()                       # load & display with current settings

    # ── label maker (annotations for training) ───────────────────────────────────
    def _ref_image_layer(self):
        """The first Image layer in the viewer, used to match shape/scale."""
        for lyr in self.viewer.layers:
            if isinstance(lyr, napari.layers.Image):
                return lyr
        return None

    def _new_label_layer(self):
        """Add an empty Labels layer matching the loaded image, ready to paint
        (APOC convention: background = 1, object = 2, unannotated = 0)."""
        ref = self._ref_image_layer()
        if ref is None:
            show_warning("Load an image first — no image layer to match.")
            return
        shape = np.asarray(ref.data).shape
        lyr = self.viewer.add_labels(
            np.zeros(shape, dtype=np.uint16), name="Annotations", scale=ref.scale,
        )
        lyr.selected_label = 2          # start painting the 'object' class
        try:
            lyr.mode = "paint"
        except Exception:
            pass
        self.viewer.layers.selection.active = lyr
        self._refresh_train_annotation()
        try:
            self.train_annotation.value = lyr.name
        except Exception:
            pass
        show_info("Empty label layer added. Paint background=1, object=2, then "
                  "'🎓 Train + preview'.")

    def _save_label_layer(self):
        """Save the selected layer as ``<opened-file>_labels.tif`` (the naming the
        trainer expects) into the Labels folder."""
        import tifffile

        fname = self.file_combo.value
        if not fname:
            show_warning("Open/select an image first.")
            return
        layer = self.viewer.layers.selection.active
        if layer is None:
            show_warning("Select the layer to save (click it in the layer list).")
            return
        data = np.asarray(layer.data)
        is_labels = isinstance(layer, napari.layers.Labels) or \
            np.issubdtype(data.dtype, np.integer)
        if not is_labels:
            show_warning("Selected layer is not a label/integer layer — pick the "
                         "annotation (Labels) layer.")
            return

        # Destination: the training Labels folder; else <input folder>/labels.
        labels_dir = str(self.train_labels_dir.value or "").strip()
        if not labels_dir or labels_dir in (".", str(self.folder.value)):
            base = str(self.folder.value) or "."
            labels_dir = str(Path(base) / "labels")
            self.train_labels_dir.value = labels_dir       # reflect + reuse for training
        Path(labels_dir).mkdir(parents=True, exist_ok=True)

        stem = Path(str(fname)).stem
        out = Path(labels_dir) / f"{stem}_labels.tif"
        try:
            tifffile.imwrite(str(out), data.astype(np.uint16))
        except Exception as exc:
            show_warning(f"Save failed: {exc}")
            return
        show_info(f"💾 Saved {out.name}  {tuple(data.shape)} → {labels_dir}")

    # ── UI construction ──────────────────────────────────────────────────────────
    def _build(self) -> Container:
        # Input / navigation
        self.folder      = FileEdit(label="Input folder", mode="d")
        self.output      = FileEdit(label="Output folder", mode="d",
                                    value=str(Path("tutorial/output").resolve()))
        self.classifier  = FileEdit(label="Cilia classifier (.cl)", mode="r",
                                    filter="APOC classifier (*.cl)",
                                    value=_DEFAULT_CLASSIFIER)
        scan_btn         = PushButton(text="🔍 Scan folder")
        scan_btn.clicked.connect(self._scan_folder)
        # Compute device (defaults to the NVIDIA GPU when present)
        _devices = _detect_gpus()
        self.gpu_combo = ComboBox(
            label="Compute device", choices=_devices, value=_default_gpu(_devices),
        )
        self.file_combo  = ComboBox(label="Image", choices=())
        self.file_combo.changed.connect(self._on_file_changed)
        prev_btn = PushButton(text="◀ Prev")
        next_btn = PushButton(text="Next ▶")
        prev_btn.clicked.connect(lambda: self._step_file(-1))
        next_btn.clicked.connect(lambda: self._step_file(+1))
        nav = Container(widgets=[prev_btn, next_btn], layout="horizontal", label="")

        # Channels
        self.ch_cilia    = SpinBox(label="Ch: Cilia",        value=1, min=0, max=9)
        self.ch_neurites = SpinBox(label="Ch: Neurites",     value=0, min=0, max=9)
        self.ch_bb       = SpinBox(label="Ch: Basal Bodies", value=2, min=0, max=9)
        self.ch_nuclei   = SpinBox(label="Ch: Nuclei",       value=3, min=0, max=9)
        self.use_mip     = CheckBox(label="MIP (2-D projection)", value=False)
        self.make_iso    = CheckBox(label="Make isotropic (downsample to Z)", value=True)

        # Normalisation (global default applied to every channel unless overridden)
        self.p_low  = SpinBox(label="Default p_low (%)",  value=0,  min=0, max=49)
        self.p_high = SpinBox(label="Default p_high (%)", value=100, min=51, max=100)
        self.p_low.changed.connect(self._refresh_norm)
        self.p_high.changed.connect(self._refresh_norm)

        # Histogram + per-channel normalisation override
        self.hist_channel = ComboBox(label="Channel", choices=())
        self.hist_channel.changed.connect(self._on_hist_channel_changed)
        self.ch_p_low  = SpinBox(label="p_low (%)",  value=0,  min=0, max=49)
        self.ch_p_high = SpinBox(label="p_high (%)", value=100, min=51, max=100)
        self.ch_p_low.changed.connect(self._on_ch_norm_changed)
        self.ch_p_high.changed.connect(self._on_ch_norm_changed)
        apply_all_btn = PushButton(text="Apply to all channels")
        reset_ch_btn  = PushButton(text="Reset channel to default")
        apply_all_btn.clicked.connect(self._apply_norm_all)
        reset_ch_btn.clicked.connect(self._reset_norm)
        self._hist_controls = Container(
            widgets=[self.hist_channel, self.ch_p_low, self.ch_p_high,
                     apply_all_btn, reset_ch_btn], labels=True,
        )
        self.hist_fig = Figure(figsize=(3.0, 1.9), tight_layout=True)
        self.hist_canvas = FigureCanvas(self.hist_fig)
        self.hist_canvas.setMinimumHeight(170)
        self.hist_ax = self.hist_fig.add_subplot(111)

        # Nuclei
        self.nuclei_method     = ComboBox(label="Nuclei method",
                                          choices=["Voronoi-Otsu", "APOC classifier"],
                                          value="Voronoi-Otsu")
        self.nuclei_classifier = FileEdit(label="Nuclei classifier (.cl)", mode="r",
                                          filter="APOC classifier (*.cl)",
                                          value=_DEFAULT_NUCLEI_CLASSIFIER)
        self.nuclei_min_size   = SpinBox(label="Nuclei min vox", value=50, min=0, max=1000000)
        self.nuclei_max_size   = SpinBox(label="Nuclei max vox (0=off)", value=0, min=0, max=10000000)
        self.nuclei_sigma         = SpinBox(label="Nuclei spot σ",    value=15, min=1, max=50)
        self.tophat_radius        = SpinBox(label="Nuclei tophat r",  value=12, min=1, max=50)
        self.nuclei_outline_sigma = SpinBox(label="Nuclei outline σ", value=3,  min=0, max=10)
        self.nuclei_log           = CheckBox(label="Nuclei log", value=False)
        # Show/hide the method-specific nuclei widgets when the method changes
        self.nuclei_method.changed.connect(self._on_nuclei_method_changed)
        # Neurites
        self.neurite_sigma = SpinBox(label="Neurite spot σ", value=5, min=1, max=20)
        self.neurite_log   = CheckBox(label="Neurite log", value=False)
        self.neurite_merge_nuclei = CheckBox(
            label="Merge nuclei into neurite before skeletonising", value=False)
        # Cilia
        self.cilia_log      = CheckBox(label="Cilia log", value=False)
        self.cilia_min_size = SpinBox(label="Cilia min vox", value=10, min=0, max=100000)
        self.cilia_max_size = SpinBox(label="Cilia max vox (0=off)", value=0, min=0, max=1000000)
        # Basal bodies
        self.bb_method        = ComboBox(label="BB method",
                                         choices=["Voronoi-Otsu", "APOC classifier"],
                                         value="APOC classifier")
        self.bb_classifier    = FileEdit(label="BB classifier (.cl)", mode="r",
                                         filter="APOC classifier (*.cl)",
                                         value=_DEFAULT_BB_CLASSIFIER)
        self.bb_spot_sigma    = FloatSpinBox(label="BB spot σ",    value=2.0, min=0.5, max=10.0, step=0.5)
        self.bb_outline_sigma = FloatSpinBox(label="BB outline σ", value=2.0, min=0.5, max=10.0, step=0.5)
        self.bb_log           = CheckBox(label="BB log", value=False)
        self.bb_min_size      = SpinBox(label="BB min vox", value=5, min=0, max=100000)
        self.bb_max_size      = SpinBox(label="BB max vox (0=off)", value=300, min=0, max=1000000)
        # Show/hide the method-specific BB widgets when the method changes
        self.bb_method.changed.connect(self._on_bb_method_changed)
        # Distances / classification
        self.max_cilia_dist = FloatSpinBox(label="Max cilia dist (µm)", value=2.0, min=0.1, max=30.0, step=0.5)
        self.max_basal_dist = FloatSpinBox(label="Max BB dist (µm)",    value=5.0, min=0.1, max=30.0, step=0.5)
        self.require_bb     = CheckBox(label="Require basal body (filter by BB distance)", value=True)
        # Axis-directed pairing: search along the cilium's major axis first.
        self.directional_pairing = CheckBox(
            label="Axis-directed BB pairing (search along cilium axis first)",
            value=False)
        self.pair_search_dist = FloatSpinBox(
            label="Axis search dist (µm)", value=5.0, min=0.5, max=30.0, step=0.5)
        self.pair_cone_angle = FloatSpinBox(
            label="Axis cone half-angle (°)", value=45.0, min=5.0, max=90.0, step=5.0)
        self.directional_pairing.changed.connect(self._on_directional_pairing_changed)
        self.ratio_from_bb  = CheckBox(label="Ratio from basal body position", value=True)
        self.ratio_epsilon  = FloatSpinBox(label="Ratio ε",             value=1.0, min=0.0, max=10.0, step=0.1)
        self.neurite_threshold = FloatSpinBox(label="Neurite log-ratio thr",  value=-0.1, min=-10, max=10.0, step=0.1)
        self.soma_threshold = FloatSpinBox(label="Soma log-ratio thr",  value=0.1, min=-10, max=10.0, step=0.1)

        # ── Train an APOC ObjectSegmenter (napari-apoc-plugin style) ────────────
        # 1) Annotate: paint a Labels layer in the viewer, pick it here.
        self.new_label_btn  = PushButton(text="🏷️ New empty label layer")
        self.new_label_btn.clicked.connect(self._new_label_layer)
        self.train_annotation = ComboBox(label="Annotation layer", choices=())
        self.refresh_ann_btn = PushButton(text="🔄 Refresh layer list")
        self.refresh_ann_btn.clicked.connect(self._refresh_train_annotation)
        self.train_channel = SpinBox(label="Train channel", value=0, min=0, max=9)
        # 2) Features (op × σ checkbox grid, apoc-style) / model.
        self.train_feat_sigmas = LineEdit(label="σ columns", value="1,2,3,5,10")
        self.apply_sigma_btn = PushButton(text="↻ Apply σ columns")
        self.apply_sigma_btn.clicked.connect(lambda *_: self._rebuild_feature_columns())
        self.train_feat_original = CheckBox(label="include 'original' intensity", value=True)
        self._build_feature_grid()                       # self.feat_table
        self.train_positive = SpinBox(label="Positive class id", value=2, min=1, max=10)
        self.train_max_depth = SpinBox(label="Tree max depth", value=2, min=1, max=10)
        self.train_num_trees = SpinBox(label="Num trees", value=100, min=10, max=500)
        self.train_output  = FileEdit(label="Output classifier (.cl)", mode="w",
                                      filter="APOC classifier (*.cl)")
        self.train_continue = CheckBox(
            label="Continue training (accumulate across images)", value=False)
        # 3) Train + preview on the current image (interactive).
        self.train_current_btn = PushButton(text="🎓 Train + preview on current image")
        self.train_current_btn.clicked.connect(self._do_train_current)
        self.predict_btn = PushButton(text="👁 Apply classifier to current image")
        self.predict_btn.clicked.connect(self._do_predict_current)
        self._build_importance_table()                   # self.imp_table

        # ── Optional: batch-train across many saved <stem>_labels.tif files ─────
        self.train_labels_dir = FileEdit(
            label="Labels folder (<stem>_labels.tif)", mode="d",
        )
        self.save_label_btn = PushButton(text="💾 Save selected layer as <file>_labels.tif")
        self.save_label_btn.clicked.connect(self._save_label_layer)
        self.train_all_labeled = CheckBox(
            label="Train on ALL images with labels (ignore view selection)", value=True)
        self.train_images = Select(label="Images (subset)", choices=())
        self.train_btn = PushButton(text="🎓 Batch-train from saved label files")
        self.train_btn.clicked.connect(self._do_train)
        # Click an image in the subset → load it into the viewer with current settings
        self._train_sel_prev: set[str] = set()
        self.train_images.changed.connect(self._on_train_image_clicked)
        # Keep the annotation dropdown in sync with the viewer's Labels layers.
        try:
            self.viewer.layers.events.inserted.connect(self._refresh_train_annotation)
            self.viewer.layers.events.removed.connect(self._refresh_train_annotation)
        except Exception:
            pass

        # Step buttons
        load_btn     = PushButton(text="1 · Load & normalise")
        cilia_btn    = PushButton(text="2 · Segment cilia")
        nuclei_btn   = PushButton(text="3 · Segment nuclei")
        neurite_btn  = PushButton(text="4 · Segment neurites")
        bb_btn       = PushButton(text="5 · Segment basal bodies")
        dist_btn     = PushButton(text="6 · Distance maps")
        assign_btn   = PushButton(text="7 · Assign & classify")
        load_btn.clicked.connect(self._do_load)
        cilia_btn.clicked.connect(self._do_cilia)
        nuclei_btn.clicked.connect(self._do_nuclei)
        neurite_btn.clicked.connect(self._do_neurites)
        bb_btn.clicked.connect(self._do_bb)
        dist_btn.clicked.connect(self._do_distances)
        assign_btn.clicked.connect(self._do_assign)
        self._step_buttons = [load_btn, cilia_btn, nuclei_btn, neurite_btn,
                              bb_btn, dist_btn, assign_btn]

        # Manual cilia editing (add / delete objects between steps).
        self.add_cilium_btn = PushButton(text="➕ Add cilium (paint)")
        self.del_cilium_btn = PushButton(text="🗑️ Delete cilium (click)")
        self.apply_edits_btn = PushButton(text="✅ Apply cilia edits")
        self.add_cilium_btn.clicked.connect(self._do_add_cilium)
        self.del_cilium_btn.clicked.connect(self._toggle_delete_cilium)
        self.apply_edits_btn.clicked.connect(self._do_apply_cilia_edits)
        self._edit_cilia_box = Container(
            widgets=[self.add_cilium_btn, self.del_cilium_btn, self.apply_edits_btn],
            layout="horizontal", labels=False,
        )

        self.run_all_btn = PushButton(text="▶ Run all steps (this image)")
        self.run_all_btn.clicked.connect(self._do_run_all)
        self.batch_btn = PushButton(text="⚡ Run BATCH (whole folder)")
        self.batch_btn.clicked.connect(self._do_batch)
        # Super batch: walk the input folder's subfolders and process each one
        # that holds .ims files as its own run, mirroring the tree under output.
        self.super_batch_btn = PushButton(text="🗂️ Run SUPER BATCH (all subfolders)")
        self.super_batch_btn.clicked.connect(self._do_super_batch)
        # Live batch visualisation: show each file's layers in the viewer and
        # save the whole-image overlay screenshots. The per-cilium ROIs are now
        # exported by the pipeline itself (fast MIP thumbnails + raw crops), so
        # they no longer depend on live capture being on.
        self.batch_capture = CheckBox(label="Capture overlay screenshots during batch", value=True)
        self.batch_rois    = CheckBox(label="Save per-cilium ROI thumbnails + crops", value=True)
        self.roi_correct   = CheckBox(label="Correct/normalise ROI display (off = raw intensities)", value=False)
        self.batch_ai      = CheckBox(label="AI-validate cilia during batch", value=False)
        self.batch_roi_only = CheckBox(
            label="ROI-only fast batch (cilia+BB → ROIs, skip analysis)", value=False)
        self.batch_xy_um = FloatSpinBox(
            label="Classifier XY µm (0 = all formats)", value=0.152,
            min=0.0, max=5.0, step=0.001)
        self.batch_roi_pct = FloatSpinBox(
            label="Export region % of area (100 = all)", value=100.0, min=1.0,
            max=100.0, step=1.0)
        self.batch_roi_tile = SpinBox(
            label="Region tile size (px)", value=512, min=32, max=8192)

        # Optional AI validation: score this image's cilia ROIs with a trained
        # validator CNN and show only the kept ones as a Labels layer.
        self.ai_model_combo  = ComboBox(label="AI model", choices=())
        self.ai_keep_thr     = FloatSpinBox(label="Keep if score ≥", value=0.6,
                                            min=0.5, max=0.99, step=0.01)
        self.ai_refresh_btn  = PushButton(text="⟳ Refresh models")
        self.ai_validate_btn = PushButton(text="🤖 AI-validate cilia (this image)")
        self.ai_refresh_btn.clicked.connect(self._refresh_ai_models)
        self.ai_validate_btn.clicked.connect(self._do_ai_validate)
        self._refresh_ai_models()

        self.status = Label(value="Open a folder to begin.")
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        self.progress.setVisible(False)

        # ── Logical parameter groups (each becomes a collapsible section) ────────
        workflow_box = Container(
            widgets=[self.folder, scan_btn, self.file_combo, nav,
                     self.gpu_combo],
            labels=True,
        )
        steps_box = Container(
            widgets=[*self._step_buttons, self._edit_cilia_box, self.run_all_btn],
            labels=False,
        )
        batch_box = Container(
            widgets=[self.output, self.batch_roi_only, self.batch_xy_um,
                     self.batch_roi_pct, self.batch_roi_tile, self.batch_capture,
                     self.batch_rois, self.roi_correct, self.batch_ai,
                     self.batch_btn, self.super_batch_btn],
            labels=True,
        )
        ai_box = Container(
            widgets=[self.ai_model_combo, self.ai_keep_thr,
                     self.ai_refresh_btn, self.ai_validate_btn], labels=True,
        )

        channels_box = Container(
            widgets=[self.ch_cilia, self.ch_neurites, self.ch_bb,
                     self.ch_nuclei, self.use_mip, self.make_iso], labels=True,
        )
        norm_box  = Container(widgets=[self.p_low, self.p_high], labels=True)
        nuclei_box = Container(
            widgets=[self.nuclei_method, self.nuclei_classifier,
                     self.nuclei_min_size, self.nuclei_max_size,
                     self.nuclei_sigma, self.tophat_radius,
                     self.nuclei_outline_sigma, self.nuclei_log], labels=True,
        )
        self._on_nuclei_method_changed()   # set initial widget visibility
        neurite_box = Container(
            widgets=[self.neurite_sigma, self.neurite_log, self.neurite_merge_nuclei],
            labels=True)
        cilia_box = Container(
            widgets=[self.classifier, self.cilia_log,
                     self.cilia_min_size, self.cilia_max_size], labels=True,
        )
        bb_box = Container(
            widgets=[self.bb_method, self.bb_classifier,
                     self.bb_spot_sigma, self.bb_outline_sigma, self.bb_log,
                     self.bb_min_size, self.bb_max_size], labels=True,
        )
        self._on_bb_method_changed()   # set initial widget visibility
        dist_box = Container(
            widgets=[self.max_cilia_dist, self.max_basal_dist, self.require_bb,
                     self.directional_pairing, self.pair_search_dist,
                     self.pair_cone_angle,
                     self.ratio_from_bb, self.ratio_epsilon,
                     self.neurite_threshold, self.soma_threshold], labels=True,
        )
        self._on_directional_pairing_changed()   # set initial widget visibility
        # The training section mixes magicgui widgets with two native Qt tables
        # (feature grid + importance), so it is assembled as a native wrapper
        # (``train_wrap``) below rather than a single magicgui Container.
        train_ann_box = Container(
            widgets=[
                Label(value="① Annotate — paint a Labels layer, pick it below"),
                self.new_label_btn, self.train_annotation, self.refresh_ann_btn,
                self.train_channel,
            ], labels=True,
        )
        train_sig_box = Container(
            widgets=[self.train_feat_sigmas, self.apply_sigma_btn,
                     self.train_feat_original], labels=True,
        )
        train_model_box = Container(
            widgets=[self.train_positive, self.train_max_depth, self.train_num_trees,
                     self.train_output, self.train_continue,
                     self.train_current_btn, self.predict_btn], labels=True,
        )
        train_batch_box = Container(
            widgets=[self.train_labels_dir, self.save_label_btn,
                     self.train_all_labeled, self.train_images, self.train_btn],
            labels=True,
        )

        # ── Assemble a native scrollable panel ──────────────────────────────────
        def _header(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight:600; margin-top:6px; color:#F5A623;")
            return lbl

        def _collapsible(title: str, box, expanded: bool = False) -> QCollapsible:
            col = QCollapsible(title)
            col.addWidget(box.native if isinstance(box, Container) else box)
            if expanded:
                col.expand(animate=False)
            return col

        # Native wrapper for the training section: containers + the two Qt tables.
        train_wrap = QWidget()
        _tw = QVBoxLayout(train_wrap)
        _tw.setContentsMargins(0, 0, 0, 0)
        _tw.setSpacing(4)
        _tw.addWidget(train_ann_box.native)
        _tw.addWidget(_header("② Features  (tick op × σ; double-click toggles)"))
        _tw.addWidget(train_sig_box.native)
        _tw.addWidget(self.feat_table)
        _tw.addWidget(_header("Model"))
        _tw.addWidget(train_model_box.native)
        _tw.addWidget(_header("③ Feature importances (after training)"))
        _tw.addWidget(self.imp_table)
        _tw.addWidget(_header("— Batch option: train from saved label files —"))
        _tw.addWidget(train_batch_box.native)

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(4)

        lay.addWidget(_header("① Input"))
        lay.addWidget(workflow_box.native)

        lay.addWidget(_header("② Step through one image"))
        lay.addWidget(steps_box.native)
        lay.addWidget(_collapsible("🤖 AI cilia validation (optional)", ai_box, False))

        lay.addWidget(_header("③ Batch (whole folder)"))
        lay.addWidget(batch_box.native)

        _sep = QFrame(); _sep.setFrameShape(QFrame.HLine); _sep.setStyleSheet("color:#444;")
        lay.addWidget(_sep)
        lay.addWidget(_header("Parameters"))

        # Histogram + per-channel normalisation section (controls + mpl canvas)
        hist_wrap = QWidget()
        _hw = QVBoxLayout(hist_wrap)
        _hw.setContentsMargins(0, 0, 0, 0)
        _hw.addWidget(self._hist_controls.native)
        _hw.addWidget(self.hist_canvas)
        hist_col = QCollapsible("📊 Histogram & per-channel norm")
        hist_col.addWidget(hist_wrap)

        for title, box, expanded in [
            ("Channels", channels_box, True),
            ("Normalisation", norm_box, False),
            ("Nuclei", nuclei_box, False),
            ("Neurites", neurite_box, False),
            ("Cilia", cilia_box, False),
            ("Basal bodies", bb_box, False),
            ("Distance & classification", dist_box, False),
            ("🎓 Train classifier", train_wrap, False),
        ]:
            lay.addWidget(_collapsible(title, box, expanded))
            if title == "Normalisation":
                lay.addWidget(hist_col)

        # Status text is surfaced as a napari notification popup (see
        # _set_status); only the progress bar stays pinned in the panel.
        lay.addWidget(self.progress)
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setMinimumWidth(360)
        return scroll


def main():
    viewer = napari.Viewer(title="LimonCELLo 🍋")
    app = LimoncelloApp(viewer)
    viewer.window.add_dock_widget(app.widget, area="right", name="LimonCELLo")
    napari.run()


if __name__ == "__main__":
    main()
