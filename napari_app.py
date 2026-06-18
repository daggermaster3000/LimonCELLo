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
import threading
from pathlib import Path

import numpy as np
import napari
from magicgui.widgets import (
    Container, PushButton, ComboBox, FileEdit, Label, CheckBox,
    SpinBox, FloatSpinBox, Select,
)
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QLabel, QFrame,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QProgressBar,
    QApplication,
)
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt, QTimer, QItemSelectionModel, QObject, Signal
from superqt import QCollapsible
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning
from scipy.ndimage import center_of_mass
from skimage.measure import regionprops_table
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
from limoncello.segmentation.train import load_training_pairs, train_object_segmenter
from limoncello.utils.assign_label_features import assign_label_features
from limoncello.analysis.pair_cilia_to_bb import pair_from_mixed_df
from limoncello.analysis.pipeline import run_pipeline3


_DEFAULT_CLASSIFIER = str(
    (Path(__file__).parent / "segmenters" / "Cilia-d38-3D.cl").resolve()
)
_DEFAULT_BB_CLASSIFIER = str(
    (Path(__file__).parent / "segmenters" / "BB-d38-3D.cl").resolve()
)

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

    return dict(raw=raw, norm=norm, voxel_size=voxel_size, n_ch=n_ch)


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
    print("[LC] Segmenting nuclei …")
    tr = p["tophat_radius"]
    ch = _clamp_ch(state["norm"].shape[0], p["ch_nuclei"], "Nuclei")
    nuclei_labels = segment_nuclei(
        state["norm"][ch],
        tophat_radius=(tr, tr, tr),
        spot_sigma=p["nuclei_sigma"],
        outline_sigma=p["nuclei_outline_sigma"],
        gaussian_sigma=(p["nuclei_gauss_z"], p["nuclei_gauss_y"], p["nuclei_gauss_x"]),
        log_transform=p["nuclei_log"],
    ).astype(np.int32)
    return dict(nuclei_labels=nuclei_labels)


def step_neurites(state: dict, p: dict) -> dict:
    print("[LC] Segmenting neurites …")
    ch = _clamp_ch(state["norm"].shape[0], p["ch_neurites"], "Neurites")
    skeleton, neurites_gpu = segment_neurites(
        state["norm"][ch],
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
            df.loc[df["log_ratio"] > p["axon_threshold"], "class"] = "axon"
            df.loc[df["log_ratio"] < p["soma_threshold"], "class"] = "soma"

    # ── Per-cilium 3-D region properties (volume / length) ──────────────────────
    if not cilia_df.empty:
        cprops = _region_props(state["cilia_labels"], state["voxel_size"])
        cilia_df["volume_voxels"] = [cprops.get(int(i), (np.nan,) * 3)[0] for i in cilia_df["cilia_id"]]
        cilia_df["volume_um3"]    = [cprops.get(int(i), (np.nan,) * 3)[1] for i in cilia_df["cilia_id"]]
        cilia_df["length_um"]     = [cprops.get(int(i), (np.nan,) * 3)[2] for i in cilia_df["cilia_id"]]

    # ── Pair cilia ↔ basal bodies (closest, strict 1:1, µm cutoff) ──────────────
    # Keep only validated cilia (paired to a BB within the cutoff) and the BBs
    # they pair with, so the table + overlays match the batch outputs.
    if not cilia_df.empty or not bb_df.empty:
        combined = pd.concat([cilia_df, bb_df], ignore_index=True)
        paired   = pair_from_mixed_df(
            combined, voxel_size=state["voxel_size"],
            max_pair_distance_um=p["max_basal_dist_um"],
        )
        cilia_df = paired[(paired["object_type"] == "cilia") & paired["validated"]].reset_index(drop=True)
        bb_df    = paired[(paired["object_type"] == "basal_body") & paired["validated"]].reset_index(drop=True)

    print(f"[LC] Done — {len(cilia_df)} validated cilia, {len(bb_df)} basal bodies paired.")
    return dict(cilia_df=cilia_df, bb_df=bb_df)


def _region_props(labels, voxel_size) -> dict:
    """{label_id: (volume_voxels, volume_um3, length_um)} using anisotropic spacing."""
    lab = np.asarray(labels).astype(np.int32)
    if lab.size == 0 or lab.max() == 0:
        return {}
    spacing = tuple(float(s) for s in voxel_size)
    vox_um3 = float(np.prod(spacing))
    out = {}
    try:
        rp = regionprops_table(lab, spacing=spacing,
                               properties=("label", "num_pixels", "axis_major_length"))
        for i, lid in enumerate(rp["label"]):
            n = int(rp["num_pixels"][i])
            out[int(lid)] = (n, n * vox_um3, float(rp["axis_major_length"][i]))
    except Exception as exc:
        print(f"[LC] region props failed ({exc}); voxel counts only.")
        counts = np.bincount(lab.ravel())
        for lid in range(1, len(counts)):
            if counts[lid]:
                out[int(lid)] = (int(counts[lid]), counts[lid] * vox_um3, float("nan"))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# LAYER UPDATERS — refresh only the layers a given step produced
# ──────────────────────────────────────────────────────────────────────────────

def _remove(viewer: napari.Viewer, *names: str) -> None:
    for n in names:
        if n in viewer.layers:
            viewer.layers.remove(n)


def show_channels(viewer, state) -> None:
    vs = state["voxel_size"]
    n_ch = state["raw"].shape[0]
    ch_keys = {"Cilia": "ch_cilia", "Neurites": "ch_neurites",
               "Basal Bodies": "ch_bb", "Nuclei": "ch_nuclei"}
    for name, cmap in zip(_CH_NAMES, _CH_COLORMAPS):
        ch = _clamp_ch(n_ch, state["channels"][ch_keys[name]], name)
        lname = f"LC: Raw {name}"
        _remove(viewer, lname)
        viewer.add_image(
            state["raw"][ch], name=lname, scale=vs,
            colormap=cmap, blending="additive", visible=(name == "Cilia"),
        )


def show_labels(viewer, state, key, lname, scale_from="voxel_size") -> None:
    _remove(viewer, lname)
    viewer.add_labels(state[key], name=lname, scale=state[scale_from])


def show_ratio(viewer, state) -> None:
    _remove(viewer, "LC: Log Ratio")
    lr = state["log_ratio_map"]
    finite = np.isfinite(lr)
    if finite.any():
        fill = float(np.nanmin(lr[finite]))
        viewer.add_image(
            np.where(finite, lr, fill), name="LC: Log Ratio",
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
        "cilia_id", "class", "log_ratio", "ratio", "distance_to_neurite_um",
        "dt_neurite", "dt_nuclei", "volume_um3", "length_um",
        "paired_id", "pair_distance_um", "pairing_status",
    ]

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.state: dict = {}          # intermediate arrays for current image
        self.files: list[str] = []     # .ims filenames in the folder
        self._busy = False
        self.cilia_pts = None          # current cilia Points layer
        self._cilia_df_view = None     # df backing the table (row order == points)
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

    # ── cilia property table ────────────────────────────────────────────────────
    def _build_table(self):
        self.table = QTableWidget()
        self.table.setColumnCount(len(self._TABLE_COLS))
        self.table.setHorizontalHeaderLabels(self._TABLE_COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        if self.viewer is not None:
            self.viewer.window.add_dock_widget(
                self.table, area="bottom", name="Cilia properties",
            )

    def _populate_table(self, cdf):
        self._cilia_df_view = cdf.reset_index(drop=True) if cdf is not None else None
        self.table.setRowCount(0)
        if cdf is None or cdf.empty:
            return
        cols = [c for c in self._TABLE_COLS if c in cdf.columns]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(cdf))
        for r in range(len(cdf)):
            for c, col in enumerate(cols):
                v = cdf.iloc[r][col]
                if isinstance(v, float):
                    txt = "—" if v != v else f"{v:.3g}"   # v!=v → NaN
                else:
                    txt = str(v)
                self.table.setItem(r, c, QTableWidgetItem(txt))
        self.table.resizeColumnsToContents()

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
            nuclei_sigma=self.nuclei_sigma.value,
            tophat_radius=self.tophat_radius.value,
            nuclei_outline_sigma=self.nuclei_outline_sigma.value,
            nuclei_log=self.nuclei_log.value,
            neurite_sigma=self.neurite_sigma.value,
            neurite_log=self.neurite_log.value,
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
            ratio_epsilon=self.ratio_epsilon.value,
            axon_threshold=self.axon_threshold.value,
            soma_threshold=self.soma_threshold.value,
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
        for b in self._step_buttons + [self.run_all_btn, self.batch_btn, self.train_btn]:
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

    def _after_assign(self, viewer, state):
        """Draw centroids + association overlays, fill the property table, wire
        the selection sync, then flash the completion image."""
        self.cilia_pts = show_centroids(viewer, state)
        show_associations(viewer, state)
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

    def _do_nuclei(self):
        self._run_step(step_nuclei, ("norm",),
                       lambda v, s: show_labels(v, s, "nuclei_labels", "LC: Nuclei Labels"),
                       "Segment nuclei")

    def _do_neurites(self):
        def _after(v, s):
            show_labels(v, s, "neurite_labels", "LC: Neurite Labels")
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
            (step_neurites,      lambda v, s: show_labels(v, s, "neurite_labels", "LC: Neurite Labels"),    "Segment neurites"),
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
    def _do_batch(self):
        if self._busy:
            return
        folder = Path(str(self.folder.value))
        out    = Path(str(self.output.value))
        if not folder.is_dir():
            show_warning("Select a valid input folder first.")
            return
        if not str(out):
            show_warning("Select an output folder first.")
            return
        out.mkdir(parents=True, exist_ok=True)
        p = self._params()
        self._set_busy(True)
        self._set_status("⚙️ Batch running over folder …")

        kwargs = dict(
            input_path=str(folder), output_path=str(out),
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
            ratio_epsilon=p["ratio_epsilon"],
            axon_threshold=p["axon_threshold"], soma_threshold=p["soma_threshold"],
        )

        bridge = self._bridge   # emit progress from the worker thread → GUI thread

        # Per-file live visualisation (optional). The callback runs in the worker
        # thread: it hands the file's arrays to the GUI thread and blocks until
        # the screenshots are taken, so napari is only ever touched on the GUI thread.
        capture     = self.batch_capture.value
        capture_rois = capture and self.batch_rois.value
        viz_bridge  = self._viz_bridge

        def _per_file(payload):
            payload["capture_rois"] = capture_rois
            done = threading.Event()
            viz_bridge.visualize.emit(payload, done)
            done.wait()

        @thread_worker
        def _work():
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

    def _capture_cilia_rois(self, state: dict, roi_dir: Path, stem: str, margin: int = 12):
        """One 3-D close-up screenshot per validated cilium: the cilium + its
        paired basal body (cropped to their bounding box) with the dot + link
        layers, auto-framed via reset_view."""
        cdf = state.get("cilia_df")
        if cdf is None or cdf.empty:
            return
        vs           = np.asarray(state["voxel_size"], dtype=float)
        cilia_labels = np.asarray(state["cilia_labels"])
        bb_labels    = np.asarray(state["bb_labels"])
        raw          = state["raw"]
        ch           = state["channels"]
        shp          = cilia_labels.shape
        cz = _clamp_ch(raw.shape[0], ch["ch_cilia"], "Cilia")
        bz = _clamp_ch(raw.shape[0], ch["ch_bb"], "Basal Bodies")

        # BB id → centroid (for the dot + pairing link)
        bb_coord = {}
        bdf = state.get("bb_df")
        if bdf is not None and not bdf.empty:
            for _, r in bdf.iterrows():
                bb_coord[int(r["cilia_id"])] = np.asarray(r["coords"], dtype=float)

        # The whole-image LC layers would clutter the close-up — start clean.
        self._clear_named_layers("LC:")

        total = len(cdf)
        saved = 0
        for k, (_, row) in enumerate(cdf.iterrows(), 1):
            self._capture_progress(f"📸 {stem} — ROI {k}/{total}")
            try:
                cid = int(row["cilia_id"])
                pid = row.get("paired_id")
                has_bb = pid is not None and not pd_isna(pid) and int(pid) in bb_coord

                mask = cilia_labels == cid
                if has_bb:
                    mask = mask | (bb_labels == int(pid))
                pts = np.argwhere(mask)
                if pts.size == 0:
                    continue
                lo = np.maximum(pts.min(0) - margin, 0)
                hi = np.minimum(pts.max(0) + margin + 1, shp)
                sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))

                self._clear_named_layers("ROI:")
                self.viewer.add_image(raw[cz][sl], name="ROI: Cilia ch", scale=vs,
                                      colormap="green", blending="additive")
                self.viewer.add_image(raw[bz][sl], name="ROI: BB ch", scale=vs,
                                      colormap="magenta", blending="additive")
                self.viewer.add_labels(cilia_labels[sl], name="ROI: Cilia", scale=vs)
                self.viewer.add_labels(bb_labels[sl],    name="ROI: BB",    scale=vs)

                c_local = np.asarray(row["coords"], dtype=float) - lo
                self.viewer.add_points(c_local[np.newaxis], name="ROI: cilium",
                                       scale=vs, size=4, face_color="cyan")
                if has_bb:
                    b_local = bb_coord[int(pid)] - lo
                    self.viewer.add_points(b_local[np.newaxis], name="ROI: bb",
                                           scale=vs, size=4, face_color="yellow")
                    self.viewer.add_shapes([np.stack([c_local, b_local])], shape_type="line",
                                           name="ROI: link", scale=vs,
                                           edge_color="yellow", edge_width=1.0)
                try:
                    self.viewer.dims.ndisplay = 3
                except Exception:
                    pass
                self.viewer.reset_view()
                self._screenshot(roi_dir / f"{stem}_cilia{cid}.png")
                saved += 1
            except Exception as exc:                      # one bad cilium must not abort the rest
                print(f"[LC] ROI capture failed for cilium {row.get('cilia_id')}: {exc}")

        self._clear_named_layers("ROI:")
        print(f"[LC] saved {saved}/{total} cilia ROIs for {stem} → {roi_dir}")

    # ── classifier training ──────────────────────────────────────────────────────
    def _do_train(self):
        """Train one APOC ObjectSegmenter on a chosen subset of images using their
        raw (un-normalised) channel + matching ``<stem>_labels.tif`` annotations."""
        if self._busy:
            return
        folder = Path(str(self.folder.value))
        if not folder.is_dir():
            show_warning("Scan a valid input folder first.")
            return
        selected = list(self.train_images.value or [])
        if not selected:
            show_warning("Select one or more images (subset) to train on.")
            return
        labels_dir = str(self.train_labels_dir.value)
        if not labels_dir or not Path(labels_dir).is_dir():
            show_warning("Select a valid labels folder (with <stem>_labels.tif files).")
            return
        out = str(self.train_output.value)
        if not out or out in (".", str(folder)):
            show_warning("Choose an output .cl path for the classifier.")
            return

        image_paths = [str(folder / name) for name in selected]
        ch          = self.train_channel.value
        gpu         = self.gpu_combo.value
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
            pairs, skipped = load_training_pairs(image_paths, labels_dir, ch)
            if not pairs:
                reasons = "; ".join(f"{s}: {r}" for s, r in skipped) or "no usable pairs"
                raise RuntimeError(f"No (image, labels) pairs to train on — {reasons}")
            train_object_segmenter(pairs, out, gpu_device=None, **params)
            return dict(n=len(pairs), skipped=skipped, out=out, channel=ch)

        def _done(res):
            self._set_busy(False)
            # Offer the freshly trained classifier to the matching segmenter field.
            if res["channel"] == self.ch_cilia.value:
                self.classifier.value = res["out"]
            elif res["channel"] == self.ch_bb.value:
                self.bb_classifier.value = res["out"]
            msg = f"✅ Trained on {res['n']} image(s) → {Path(res['out']).name}"
            if res["skipped"]:
                msg += f"  ({len(res['skipped'])} skipped)"
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
            np.zeros(shape, dtype=np.uint16), name="labels", scale=ref.scale,
        )
        lyr.selected_label = 2          # start painting the 'object' class
        try:
            lyr.mode = "paint"
        except Exception:
            pass
        self.viewer.layers.selection.active = lyr
        show_info("Empty label layer added. Paint background=1, object=2, then Save.")

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
        self.ch_cilia    = SpinBox(label="Ch: Cilia",        value=0, min=0, max=9)
        self.ch_neurites = SpinBox(label="Ch: Neurites",     value=1, min=0, max=9)
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
        self.nuclei_sigma         = SpinBox(label="Nuclei spot σ",    value=15, min=1, max=50)
        self.tophat_radius        = SpinBox(label="Nuclei tophat r",  value=12, min=1, max=50)
        self.nuclei_outline_sigma = SpinBox(label="Nuclei outline σ", value=3,  min=0, max=10)
        self.nuclei_log           = CheckBox(label="Nuclei log", value=False)
        # Neurites
        self.neurite_sigma = SpinBox(label="Neurite spot σ", value=5, min=1, max=20)
        self.neurite_log   = CheckBox(label="Neurite log", value=False)
        # Cilia
        self.cilia_log      = CheckBox(label="Cilia log", value=False)
        self.cilia_min_size = SpinBox(label="Cilia min vox", value=10, min=0, max=100000)
        self.cilia_max_size = SpinBox(label="Cilia max vox (0=off)", value=70, min=0, max=1000000)
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
        self.ratio_epsilon  = FloatSpinBox(label="Ratio ε",             value=1.0, min=0.0, max=10.0, step=0.1)
        self.axon_threshold = FloatSpinBox(label="Axon log-ratio thr",  value=2.5, min=0.0, max=10.0, step=0.1)
        self.soma_threshold = FloatSpinBox(label="Soma log-ratio thr",  value=1.0, min=0.0, max=10.0, step=0.1)

        # ── Train an APOC classifier on a subset of images (raw, no normalisation) ─
        self.train_images = Select(label="Images (subset)", choices=())
        self.train_labels_dir = FileEdit(
            label="Labels folder (<stem>_labels.tif)", mode="d",
        )
        # Label maker: save the selected napari layer as <opened-file>_labels.tif
        self.new_label_btn  = PushButton(text="🏷️ New empty label layer")
        self.new_label_btn.clicked.connect(self._new_label_layer)
        self.save_label_btn = PushButton(text="💾 Save selected layer as <file>_labels.tif")
        self.save_label_btn.clicked.connect(self._save_label_layer)
        self.train_channel = SpinBox(label="Train channel", value=0, min=0, max=9)
        self.train_output  = FileEdit(label="Output classifier (.cl)", mode="w",
                                      filter="APOC classifier (*.cl)")
        self.train_positive = SpinBox(label="Positive class id", value=2, min=1, max=10)
        self.train_max_depth = SpinBox(label="Tree max depth", value=2, min=1, max=10)
        self.train_num_trees = SpinBox(label="Num trees", value=100, min=10, max=500)
        self.train_btn = PushButton(text="🎓 Train classifier (subset)")
        self.train_btn.clicked.connect(self._do_train)

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

        self.run_all_btn = PushButton(text="▶ Run all steps (this image)")
        self.run_all_btn.clicked.connect(self._do_run_all)
        self.batch_btn = PushButton(text="⚡ Run BATCH (whole folder)")
        self.batch_btn.clicked.connect(self._do_batch)
        # Live batch visualisation: show each file's layers in the viewer and
        # save overlay screenshots (+ optional per-cilium 3-D ROI close-ups).
        self.batch_capture = CheckBox(label="Capture screenshots during batch", value=True)
        self.batch_rois    = CheckBox(label="↳ also save per-cilium 3-D ROIs", value=True)
        self.batch_capture.changed.connect(
            lambda *_: setattr(self.batch_rois, "enabled", self.batch_capture.value)
        )

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
            widgets=[*self._step_buttons, self.run_all_btn], labels=False,
        )
        batch_box = Container(
            widgets=[self.output, self.batch_capture, self.batch_rois, self.batch_btn],
            labels=True,
        )

        channels_box = Container(
            widgets=[self.ch_cilia, self.ch_neurites, self.ch_bb,
                     self.ch_nuclei, self.use_mip, self.make_iso], labels=True,
        )
        norm_box  = Container(widgets=[self.p_low, self.p_high], labels=True)
        nuclei_box = Container(
            widgets=[self.nuclei_sigma, self.tophat_radius,
                     self.nuclei_outline_sigma, self.nuclei_log], labels=True,
        )
        neurite_box = Container(widgets=[self.neurite_sigma, self.neurite_log], labels=True)
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
            widgets=[self.max_cilia_dist, self.max_basal_dist, self.ratio_epsilon,
                     self.axon_threshold, self.soma_threshold], labels=True,
        )
        train_box = Container(
            widgets=[self.train_images, self.train_labels_dir,
                     self.new_label_btn, self.save_label_btn,
                     self.train_channel, self.train_output, self.train_positive,
                     self.train_max_depth, self.train_num_trees, self.train_btn],
            labels=True,
        )

        # ── Assemble a native scrollable panel ──────────────────────────────────
        def _header(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight:600; margin-top:6px; color:#F5A623;")
            return lbl

        def _collapsible(title: str, box: Container, expanded: bool = False) -> QCollapsible:
            col = QCollapsible(title)
            col.addWidget(box.native)
            if expanded:
                col.expand(animate=False)
            return col

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(4)

        lay.addWidget(_header("① Input"))
        lay.addWidget(workflow_box.native)

        lay.addWidget(_header("② Step through one image"))
        lay.addWidget(steps_box.native)

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
            ("🎓 Train classifier", train_box, False),
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
