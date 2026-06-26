"""
LimonCELLo — proposal-driven (fully-convolutional) cilia app.

A sibling of ``napari_app.py`` that swaps the "segment-everything-then-filter"
front end for a **proposal-driven** one, so you can compare the two. The existing
app is left untouched.

How it works
------------
1. FULLY-CONVOLUTIONAL CONVERSION.  The trained ROI-validator CNN
   (``limoncello/ml/roi_validator.py``) is a 3×96×96 → P(real cilium) classifier:
   4 conv blocks (3→32→64→128→256, ×16 downsample → 6×6) → global-average-pool →
   dense 256→128 → dense 128→2 → softmax. We keep the conv blocks and their
   trained weights, **drop the GAP+flatten**, and turn each dense layer into an
   equivalent 1×1 convolution (dense ``[out,in]`` → conv weight ``[out,in,1,1]``).
   The all-conv net then runs **once** over an H×W image and emits a dense
   ``[2, H/16, W/16]`` logit map → softmax → a P(cilium) heatmap at 1/16 scale.
   Because the conv stack is untouched, this reuses the trained weights exactly;
   ``verify_fcn_transplant`` asserts the FCN reproduces the classifier on a single
   receptive field.  (The FCN math is shared with ``napari_cnn_app.py``.)

2. SCALE MATCHING — *the main thing that breaks on a new microscope.*  The CNN
   learnt cilia at the µm/pixel of its training data. Before running the FCN we
   resample the image so ``um_per_pixel_image`` matches ``um_per_pixel_train``;
   proposal coordinates are mapped back to native pixels afterwards. If the two
   scales are equal/unset, resampling is skipped. The factor is logged.

3. INTENSITY NORMALIZATION — the CNN trained on per-crop **raw min-max** of the
   cilia (green) + basal-body (magenta) channels (``correct_display=False``
   thumbnails). The whole-image input is built the same way; a percentile mode is
   exposed for a new microscope's intensity regime.

4. PIPELINE (bb-anchored, reused).  Heatmap → threshold + peak-find → ROI centres
   → APOC segments the cilia and basal-body channels → the existing greedy
   ``pair_from_mixed_df`` assigns one basal body per cilium. A proposal is a real
   cilium only if a basal body is assigned to it (``validated``); cilium signal
   with no owning basal body is rejected as artifact.

Run with:  python napari_app_fcn.py
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import napari
import torch
from magicgui.widgets import (
    Container, PushButton, ComboBox, FileEdit, Label, FloatSlider, SpinBox,
    FloatSpinBox, CheckBox,
)
from qtpy.QtCore import QObject, Signal
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning
from scipy.ndimage import center_of_mass

# Conventions + loaders shared with the existing app.
from napari_app import (
    step_load, _detect_gpus, _default_gpu, _select_device, _add_image_safe,
    _DEFAULT_CLASSIFIER, _DEFAULT_BB_CLASSIFIER,
)
# Verified FCN math (conversion, heatmap, peak-find, coord mapping).
from napari_cnn_app import (
    to_fully_convolutional, cnn_heatmap, peak_find, heatmap_peaks_to_image_xy,
    _clamp_ch,
)
from limoncello.segmentation.cilia import segment_cilia_ml
from limoncello.segmentation.basal_bodies import segment_basal_bodies_ml
from limoncello.analysis.pair_cilia_to_bb import pair_from_mixed_df
from limoncello.ml.roi_validator import load_bundle, DEFAULT_SIZE

_MODELS_DIR = str((Path(__file__).parent / "models").resolve())
_NORM_MODES = ("raw min-max (training)", "percentile")


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation + scale matching (pure compute)
# ──────────────────────────────────────────────────────────────────────────────
def _minmax(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    return np.clip((a - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def _percentile(a: np.ndarray, lo_p: float, hi_p: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.percentile(a, [lo_p, hi_p])
    return np.clip((a - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def build_fcn_rgb(raw: np.ndarray, ch_cilia: int, ch_bb: int, *,
                  mode: str = _NORM_MODES[0], p_low: float = 2.0,
                  p_high: float = 99.5) -> np.ndarray:
    """Whole-image ``(3, H, W)`` RGB in the layout the CNN trained on: cilia→green,
    basal body→magenta, on the XY max-projection. ``mode`` picks the per-image
    normalisation (raw min-max = training default; percentile for a new scope)."""
    n = raw.shape[0]
    cil = raw[_clamp_ch(n, ch_cilia)].max(axis=0)
    bb = raw[_clamp_ch(n, ch_bb)].max(axis=0)
    if mode == "percentile":
        cil, bb = _percentile(cil, p_low, p_high), _percentile(bb, p_low, p_high)
    else:
        cil, bb = _minmax(cil), _minmax(bb)
    rgb = np.zeros((3, *cil.shape), dtype=np.float32)
    rgb[1] = cil                      # green
    rgb[0] = bb; rgb[2] = bb          # magenta
    return rgb


def resample_to_train_scale(rgb: np.ndarray, um_image: float,
                            um_train: float) -> tuple[np.ndarray, float]:
    """Resample ``(3,H,W)`` so its µm/pixel matches the training scale.

    ``scale = um_image / um_train`` (>1 upsamples a coarser image, <1 downsamples
    a finer one). Returns ``(rgb_scaled, scale)``; ``scale == 1`` (no-op) when
    either value is unset (≤0) or the two are equal.
    """
    if um_image <= 0 or um_train <= 0:
        return rgb, 1.0
    scale = float(um_image) / float(um_train)
    if abs(scale - 1.0) < 1e-3:
        return rgb, 1.0
    from PIL import Image
    h, w = rgb.shape[1:]
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    img = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
    img = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    return np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1)), scale


# ──────────────────────────────────────────────────────────────────────────────
# FCN correctness check
# ──────────────────────────────────────────────────────────────────────────────
def verify_fcn_transplant(model: torch.nn.Module, fcn: torch.nn.Module,
                          stride: int) -> tuple[float, float]:
    """Assert the FCN reproduces the classifier on a single receptive field.

    Feeding a patch of side ``stride`` collapses the conv stack to a 1×1 feature
    map, so the classifier's GAP is the identity and the FCN's single output cell
    must equal the classifier output (same transplanted weights). Prints both and
    asserts the gap is < 1e-4.
    """
    model = model.eval(); fcn = fcn.eval()
    x = torch.randn(1, 3, stride, stride)
    with torch.no_grad():
        p_clf = torch.softmax(model(x), 1)[0, 1].item()
        out = torch.softmax(fcn(x), 1)[0, 1]
        p_fcn = out.reshape(-1)[0].item()
    delta = abs(p_clf - p_fcn)
    print(f"[FCN] transplant check: classifier P={p_clf:.6f}  FCN P={p_fcn:.6f}  "
          f"Δ={delta:.2e}  {'OK' if delta < 1e-4 else 'MISMATCH'}")
    assert delta < 1e-4, f"FCN transplant mismatch (Δ={delta:.2e})"
    return p_clf, p_fcn


# ──────────────────────────────────────────────────────────────────────────────
# Detection + bb-anchored assignment (pure compute; shared by single + batch)
# ──────────────────────────────────────────────────────────────────────────────
def _labels_df(labels: np.ndarray, object_type: str):
    """Build the minimal dataframe ``pair_from_mixed_df`` needs from a label
    volume: one row per object with ``cilia_id``, ``coords`` (z,y,x), and a
    ``centroid_yx`` cache for gating."""
    import pandas as pd
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if ids.size == 0:
        return pd.DataFrame(columns=["cilia_id", "coords", "object_type",
                                     "centroid_yx"]), np.empty((0, 2))
    cents = np.atleast_2d(center_of_mass(labels > 0, labels=labels, index=ids))
    rows = [{"cilia_id": int(i), "coords": [float(c[0]), float(c[1]), float(c[2])],
             "object_type": object_type}
            for i, c in zip(ids, cents)]
    return pd.DataFrame(rows), cents[:, 1:]          # (y, x) per object


def detect_and_assign(raw: np.ndarray, voxel_size, fcn: torch.nn.Module,
                      stride: int, size: int, device: torch.device, p: dict):
    """Full proposal→segment→assign for one image.

    Returns ``dict(heatmap, in_scale, proposals_yx, scores, cilia_df, bb_df,
    cilia_labels, bb_labels)``. ``cilia_df``/``bb_df`` carry the reused
    ``pair_from_mixed_df`` columns; a cilium is kept iff ``validated`` (owns a
    basal body).
    """
    import pandas as pd
    rgb = build_fcn_rgb(raw, p["ch_cilia"], p["ch_bb"], mode=p["norm_mode"],
                        p_low=p["p_low"], p_high=p["p_high"])
    rgb, in_scale = resample_to_train_scale(rgb, p["um_image"], p["um_train"])
    heatmap = cnn_heatmap(fcn, rgb, device)
    rc, scores = peak_find(heatmap, p["threshold"], p["min_dist"])
    # FCN 'same'-conv grid → cell centre offset = stride/2 (not size/2).
    yx = heatmap_peaks_to_image_xy(rc, stride, in_scale, stride / 2.0)

    # APOC segmentation of both channels over the full volume.
    cl = np.asarray(segment_cilia_ml(
        raw[_clamp_ch(raw.shape[0], p["ch_cilia"])],
        classifier_path=p["cilia_cl"], gaussian_sigma=(0.0, 0.0, 0.0),
        log_transform=False, min_size=p["cilia_min"], max_size=p["cilia_max"],
    )).astype(np.int32)
    bb = np.asarray(segment_basal_bodies_ml(
        raw[_clamp_ch(raw.shape[0], p["ch_bb"])],
        classifier_path=p["bb_cl"], gaussian_sigma=(0.0, 0.0, 0.0),
        log_transform=False, min_size=p["bb_min"], max_size=p["bb_max"],
    )).astype(np.int32)

    cilia_df, cil_yx = _labels_df(cl, "cilia")
    bb_df, _ = _labels_df(bb, "basal_body")

    # Gate cilia to the FCN proposals: keep only APOC cilia whose XY centroid is
    # near a proposed centre (the CNN says "look here"); tag each with the nearest
    # proposal's heatmap score.
    if len(cilia_df) and len(yx):
        d = np.hypot(cil_yx[:, 0][:, None] - yx[None, :, 0],
                     cil_yx[:, 1][:, None] - yx[None, :, 1])
        nearest = d.argmin(axis=1)
        keep = d[np.arange(len(cilia_df)), nearest] <= p["gate_radius"]
        cilia_df = cilia_df.loc[keep].reset_index(drop=True)
        cilia_df["proposal_score"] = scores[nearest[keep]] if scores.size else np.nan
    else:
        cilia_df = cilia_df.iloc[0:0].copy()
        cilia_df["proposal_score"] = np.array([], dtype=float)

    # Reuse the existing greedy one-bb-per-cilium assignment.
    if len(cilia_df) and len(bb_df):
        combined = pd.concat([cilia_df, bb_df], ignore_index=True)
        paired = pair_from_mixed_df(combined, voxel_size=voxel_size,
                                    max_pair_distance_um=p["max_pair_um"])
        cilia_df = paired[paired["object_type"] == "cilia"].reset_index(drop=True)
        bb_df = paired[paired["object_type"] == "basal_body"].reset_index(drop=True)
    else:
        for c in ("paired_id", "pair_distance_um", "pairing_status", "validated"):
            cilia_df[c] = np.nan if c != "validated" else False

    return dict(heatmap=heatmap, in_scale=in_scale, proposals_yx=yx, scores=scores,
                cilia_df=cilia_df, bb_df=bb_df, cilia_labels=cl, bb_labels=bb)


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
class _ProgressBridge(QObject):
    progressed = Signal(int, int, str)


class LimoncelloFCNApp:
    """Dock widget: trained CNN → FCN → heatmap → proposals → APOC → bb-anchored
    cilia, single-image (reactive) or batch over a folder."""

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.state: dict = {}
        self.files: list[str] = []
        self.fcn: torch.nn.Module | None = None
        self.fcn_stride = 16
        self.model_size = DEFAULT_SIZE["big"]
        self._busy = False
        self._bridge = _ProgressBridge()
        self._bridge.progressed.connect(self._on_batch_progress)
        self.widget = self._build()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build(self) -> Container:
        devices = _detect_gpus()
        self.gpu_combo = ComboBox(label="Compute device",
                                  choices=[(l, v) for l, v in devices],
                                  value=_default_gpu(devices))
        self.model_combo = ComboBox(label="ROI model (.pt)",
                                    choices=self._discover_models())
        self.build_btn = PushButton(text="① Build FCN from model")
        self.build_btn.clicked.connect(self._on_build)
        self.fcn_status = Label(value="No FCN built.")

        self.folder = FileEdit(label="Image folder", mode="d")
        self.scan_btn = PushButton(text="Scan folder")
        self.scan_btn.clicked.connect(self._scan_folder)
        self.file_combo = ComboBox(label="Image", choices=())
        self.output = FileEdit(label="Output folder (batch)", mode="d")

        self.ch_cilia = SpinBox(label="Cilia channel", value=1, min=0, max=16)
        self.ch_bb = SpinBox(label="Basal-body channel", value=2, min=0, max=16)

        self.norm_mode = ComboBox(label="Normalization", choices=list(_NORM_MODES),
                                  value=_NORM_MODES[0])
        self.p_low = FloatSpinBox(label="pct low", value=2.0, min=0, max=100)
        self.p_high = FloatSpinBox(label="pct high", value=99.5, min=0, max=100)

        # Scale matching — the cross-microscope knob.
        self.um_image = FloatSpinBox(label="µm/px image", value=0.0, min=0.0,
                                     max=10.0, step=0.001)
        self.um_train = FloatSpinBox(label="µm/px train", value=0.0, min=0.0,
                                     max=10.0, step=0.001)

        self.load_btn = PushButton(text="② Load image")
        self.load_btn.clicked.connect(self._on_load)

        self.threshold = FloatSlider(label="Heatmap threshold", value=0.5,
                                     min=0.0, max=1.0, step=0.01)
        self.min_dist = SpinBox(label="Peak min-dist (cells)", value=2, min=1, max=50)
        self.run_btn = PushButton(text="③ Run heatmap + detect")
        self.run_btn.clicked.connect(self._on_run)
        # Reactive: threshold / min-dist re-peak-find from the cached heatmap.
        self.threshold.changed.connect(self._reactive_detect)
        self.min_dist.changed.connect(self._reactive_detect)

        self.cilia_cl = FileEdit(label="APOC cilia .cl",
                                 value=_DEFAULT_CLASSIFIER, mode="r")
        self.bb_cl = FileEdit(label="APOC basal-body .cl",
                              value=_DEFAULT_BB_CLASSIFIER, mode="r")
        self.cilia_min = SpinBox(label="Cilia min size", value=10, min=0, max=1_000_000)
        self.cilia_max = SpinBox(label="Cilia max size", value=0, min=0, max=10_000_000)
        self.bb_min = SpinBox(label="BB min size", value=5, min=0, max=1_000_000)
        self.bb_max = SpinBox(label="BB max size", value=0, min=0, max=10_000_000)
        self.gate_radius = FloatSpinBox(label="Proposal gate (px)", value=20.0,
                                        min=0, max=500)
        self.max_pair_um = FloatSpinBox(label="Max bb pair (µm)", value=2.0,
                                        min=0, max=50)
        self.segment_btn = PushButton(text="④ Segment + assign (bb-anchored)")
        self.segment_btn.clicked.connect(self._on_segment)
        self.seg_status = Label(value="—")

        self.batch_btn = PushButton(text="⑤ Batch folder → output")
        self.batch_btn.clicked.connect(self._on_batch)

        self._buttons = [self.build_btn, self.load_btn, self.run_btn,
                         self.segment_btn, self.batch_btn]

        return Container(widgets=[
            Label(value="<b>Proposal-driven cilia (FCN)</b>"),
            self.gpu_combo,
            Label(value="<b>1 · Model → FCN</b>"),
            self.model_combo, self.build_btn, self.fcn_status,
            Label(value="<b>2 · Image</b>"),
            self.folder, self.scan_btn, self.file_combo, self.output,
            self.ch_cilia, self.ch_bb,
            self.norm_mode, self.p_low, self.p_high,
            self.um_image, self.um_train, self.load_btn,
            Label(value="<b>3 · Heatmap → proposals</b>"),
            self.threshold, self.min_dist, self.run_btn,
            Label(value="<b>4 · APOC + bb assignment</b>"),
            self.cilia_cl, self.bb_cl,
            self.cilia_min, self.cilia_max, self.bb_min, self.bb_max,
            self.gate_radius, self.max_pair_um, self.segment_btn, self.seg_status,
            Label(value="<b>5 · Batch</b>"),
            self.batch_btn,
        ], labels=True, scrollable=True)

    # ── helpers ─────────────────────────────────────────────────────────────
    def _discover_models(self) -> list[str]:
        if not os.path.isdir(_MODELS_DIR):
            return []
        return sorted(f for f in os.listdir(_MODELS_DIR) if f.endswith(".pt"))

    def _set_busy(self, busy: bool):
        self._busy = busy
        for b in self._buttons:
            b.enabled = not busy

    def _params(self) -> dict:
        return dict(
            ch_cilia=self.ch_cilia.value, ch_bb=self.ch_bb.value,
            norm_mode=self.norm_mode.value,
            p_low=float(self.p_low.value), p_high=float(self.p_high.value),
            um_image=float(self.um_image.value), um_train=float(self.um_train.value),
            threshold=float(self.threshold.value), min_dist=int(self.min_dist.value),
            cilia_cl=str(self.cilia_cl.value), bb_cl=str(self.bb_cl.value),
            cilia_min=int(self.cilia_min.value), cilia_max=int(self.cilia_max.value),
            bb_min=int(self.bb_min.value), bb_max=int(self.bb_max.value),
            gate_radius=float(self.gate_radius.value),
            max_pair_um=float(self.max_pair_um.value),
        )

    def _device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── build FCN ───────────────────────────────────────────────────────────
    def _on_build(self):
        name = self.model_combo.value
        if not name:
            show_warning("Pick a trained .pt model first.")
            return
        try:
            model, meta = load_bundle(os.path.join(_MODELS_DIR, str(name)))
            self.fcn = to_fully_convolutional(model)
            self.fcn_stride = int(getattr(self.fcn, "stride", 16))
            self.model_size = int(meta.get("size",
                                  DEFAULT_SIZE.get(meta.get("arch", "big"), 64)))
            verify_fcn_transplant(model, self.fcn, self.fcn_stride)
        except Exception as exc:                              # noqa: BLE001
            show_warning(f"Build/verify failed: {exc}")
            return
        self.fcn_status.value = (f"FCN ready · arch={meta.get('arch')} · "
                                 f"size={self.model_size} · stride={self.fcn_stride} "
                                 f"· transplant verified ✓")
        show_info("FCN built and transplant verified (see console).")

    # ── folder / image ──────────────────────────────────────────────────────
    def _scan_folder(self):
        folder = Path(str(self.folder.value))
        if not folder.is_dir():
            show_warning("Select a valid image folder first.")
            return
        self.files = sorted(f for f in os.listdir(folder)
                            if f.lower().endswith(".ims"))
        if not self.files:
            show_warning(f"No .ims files in {folder}")
            self.file_combo.choices = ()
            return
        self.file_combo.choices = self.files
        self.file_combo.value = self.files[0]
        show_info(f"Found {len(self.files)} .ims file(s).")

    def _load_params(self) -> dict:
        p = dict(gpu_device=self.gpu_combo.value,
                 ch_cilia=self.ch_cilia.value, ch_neurites=0,
                 ch_bb=self.ch_bb.value, ch_nuclei=0,
                 # Match the pipeline that made the CNN's training crops (isotropic).
                 use_mip=False, make_isotropic=True,
                 p_low=2.0, p_high=98.0, ch_norm={})
        if self.file_combo.value and self.folder.value:
            p["ims_path"] = str(Path(str(self.folder.value)) / self.file_combo.value)
        else:
            p["ims_path"] = ""
        return p

    def _on_load(self):
        if self._busy:
            return
        p = self._load_params()
        if not p["ims_path"]:
            show_warning("Scan a folder and pick an image first.")
            return
        self._set_busy(True)
        show_info("Loading image …")

        @thread_worker
        def _work():
            _select_device(p["gpu_device"])
            return step_load(self.state, p)

        def _done(updates):
            self.state.update(updates)
            # Auto-fill the image µm/pixel (XY) from the voxel size if unset.
            if float(self.um_image.value) <= 0:
                self.um_image.value = float(self.state["voxel_size"][1])
            self._show_channels()
            self._set_busy(False)
            show_info("Image loaded.")

        def _err(e):
            self._set_busy(False); show_warning(f"Load failed: {e}")

        w = _work(); w.returned.connect(_done); w.errored.connect(_err); w.start()

    def _show_channels(self):
        vs = self.state["voxel_size"]
        n = self.state["raw"].shape[0]
        for name, cmap, ch in (("Cilia", "green", self.ch_cilia.value),
                               ("Basal Bodies", "magenta", self.ch_bb.value)):
            lname = f"LC-FCN: Raw {name}"
            if lname in self.viewer.layers:
                self.viewer.layers.remove(lname)
            _add_image_safe(self.viewer, self.state["raw"][_clamp_ch(n, ch)],
                            name=lname, scale=vs, colormap=cmap,
                            blending="additive", visible=(name == "Cilia"))

    # ── heatmap + proposals ──────────────────────────────────────────────────
    def _on_run(self):
        if self._busy:
            return
        if self.fcn is None:
            show_warning("Build the FCN first (step ①).")
            return
        if "raw" not in self.state:
            show_warning("Load an image first (step ②).")
            return
        self._set_busy(True)
        show_info("Running FCN heatmap …")
        p = self._params()
        rgb = build_fcn_rgb(self.state["raw"], p["ch_cilia"], p["ch_bb"],
                            mode=p["norm_mode"], p_low=p["p_low"], p_high=p["p_high"])
        rgb, in_scale = resample_to_train_scale(rgb, p["um_image"], p["um_train"])
        if abs(in_scale - 1.0) > 1e-3:
            print(f"[FCN] scale-matching: image {p['um_image']:.4f} → train "
                  f"{p['um_train']:.4f} µm/px ⇒ resample ×{in_scale:.3f}")

        @thread_worker
        def _work():
            _select_device(self.gpu_combo.value)
            return cnn_heatmap(self.fcn, rgb, self._device()), in_scale

        def _done(res):
            heatmap, scl = res
            self.state["heatmap"] = heatmap
            self.state["in_scale"] = scl
            self._show_heatmap(heatmap)
            self._detect_from_heatmap()
            self._set_busy(False)
            show_info(f"Heatmap {heatmap.shape} · max P={heatmap.max():.2f}")

        def _err(e):
            self._set_busy(False); show_warning(f"Heatmap failed: {e}")

        w = _work(); w.returned.connect(_done); w.errored.connect(_err); w.start()

    def _show_heatmap(self, heatmap: np.ndarray):
        from PIL import Image
        oh, ow = self.state["raw"].shape[-2:]
        up = np.asarray(Image.fromarray((heatmap * 255).astype(np.uint8))
                        .resize((ow, oh), Image.BILINEAR)).astype(np.float32) / 255.0
        lname = "LC-FCN: P(cilium) heatmap"
        if lname in self.viewer.layers:
            self.viewer.layers.remove(lname)
        self.viewer.add_image(up, name=lname, scale=self.state["voxel_size"][-2:],
                              colormap="magma", blending="additive", opacity=0.6,
                              contrast_limits=(0.0, 1.0))

    def _detect_from_heatmap(self):
        """Peak-find on the cached heatmap → proposal points (no re-run of FCN)."""
        if "heatmap" not in self.state:
            return
        rc, scores = peak_find(self.state["heatmap"], float(self.threshold.value),
                               int(self.min_dist.value))
        yx = heatmap_peaks_to_image_xy(rc, self.fcn_stride,
                                       float(self.state.get("in_scale", 1.0)),
                                       self.fcn_stride / 2.0)
        self.state["proposals_yx"] = yx
        self.state["proposal_scores"] = scores
        self._show_proposals(yx, scores)

    def _reactive_detect(self, *_):
        if not self._busy and "heatmap" in self.state:
            self._detect_from_heatmap()

    def _show_proposals(self, yx: np.ndarray, scores: np.ndarray):
        lname = "LC-FCN: Proposals"
        if lname in self.viewer.layers:
            self.viewer.layers.remove(lname)
        if yx.size == 0:
            return
        zmid = self.state["raw"].shape[1] / 2.0
        coords = np.column_stack([np.full(len(yx), zmid), yx[:, 0], yx[:, 1]])
        self.viewer.add_points(coords, name=lname, scale=self.state["voxel_size"],
                               size=8, face_color="yellow", border_color="black",
                               symbol="ring", properties={"score": scores})

    # ── segment + bb-anchored assignment ─────────────────────────────────────
    def _on_segment(self):
        if self._busy:
            return
        if self.fcn is None or "raw" not in self.state:
            show_warning("Build FCN + load an image first.")
            return
        self._set_busy(True)
        show_info("APOC segmentation + bb assignment …")
        p = self._params()
        raw = self.state["raw"]; vs = self.state["voxel_size"]

        @thread_worker
        def _work():
            _select_device(self.gpu_combo.value)
            return detect_and_assign(raw, vs, self.fcn, self.fcn_stride,
                                     self.model_size, self._device(), p)

        def _done(res):
            self.state.update(res)
            self._show_heatmap(res["heatmap"])
            self._show_proposals(res["proposals_yx"], res["scores"])
            self._show_results(res["cilia_df"], vs)
            cdf = res["cilia_df"]
            n_keep = int(cdf["validated"].sum()) if len(cdf) else 0
            self.seg_status.value = (f"{len(res['proposals_yx'])} proposals · "
                                     f"{n_keep} kept / {len(cdf) - n_keep} rejected")
            self._set_busy(False)
            show_info(f"Kept {n_keep} cilia (own a basal body), "
                      f"rejected {len(cdf) - n_keep}.")

        def _err(e):
            self._set_busy(False); show_warning(f"Segment/assign failed: {e}")

        w = _work(); w.returned.connect(_done); w.errored.connect(_err); w.start()

    def _show_results(self, cilia_df, vs):
        """Kept (green) vs rejected (red) cilia as two points layers."""
        for nm in ("LC-FCN: Kept cilia", "LC-FCN: Rejected cilia",
                   "LC-FCN: APOC cilia labels", "LC-FCN: APOC bb labels"):
            if nm in self.viewer.layers:
                self.viewer.layers.remove(nm)
        if "cilia_labels" in self.state:
            self.viewer.add_labels(self.state["cilia_labels"],
                                   name="LC-FCN: APOC cilia labels", scale=vs,
                                   visible=False)
        if "bb_labels" in self.state:
            self.viewer.add_labels(self.state["bb_labels"],
                                   name="LC-FCN: APOC bb labels", scale=vs,
                                   visible=False)
        if cilia_df is None or not len(cilia_df):
            return

        def _pts(sub, name, color):
            if not len(sub):
                return
            coords = np.asarray([list(map(float, c)) for c in sub["coords"]])
            self.viewer.add_points(coords, name=name, scale=vs, size=10,
                                   symbol="ring", face_color=color)

        _pts(cilia_df[cilia_df["validated"]], "LC-FCN: Kept cilia", "lime")
        _pts(cilia_df[~cilia_df["validated"].astype(bool)],
             "LC-FCN: Rejected cilia", "red")

    # ── batch ────────────────────────────────────────────────────────────────
    def _on_batch_progress(self, i: int, n: int, fn: str):
        show_info(f"Batch [{i + 1}/{n}] — {fn}")

    def _on_batch(self):
        if self._busy:
            return
        if self.fcn is None:
            show_warning("Build the FCN first (step ①).")
            return
        folder = Path(str(self.folder.value))
        out = Path(str(self.output.value))
        if not folder.is_dir() or not str(out):
            show_warning("Need a valid input folder and an output folder.")
            return
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".ims"))
        if not files:
            show_warning("No .ims files in the input folder.")
            return
        self._set_busy(True)
        show_info(f"Batch over {len(files)} files …")
        p = self._params()
        gpu = self.gpu_combo.value
        load_base = self._load_params()
        bridge = self._bridge

        @thread_worker
        def _work():
            import pandas as pd
            _select_device(gpu)
            device = self._device()
            csv_dir = out / "csv"; csv_dir.mkdir(parents=True, exist_ok=True)
            all_dfs, counts = [], []
            for i, f in enumerate(files):
                bridge.progressed.emit(i, len(files), f)
                lp = dict(load_base, ims_path=str(folder / f))
                loaded = step_load({}, lp)
                res = detect_and_assign(loaded["raw"], loaded["voxel_size"],
                                        self.fcn, self.fcn_stride, self.model_size,
                                        device, p)
                stem = os.path.splitext(f)[0]
                cdf, bdf = res["cilia_df"].copy(), res["bb_df"].copy()
                for df in (cdf, bdf):
                    df["filename"] = f
                    df["file_short"] = stem
                n_keep = int(cdf["validated"].sum()) if len(cdf) else 0
                counts.append({"filename": f, "n_proposals": len(res["proposals_yx"]),
                               "n_cilia": len(cdf), "n_kept": n_keep,
                               "n_rejected": len(cdf) - n_keep, "n_bb": len(bdf)})
                if len(cdf):
                    all_dfs.append(cdf)
                if len(bdf):
                    all_dfs.append(bdf)
            final = (pd.concat(all_dfs, ignore_index=True) if all_dfs
                     else pd.DataFrame())
            # Same primary artifact the existing batch writes: all_data sheet.
            with pd.ExcelWriter(csv_dir / "all_cilia_features.xlsx") as w:
                final.to_excel(w, sheet_name="all_data", index=False)
            pd.DataFrame(counts).to_csv(csv_dir / "per_image_counts.csv", index=False)
            return str(out), int(sum(c["n_kept"] for c in counts))

        def _done(res):
            out_dir, total = res
            self._set_busy(False)
            show_info(f"Batch done — {total} kept cilia. Wrote {out_dir}/csv/.")

        def _err(e):
            self._set_busy(False); show_warning(f"Batch failed: {e}")

        w = _work(); w.returned.connect(_done); w.errored.connect(_err); w.start()


def main():
    viewer = napari.Viewer(title="LimonCELLo — FCN proposal cilia app")
    app = LimoncelloFCNApp(viewer)
    viewer.window.add_dock_widget(app.widget, area="right", name="FCN cilia app")
    napari.run()


if __name__ == "__main__":
    main()
