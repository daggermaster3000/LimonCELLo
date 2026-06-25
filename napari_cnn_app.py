"""
LimonCELLo — fully-convolutional cilia detector (the "fast CNN" sliding window).

Idea
----
The ROI validator (``limoncello/ml/roi_validator.py``) is a small image
classifier: a stack of conv blocks, a global-average-pool, then dense layers that
output ``P(real cilium)`` for a single 96×96 (big) / 64×64 (tiny) ROI crop.

Sliding that classifier over a whole image with a window would re-run the conv
stack for every window — wasteful, because neighbouring windows share almost all
their convolutions. The standard trick is to make the network **fully
convolutional**: drop the GAP + flatten and turn each dense layer into a 1×1
convolution (a dense ``256 → 128`` becomes a 1×1 conv ``256 → 128``; the final
``128 → 2`` becomes a 1×1 conv ``128 → 2``). The all-conv net then runs **once**
over an image of any size and emits a coarse heatmap of ``P(cilium)`` — one output
pixel per receptive field, spaced by the network's total stride (16 px for big,
4 px for tiny). This reuses the trained conv weights directly.

We then threshold + peak-find the heatmap to get candidate ROI centres, and hand
those to APOC (``segment_cilia_ml``) — i.e. the CNN proposes *where* to look and
APOC does the precise segmentation, gated to the proposed centres.

Caveats (this is a *test* harness)
----------------------------------
* The classifier was trained on per-cilium crops where the cilium roughly fills
  the 96 px frame. In a whole image a cilium is only ~10-20 px, so there is a
  scale mismatch — use the **Input scale** control to upsample the image until a
  cilium approaches the training scale.
* The FCN input is built like the training thumbnails (cilia→green,
  BB→magenta, XY-MIP, raw min-max), so the network sees the same distribution.

Run with:  python napari_cnn_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import napari
import torch
from torch import nn
from magicgui.widgets import (
    Container, PushButton, ComboBox, FileEdit, Label, FloatSlider, SpinBox,
    FloatSpinBox, CheckBox,
)
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning
from scipy.ndimage import maximum_filter, center_of_mass

# Reuse the pipeline app's loader + device helpers so behaviour matches.
from napari_app import (
    step_load, _detect_gpus, _default_gpu, _select_device, _add_image_safe,
    _clamp_ch,
)
from limoncello.segmentation.cilia import segment_cilia_ml
from limoncello.ml.roi_validator import (
    ROINetBig, TinyROINet, load_bundle, DEFAULT_SIZE,
)

_DEFAULT_CLASSIFIER = str(
    (Path(__file__).parent / "segmenters" / "Cilia-d38-3D.cl").resolve()
)
_MODELS_DIR = str((Path(__file__).parent / "models").resolve())


# ──────────────────────────────────────────────────────────────────────────────
# Fully-convolutional conversion — reuse the trained conv weights, replace the
# GAP + flatten + dense head with 1×1 (and, for tiny, k×k) convolutions.
# ──────────────────────────────────────────────────────────────────────────────
class BigFCN(nn.Module):
    """``ROINetBig`` as a fully-convolutional net.

    Keeps the 4 conv blocks (drops the final ``AdaptiveAvgPool2d(1)``) and turns
    ``Linear(256→128)`` and ``Linear(128→2)`` into 1×1 convolutions carrying the
    exact trained weights. Output is ``(2, H/16, W/16)`` class logits; total
    downsampling stride is 16 (4 max-pools).
    """

    stride = 16

    def __init__(self, src: ROINetBig):
        super().__init__()
        self.features = src.features[:-1]          # 4 blocks, no GAP
        lin1: nn.Linear = src.head[2]              # Linear(256, 128)
        lin2: nn.Linear = src.head[5]              # Linear(128, 2)
        self.conv1 = nn.Conv2d(lin1.in_features, lin1.out_features, 1)
        self.conv2 = nn.Conv2d(lin2.in_features, lin2.out_features, 1)
        with torch.no_grad():
            # Linear weight (out, in) → conv weight (out, in, 1, 1).
            self.conv1.weight.copy_(lin1.weight[:, :, None, None])
            self.conv1.bias.copy_(lin1.bias)
            self.conv2.weight.copy_(lin2.weight[:, :, None, None])
            self.conv2.bias.copy_(lin2.bias)

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(self.conv1(x))
        return self.conv2(x)


class TinyFCN(nn.Module):
    """``TinyROINet`` as a fully-convolutional net.

    Drops the final ``AdaptiveAvgPool2d(4)``; the dense ``Linear(32·4·4 → 32)``
    that consumed the flattened 4×4×32 block becomes a **4×4 conv** (reshaping the
    flattened weight back to ``(32, 32, 4, 4)``), and ``Linear(32 → 2)`` becomes a
    1×1 conv. Total stride is 4 (2 max-pools); the 4×4 head adds receptive field
    but no stride.
    """

    stride = 4

    def __init__(self, src: TinyROINet):
        super().__init__()
        self.features = src.features[:-1]          # conv/relu/pool ×3, no GAP
        lin1: nn.Linear = src.head[2]              # Linear(32*4*4, 32)
        lin2: nn.Linear = src.head[4]              # Linear(32, 2)
        self.conv1 = nn.Conv2d(32, lin1.out_features, kernel_size=4)
        self.conv2 = nn.Conv2d(lin2.in_features, lin2.out_features, 1)
        with torch.no_grad():
            # Flatten order is (C, H, W) → reshape (out, 512) to (out, 32, 4, 4).
            self.conv1.weight.copy_(lin1.weight.view(lin1.out_features, 32, 4, 4))
            self.conv1.bias.copy_(lin1.bias)
            self.conv2.weight.copy_(lin2.weight[:, :, None, None])
            self.conv2.bias.copy_(lin2.bias)

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(self.conv1(x))
        return self.conv2(x)


def to_fully_convolutional(model: nn.Module) -> nn.Module:
    """Wrap a trained classifier as its fully-convolutional twin."""
    if isinstance(model, ROINetBig):
        return BigFCN(model)
    if isinstance(model, TinyROINet):
        return TinyFCN(model)
    raise TypeError(f"No FCN conversion for {type(model).__name__}")


# ──────────────────────────────────────────────────────────────────────────────
# FCN input + heatmap + peak finding (pure compute)
# ──────────────────────────────────────────────────────────────────────────────
def _norm_raw(a: np.ndarray) -> np.ndarray:
    """Plain linear min-max to [0, 1] — matches the training thumbnails rendered
    with ``correct_display=False`` (raw intensities, no gamma)."""
    a = np.asarray(a, dtype=np.float32)
    if a.size == 0:
        return a
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def build_fcn_input(raw: np.ndarray, ch_cilia: int, ch_bb: int,
                    scale: float = 1.0) -> np.ndarray:
    """Whole-image RGB the network was trained on: cilia→green, BB→magenta, on
    the XY max-projection with per-channel raw min-max. Returns ``(3, H, W)``
    float32 in [0, 1], optionally rescaled by ``scale`` to match training object
    size."""
    n_ch = raw.shape[0]
    cil = raw[_clamp_ch(n_ch, ch_cilia)].max(axis=0)        # (Y, X) from MIP
    bb = raw[_clamp_ch(n_ch, ch_bb)].max(axis=0)
    cil, bb = _norm_raw(cil), _norm_raw(bb)
    rgb = np.zeros((3, *cil.shape), dtype=np.float32)
    rgb[1] = cil                                            # green
    rgb[0] = bb; rgb[2] = bb                                # magenta
    if abs(scale - 1.0) > 1e-3:
        from PIL import Image
        h, w = cil.shape
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        img = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
        img = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
        rgb = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
    return rgb


def cnn_heatmap(fcn: nn.Module, rgb: np.ndarray,
                device: torch.device) -> np.ndarray:
    """Run the FCN once over the whole RGB image → ``P(cilium)`` heatmap
    ``(Hc, Wc)`` in [0, 1] (softmax over the 2 class channels, keep class 1)."""
    fcn = fcn.to(device).eval()
    x = torch.from_numpy(rgb[None]).to(device)              # (1, 3, H, W)
    with torch.no_grad():
        logits = fcn(x)                                     # (1, 2, Hc, Wc)
        prob = torch.softmax(logits, dim=1)[0, 1]
    return prob.float().cpu().numpy()


def peak_find(heatmap: np.ndarray, threshold: float,
              min_distance: int) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of ``heatmap`` ≥ ``threshold``, separated by at least
    ``min_distance`` heatmap pixels. Returns ``(rc, scores)`` where ``rc`` is an
    ``(N, 2)`` array of (row, col) heatmap indices."""
    if heatmap.size == 0:
        return np.empty((0, 2), int), np.empty((0,), float)
    size = max(1, int(min_distance))
    mx = maximum_filter(heatmap, size=size, mode="nearest")
    peaks = (heatmap == mx) & (heatmap >= threshold)
    rs, cs = np.where(peaks)
    return np.stack([rs, cs], axis=1), heatmap[rs, cs]


def heatmap_peaks_to_image_xy(rc: np.ndarray, stride: float, in_scale: float,
                              size: int) -> np.ndarray:
    """Map heatmap (row, col) indices back to **input image** (y, x) pixel
    centres. Each output cell covers a ``size``-wide receptive field stepped by
    ``stride`` on the (possibly up-scaled) input, so the receptive-field centre is
    ``cell * stride + size/2``; divide by ``in_scale`` to undo the input rescale.
    """
    if rc.size == 0:
        return np.empty((0, 2), float)
    yx = rc.astype(np.float64) * stride + size / 2.0
    return yx / max(in_scale, 1e-6)


def apoc_gated_by_centres(raw: np.ndarray, ch_cilia: int, classifier_path: str,
                          centres_yx: np.ndarray, gate_radius_px: float,
                          *, min_size: int, max_size: int):
    """Segment cilia with APOC over the full 3-D channel, then keep only labels
    whose XY centroid lies within ``gate_radius_px`` of a CNN-proposed centre.
    Returns ``(gated_labels, n_total, n_kept)``."""
    ch = _clamp_ch(raw.shape[0], ch_cilia)
    labels = np.asarray(segment_cilia_ml(
        raw[ch], classifier_path=classifier_path,
        gaussian_sigma=(0.0, 0.0, 0.0), log_transform=False,
        min_size=min_size, max_size=max_size,
    )).astype(np.int32)
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if ids.size == 0 or centres_yx.size == 0:
        return np.zeros_like(labels), int(ids.size), 0
    cents = center_of_mass(labels > 0, labels=labels, index=ids)  # (z, y, x)
    keep_ids = []
    for lid, c in zip(ids, np.atleast_2d(cents)):
        cy, cx = float(c[1]), float(c[2])
        d = np.hypot(centres_yx[:, 0] - cy, centres_yx[:, 1] - cx)
        if d.min() <= gate_radius_px:
            keep_ids.append(int(lid))
    gated = np.where(np.isin(labels, keep_ids), labels, 0).astype(np.int32)
    return gated, int(ids.size), len(keep_ids)


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
class LimoncelloCNNApp:
    """Dock widget: load a trained ROI model, convert it to fully-convolutional,
    run the heatmap over an image, peak-find ROI centres, gate APOC by them."""

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.state: dict = {}
        self.files: list[str] = []
        self.fcn: nn.Module | None = None
        self.fcn_stride: int = 16
        self.model_size: int = DEFAULT_SIZE["big"]
        self._busy = False
        self.widget = self._build()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build(self) -> Container:
        devices = _detect_gpus()
        self.gpu_combo = ComboBox(
            label="Compute device",
            choices=[(lbl, val) for lbl, val in devices],
            value=_default_gpu(devices),
        )

        self.model_combo = ComboBox(label="ROI model (.pt)",
                                    choices=self._discover_models())
        self.refresh_models = PushButton(text="↻ Rescan models")
        self.refresh_models.clicked.connect(self._on_rescan_models)
        self.build_fcn_btn = PushButton(text="① Build fully-convolutional net")
        self.build_fcn_btn.clicked.connect(self._on_build_fcn)
        self.fcn_status = Label(value="No FCN built yet.")

        self.folder = FileEdit(label="Image folder", mode="d")
        self.scan_btn = PushButton(text="Scan folder")
        self.scan_btn.clicked.connect(self._scan_folder)
        self.file_combo = ComboBox(label="Image", choices=())

        self.ch_cilia = SpinBox(label="Cilia channel", value=0, min=0, max=16)
        self.ch_bb = SpinBox(label="Basal-body channel", value=2, min=0, max=16)
        self.p_low = FloatSpinBox(label="Norm p-low", value=2.0, min=0, max=100)
        self.p_high = FloatSpinBox(label="Norm p-high", value=98.0, min=0, max=100)

        self.in_scale = FloatSlider(label="Input scale ×", value=1.0,
                                    min=0.5, max=6.0, step=0.5)
        self.load_btn = PushButton(text="② Load image")
        self.load_btn.clicked.connect(self._on_load)
        self.run_btn = PushButton(text="③ Run CNN heatmap")
        self.run_btn.clicked.connect(self._on_run_heatmap)

        self.thr = FloatSlider(label="Peak threshold", value=0.5, min=0.0,
                               max=1.0, step=0.01)
        self.min_dist = SpinBox(label="Min peak dist (cells)", value=2, min=1, max=50)
        self.detect_btn = PushButton(text="④ Detect ROI centres")
        self.detect_btn.clicked.connect(self._on_detect)
        self.detect_status = Label(value="—")

        self.classifier = FileEdit(label="APOC cilia .cl",
                                   value=_DEFAULT_CLASSIFIER, mode="r")
        self.cilia_min = SpinBox(label="Cilia min size", value=10, min=0, max=100000)
        self.cilia_max = SpinBox(label="Cilia max size", value=10000, min=0,
                                 max=10_000_000)
        self.gate_radius = FloatSpinBox(label="Gate radius (px)", value=20.0,
                                        min=0, max=500)
        self.apoc_btn = PushButton(text="⑤ Segment cilia (APOC) gated by CNN")
        self.apoc_btn.clicked.connect(self._on_apoc)
        self.apoc_status = Label(value="—")

        self._buttons = [self.build_fcn_btn, self.load_btn, self.run_btn,
                         self.detect_btn, self.apoc_btn]

        return Container(widgets=[
            Label(value="<b>Fully-convolutional cilia detector</b>"),
            self.gpu_combo,
            Label(value="<b>1 · Model → FCN</b>"),
            self.model_combo, self.refresh_models, self.build_fcn_btn,
            self.fcn_status,
            Label(value="<b>2 · Image</b>"),
            self.folder, self.scan_btn, self.file_combo,
            self.ch_cilia, self.ch_bb, self.p_low, self.p_high,
            self.in_scale, self.load_btn,
            Label(value="<b>3 · Heatmap → centres</b>"),
            self.run_btn, self.thr, self.min_dist, self.detect_btn,
            self.detect_status,
            Label(value="<b>4 · APOC gated by CNN</b>"),
            self.classifier, self.cilia_min, self.cilia_max, self.gate_radius,
            self.apoc_btn, self.apoc_status,
        ], labels=True, scrollable=True)

    # ── model discovery ────────────────────────────────────────────────────────
    def _discover_models(self) -> list[str]:
        if not os.path.isdir(_MODELS_DIR):
            return []
        return sorted(f for f in os.listdir(_MODELS_DIR) if f.endswith(".pt"))

    def _on_rescan_models(self):
        self.model_combo.choices = self._discover_models()

    def _set_busy(self, busy: bool):
        self._busy = busy
        for b in self._buttons:
            b.enabled = not busy

    # ── build FCN ───────────────────────────────────────────────────────────────
    def _on_build_fcn(self):
        name = self.model_combo.value
        if not name:
            show_warning("Pick a trained .pt model first.")
            return
        path = os.path.join(_MODELS_DIR, str(name))
        try:
            model, meta = load_bundle(path)
            self.fcn = to_fully_convolutional(model)
            self.fcn_stride = int(getattr(self.fcn, "stride", 16))
            self.model_size = int(meta.get("size", DEFAULT_SIZE.get(
                meta.get("arch", "big"), 64)))
        except Exception as exc:                              # noqa: BLE001
            show_warning(f"Could not build FCN: {exc}")
            return
        self.fcn_status.value = (
            f"FCN ready · arch={meta.get('arch')} · size={self.model_size} · "
            f"stride={self.fcn_stride}")
        show_info("Fully-convolutional net built from trained weights.")

    # ── folder / image ──────────────────────────────────────────────────────────
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

    def _params(self) -> dict:
        p = dict(
            gpu_device=self.gpu_combo.value,
            ch_cilia=self.ch_cilia.value, ch_neurites=0,
            ch_bb=self.ch_bb.value, ch_nuclei=0,
            use_mip=False, make_isotropic=False,
            p_low=self.p_low.value, p_high=self.p_high.value, ch_norm={},
        )
        if self.file_combo.value and self.folder.value:
            p["ims_path"] = str(Path(str(self.folder.value)) / self.file_combo.value)
        else:
            p["ims_path"] = ""
        return p

    def _on_load(self):
        if self._busy:
            return
        p = self._params()
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
            self._show_channels()
            self._set_busy(False)
            show_info("Image loaded.")

        def _err(e):
            self._set_busy(False)
            show_warning(f"Load failed: {e}")

        w = _work(); w.returned.connect(_done); w.errored.connect(_err); w.start()

    def _show_channels(self):
        vs = self.state["voxel_size"]
        n = self.state["raw"].shape[0]
        for name, cmap, ch in (("Cilia", "green", self.ch_cilia.value),
                               ("Basal Bodies", "magenta", self.ch_bb.value)):
            c = _clamp_ch(n, ch)
            lname = f"LC-CNN: Raw {name}"
            if lname in self.viewer.layers:
                self.viewer.layers.remove(lname)
            _add_image_safe(self.viewer, self.state["raw"][c], name=lname,
                            scale=vs, colormap=cmap, blending="additive",
                            visible=(name == "Cilia"))

    # ── heatmap ─────────────────────────────────────────────────────────────────
    def _on_run_heatmap(self):
        if self._busy:
            return
        if self.fcn is None:
            show_warning("Build the fully-convolutional net first (step ①).")
            return
        if "raw" not in self.state:
            show_warning("Load an image first (step ②).")
            return
        self._set_busy(True)
        show_info("Running CNN heatmap …")
        scale = float(self.in_scale.value)
        rgb = build_fcn_input(self.state["raw"], self.ch_cilia.value,
                              self.ch_bb.value, scale)

        @thread_worker
        def _work():
            _select_device(self.gpu_combo.value)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            return cnn_heatmap(self.fcn, rgb, device), rgb.shape[1:]

        def _done(res):
            heatmap, in_hw = res
            self.state["heatmap"] = heatmap
            self.state["fcn_in_scale"] = scale
            self._show_heatmap(heatmap, in_hw)
            self._set_busy(False)
            show_info(f"Heatmap {heatmap.shape} · max P={heatmap.max():.2f}")

        def _err(e):
            self._set_busy(False)
            show_warning(f"Heatmap failed: {e}")

        w = _work(); w.returned.connect(_done); w.errored.connect(_err); w.start()

    def _show_heatmap(self, heatmap: np.ndarray, in_hw):
        """Display the coarse heatmap upsampled to the input grid (so it overlays
        the channels), accounting for the input rescale."""
        lname = "LC-CNN: P(cilium) heatmap"
        if lname in self.viewer.layers:
            self.viewer.layers.remove(lname)
        # Upsample heatmap → input pixels → undo input scale → original image px.
        from PIL import Image
        ih, iw = int(in_hw[0]), int(in_hw[1])
        up = np.asarray(Image.fromarray((heatmap * 255).astype(np.uint8))
                        .resize((iw, ih), Image.BILINEAR)).astype(np.float32) / 255.0
        scale = float(self.state.get("fcn_in_scale", 1.0))
        oh, ow = self.state["raw"].shape[-2:]
        if (ih, iw) != (oh, ow):
            up = np.asarray(Image.fromarray((up * 255).astype(np.uint8))
                            .resize((ow, oh), Image.BILINEAR)).astype(np.float32) / 255.0
        vs = self.state["voxel_size"]
        self.viewer.add_image(up, name=lname, scale=vs[-2:], colormap="inferno",
                              blending="additive", opacity=0.6,
                              contrast_limits=(0.0, 1.0))

    # ── detect centres ──────────────────────────────────────────────────────────
    def _on_detect(self):
        if "heatmap" not in self.state:
            show_warning("Run the heatmap first (step ③).")
            return
        rc, scores = peak_find(self.state["heatmap"], float(self.thr.value),
                               int(self.min_dist.value))
        yx = heatmap_peaks_to_image_xy(
            rc, self.fcn_stride, float(self.state.get("fcn_in_scale", 1.0)),
            self.model_size)
        self.state["centres_yx"] = yx
        self._show_centres(yx, scores)
        self.detect_status.value = f"{len(yx)} ROI centres ≥ {self.thr.value:.2f}"
        show_info(f"Detected {len(yx)} candidate cilia.")

    def _show_centres(self, yx: np.ndarray, scores: np.ndarray):
        lname = "LC-CNN: ROI centres"
        if lname in self.viewer.layers:
            self.viewer.layers.remove(lname)
        if yx.size == 0:
            return
        # Place points at mid-Z so they sit inside the 3-D volume for display.
        zmid = self.state["raw"].shape[1] / 2.0
        coords = np.column_stack([np.full(len(yx), zmid), yx[:, 0], yx[:, 1]])
        vs = self.state["voxel_size"]
        self.viewer.add_points(
            coords, name=lname, scale=vs, size=8, face_color="yellow",
            border_color="black", symbol="ring",
            properties={"score": scores},
        )

    # ── APOC gated ──────────────────────────────────────────────────────────────
    def _on_apoc(self):
        if self._busy:
            return
        if "centres_yx" not in self.state:
            show_warning("Detect ROI centres first (step ④).")
            return
        self._set_busy(True)
        show_info("Running APOC, gating by CNN centres …")
        p = dict(
            ch_cilia=self.ch_cilia.value,
            classifier_path=str(self.classifier.value),
            centres=self.state["centres_yx"],
            gate=float(self.gate_radius.value),
            min_size=int(self.cilia_min.value), max_size=int(self.cilia_max.value),
            gpu=self.gpu_combo.value,
        )
        raw = self.state["raw"]

        @thread_worker
        def _work():
            _select_device(p["gpu"])
            return apoc_gated_by_centres(
                raw, p["ch_cilia"], p["classifier_path"], p["centres"],
                p["gate"], min_size=p["min_size"], max_size=p["max_size"])

        def _done(res):
            gated, n_total, n_kept = res
            lname = "LC-CNN: APOC cilia (gated)"
            if lname in self.viewer.layers:
                self.viewer.layers.remove(lname)
            self.viewer.add_labels(gated, name=lname, scale=self.state["voxel_size"])
            self.apoc_status.value = f"kept {n_kept} / {n_total} APOC cilia"
            self._set_busy(False)
            show_info(f"APOC: kept {n_kept} of {n_total} cilia near CNN centres.")

        def _err(e):
            self._set_busy(False)
            show_warning(f"APOC failed: {e}")

        w = _work(); w.returned.connect(_done); w.errored.connect(_err); w.start()


def main():
    viewer = napari.Viewer(title="LimonCELLo — Fast CNN cilia detector")
    app = LimoncelloCNNApp(viewer)
    viewer.window.add_dock_widget(app.widget, area="right",
                                  name="Fast CNN detector")
    napari.run()


if __name__ == "__main__":
    main()
