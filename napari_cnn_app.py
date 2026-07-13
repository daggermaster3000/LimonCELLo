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
import hashlib
import tempfile
import uuid
from pathlib import Path

import numpy as np
import napari
import torch
from torch import nn
from magicgui.widgets import (
    Container, PushButton, ComboBox, FileEdit, Label, FloatSlider, SpinBox,
    FloatSpinBox, CheckBox, LineEdit,
)
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning
from scipy.ndimage import maximum_filter, center_of_mass
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:                                              # older matplotlib
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Reuse the pipeline app's loader + device helpers so behaviour matches.
from napari_app import (
    step_load, _detect_gpus, _default_gpu, _select_device, _add_image_safe,
)
from limoncello.segmentation.cilia import segment_cilia_ml
from limoncello.ml.roi_validator import (
    ROINetBig, TinyROINet, load_bundle, DEFAULT_SIZE, get_device,
)
from limoncello.analysis.pair_cilia_to_bb import parse_coords_string
from limoncello.utils.app_helpers import load_run_json

_DEFAULT_CLASSIFIER = str(
    (Path(__file__).parent / "segmenters" / "Cilia-d38-3D.cl").resolve()
)
_MODELS_DIR = str((Path(__file__).parent / "models").resolve())


def _clamp_ch(n_ch: int, idx) -> int:
    """Clamp a channel index into ``[0, n_ch - 1]``."""
    try:
        return int(min(max(int(idx), 0), n_ch - 1))
    except (TypeError, ValueError):
        return 0


_ISO_CACHE = os.path.join(tempfile.gettempdir(), "lc_iso_cache")


def cached_load(p: dict) -> dict:
    """``step_load`` with a disk cache of the (isotropic) raw stack.

    The raw stack depends only on the file + ``use_mip`` + ``make_isotropic`` (not
    the percentile params, which only affect the unused ``norm``), so we key on
    those and skip re-reading the ``.ims`` + resampling on repeat loads — a big win
    when training reloads every image. Cached as ``.npz`` in the system temp dir;
    returns ``dict(raw, voxel_size, n_ch)``.
    """
    ims = p.get("ims_path", "")
    try:
        sig = (f"{ims}|{os.path.getmtime(ims)}|{int(bool(p.get('use_mip')))}|"
               f"{int(bool(p.get('make_isotropic')))}")
    except OSError:
        return step_load({}, p)
    key = hashlib.md5(sig.encode()).hexdigest()
    os.makedirs(_ISO_CACHE, exist_ok=True)
    cpath = os.path.join(_ISO_CACHE, f"{key}.npz")
    if os.path.exists(cpath):
        try:
            d = np.load(cpath, allow_pickle=False)
            return dict(raw=d["raw"],
                        voxel_size=tuple(float(v) for v in d["voxel_size"]),
                        n_ch=int(d["n_ch"]))
        except Exception:                                     # noqa: BLE001
            pass
    out = step_load({}, p)
    try:
        np.savez(cpath, raw=out["raw"],
                 voxel_size=np.asarray(out["voxel_size"], dtype=float),
                 n_ch=int(out["n_ch"]))
        print(f"[cache] saved isotropic stack → {cpath}")
    except Exception as exc:                                  # noqa: BLE001
        print(f"[cache] could not write {cpath}: {exc}")
    return out


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


def model_stride(model: nn.Module) -> int:
    """Total downsampling stride of a trained classifier's conv stack
    (max-pools): 16 for the big net (4 pools), 4 for tiny (2 pools)."""
    return 16 if isinstance(model, ROINetBig) else 4


def cam_heatmap(model: nn.Module, rgb: np.ndarray,
                device: torch.device) -> np.ndarray:
    """Coarse class-activation-style ``P(cilium)`` map **without building the
    FCN** — the cheap, free localization signal.

    Run the trained classifier's conv blocks (drop the trailing GAP in
    ``features``) to get the pre-pool feature map ``(C, H/16, W/16)``, then apply
    the head's dense weights as 1×1 convolutions straight onto that map (dense
    ``[out,in]`` → conv ``[out,in,1,1]``; for the tiny net the first dense is a
    4×4 conv). Softmax over the 2 class channels → a 6×6-per-patch CAM. This is
    mathematically the same as the surgically-built FCN, just computed directly
    off the original model's features + classification weights — no new module.
    """
    import torch.nn.functional as F
    model = model.to(device).eval()
    x = torch.from_numpy(rgb[None]).to(device)
    feats = model.features[:-1]                 # conv blocks, no GAP
    lins = [m for m in model.head if isinstance(m, nn.Linear)]
    with torch.no_grad():
        feat = feats(x)                         # (1, C, h, w)
        c = feat.shape[1]
        w1, b1 = lins[0].weight, lins[0].bias
        w2, b2 = lins[-1].weight, lins[-1].bias
        if w1.shape[1] == c:                    # big: dense 256→128 is a 1×1 conv
            h = F.relu(F.conv2d(feat, w1[:, :, None, None], b1))
        else:                                   # tiny: dense 512→32 is a 4×4 conv
            k = int(round((w1.shape[1] // c) ** 0.5))
            h = F.relu(F.conv2d(feat, w1.view(w1.shape[0], c, k, k), b1))
        logits = F.conv2d(h, w2[:, :, None, None], b2)
        prob = torch.softmax(logits, dim=1)[0, 1]
    return prob.float().cpu().numpy()


def sliding_window_heatmap(model: nn.Module, rgb: np.ndarray, size: int,
                           stride: int, device: torch.device,
                           batch_size: int = 256) -> np.ndarray:
    """Brute-force sliding window — the simplest detector, no architecture change.

    Slide the **original** ``size×size`` classifier over ``rgb (3, H, W)`` with
    ``stride``, score each window's ``P(keep)``, and assemble them into a grid
    heatmap ``(n_rows, n_cols)`` on the same ``cell * stride + size/2`` geometry as
    the FCN/CAM maps (so the same peak-find + coord mapping apply). Windows are
    batched for throughput, but this still recomputes the shared convolutions for
    every overlapping window — exactly the redundant work the fully-convolutional
    version avoids. Fine for offline screening.
    """
    model = model.to(device).eval()
    _, h, w = rgb.shape
    # Pad so at least one full window fits in each axis.
    pad_h, pad_w = max(0, size - h), max(0, size - w)
    if pad_h or pad_w:
        rgb = np.pad(rgb, ((0, 0), (0, pad_h), (0, pad_w)))
        _, h, w = rgb.shape
    ys = list(range(0, h - size + 1, stride)) or [0]
    xs = list(range(0, w - size + 1, stride)) or [0]
    cells = [(i, j, y, x) for i, y in enumerate(ys) for j, x in enumerate(xs)]
    hm = np.zeros((len(ys), len(xs)), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(cells), batch_size):
            chunk = cells[b:b + batch_size]
            batch = np.stack([rgb[:, y:y + size, x:x + size] for *_, y, x in chunk])
            probs = torch.softmax(
                model(torch.from_numpy(batch).to(device)), dim=1)[:, 1]
            probs = probs.float().cpu().numpy()
            for (i, j, _, _), pv in zip(chunk, probs):
                hm[i, j] = pv
    return hm


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
                              offset: float) -> np.ndarray:
    """Map heatmap (row, col) indices back to **input image** (y, x) pixel
    centres: ``(rc * stride + offset) / in_scale``.

    ``offset`` is the per-cell centre offset (in scaled-input px) and differs by
    detector:

    * FCN / CAM — the conv blocks are ``padding=1`` ('same') with stride-2 pools,
      so output cell ``r`` maps to the input block ``[r*stride, (r+1)*stride)``,
      centred at ``offset = stride/2``. (Using ``size/2`` here — the
      receptive-field *width* — would shift every centre down-right by
      ``size/2 - stride/2``.)
    * Sliding window — cell ``r`` is an explicit window with top-left ``r*stride``,
      so its centre is ``offset = size/2``.

    Divide by ``in_scale`` to undo the input rescale → native image pixels.
    """
    if rc.size == 0:
        return np.empty((0, 2), float)
    yx = rc.astype(np.float64) * stride + float(offset)
    return yx / max(in_scale, 1e-6)


CILIAI_DEFAULT_URL = "http://localhost:5007"


def _rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    """``(3, H, W)`` float [0,1] → PNG bytes (``H, W, 3`` uint8)."""
    from PIL import Image
    import io
    img = (np.clip(np.transpose(rgb, (1, 2, 0)), 0.0, 1.0) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def ciliai_detect_centres(rgb: np.ndarray, server_url: str,
                          score_thr: float = 0.5, timeout: float = 180.0
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Detect cilia centres via a CiliAI (datamarkin/ciliai) detector server.

    POSTs the whole-image RGB as a PNG in a multipart ``file`` field to
    ``<server_url>/predict/cilia-detector`` and turns every returned object with
    ``bbox_score >= score_thr`` into its bbox centre. CiliAI reports boxes in the
    **uploaded image's** pixel space, so — when ``rgb`` is built at input scale
    1.0 — the centres are already native image pixels (no rescale needed).

    Returns ``(centres_yx (N,2), scores (N,))``. Uses only the Python stdlib for
    the request (no ``requests``/``detectron2`` dependency in LimonCELLo — you run
    the CiliAI Flask server separately)."""
    import json
    import urllib.request

    png = _rgb_to_png_bytes(rgb)
    boundary = "----LCciliai" + uuid.uuid4().hex
    pre = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="image.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8")
    post = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = pre + png + post
    url = server_url.rstrip("/") + "/predict/cilia-detector"
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Content-Length": str(len(body))})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    objs = ((data or {}).get("predictions") or {}).get("objects") or []
    ys, xs, sc = [], [], []
    for o in objs:
        try:
            s = float(o.get("bbox_score", o.get("score", 1.0)))
        except (TypeError, ValueError):
            s = 1.0
        if s < score_thr:
            continue
        b = o.get("bbox")
        if not b or len(b) < 4:
            continue
        x0, y0, x1, y1 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        ys.append((y0 + y1) / 2.0)
        xs.append((x0 + x1) / 2.0)
        sc.append(s)
    if not ys:
        return np.empty((0, 2), float), np.empty((0,), float)
    return np.column_stack([ys, xs]).astype(float), np.asarray(sc, float)


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
# Whole-image detector — TRAINED on full images (not a transplant)
# ──────────────────────────────────────────────────────────────────────────────
# The transplant/CAM/sliding paths reuse a classifier trained on tight 96×96 ROI
# crops, so they inherit that object-fills-the-frame scale. Here we instead train
# a fully-convolutional detector *directly on whole images*: targets are
# reconstructed by stamping each kept cilium (from the run's ``all_data`` coords +
# ``human_validation.csv`` keep/reject) as a Gaussian onto a 1/16-resolution
# heatmap over the full image. The net then learns cilia at their true
# whole-image scale/context. Conv blocks can be warm-started from the ROI
# classifier; the head is trained fresh.
class FCNDetector(torch.nn.Module):
    """Fully-convolutional cilium detector: ROINetBig conv blocks (no GAP) → 1×1
    conv head → single-channel logit heatmap (sigmoid → P(cilium)), stride 16."""

    stride = 16

    def __init__(self):
        super().__init__()
        self.features = ROINetBig().features[:-1]      # 4 conv blocks, no GAP
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(128, 1, 1))

    def forward(self, x):
        return self.head(self.features(x))

    def warm_start_from(self, classifier: torch.nn.Module) -> bool:
        """Copy the trained conv-block weights from a ROINetBig classifier."""
        if isinstance(classifier, ROINetBig):
            self.features.load_state_dict(classifier.features[:-1].state_dict())
            return True
        return False


def _stamp_gaussian(target: np.ndarray, gy: int, gx: int, sigma: float) -> None:
    """Max-blend a unit Gaussian centred at ``(gy, gx)`` into ``target`` (in grid
    cells), so overlapping cilia don't sum past 1."""
    h, w = target.shape
    r = max(1, int(round(3 * sigma)))
    ys = np.arange(max(0, gy - r), min(h, gy + r + 1))
    xs = np.arange(max(0, gx - r), min(w, gx + r + 1))
    if ys.size == 0 or xs.size == 0:
        return
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    g = np.exp(-((yy - gy) ** 2 + (xx - gx) ** 2) / (2.0 * sigma * sigma))
    sub = target[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
    np.maximum(sub, g.astype(np.float32), out=sub)


def run_file_map(run_dir: str) -> dict[str, str]:
    """``{basename: full_path}`` for every input image of a finished run.

    Rebuilt from the run's ``csv/run_parameters.json`` ``input_path`` exactly the
    way the pipeline enumerated its inputs: a ``.txt`` manifest is read line by
    line (blank / ``#`` lines skipped) so images scattered across folders resolve
    to their real paths; otherwise the folder is listed for ``.ims`` files. The
    keys are basenames because that's what ``filename`` is in the tables."""
    j = load_run_json(run_dir)
    if not j:
        return {}
    input_path = j.get("input_path")
    if not input_path:
        return {}
    fmap: dict[str, str] = {}
    if str(input_path).lower().endswith(".txt"):
        try:
            with open(input_path, encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln and not ln.lstrip().startswith("#"):
                        fmap[os.path.basename(ln)] = ln
        except OSError:
            return {}
    elif os.path.isdir(input_path):
        for f in sorted(os.listdir(input_path)):
            if f.endswith(".ims"):
                fmap[f] = os.path.join(input_path, f)
    return fmap


def reconstruct_targets(run_dir: str, image_folder: str | None, p: dict,
                        log=print) -> list:
    """Reconstruct full-image training targets from a finished run.

    Reads the run's ``csv/all_cilia_features.xlsx`` (cilium coords per image) and
    ``csv/human_validation.csv`` (keep/reject). Each image's raw volume is located
    from the run's own **file list** (``run_parameters.json`` → the folder or
    ``.txt`` manifest the run was computed from), so images spread across folders
    still resolve; ``image_folder`` is only a fallback when a file isn't found in
    that list. For each located image it loads the volume (same isotropic loader
    as inference), builds the whole-image RGB, and stamps every **kept** cilium as
    a Gaussian on a 1/``stride`` target heatmap. Returns ``[(rgb (3,H,W), target
    (Hc,Wc), filename, n_pos), …]``.
    """
    import pandas as pd
    xls = os.path.join(run_dir, "csv", "all_cilia_features.xlsx")
    if not os.path.exists(xls):
        raise FileNotFoundError(f"No all_cilia_features.xlsx in {run_dir}/csv")
    df = pd.read_excel(xls, sheet_name="all_data")
    if "object_type" in df.columns:
        df = df[df["object_type"] == "cilia"]
    keep: dict[tuple[str, int], bool] = {}
    vfile = os.path.join(run_dir, "csv", "human_validation.csv")
    if os.path.exists(vfile):
        v = pd.read_csv(vfile)
        keep = {(str(r.filename), int(r.cilia_id)): bool(r.human_validated)
                for r in v.itertuples()}

    fmap = run_file_map(run_dir)
    log(f"  run file list: {len(fmap)} image(s) from run_parameters.json")

    stride, sigma = int(p["stride"]), float(p["sigma"])
    crop = float(p.get("crop_factor", 0.5))
    samples = []
    for fn, g in df.groupby("filename"):
        # Prefer the run's own file list; fall back to the Image folder only when
        # the file isn't listed there (or its recorded path no longer exists).
        ipath = fmap.get(str(fn))
        if not ipath or not os.path.exists(ipath):
            ipath = (os.path.join(image_folder, str(fn))
                     if image_folder else None)
        if not ipath or not os.path.exists(ipath):
            log(f"  skip {fn} — not in run file list or image folder"); continue
        loaded = cached_load(dict(p["load"], ims_path=ipath))
        rgb = build_fcn_input(loaded["raw"], p["ch_cilia"], p["ch_bb"], 1.0)
        _, h, w = rgb.shape
        ph, pw = (-h) % stride, (-w) % stride          # pad to multiple of stride
        if ph or pw:
            rgb = np.pad(rgb, ((0, 0), (0, ph), (0, pw)))
        _, h, w = rgb.shape
        hc, wc = h // stride, w // stride
        target = np.zeros((hc, wc), dtype=np.float32)
        coords = parse_coords_string(g["coords"])      # (n, 3) z, y, x
        vxy = float(loaded["voxel_size"][1])           # µm per XY pixel
        n_pos = 0
        for (_, row), c in zip(g.iterrows(), coords):
            if not keep.get((str(fn), int(row["cilia_id"])), True):
                continue                               # rejected → background
            gy = min(max(int(round(float(c[1]) / stride - 0.5)), 0), hc - 1)
            gx = min(max(int(round(float(c[2]) / stride - 0.5)), 0), wc - 1)
            # Footprint from the cilium's own ROI bbox (length_um → radius in
            # grid cells), shrunk by `crop`; fall back to the fixed sigma. Clamp
            # so a positive is always at least ~half a cell wide.
            s = sigma
            lum = row.get("length_um", np.nan) if hasattr(row, "get") else np.nan
            if np.isfinite(lum) and vxy > 0:
                r_cells = (float(lum) / 2.0 / vxy) / stride
                s = min(sigma, max(0.4, r_cells * crop))
            _stamp_gaussian(target, gy, gx, s)
            n_pos += 1
        samples.append((rgb.astype(np.float32), target, str(fn), n_pos))
        log(f"  {fn}: {n_pos} positives · grid {hc}×{wc}")
    return samples


def train_fcn_detector(samples: list, *, epochs: int = 30, lr: float = 1e-3,
                       warm: torch.nn.Module | None = None,
                       progress=None) -> tuple[FCNDetector, list]:
    """Train an ``FCNDetector`` on reconstructed full-image targets.

    Weighted BCE on the dense heatmap (positives are sparse), Adam, horizontal/
    vertical flip augmentation (kept consistent between image and target;
    batch = 1 so arbitrary image sizes are fine). Returns ``(model, history)``.
    """
    if not samples:
        raise ValueError("No training samples (no images matched / no positives).")
    device = get_device()
    model = FCNDetector().to(device)
    if warm is not None:
        model.warm_start_from(warm)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    tot = float(sum(t.size for _, t, _, _ in samples))
    pos = float(sum((t > 0.5).sum() for _, t, _, _ in samples))
    pw = torch.tensor([max(1.0, (tot - pos) / max(pos, 1.0))], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    rng = np.random.default_rng(0)
    history = []
    for ep in range(epochs):
        model.train()
        run = 0.0
        for i in rng.permutation(len(samples)):
            rgb, target, _, _ = samples[i]
            xb = torch.from_numpy(rgb[None]).to(device)
            tb = torch.from_numpy(target[None, None]).to(device)
            if rng.random() < 0.5:
                xb = torch.flip(xb, [3]); tb = torch.flip(tb, [3])
            if rng.random() < 0.5:
                xb = torch.flip(xb, [2]); tb = torch.flip(tb, [2])
            opt.zero_grad()
            loss = loss_fn(model(xb), tb)
            loss.backward()
            opt.step()
            run += float(loss.item())
        history.append(run / len(samples))
        if progress:
            progress((ep + 1) / epochs,
                     f"epoch {ep + 1}/{epochs} · BCE {history[-1]:.4f} ({device.type})")
    return model, history


def detector_heatmap(model: FCNDetector, rgb: np.ndarray,
                     device: torch.device) -> np.ndarray:
    """Run a trained ``FCNDetector`` over a whole image → P(cilium) heatmap
    (sigmoid of the single-channel logit map)."""
    model = model.to(device).eval()
    x = torch.from_numpy(rgb[None]).to(device)
    with torch.no_grad():
        hm = torch.sigmoid(model(x))[0, 0]
    return hm.float().cpu().numpy()


def save_detector(path: str, model: FCNDetector, meta: dict) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "fcn_detector", "stride": int(model.stride),
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "meta": meta}, path)
    return path


def load_detector(path: str) -> tuple[FCNDetector, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "fcn_detector":
        raise ValueError(f"{path} is not an FCN detector bundle.")
    model = FCNDetector()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("meta", {})


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
        self.detector: FCNDetector | None = None
        self._busy = False
        # Detector training-loss plot (filled live during training).
        self.train_fig = Figure(figsize=(4, 2.2), tight_layout=True)
        self.train_ax = self.train_fig.add_subplot(111)
        self.train_canvas = FigureCanvas(self.train_fig)
        self._plot_training([])
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

        self.ch_cilia = SpinBox(label="Cilia channel", value=1, min=0, max=16)
        self.ch_bb = SpinBox(label="Basal-body channel", value=2, min=0, max=16)
        self.p_low = FloatSpinBox(label="Norm p-low", value=0.0, min=0, max=100)
        self.p_high = FloatSpinBox(label="Norm p-high", value=100.0, min=0, max=100)

        self.in_scale = FloatSlider(label="Input scale ×", value=1.0,
                                    min=0.5, max=6.0, step=0.5)
        self.load_btn = PushButton(text="② Load image")
        self.load_btn.clicked.connect(self._on_load)
        self.run_btn = PushButton(text="③ Run CNN heatmap")
        self.run_btn.clicked.connect(self._on_run_heatmap)
        self.cam_btn = PushButton(text="③′ Run CAM heatmap (no FCN build)")
        self.cam_btn.clicked.connect(self._on_run_cam)
        self.sw_stride = SpinBox(label="Sliding stride (px)", value=32, min=4, max=256)
        self.sw_btn = PushButton(text="③″ Run sliding window (no FCN, slow)")
        self.sw_btn.clicked.connect(self._on_run_sliding)

        self.thr = FloatSlider(label="Peak threshold", value=0.5, min=0.0,
                               max=1.0, step=0.01)
        self.min_dist = SpinBox(label="Min peak dist (cells)", value=2, min=1, max=50)
        self.detect_btn = PushButton(text="④ Detect ROI centres")
        self.detect_btn.clicked.connect(self._on_detect)
        self.detect_status = Label(value="—")

        # Alternative detector: query a CiliAI (datamarkin/ciliai) Mask R-CNN
        # server over HTTP for cilia boxes → centres (reuses "Peak threshold" as
        # the box-score cutoff). Needs the image loaded (step ②); no FCN build.
        self.ciliai_url = LineEdit(label="CiliAI server URL",
                                   value=CILIAI_DEFAULT_URL)
        self.ciliai_btn = PushButton(text="④′ Detect via CiliAI (HTTP)")
        self.ciliai_btn.clicked.connect(self._on_ciliai_detect)

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

        # ── Train a whole-image detector (no transplant) ──────────────────────
        self.run_dir = FileEdit(label="Run output dir (csv/)", mode="d")
        self.det_epochs = SpinBox(label="Detector epochs", value=30, min=1, max=500)
        self.det_sigma = FloatSpinBox(label="Target σ max (cells)", value=1.5,
                                      min=0.3, max=8.0, step=0.1)
        self.det_crop = FloatSpinBox(label="ROI crop factor", value=0.5,
                                     min=0.1, max=1.5, step=0.05)
        self.det_warm = CheckBox(label="Warm-start conv from ROI model", value=True)
        self.det_name = LineEdit(label="Save as", value="fcn_detector.pt")
        self.train_det_btn = PushButton(text="Ⓣ Train whole-image detector")
        self.train_det_btn.clicked.connect(self._on_train_detector)
        self.det_pick = ComboBox(label="Detector (.pt)",
                                 choices=self._discover_detectors())
        self.load_det_btn = PushButton(text="Load detector")
        self.load_det_btn.clicked.connect(self._on_load_detector)
        self.run_det_btn = PushButton(text="③‴ Run trained detector heatmap")
        self.run_det_btn.clicked.connect(self._on_run_detector)
        self.det_status = Label(value="No detector trained/loaded.")

        self._buttons = [self.build_fcn_btn, self.load_btn, self.run_btn,
                         self.cam_btn, self.sw_btn, self.detect_btn,
                         self.ciliai_btn, self.apoc_btn,
                         self.train_det_btn, self.load_det_btn, self.run_det_btn]

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
            self.run_btn, self.cam_btn, self.sw_stride, self.sw_btn,
            self.thr, self.min_dist, self.detect_btn, self.detect_status,
            self.ciliai_url, self.ciliai_btn,
            Label(value="<b>4 · APOC gated by CNN</b>"),
            self.classifier, self.cilia_min, self.cilia_max, self.gate_radius,
            self.apoc_btn, self.apoc_status,
            Label(value="<b>T · Train whole-image detector (no transplant)</b>"),
            self.run_dir, self.det_epochs, self.det_sigma, self.det_crop,
            self.det_warm,
            self.det_name, self.train_det_btn,
            self.det_pick, self.load_det_btn, self.run_det_btn, self.det_status,
        ], labels=True, scrollable=False)   # outer QScrollArea (main) handles scroll

    # ── model discovery ────────────────────────────────────────────────────────
    def _discover_models(self) -> list[str]:
        """ROI-classifier ``.pt`` bundles in models/ (i.e. NOT FCN-detector
        bundles). A detector picked here would crash ``load_bundle`` / the FCN
        build, so they're filtered out."""
        if not os.path.isdir(_MODELS_DIR):
            return []
        out = []
        for f in sorted(os.listdir(_MODELS_DIR)):
            if not f.endswith(".pt"):
                continue
            try:
                pl = torch.load(os.path.join(_MODELS_DIR, f),
                                map_location="cpu", weights_only=False)
                if isinstance(pl, dict) and pl.get("kind") == "fcn_detector":
                    continue                                  # skip detectors
            except Exception:                                 # noqa: BLE001
                continue                                      # unreadable → skip
            out.append(f)
        return out

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
            # Match the pipeline that made the CNN's training crops (isotropic).
            use_mip=False, make_isotropic=True,
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
            return cached_load(p)

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
            self.state["peak_offset"] = self.fcn_stride / 2.0   # 'same'-conv grid
            self._show_heatmap(heatmap, in_hw)
            self._set_busy(False)
            show_info(f"Heatmap {heatmap.shape} · max P={heatmap.max():.2f}")

        def _err(e):
            self._set_busy(False)
            show_warning(f"Heatmap failed: {e}")

        w = _work(); w.returned.connect(_done); w.errored.connect(_err); w.start()

    def _on_run_cam(self):
        """CAM heatmap straight off the trained classifier — no FCN build needed.
        Loads the selected .pt, runs its conv stack, applies the head weights as
        1×1 convs (see ``cam_heatmap``). Same downstream detect/APOC steps."""
        if self._busy:
            return
        if "raw" not in self.state:
            show_warning("Load an image first (step ②).")
            return
        name = self.model_combo.value
        if not name:
            show_warning("Pick a trained .pt model first.")
            return
        self._set_busy(True)
        show_info("Running CAM heatmap (no FCN build) …")
        scale = float(self.in_scale.value)
        rgb = build_fcn_input(self.state["raw"], self.ch_cilia.value,
                              self.ch_bb.value, scale)
        path = os.path.join(_MODELS_DIR, str(name))

        @thread_worker
        def _work():
            _select_device(self.gpu_combo.value)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, meta = load_bundle(path)
            stride = model_stride(model)
            size = int(meta.get("size", DEFAULT_SIZE.get(meta.get("arch", "big"), 64)))
            return cam_heatmap(model, rgb, device), rgb.shape[1:], stride, size

        def _done(res):
            heatmap, in_hw, stride, size = res
            self.fcn_stride, self.model_size = stride, size
            self.state["heatmap"] = heatmap
            self.state["fcn_in_scale"] = scale
            self.state["peak_offset"] = stride / 2.0            # 'same'-conv grid
            self._show_heatmap(heatmap, in_hw)
            self._set_busy(False)
            show_info(f"CAM heatmap {heatmap.shape} · max P={heatmap.max():.2f} "
                      f"(stride {stride}).")

        def _err(e):
            self._set_busy(False)
            show_warning(f"CAM failed: {e}")

        w = _work(); w.returned.connect(_done); w.errored.connect(_err); w.start()

    def _on_run_sliding(self):
        """Brute-force sliding-window detector — slide the original classifier at
        the chosen stride. No FCN/CAM, no architecture change; slow but simple.
        Feeds the same detect/APOC steps via the same peak geometry."""
        if self._busy:
            return
        if "raw" not in self.state:
            show_warning("Load an image first (step ②).")
            return
        name = self.model_combo.value
        if not name:
            show_warning("Pick a trained .pt model first.")
            return
        self._set_busy(True)
        stride = int(self.sw_stride.value)
        show_info(f"Sliding window (stride {stride}px) — this is the slow path …")
        scale = float(self.in_scale.value)
        rgb = build_fcn_input(self.state["raw"], self.ch_cilia.value,
                              self.ch_bb.value, scale)
        path = os.path.join(_MODELS_DIR, str(name))

        @thread_worker
        def _work():
            _select_device(self.gpu_combo.value)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, meta = load_bundle(path)
            size = int(meta.get("size", DEFAULT_SIZE.get(meta.get("arch", "big"), 64)))
            hm = sliding_window_heatmap(model, rgb, size, stride, device)
            return hm, rgb.shape[1:], stride, size

        def _done(res):
            heatmap, in_hw, sd, size = res
            # Sliding-window geometry: grid spacing == window stride.
            self.fcn_stride, self.model_size = sd, size
            self.state["heatmap"] = heatmap
            self.state["fcn_in_scale"] = scale
            # Window top-left at r*stride → centre at size/2.
            self.state["peak_offset"] = size / 2.0
            self._show_heatmap(heatmap, in_hw)
            self._set_busy(False)
            show_info(f"Sliding-window heatmap {heatmap.shape} · "
                      f"max P={heatmap.max():.2f} (stride {sd}px).")

        def _err(e):
            self._set_busy(False)
            show_warning(f"Sliding window failed: {e}")

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
        # Offset depends on which detector produced the heatmap (FCN/CAM →
        # stride/2; sliding window → size/2); set when the heatmap was computed.
        offset = float(self.state.get("peak_offset", self.fcn_stride / 2.0))
        yx = heatmap_peaks_to_image_xy(
            rc, self.fcn_stride, float(self.state.get("fcn_in_scale", 1.0)), offset)
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

    # ── CiliAI detector (HTTP) ────────────────────────────────────────────────
    def _on_ciliai_detect(self):
        if self._busy:
            return
        if "raw" not in self.state:
            show_warning("Load an image first (step ②).")
            return
        url = str(self.ciliai_url.value or "").strip()
        if not url:
            show_warning("Set the CiliAI server URL (e.g. http://localhost:5007).")
            return
        thr = float(self.thr.value)
        # Build the whole-image RGB at native scale so returned boxes are already
        # in image pixels (no rescale needed to map back to centres_yx).
        rgb = build_fcn_input(self.state["raw"], self.ch_cilia.value,
                              self.ch_bb.value, 1.0)
        self._set_busy(True)
        show_info("Querying CiliAI detector — see console …")

        @thread_worker
        def _work():
            return ciliai_detect_centres(rgb, url, thr)

        def _done(res):
            yx, scores = res
            self.state["centres_yx"] = yx
            self.state["fcn_in_scale"] = 1.0          # native px; no heatmap step
            self._show_centres(yx, scores)
            self.detect_status.value = f"CiliAI: {len(yx)} centres ≥ {thr:.2f}"
            self._set_busy(False)
            show_info(f"CiliAI detected {len(yx)} cilia. Run ⑤ to segment.")

        def _err(e):
            self._set_busy(False)
            show_warning(f"CiliAI detect failed: {e}")

        w = _work(); w.returned.connect(_done); w.errored.connect(_err); w.start()

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

    # ── train / load / run whole-image detector ──────────────────────────────
    def _plot_training(self, history: list) -> None:
        """Draw the detector's per-epoch BCE loss curve in the bottom dock."""
        self.train_ax.clear()
        if history:
            self.train_ax.plot(range(1, len(history) + 1), history, "-o",
                               color="#0C5DA5", ms=3, lw=1.4)
            self.train_ax.set_title(f"Detector training — final BCE "
                                    f"{history[-1]:.4f}", fontsize=9)
        else:
            self.train_ax.set_title("Detector training (no run yet)", fontsize=9)
        self.train_ax.set_xlabel("epoch", fontsize=8)
        self.train_ax.set_ylabel("BCE loss", fontsize=8)
        self.train_ax.tick_params(labelsize=7)
        self.train_canvas.draw_idle()

    def _discover_detectors(self) -> list[str]:
        """``.pt`` files in models/ that are FCN-detector bundles."""
        out = []
        if os.path.isdir(_MODELS_DIR):
            for f in sorted(os.listdir(_MODELS_DIR)):
                if not f.endswith(".pt"):
                    continue
                try:
                    pl = torch.load(os.path.join(_MODELS_DIR, f),
                                    map_location="cpu", weights_only=False)
                    if isinstance(pl, dict) and pl.get("kind") == "fcn_detector":
                        out.append(f)
                except Exception:                             # noqa: BLE001
                    pass
        return out

    def _on_train_detector(self):
        if self._busy:
            return
        run_dir = str(self.run_dir.value or "")
        folder = str(self.folder.value or "")
        if not os.path.isdir(run_dir):
            show_warning("Pick the run output dir (the lc-analysis-* folder with csv/).")
            return
        # Raw volumes are located from the run's own file list
        # (run_parameters.json); the Image folder is only a fallback, so it is
        # optional. Bail only if neither source can supply the originals.
        if not run_file_map(run_dir) and not os.path.isdir(folder):
            show_warning("This run has no usable file list in run_parameters.json "
                         "and no Image folder is set — can't locate the raw images.")
            return
        self._set_busy(True)
        show_info("Reconstructing targets + training detector — see console …")
        load_base = {k: v for k, v in self._params().items() if k != "ims_path"}
        ptrain = dict(stride=16, sigma=float(self.det_sigma.value),
                      crop_factor=float(self.det_crop.value),
                      ch_cilia=self.ch_cilia.value, ch_bb=self.ch_bb.value,
                      load=load_base)
        epochs = int(self.det_epochs.value)
        warm_name = str(self.model_combo.value) if self.det_warm.value else ""
        save_name = str(self.det_name.value or "fcn_detector.pt")
        if not save_name.endswith(".pt"):
            save_name += ".pt"

        @thread_worker
        def _work():
            _select_device(self.gpu_combo.value)
            print(f"[detector] reconstructing targets from {run_dir} …")
            samples = reconstruct_targets(run_dir, folder or None, ptrain)
            n_pos = sum(s[3] for s in samples)
            print(f"[detector] {len(samples)} images · {n_pos} positives total")
            warm = None
            if warm_name.endswith(".pt"):
                # Warm-start only from a ROINetBig ROI classifier; a wrong pick
                # (e.g. a detector or tiny net) must not abort training.
                try:
                    _w, _ = load_bundle(os.path.join(_MODELS_DIR, warm_name))
                    if isinstance(_w, ROINetBig):
                        warm = _w
                    else:
                        print(f"[detector] warm-start skipped — {warm_name} is "
                              f"not a 'big' ROI classifier")
                except Exception as _exc:                     # noqa: BLE001
                    print(f"[detector] warm-start skipped — could not load "
                          f"{warm_name}: {_exc}")
            model, hist = train_fcn_detector(
                samples, epochs=epochs, warm=warm,
                progress=lambda f, m: print(f"[detector] {m}"))
            out = save_detector(os.path.join(_MODELS_DIR, save_name), model,
                                {"run_dir": run_dir, "epochs": epochs,
                                 "sigma": ptrain["sigma"], "n_images": len(samples),
                                 "n_positives": int(n_pos), "final_loss": hist[-1],
                                 "history": hist})
            return model, out, len(samples), int(n_pos), hist

        def _done(res):
            model, out, n_img, n_pos, hist = res
            self.detector = model
            self.fcn_stride, self.model_size = model.stride, 96
            self.det_pick.choices = self._discover_detectors()
            self.det_pick.value = os.path.basename(out)
            self._plot_training(hist)
            self.det_status.value = (f"Trained on {n_img} imgs / {n_pos} cilia · "
                                     f"BCE {hist[-1]:.4f} · saved {os.path.basename(out)}")
            self._set_busy(False)
            show_info(f"Detector trained ({n_img} images, {n_pos} cilia). Saved {out}.")

        def _err(e):
            self._set_busy(False); show_warning(f"Detector training failed: {e}")

        w = _work(); w.returned.connect(_done); w.errored.connect(_err); w.start()

    def _on_load_detector(self):
        name = self.det_pick.value
        if not name:
            show_warning("No detector .pt to load (train one first).")
            return
        try:
            self.detector, meta = load_detector(os.path.join(_MODELS_DIR, str(name)))
            self.fcn_stride, self.model_size = self.detector.stride, 96
        except Exception as exc:                              # noqa: BLE001
            show_warning(f"Load failed: {exc}")
            return
        if meta.get("history"):                               # show its loss curve
            self._plot_training(meta["history"])
        self.det_status.value = f"Loaded {name} · {meta.get('n_images', '?')} imgs"
        show_info(f"Detector {name} loaded.")

    def _on_run_detector(self):
        if self._busy:
            return
        if self.detector is None:
            show_warning("Train or load a detector first.")
            return
        if "raw" not in self.state:
            show_warning("Load an image first (step ②).")
            return
        self._set_busy(True)
        show_info("Running trained detector heatmap …")
        scale = float(self.in_scale.value)
        rgb = build_fcn_input(self.state["raw"], self.ch_cilia.value,
                              self.ch_bb.value, scale)

        @thread_worker
        def _work():
            _select_device(self.gpu_combo.value)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return detector_heatmap(self.detector, rgb, device), rgb.shape[1:]

        def _done(res):
            heatmap, in_hw = res
            self.fcn_stride, self.model_size = self.detector.stride, 96
            self.state["heatmap"] = heatmap
            self.state["fcn_in_scale"] = scale
            self.state["peak_offset"] = self.detector.stride / 2.0   # 'same'-conv grid
            self._show_heatmap(heatmap, in_hw)
            self._set_busy(False)
            show_info(f"Detector heatmap {heatmap.shape} · max P={heatmap.max():.2f}")

        def _err(e):
            self._set_busy(False); show_warning(f"Detector run failed: {e}")

        w = _work(); w.returned.connect(_done); w.errored.connect(_err); w.start()


def main():
    from qtpy.QtWidgets import QScrollArea
    viewer = napari.Viewer(title="LimonCELLo — Fast CNN cilia detector")
    app = LimoncelloCNNApp(viewer)
    # Wrap the panel in a real QScrollArea — magicgui's own ``scrollable`` often
    # doesn't take when docked, so the long control list gets clipped otherwise.
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(app.widget.native)
    viewer.window.add_dock_widget(scroll, area="right", name="Fast CNN detector")
    # Detector training-loss plot in its own bottom dock.
    viewer.window.add_dock_widget(app.train_canvas, area="bottom",
                                  name="Detector training")
    napari.run()


if __name__ == "__main__":
    main()
