import os
os.environ["PYOPENCL_NO_CACHE"] = "1"

import ast
import json
import re
import sys
import time
import glob
import threading
import warnings
from io import BytesIO

# Silence noisy GPU/OpenCL warnings that otherwise flood the in-app log stream.
warnings.filterwarnings("ignore", message=r".*PyOpenCL compiler caching failed.*")
warnings.filterwarnings("ignore", message=r".*Cannot deduce a name.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.spatial import KDTree
from scipy.ndimage import center_of_mass
from skimage.measure import regionprops_table

from limoncello.analysis.pipeline import run_pipeline3
from limoncello.utils.app_helpers import find_latest_run_dir as _find_latest_run_dir

try:
    import plotly.express as _px
    import plotly.figure_factory as _ff
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Limoncello 🍋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
_CLASS_PALETTE = {"axon": "#2ecc71", "soma": "#e74c3c", "ambiguous": "#f39c12"}

_PF_DEFAULTS = {
    "pf_axon_thr":     2.5,
    "pf_soma_thr":     1.0,
    "pf_nn_min_dist":  0.0,
    "pf_nn_radius":    50.0,
    "pf_nn_max_count": 1000,
}

# ─────────────────────────────────────────────────────────────────────────────
# LOG-CAPTURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\r")

# Lines matching this are GPU/OpenCL plumbing noise — hidden from the app log
# stream so users see only meaningful pipeline messages.
_LOG_NOISE_RE = re.compile(
    r"PyOpenCL compiler caching failed"
    r"|\[begin exception\]|\[end exception\]"
    r"|Traceback \(most recent call last\)"
    r"|create_built_program_from_source_cached"
    r"|pyopencl_defeat_cache"
    r"|%b requires a bytes-like object"
    r"|^\s*src = src \+ b"
    r"|^\s*lambda: create_built_program"
    r"|^\s*File \".*[\\/](pyopencl|pyclesperanto)"
    r"|UserWarning"
    r"|^\s*warnings\.warn"
    r"|^size:\s*\d+"
    r"|cl_amd_printf"
)


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


class _LogCapture:
    """Tees writes to the original stream and appends stripped lines to a list."""

    def __init__(self, log_list: list, original):
        self._list = log_list
        self._orig = original
        self._buf = ""

    def write(self, text: str):
        try:
            self._orig.write(text)
        except Exception:
            pass
        text = _strip_ansi(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if _LOG_NOISE_RE.search(line):
                continue                       # drop GPU/OpenCL plumbing noise
            if not line.strip() and self._list and not self._list[-1].strip():
                continue                       # collapse consecutive blank lines
            self._list.append(line)

    def flush(self):
        if self._buf:
            self._list.append(_strip_ansi(self._buf))
            self._buf = ""
        try:
            self._orig.flush()
        except Exception:
            pass

    def isatty(self):
        return False


def _run_pipeline_thread(params: dict, log_list: list, result: dict):
    """Runs in a background thread — must NOT access st.session_state."""
    _orig_out, _orig_err = sys.stdout, sys.stderr

    def _progress_cb(file_idx: int, n_files: int, filename: str):
        result["n_files"] = n_files
        result["file_idx"] = file_idx
        result["current_file"] = filename
        result["progress"] = file_idx / n_files if n_files > 0 else 0.0

    try:
        sys.stdout = _LogCapture(log_list, _orig_out)
        sys.stderr = _LogCapture(log_list, _orig_err)
        run_pipeline3(**params, progress_callback=_progress_cb)
        result["status"] = "completed"
    except Exception as exc:
        import traceback
        result["status"] = "error"
        result["error"] = str(exc)
        for line in traceback.format_exc().splitlines():
            log_list.append(line)
    finally:
        sys.stdout = _orig_out
        sys.stderr = _orig_err
        result["elapsed"] = time.time() - result["start_time"]
        result["running"] = False


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_excel(path: str, mtime: float, sheet: str) -> pd.DataFrame:
    _ = mtime  # cache-bust key — forces reload when file changes on disk
    return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")


@st.cache_data(ttl=300, show_spinner=False)
def _get_first_ims_info(input_path: str) -> dict | None:
    """Return shape metadata of the first .ims file without loading pixel data."""
    if not input_path or not os.path.exists(input_path):
        return None
    ims_files = sorted(f for f in os.listdir(input_path) if f.endswith(".ims"))
    if not ims_files:
        return None
    try:
        from imaris_ims_file_reader.ims import ims as _ImsReader
        img = _ImsReader(os.path.join(input_path, ims_files[0]))
        _, c, z, y, x = img.shape
        return {"filename": ims_files[0], "n_channels": c, "z": z, "y": y, "x": x}
    except Exception:
        return None


@st.cache_data(show_spinner="Loading channel preview…")
def _load_channel_preview(input_path: str, filename: str) -> np.ndarray | None:
    """Load a normalised mid-Z slice of every channel. Returns float32 (C, Y, X)."""
    try:
        from imaris_ims_file_reader.ims import ims as _ImsReader
        img = _ImsReader(os.path.join(input_path, filename))
        _, c, z, *_ = img.shape
        mid_z = z // 2
        slices = []
        for ch in range(c):
            sl = np.array(img[0, ch, mid_z]).astype(np.float32)
            p_lo = np.percentile(sl, 0)
            p_hi = np.percentile(sl, 99.9)
            sl = (sl - p_lo) / (p_hi - p_lo + 1e-8)  # no hard clip; imshow clamps
            slices.append(sl)
        return np.stack(slices)
    except Exception:
        return None


_DIALOG_ROLES: list[tuple[str, str, int]] = [
    ("Cilia",         "ch_cilia",    0),
    ("Neurites",      "ch_neurites", 1),
    ("Basal bodies",  "ch_bb",       2),
    ("Nuclei (DAPI)", "ch_nuclei",   3),
]


@st.dialog("🔭 Channel Viewer & Assignment", width="large")
def _channel_assignment_dialog(input_path: str, filename: str, n_channels: int):
    """Interactive channel browser with arrow navigation and role assignment."""
    # ── Navigation state ──────────────────────────────────────────────────────
    if "dialog_ch_idx" not in st.session_state:
        st.session_state.dialog_ch_idx = 0
    ch = max(0, min(int(st.session_state.dialog_ch_idx), n_channels - 1))

    previews = _load_channel_preview(input_path, filename)

    # Build channel → role label map from current sidebar state
    _ch_role: dict[int, str] = {}
    for _lbl, _sk, _df in _DIALOG_ROLES:
        _ch_role[int(st.session_state.get(_sk, _df))] = _lbl

    # ── Thumbnail strip ───────────────────────────────────────────────────────
    st.caption(f"Mid-Z slice — **{filename}**   |   click a thumbnail or use arrows to navigate")
    _tcols = st.columns(n_channels)
    for i in range(n_channels):
        with _tcols[i]:
            if previews is not None:
                _tfig, _tax = plt.subplots(figsize=(2, 2))
                _tax.imshow(previews[i], cmap="gray")
                _tax.axis("off")
                if i == ch:
                    for _sp in _tax.spines.values():
                        _sp.set_visible(True)
                        _sp.set_edgecolor("#F5A623")
                        _sp.set_linewidth(3)
                plt.tight_layout(pad=0.1)
                st.pyplot(_tfig, width="stretch")
                plt.close(_tfig)
            _rlbl = _ch_role.get(i, "—")
            st.caption(f"Ch {i}" + (f" · {_rlbl}" if _rlbl != "—" else ""))
            # Button click triggers a rerun WITHOUT st.rerun() so dialog stays open
            if st.button("Select", key=f"_dsel_{i}", width="stretch",
                         type="primary" if i == ch else "secondary"):
                st.session_state.dialog_ch_idx = i

    st.divider()

    # ── Large view + arrow navigation ─────────────────────────────────────────
    _nl, _nc, _nr = st.columns([1, 6, 1])
    with _nl:
        if st.button("◀", key="_dprev", disabled=(ch == 0), width="stretch"):
            st.session_state.dialog_ch_idx = ch - 1
    with _nc:
        _current_role = _ch_role.get(ch, "unassigned")
        st.markdown(f"#### Channel {ch} — *{_current_role}*")
    with _nr:
        if st.button("▶", key="_dnext", disabled=(ch == n_channels - 1), width="stretch"):
            st.session_state.dialog_ch_idx = ch + 1

    if previews is not None:
        _mfig, _max = plt.subplots(figsize=(9, 5))
        _max.imshow(previews[ch], cmap="gray", aspect="equal")
        _max.axis("off")
        plt.tight_layout(pad=0.1)
        st.pyplot(_mfig, width="stretch")
        plt.close(_mfig)
    else:
        st.error("Could not load image data.")

    st.divider()

    # ── Assignment row ────────────────────────────────────────────────────────
    st.subheader("Assign structures to channels")
    _acols = st.columns(len(_DIALOG_ROLES))
    for _col, (_lbl, _sk, _df) in zip(_acols, _DIALOG_ROLES):
        with _col:
            st.selectbox(
                _lbl,
                options=list(range(n_channels)),
                index=int(st.session_state.get(f"_dassign_{_sk}",
                                                st.session_state.get(_sk, _df))),
                key=f"_dassign_{_sk}",
                format_func=lambda x: f"Channel {x}",
            )

    if st.button("✅ Apply Assignments", type="primary", width="stretch"):
        st.session_state["_pending_channels"] = {
            _sk: int(st.session_state[f"_dassign_{_sk}"])
            for _, _sk, _ in _DIALOG_ROLES
        }
        st.rerun()  # closes the dialog and triggers early-apply block


@st.cache_data(show_spinner=False)
def _load_mips(mip_dir: str, stem: str, mtime: float) -> dict | None:
    _ = mtime  # cache-bust key
    result = {}
    for key, suffix in [
        ("neurite", "neurite_mip"),
        ("cilia",   "cilia_mip"),
        ("nuclei",  "nuclei_mip"),
        ("ratio",   "ratio_mid"),
    ]:
        path = os.path.join(mip_dir, f"{stem}_{suffix}.npy")
        if not os.path.exists(path):
            return None
        result[key] = np.load(path)
    # Optional MIPs — present only in runs produced after the label-MIP update
    for key, suffix, cast in [
        ("neurite_mask",   "neurite_mask_mip",   lambda a: a.astype(bool)),
        ("bb",             "bb_mip",             None),
        ("cilia_labels",   "cilia_labels_mip",   None),
        ("nuclei_labels",  "nuclei_labels_mip",  None),
        ("bb_labels",      "bb_labels_mip",      None),
        ("neurite_labels", "neurite_labels_mip", None),
    ]:
        _p = os.path.join(mip_dir, f"{stem}_{suffix}.npy")
        if os.path.exists(_p):
            arr = np.load(_p)
            result[key] = cast(arr) if cast else arr
    return result


def _fig_to_png(fig: plt.Figure) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def _fig_to_pdf(fig: plt.Figure) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# QC PDF REPORT
# ─────────────────────────────────────────────────────────────────────────────
def _label_stats(label_mip):
    """For a 2-D label MIP return (n_objects, mean_area_px, centroids_yx, ids)."""
    if label_mip is None:
        return 0, float("nan"), np.empty((0, 2)), np.empty((0,), dtype=int)
    lab = np.asarray(label_mip)
    ids = np.unique(lab)
    ids = ids[ids != 0]
    if ids.size == 0:
        return 0, float("nan"), np.empty((0, 2)), np.empty((0,), dtype=int)
    areas = np.bincount(lab.ravel())[ids]
    cents = np.atleast_2d(np.array(center_of_mass(lab > 0, labels=lab, index=ids)))
    return int(ids.size), float(areas.mean()), cents, ids.astype(int)  # cents = (y, x)


def _qc_show_gray(ax, img, title):
    if img is None:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", color="#999")
    else:
        _lo, _hi = np.percentile(img, (1, 99.5))
        ax.imshow(img, cmap="gray", vmin=_lo, vmax=_hi + 1e-8)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _qc_show_labels(ax, lab, title, mask=None):
    if lab is None and mask is None:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", color="#999")
    elif lab is not None:
        _disp = np.ma.masked_where(np.asarray(lab) == 0, np.asarray(lab))
        ax.imshow(_disp, cmap="nipy_spectral", interpolation="nearest")
    else:
        ax.imshow(np.asarray(mask).astype(float), cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _render_cilia_roi_pages(pdf, label, mips, kept_ids=None, props3d=None,
                            roi_half=26, cols=3, rows=3, max_rois=360,
                            max_window=70):
    """Large close-up ROI of every detected cilium drawn on the raw cilia channel
    with its outline, plus its **nearest basal body** (the spatially-associated
    one) outlined and marked, with the cilium→BB distance annotated. Each tile is
    labelled with region properties — true 3-D volume/length when available
    (``props3d``), otherwise 2-D projected area/length. Cilia are split into kept
    and non-kept groups."""
    chan = mips.get("cilia")
    labmip = mips.get("cilia_labels")
    if chan is None or labmip is None:
        return
    chan = np.asarray(chan, dtype=float)
    labmip = np.asarray(labmip)
    if chan.shape != labmip.shape:
        return
    bb_labmip = mips.get("bb_labels")
    if bb_labmip is not None:
        bb_labmip = np.asarray(bb_labmip)
        if bb_labmip.shape != labmip.shape:
            bb_labmip = None
    props3d = props3d or {}

    ids = np.unique(labmip)
    ids = ids[ids != 0]
    if ids.size == 0:
        return

    # Per-cilium 2-D props + centroids
    try:
        rp = regionprops_table(
            labmip.astype(np.int32),
            properties=("label", "area", "major_axis_length",
                        "eccentricity", "centroid"),
        )
    except Exception:
        rp = {"label": ids, "area": np.full(len(ids), np.nan),
              "major_axis_length": np.full(len(ids), np.nan),
              "eccentricity": np.full(len(ids), np.nan),
              "centroid-0": np.zeros(len(ids)), "centroid-1": np.zeros(len(ids))}
    prop_of = {}
    for _i, _lab in enumerate(rp["label"]):
        prop_of[int(_lab)] = {
            "area": float(rp["area"][_i]),
            "length": float(rp["major_axis_length"][_i]),
            "ecc": float(rp["eccentricity"][_i]),
            "cent": (float(rp["centroid-0"][_i]), float(rp["centroid-1"][_i])),
        }

    # Basal-body centroids (for nearest-BB association)
    bb_cents = np.empty((0, 2))
    if bb_labmip is not None:
        bb_ids = np.unique(bb_labmip)
        bb_ids = bb_ids[bb_ids != 0]
        if bb_ids.size:
            bb_cents = np.atleast_2d(np.array(
                center_of_mass(bb_labmip > 0, labels=bb_labmip, index=bb_ids)))

    H, W = labmip.shape
    lo, hi = np.percentile(chan, (1, 99.5))
    hi = hi + 1e-8
    per_page = cols * rows

    if kept_ids is None:
        groups = [("Detected cilia", [int(i) for i in ids], "#00FF88")]
    else:
        kept_set = {int(k) for k in kept_ids}
        kept = [int(i) for i in ids if int(i) in kept_set]
        nonkept = [int(i) for i in ids if int(i) not in kept_set]
        groups = [("Kept cilia", kept, "#00FF88"),
                  ("Non-kept cilia", nonkept, "#FF5555")]

    def _metric_str(cid):
        if cid in props3d:
            _v, _l = props3d[cid]
            _vs = f"vol {_v:.2f} µm³" if np.isfinite(_v) else "vol —"
            _ls = f"len {_l:.2f} µm" if np.isfinite(_l) else "len —"
            return f"{_vs} · {_ls}"
        p = prop_of.get(cid, {})
        _a = f"area {p.get('area', float('nan')):.0f} px²"
        _l = f"len {p.get('length', float('nan')):.1f} px"
        _e = f"ecc {p['ecc']:.2f}" if np.isfinite(p.get("ecc", np.nan)) else "ecc —"
        return f"{_a} · {_l} · {_e}"

    def _draw_roi(ax, cid, accent):
        cy, cx = prop_of.get(cid, {}).get("cent", (H / 2, W / 2))
        # nearest basal body
        bb_d, bb_yx = None, None
        if bb_cents.size:
            _d = np.hypot(bb_cents[:, 0] - cy, bb_cents[:, 1] - cx)
            _j = int(np.argmin(_d))
            bb_d, bb_yx = float(_d[_j]), bb_cents[_j]
        # window large enough to include the nearest BB (capped)
        half = roi_half
        if bb_d is not None:
            half = int(min(max_window, max(roi_half, bb_d + 10)))
        y0 = max(0, int(round(cy)) - half)
        x0 = max(0, int(round(cx)) - half)
        y1 = min(H, y0 + 2 * half)
        x1 = min(W, x0 + 2 * half)
        ax.imshow(chan[y0:y1, x0:x1], cmap="gray", vmin=lo, vmax=hi)
        _cil = labmip[y0:y1, x0:x1] == cid
        if _cil.any():
            ax.contour(_cil.astype(float), levels=[0.5], colors=accent, linewidths=1.3)
        if bb_labmip is not None:
            _bb = bb_labmip[y0:y1, x0:x1] > 0
            if _bb.any():
                ax.contour(_bb.astype(float), levels=[0.5],
                           colors="#FF8800", linewidths=1.3)
            # mark the nearest BB centroid so tiny basal bodies stay visible
            if bb_yx is not None and y0 <= bb_yx[0] < y1 and x0 <= bb_yx[1] < x1:
                ax.plot(bb_yx[1] - x0, bb_yx[0] - y0, marker="+",
                        color="#FF8800", markersize=9, markeredgewidth=1.5)
        _bb_str = f"BB {bb_d:.0f} px" if bb_d is not None else "no BB"
        ax.set_title(f"#{cid}   {_metric_str(cid)}\nnearest {_bb_str}",
                     fontsize=8, pad=3)

    for gname, gids, accent in groups:
        if not gids:
            continue
        n_total = len(gids)
        n_show = min(n_total, max_rois)
        truncated = n_total > max_rois
        for start in range(0, n_show, per_page):
            page = gids[start:min(start + per_page, n_show)]
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.5))
            axes = np.atleast_2d(axes)
            fig.suptitle(
                f"{gname} close-ups — {label}   "
                f"(#{start + 1}–{start + len(page)} of {n_total})\n"
                "cilium outline coloured · nearest basal body orange (＋ = its centre)",
                fontsize=13, fontweight="bold",
            )
            for slot in range(per_page):
                ax = axes.flat[slot]
                ax.axis("off")
                if slot < len(page):
                    _draw_roi(ax, page[slot], accent)
            if truncated and start + per_page >= n_show:
                fig.text(0.5, 0.005,
                         f"… showing first {max_rois} of {n_total} {gname.lower()}",
                         ha="center", fontsize=9, color="#c0392b")
            fig.tight_layout(rect=[0, 0.01, 1, 0.94])
            pdf.savefig(fig)
            plt.close(fig)


def _flag_deviant(records, z_thresh=3.5):
    """Flag samples whose metrics are robust outliers (modified z-score on
    median/MAD). Returns {fname: [reasons]}. Needs >=4 samples to judge."""
    flags = {r["fname"]: [] for r in records}
    if len(records) < 4:
        return flags
    metrics = [
        ("n_cil_det",  "detected cilia"),
        ("n_cil_kept", "kept cilia"),
        ("area_cil",   "cilia area"),
        ("n_bb_det",   "detected BB"),
        ("mean_lr",    "mean log_ratio"),
    ]
    for key, nice in metrics:
        vals = np.array([r.get(key, np.nan) for r in records], dtype=float)
        fin = np.isfinite(vals)
        if fin.sum() < 4:
            continue
        med = np.median(vals[fin])
        mad = np.median(np.abs(vals[fin] - med))
        if mad > 0:
            score, thr = 0.6745 * (vals - med) / mad, z_thresh
        else:
            sd = vals[fin].std()
            if sd == 0:
                continue
            score, thr = (vals - vals[fin].mean()) / sd, 2.5
        for i, r in enumerate(records):
            if fin[i] and abs(score[i]) > thr:
                flags[r["fname"]].append(f"{nice} {'high' if vals[i] > med else 'low'}")
    return flags


def _render_qc_overview(pdf, records, flags):
    """First page: aggregate metrics, per-sample kept-cilia bar chart, and the
    list of flagged deviant samples."""
    n = len(records)
    labels = [r["label"] for r in records]
    kept = np.array([r["n_cil_kept"] for r in records], dtype=float)
    det = np.array([r["n_cil_det"] for r in records], dtype=float)
    lr = np.array([r["mean_lr"] for r in records], dtype=float)
    lr_fin = lr[np.isfinite(lr)]
    glob_cls = {}
    for r in records:
        for k, v in r["cls"].items():
            glob_cls[k] = glob_cls.get(k, 0) + v
    flagged = [r for r in records if flags.get(r["fname"])]

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("QC Overview — all samples", fontsize=17, fontweight="bold")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.18)

    # Aggregate metrics (top-left)
    ax_txt = fig.add_subplot(gs[0, 0]); ax_txt.axis("off")
    _kept_ms = f"{kept.mean():.1f} ± {kept.std():.1f}" if n else "—"
    _det_ms = f"{det[np.isfinite(det)].mean():.1f}" if np.isfinite(det).any() else "—"
    _lr_ms = f"{lr_fin.mean():.2f} ± {lr_fin.std():.2f}" if lr_fin.size else "—"
    _cls_txt = "  ".join(f"{k}={glob_cls.get(k, 0)}"
                         for k in ("axon", "soma", "ambiguous"))
    ax_txt.text(
        0.0, 1.0,
        f"Samples              : {n}\n"
        f"Flagged samples      : {len(flagged)}\n"
        f"{'─' * 34}\n"
        f"Detected cilia (tot) : {int(np.nansum(det))}   (mean {_det_ms}/sample)\n"
        f"Kept cilia (tot)     : {int(kept.sum())}   (mean {_kept_ms}/sample)\n"
        f"Detected BB (tot)    : {sum(r['n_bb_det'] for r in records)}\n"
        f"Kept BB (tot)        : {sum(r['n_bb_kept'] for r in records)}\n"
        f"{'─' * 34}\n"
        f"Mean log_ratio       : {_lr_ms}\n"
        f"Class distribution   : {_cls_txt}",
        va="top", ha="left", family="monospace", fontsize=11,
        transform=ax_txt.transAxes,
    )

    # Per-sample kept-cilia bar chart (top-right), flagged bars in red
    ax_bar = fig.add_subplot(gs[0, 1])
    _colors = ["#e74c3c" if flags.get(r["fname"]) else "#34495e" for r in records]
    ax_bar.bar(range(n), kept, color=_colors)
    if n:
        _med = float(np.median(kept))
        ax_bar.axhline(_med, ls="--", lw=1, color="gray", label=f"median={_med:.0f}")
        ax_bar.legend(fontsize=8, loc="upper right")
    ax_bar.set_xticks(range(n))
    ax_bar.set_xticklabels(labels, rotation=90, fontsize=7)
    ax_bar.set_ylabel("Kept cilia")
    ax_bar.set_title("Kept cilia per sample  (red = flagged)", fontsize=10)
    ax_bar.spines[["top", "right"]].set_visible(False)

    # Flagged-sample list (bottom, full width)
    ax_flag = fig.add_subplot(gs[1, :]); ax_flag.axis("off")
    if n < 4:
        ax_flag.text(0.0, 1.0,
                     "Too few samples (<4) to flag statistical outliers.",
                     va="top", ha="left", color="#7f8c8d", fontsize=12,
                     transform=ax_flag.transAxes)
    elif flagged:
        _lines = ["⚠  Deviant samples flagged (robust modified z-score > 3.5):", ""]
        for r in records:
            fr = flags.get(r["fname"], [])
            if fr:
                _lines.append(f"   • {r['label']:<14s}  →  " + ", ".join(fr))
        ax_flag.text(0.0, 1.0, "\n".join(_lines),
                     va="top", ha="left", color="#c0392b",
                     family="monospace", fontsize=12,
                     transform=ax_flag.transAxes)
    else:
        ax_flag.text(0.0, 1.0,
                     "✓  No deviant samples detected — all metrics within robust range.",
                     va="top", ha="left", color="#27ae60", fontsize=13,
                     transform=ax_flag.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def _build_qc_report_pdf(mip_dir, files, df_kept, df_bb, x_col):
    """Build a multi-page QC PDF: overview/flags page, one page per sample,
    and a detailed summary table. Returns bytes."""
    def _sub(df, fname):
        if df is None or df.empty or "filename" not in df.columns:
            return pd.DataFrame()
        return df[df["filename"] == fname].copy()

    def _label_for(fname):
        sub = _sub(df_kept, fname)
        if sub.empty:
            sub = _sub(df_bb, fname)
        if not sub.empty and x_col in sub.columns:
            return str(sub[x_col].iloc[0])
        return os.path.splitext(os.path.basename(fname))[0]

    def _stats_for(fname):
        stem = os.path.splitext(os.path.basename(fname))[0]
        probe = os.path.join(mip_dir, f"{stem}_neurite_mip.npy")
        mips = (
            _load_mips(mip_dir, stem, os.path.getmtime(probe))
            if os.path.exists(probe) else None
        ) or {}
        fk = _sub(df_kept, fname)
        bb_sub = _sub(df_bb, fname)
        n_cil_det, area_cil, cil_cents, cil_ids = _label_stats(mips.get("cilia_labels"))
        n_bb_det,  area_bb,  bb_cents,  _bb_ids = _label_stats(mips.get("bb_labels"))
        _lr = fk["log_ratio"].to_numpy(dtype=float) if "log_ratio" in fk else np.array([])
        _lr = _lr[np.isfinite(_lr)]
        return dict(
            fname=fname, label=_label_for(fname), mips=mips, fk=fk, bb_sub=bb_sub,
            n_cil_det=n_cil_det, area_cil=area_cil, cil_cents=cil_cents, cil_ids=cil_ids,
            n_bb_det=n_bb_det, area_bb=area_bb, bb_cents=bb_cents,
            n_cil_kept=len(fk), n_bb_kept=len(bb_sub),
            mean_lr=float(_lr.mean()) if _lr.size else float("nan"),
            cls=(fk["class"].value_counts().to_dict() if "class" in fk else {}),
        )

    # ── Pass 1: gather per-sample numeric stats, then flag outliers ───────────
    records = [_stats_for(f) for f in files]
    flags = _flag_deviant(records)

    summary_rows = []
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # ── Overview / flags page (first) ─────────────────────────────────────
        _render_qc_overview(pdf, records, flags)

        # ── Per-sample pages ──────────────────────────────────────────────────
        for rec in records:
            fname, label, mips = rec["fname"], rec["label"], rec["mips"]
            fk = rec["fk"]
            n_cil_det, area_cil, cil_cents = rec["n_cil_det"], rec["area_cil"], rec["cil_cents"]
            cil_ids = rec["cil_ids"]
            n_bb_det, area_bb, bb_cents = rec["n_bb_det"], rec["area_bb"], rec["bb_cents"]
            n_cil_kept, n_bb_kept, mean_lr = rec["n_cil_kept"], rec["n_bb_kept"], rec["mean_lr"]
            _cls = rec["cls"]

            summary_rows.append({
                "Sample": label,
                "Cilia det.": n_cil_det,
                "Cilia kept": n_cil_kept,
                "Cilia area": f"{area_cil:.0f}" if np.isfinite(area_cil) else "—",
                "BB det.": n_bb_det,
                "BB kept": n_bb_kept,
                "BB area": f"{area_bb:.0f}" if np.isfinite(area_bb) else "—",
                "mean log_ratio": f"{mean_lr:.2f}" if np.isfinite(mean_lr) else "—",
                "axon": _cls.get("axon", 0),
                "soma": _cls.get("soma", 0),
                "amb.": _cls.get("ambiguous", 0),
                "flag": "⚠" if flags.get(fname) else "",
            })

            fig, axes = plt.subplots(3, 4, figsize=(16, 11))
            _fr = flags.get(fname, [])
            _title = f"QC report — {label}\n{os.path.basename(fname)}"
            if _fr:
                _title += "\n⚠ FLAGGED: " + ", ".join(_fr)
            fig.suptitle(_title, fontsize=14, fontweight="bold",
                         color=("#c0392b" if _fr else "black"))

            # Row 0 — original channels
            _qc_show_gray(axes[0, 0], mips.get("cilia"),   "Cilia (raw MIP)")
            _qc_show_gray(axes[0, 1], mips.get("neurite"), "Neurites (raw MIP)")
            _qc_show_gray(axes[0, 2], mips.get("bb"),      "Basal Bodies (raw MIP)")
            _qc_show_gray(axes[0, 3], mips.get("nuclei"),  "Nuclei (raw MIP)")

            # Row 1 — segmentations
            _qc_show_labels(axes[1, 0], mips.get("cilia_labels"),   "Cilia segmentation")
            _qc_show_labels(axes[1, 1], mips.get("neurite_labels"), "Neurite segmentation",
                            mask=mips.get("neurite_mask"))
            _qc_show_labels(axes[1, 2], mips.get("bb_labels"),      "Basal Body segmentation")
            _qc_show_labels(axes[1, 3], mips.get("nuclei_labels"),  "Nuclei segmentation")

            # Row 2 — detections + stats (cilia annotated with their label IDs)
            _qc_show_gray(axes[2, 0], mips.get("cilia"), f"All detected cilia (n={n_cil_det})")
            if cil_cents.size:
                axes[2, 0].scatter(cil_cents[:, 1], cil_cents[:, 0], s=18,
                                   facecolors="none", edgecolors="cyan", linewidths=0.8)
                for _cc, _cid in zip(cil_cents, cil_ids):
                    axes[2, 0].text(_cc[1] + 4, _cc[0] - 4, str(int(_cid)),
                                    color="cyan", fontsize=4, ha="left", va="bottom")

            _qc_show_gray(axes[2, 1], mips.get("cilia"), f"Kept cilia (n={n_cil_kept})")
            if not fk.empty and "coords" in fk.columns:
                _c = np.vstack([_parse_coords(c) for c in fk["coords"]])
                _cols = [_CLASS_PALETTE.get(c, "#888") for c in fk["class"]] \
                    if "class" in fk.columns else "yellow"
                axes[2, 1].scatter(_c[:, 2], _c[:, 1], s=18, c=_cols,
                                   edgecolors="black", linewidths=0.3)
                if "cilia_id" in fk.columns:
                    for _cc, _cid in zip(_c, fk["cilia_id"].tolist()):
                        axes[2, 1].text(_cc[2] + 4, _cc[1] - 4, str(int(_cid)),
                                        color="white", fontsize=4, ha="left", va="bottom")

            _qc_show_gray(axes[2, 2], mips.get("bb"), f"All detected BB (n={n_bb_det})")
            if bb_cents.size:
                axes[2, 2].scatter(bb_cents[:, 1], bb_cents[:, 0], s=18,
                                   facecolors="none", edgecolors="orange", linewidths=0.8)

            axes[2, 3].axis("off")
            _area_cil_s = f"{area_cil:.1f} px" if np.isfinite(area_cil) else "—"
            _area_bb_s  = f"{area_bb:.1f} px"  if np.isfinite(area_bb)  else "—"
            _mean_lr_s  = f"{mean_lr:.2f}"     if np.isfinite(mean_lr)  else "—"
            _cls_txt = "  ".join(f"{k}={_cls.get(k, 0)}"
                                 for k in ("axon", "soma", "ambiguous"))
            _stats_txt = (
                f"Counts\n{'─' * 24}\n"
                f"Detected cilia : {n_cil_det}\n"
                f"Kept cilia     : {n_cil_kept}\n"
                f"Mean cilia area: {_area_cil_s}\n\n"
                f"Detected BB    : {n_bb_det}\n"
                f"Kept BB        : {n_bb_kept}\n"
                f"Mean BB area   : {_area_bb_s}\n\n"
                f"Mean log_ratio : {_mean_lr_s}\n"
                f"Classes        : {_cls_txt}"
            )
            axes[2, 3].text(
                0.0, 1.0, _stats_txt,
                va="top", ha="left", family="monospace", fontsize=10,
                transform=axes[2, 3].transAxes,
            )

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

            # Close-up ROIs of every detected cilium (+ nearest basal body),
            # split into kept vs non-kept, with 3-D volume/length when available
            _kept_ids = (
                fk["cilia_id"].tolist()
                if not fk.empty and "cilia_id" in fk.columns else []
            )
            _props3d = {}
            if not fk.empty and {"cilia_id", "volume_um3", "length_um"} <= set(fk.columns):
                for _cid, _v, _l in zip(fk["cilia_id"], fk["volume_um3"], fk["length_um"]):
                    try:
                        _props3d[int(_cid)] = (float(_v), float(_l))
                    except (TypeError, ValueError):
                        pass
            _render_cilia_roi_pages(pdf, label, mips, kept_ids=_kept_ids, props3d=_props3d)

        # ── Summary page ──────────────────────────────────────────────────────
        _sdf = pd.DataFrame(summary_rows)
        n = max(1, len(_sdf))
        fig, ax = plt.subplots(figsize=(16, 1.4 + 0.45 * n))
        ax.axis("off")
        ax.set_title("Summary — all samples  (⚠ = flagged as deviant)",
                     fontsize=14, fontweight="bold", loc="left")
        if not _sdf.empty:
            tbl = ax.table(cellText=_sdf.values, colLabels=_sdf.columns,
                           loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.4)
            for _j in range(len(_sdf.columns)):
                tbl[0, _j].set_facecolor("#34495e")
                tbl[0, _j].set_text_props(color="white", fontweight="bold")
            # Tint flagged rows
            for _i in range(len(_sdf)):
                if _sdf.iloc[_i]["flag"]:
                    for _j in range(len(_sdf.columns)):
                        tbl[_i + 1, _j].set_facecolor("#fdecea")
                        tbl[_i + 1, _j].set_text_props(color="#c0392b")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return buf.getvalue()


def _parse_coords(val) -> np.ndarray:
    """Parse a centroid stored as a list or as its string repr from Excel.
    Handles both clean '[z, y, x]' and numpy repr '[np.float64(z), ...]'."""
    if isinstance(val, (list, np.ndarray)):
        return np.array(val, dtype=float)
    s = str(val)
    # Strip numpy type wrappers produced by older runs: np.float64(x) → x
    s = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', s)
    try:
        return np.array(ast.literal_eval(s), dtype=float)
    except Exception:
        return np.zeros(3)


def _reclassify(series: pd.Series, axon_thr: float, soma_thr: float) -> pd.Series:
    def _cls(r):
        if pd.isna(r):
            return "ambiguous"
        return "axon" if r > axon_thr else ("soma" if r < soma_thr else "ambiguous")
    return series.apply(_cls)


def _apply_post_filters(
    df_cil: pd.DataFrame,
    axon_thr: float,
    soma_thr: float,
    nn_min_dist: float,
    nn_radius: float,
    nn_max_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reclassify and apply NN filter per-file.
    Returns (df_kept, df_nn_removed). Never triggers a pipeline rerun.
    """
    if df_cil.empty:
        return df_cil.copy(), df_cil.iloc[:0].copy()

    kept_parts, removed_parts = [], []
    nn_active = (nn_min_dist > 0.0) or (nn_max_count < 1000)

    for _, group in df_cil.groupby("filename"):
        group = group.copy()
        if "log_ratio" in group.columns:
            group["class"] = _reclassify(group["log_ratio"], axon_thr, soma_thr)

        if nn_active and "coords" in group.columns and len(group) >= 2:
            try:
                coords = np.array([_parse_coords(c) for c in group["coords"]])
                yx = coords[:, 1:3]  # Y, X pixel space

                tree = KDTree(yx)
                k = min(2, len(yx))
                dists, _ = tree.query(yx, k=k)
                nn_dist = dists[:, 1] if k == 2 else np.full(len(yx), np.inf)

                neighbors = tree.query_ball_point(yx, r=max(nn_radius, 1e-9))
                nn_count = np.array([len(n) - 1 for n in neighbors])

                keep = (nn_dist >= nn_min_dist) & (nn_count <= nn_max_count)
                kept_parts.append(group[keep])
                removed_parts.append(group[~keep])
                continue
            except Exception:
                pass

        kept_parts.append(group)

    df_kept = pd.concat(kept_parts, ignore_index=True) if kept_parts else df_cil.iloc[:0].copy()
    df_removed = pd.concat(removed_parts, ignore_index=True) if removed_parts else df_cil.iloc[:0].copy()
    return df_kept, df_removed


# ─────────────────────────────────────────────────────────────────────────────
# GPU DETECTION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def _detect_gpus() -> list[tuple[str, str | None]]:
    """Return [(display_label, cle_device_name), ...] using pyclesperanto's own
    OpenCL enumeration — the only names that work with cle.select_device().
    CPU fallback is always appended last."""
    devices: list[tuple[str, str | None]] = []

    try:
        import pyclesperanto_prototype as cle
        for name in cle.available_device_names(dev_type="gpu"):
            if name:
                devices.append((name, name))
    except Exception:
        pass

    devices.append(("CPU only (no explicit device selection)", None))
    return devices


def _vram_badge(display_label: str) -> str | None:
    """Return VRAM usage string or None if torch/CUDA not available."""
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            if torch.cuda.get_device_name(i) in display_label:
                used_mb = torch.cuda.memory_reserved(i) / 1024 ** 2
                total_mb = torch.cuda.get_device_properties(i).total_memory / 1024 ** 2
                return f"{used_mb:.0f} / {total_mb:.0f} MB  ({used_mb / total_mb * 100:.0f}%)"
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# RECENT PATH HISTORY
# ─────────────────────────────────────────────────────────────────────────────
_PATH_HISTORY_FILE = os.path.join(os.path.expanduser("~"), ".limoncello_paths.json")
_PATH_HISTORY_MAX = 8
_NEW_PATH_SENTINEL = "✏️  Enter new path…"


def _load_path_history() -> dict:
    try:
        with open(_PATH_HISTORY_FILE) as _fh:
            return json.load(_fh)
    except Exception:
        return {}


def _save_to_path_history(key: str, value: str) -> None:
    if not value:
        return
    h = _load_path_history()
    lst = [v for v in h.get(key, []) if v != value]
    h[key] = [value] + lst[:_PATH_HISTORY_MAX - 1]
    try:
        with open(_PATH_HISTORY_FILE, "w") as _fh:
            json.dump(h, _fh, indent=2)
    except Exception:
        pass


_JSON_SS_MAP = [
    # (json_path_tuple,                                    ss_key,                 cast)
    (("use_mip",),                                         "use_mip",              bool),
    (("channels", "cilia"),                                "ch_cilia",             int),
    (("channels", "neurites"),                             "ch_neurites",          int),
    (("channels", "basal_bodies"),                         "ch_bb",                int),
    (("channels", "nuclei"),                               "ch_nuclei",            int),
    (("intensity_normalization", "p_low"),                 "p_low",                int),
    (("intensity_normalization", "p_high"),                "p_high",               int),
    (("nuclei", "spot_sigma"),                             "nuclei_sigma",         int),
    (("nuclei", "tophat_radius"),                          "tophat_radius",        int),
    (("nuclei", "outline_sigma"),                          "nuclei_outline_sigma", int),
    (("neurites", "spot_sigma"),                           "neurite_sigma",        int),
    (("cilia", "min_size"),                                "cilia_min_size",       int),
    (("cilia", "max_size"),                                "cilia_max_size",       int),
    # cilia/basal_bodies "gaussian_sigma" are 3-element lists handled separately below
    (("basal_bodies", "spot_sigma"),                       "bb_spot_sigma",        float),
    (("basal_bodies", "outline_sigma"),                    "bb_outline_sigma",     float),
    (("basal_bodies", "min_size"),                         "bb_min_size",          int),
    (("basal_bodies", "max_size"),                         "bb_max_size",          int),
    (("distance_thresholds", "max_cilia_um"),              "max_cilia",            float),
    (("distance_thresholds", "max_basal_body_um"),         "max_basal",            float),
    (("distance_thresholds", "require_basal_body"),        "require_bb",           bool),
    (("distance_thresholds", "ratio_from_basal_body"),     "ratio_from_bb",        bool),
    (("distance_thresholds", "ratio_epsilon"),             "ratio_epsilon",        float),
    (("classification", "neurite_threshold"),             "pf_axon_thr",          float),
    (("classification", "soma_threshold"),                 "pf_soma_thr",          float),
]


def _apply_json_params(data: dict) -> None:
    """Queue parameters to be applied at the top of the next rerun.

    Streamlit forbids writing to widget-bound session-state keys after those
    widgets have been instantiated.  We store the payload in a staging key;
    the early-apply block (just after _SS_DEFAULTS init) consumes it before
    any widget is created.
    """
    st.session_state["_pending_params"] = data


def _path_input(label: str, history_key: str, help_text: str = "",
                default: str = "") -> str:
    """Selectbox of recent paths; falls back to a text input when empty."""
    recent = _load_path_history().get(history_key, [])
    if recent:
        sel = st.selectbox(label, [_NEW_PATH_SENTINEL] + recent,
                           key=f"_sel_{history_key}", help=help_text)
        if sel == _NEW_PATH_SENTINEL:
            return st.text_input(label, key=f"_new_{history_key}",
                                 label_visibility="collapsed",
                                 placeholder="Paste or type path…")
        return sel
    return st.text_input(label, key=f"_new_{history_key}",
                         help=help_text,
                         value=st.session_state.get(f"_new_{history_key}", default))


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_SS_DEFAULTS: dict = {
    "pipeline_running":    False,
    "pipeline_status":     "idle",
    "pipeline_error":      None,
    "pipeline_elapsed":    None,
    "pipeline_start_time": None,
    "logs":                [],
    "logs_paused":         False,
    "custom_plot_png":     None,
    "thread_result":       {},
    "_custom_ov_bytes":    None,
    "_custom_ov_ext":      "png",
    "_custom_ov_mime":     "image/png",
    # Segmentation parameters (keyed so JSON reload works)
    "p_low":               2,
    "p_high":              98,
    "nuclei_sigma":        15,
    "tophat_radius":       12,
    "nuclei_outline_sigma": 3,
    "nuclei_gauss_on":     False,
    "nuclei_gauss_z":      1.0,
    "nuclei_gauss_y":      1.0,
    "nuclei_gauss_x":      1.0,
    "nuclei_log":          False,
    "neurite_sigma":       5,
    "neurite_gauss_on":    False,
    "neurite_gauss_z":     1.0,
    "neurite_gauss_y":     1.0,
    "neurite_gauss_x":     1.0,
    "neurite_log":         False,
    "cilia_log":           False,
    "bb_log":              False,
    "cilia_gauss_on":      True,
    "cilia_gauss_z":       1.0,
    "cilia_gauss_y":       1.0,
    "cilia_gauss_x":       1.0,
    "cilia_min_size":      20,
    "cilia_max_size":      0,
    "bb_spot_sigma":       2.0,
    "bb_outline_sigma":    2.0,
    "bb_gauss_on":         True,
    "bb_gauss_z":          1.0,
    "bb_gauss_y":          1.0,
    "bb_gauss_x":          0.0,
    "bb_min_size":         5,
    "bb_max_size":         0,
    "max_cilia":           2.0,
    "max_basal":           2.0,
    "require_bb":          True,
    "ratio_from_bb":       True,
    "ratio_epsilon":       1.0,
    "use_mip":             False,
    **_PF_DEFAULTS,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Apply queued channel assignments from the channel viewer dialog
if "_pending_channels" in st.session_state:
    for _sk, _cv in st.session_state.pop("_pending_channels").items():
        st.session_state[_sk] = int(_cv)

# Apply any queued JSON parameter reload — must happen here, before widgets are
# instantiated, so Streamlit allows writing to widget-bound session-state keys.
if "_pending_params" in st.session_state:
    _qd = st.session_state.pop("_pending_params")
    for _path, _ss_key, _cast in _JSON_SS_MAP:
        _node = _qd
        for _k in _path:
            _node = _node.get(_k) if isinstance(_node, dict) else None
        if _node is not None:
            st.session_state[_ss_key] = _cast(_node)
    # Per-channel Gaussian pre-blur + log transform: restore from JSON
    for _sec, _pre in (("cilia", "cilia"), ("basal_bodies", "bb"),
                       ("nuclei", "nuclei"), ("neurites", "neurite")):
        _gs = _qd.get(_sec, {}).get("gaussian_sigma")
        if _gs and len(_gs) >= 3:
            st.session_state[f"{_pre}_gauss_z"] = float(_gs[0])
            st.session_state[f"{_pre}_gauss_y"] = float(_gs[1])
            st.session_state[f"{_pre}_gauss_x"] = float(_gs[2])
            st.session_state[f"{_pre}_gauss_on"] = any(float(s) > 0 for s in _gs[:3])
        _lt = _qd.get(_sec, {}).get("log_transform")
        if _lt is not None:
            st.session_state[f"{_pre}_log"] = bool(_lt)

# Sync completed thread result → session_state (thread cannot write session_state directly)
if st.session_state.pipeline_running:
    _tr = st.session_state.thread_result
    if _tr and not _tr.get("running", True):
        st.session_state.pipeline_running = False
        st.session_state.pipeline_status = _tr.get("status", "completed")
        st.session_state.pipeline_error = _tr.get("error")
        st.session_state.pipeline_elapsed = _tr.get("elapsed")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:2px'>Limoncello 🍋</h1>"
    "<p style='color:#888;margin-top:0'>Interactive cilia–neurite analysis pipeline</p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT STATUS BANNER
# ─────────────────────────────────────────────────────────────────────────────
_status = st.session_state.pipeline_status
_running_elapsed = (
    time.time() - st.session_state.pipeline_start_time
    if st.session_state.pipeline_running and st.session_state.pipeline_start_time
    else None
)

if _status == "idle":
    st.info("⏸ **Idle** — configure parameters in the sidebar and click **Run Pipeline**.")
elif _status == "running":
    st.warning(f"⚙️ **Running** — {_running_elapsed:.0f}s elapsed…")
elif _status == "completed":
    st.success(f"✅ **Completed** in {st.session_state.pipeline_elapsed:.1f}s")
elif _status == "error":
    st.error(f"❌ **Error** — {st.session_state.pipeline_error}")

st.markdown("<hr style='margin:6px 0 12px'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Settings")

    # ── File Paths ────────────────────────────────────────────────────────────
    st.subheader("📁 File Paths")
    input_path = _path_input(
        "Input folder", "input",
        help_text="Folder containing .ims files to process",
    )
    output_path = _path_input(
        "Output folder", "output",
        help_text="Results (figures, CSVs, Excel) saved here",
        default="tutorial/output",
    )

    # ── Run selector ──────────────────────────────────────────────────────────
    _sidebar_runs = sorted(
        (d for d in (os.listdir(output_path) if output_path and os.path.isdir(output_path) else [])
         if d.startswith("lc-analysis-") and os.path.isdir(os.path.join(output_path, d))),
        reverse=True,
    )
    if _sidebar_runs:
        st.selectbox(
            "Analysis run to explore",
            options=_sidebar_runs,
            key="_sel_run",
            format_func=lambda d: d.replace("lc-analysis-", "").replace("_", "  "),
            help="Which lc-analysis run to show in Data Tables, Graphs, and Overlays",
        )
    elif output_path:
        st.caption("No runs found in output folder yet.")

    classifier_path = _path_input(
        "Cilia classifier", "classifier",
        help_text="Path to the trained .cl cilia segmentation model",
        default=r"segmenters\cilia-segmenter.cl",
    )

    # Expose resolved paths so other pages (e.g. Preview) can reuse them
    st.session_state["_resolved_input"] = input_path
    st.session_state["_resolved_output"] = output_path
    st.session_state["_resolved_classifier"] = classifier_path

    st.markdown("---")

    # ── Channel Assignment ────────────────────────────────────────────────────
    st.subheader(
        "🔬 Channel Assignment",
        help="🔄 Map image channels to biological structures. Requires re-running the pipeline.",
    )

    _file_info = _get_first_ims_info(input_path)
    _n_ch = _file_info["n_channels"] if _file_info else 4
    _ch_opts = list(range(_n_ch))

    # Current assignment summary
    _asum = "  ·  ".join(
        f"{lbl}: Ch {st.session_state.get(sk, df)}"
        for lbl, sk, df in _DIALOG_ROLES
    )
    st.caption(_asum)

    # Primary: full visual viewer
    if _file_info:
        if st.button("🔭 Open Channel Viewer", key="ch_viewer_btn",
                     width="stretch", type="primary"):
            # Clear dialog assignment state so it initialises from current sidebar values
            for _, _sk, _ in _DIALOG_ROLES:
                st.session_state.pop(f"_dassign_{_sk}", None)
            _channel_assignment_dialog(input_path, _file_info["filename"], _n_ch)
    else:
        st.info("Enter a valid input folder to open the channel viewer.")

    # Fallback: compact selectboxes for quick edits without opening the dialog
    with st.expander("⚙️ Manual override", expanded=False):
        cilia_channel = st.selectbox(
            "Cilia", _ch_opts, key="ch_cilia",
            help="🔄 Raw channel fed to the ML cilia segmenter",
        )
        neurites_channel = st.selectbox(
            "Neurites", _ch_opts, key="ch_neurites",
            help="🔄 Normalised channel used for neurite tracing",
        )
        basal_bodies_channel = st.selectbox(
            "Basal bodies", _ch_opts, key="ch_bb",
            help="🔄 Raw channel used for basal body detection",
        )
        nuclei_channel = st.selectbox(
            "Nuclei (DAPI)", _ch_opts, key="ch_nuclei",
            help="🔄 Normalised channel used for nucleus segmentation",
        )

    st.markdown("---")

    # ── Compute Device ────────────────────────────────────────────────────────
    st.subheader("🖥️ Compute Device")
    _gpu_options = _detect_gpus()
    _gpu_labels = [d[0] for d in _gpu_options]
    _gpu_map = {d[0]: d[1] for d in _gpu_options}

    # Default to first real GPU
    _default_gpu_idx = next(
        (i for i, (_, nm) in enumerate(_gpu_options) if nm is not None), 0
    )
    _gpu_sel = st.selectbox(
        "Device", _gpu_labels, index=_default_gpu_idx,
        help="GPU used for pyclesperanto segmentation operations"
    )
    gpu_device = _gpu_map[_gpu_sel]

    _vram = _vram_badge(_gpu_sel)
    if _vram:
        st.caption(f"VRAM: {_vram}")
    elif gpu_device is not None:
        st.caption("VRAM unavailable (install PyTorch for usage stats)")
    else:
        st.warning("No GPU detected — pipeline will use default OpenCL device.", icon="⚠️")

    st.markdown("---")

    # ── 🔄 Segmentation Parameters ────────────────────────────────────────────
    st.subheader(
        "🔄 Segmentation Parameters",
        help="Changing any parameter in this section requires re-running the full pipeline."
    )

    use_mip = st.checkbox(
        "MIP mode (project Z → 2-D before segmentation)",
        key="use_mip",
        help="🔄 Use pyclesperanto maximum_z_projection on each channel before segmentation. "
             "Useful for thin samples or when Z resolution is poor.",
    )

    with st.expander("🔆 Intensity Normalization", expanded=False):
        p_low = st.slider(
            "Percentile low", 0, 10, step=1, key="p_low",
            help="🔄 Lower percentile for min-max intensity clipping"
        )
        p_high = st.slider(
            "Percentile high", 90, 100, step=1, key="p_high",
            help="🔄 Upper percentile for min-max intensity clipping"
        )

    with st.expander("🟡 Nuclei", expanded=True):
        nuclei_sigma = st.slider(
            "Spot sigma", 1, 50, step=1, key="nuclei_sigma",
            help="🔄 Voronoi-Otsu object-separation scale for nuclei"
        )
        tophat_radius = st.slider(
            "Tophat radius", 1, 50, step=1, key="tophat_radius",
            help="🔄 Morphological top-hat background subtraction radius (voxels)"
        )
        outline_sigma = st.slider(
            "Outline sigma", 0, 10, step=1, key="nuclei_outline_sigma",
            help="🔄 Boundary-precision smoothing for nuclei labeling"
        )
        nuclei_log = st.checkbox(
            "Log-normalize intensities", key="nuclei_log",
            help="🔄 Apply a log1p transform to the nuclei channel before "
                 "segmentation (compresses dynamic range)."
        )
        nuclei_gauss_on = st.checkbox(
            "Apply Gaussian blur", key="nuclei_gauss_on",
            help="🔄 Optional Gaussian pre-blur before nuclei segmentation"
        )
        nuclei_gauss_z = st.slider(
            "Gaussian σ_z", 0.0, 5.0, step=0.5, key="nuclei_gauss_z",
            disabled=not nuclei_gauss_on, help="🔄 Gaussian blur sigma along Z"
        )
        nuclei_gauss_y = st.slider(
            "Gaussian σ_y", 0.0, 5.0, step=0.5, key="nuclei_gauss_y",
            disabled=not nuclei_gauss_on, help="🔄 Gaussian blur sigma along Y"
        )
        nuclei_gauss_x = st.slider(
            "Gaussian σ_x", 0.0, 5.0, step=0.5, key="nuclei_gauss_x",
            disabled=not nuclei_gauss_on, help="🔄 Gaussian blur sigma along X"
        )

    with st.expander("🧵 Neurites", expanded=True):
        neurite_sigma = st.slider(
            "Spot sigma", 1, 20, step=1, key="neurite_sigma",
            help="🔄 Voronoi-Otsu object-separation scale for neurite detection"
        )
        neurite_log = st.checkbox(
            "Log-normalize intensities", key="neurite_log",
            help="🔄 Apply a log1p transform to the neurite channel before "
                 "segmentation (compresses dynamic range)."
        )
        neurite_gauss_on = st.checkbox(
            "Apply Gaussian blur", key="neurite_gauss_on",
            help="🔄 Optional Gaussian pre-blur before neurite segmentation"
        )
        neurite_gauss_z = st.slider(
            "Gaussian σ_z", 0.0, 5.0, step=0.5, key="neurite_gauss_z",
            disabled=not neurite_gauss_on, help="🔄 Gaussian blur sigma along Z"
        )
        neurite_gauss_y = st.slider(
            "Gaussian σ_y", 0.0, 5.0, step=0.5, key="neurite_gauss_y",
            disabled=not neurite_gauss_on, help="🔄 Gaussian blur sigma along Y"
        )
        neurite_gauss_x = st.slider(
            "Gaussian σ_x", 0.0, 5.0, step=0.5, key="neurite_gauss_x",
            disabled=not neurite_gauss_on, help="🔄 Gaussian blur sigma along X"
        )

    with st.expander("🎯 Cilia", expanded=True):
        cilia_log = st.checkbox(
            "Log-normalize intensities", key="cilia_log",
            help="🔄 Apply a log1p transform before the APOC classifier. "
                 "⚠️ The classifier is trained on raw intensities — enabling this "
                 "changes the feature space and may hurt results."
        )
        cilia_gauss_on = st.checkbox(
            "Apply Gaussian blur", key="cilia_gauss_on",
            help="🔄 Optional Gaussian pre-blur before the APOC classifier"
        )
        cilia_gauss_z = st.slider(
            "Gaussian σ_z", 0.0, 5.0, step=0.5, key="cilia_gauss_z",
            disabled=not cilia_gauss_on,
            help="🔄 Gaussian blur sigma along Z before cilia segmentation"
        )
        cilia_gauss_y = st.slider(
            "Gaussian σ_y", 0.0, 5.0, step=0.5, key="cilia_gauss_y",
            disabled=not cilia_gauss_on,
            help="🔄 Gaussian blur sigma along Y before cilia segmentation"
        )
        cilia_gauss_x = st.slider(
            "Gaussian σ_x", 0.0, 5.0, step=0.5, key="cilia_gauss_x",
            disabled=not cilia_gauss_on,
            help="🔄 Gaussian blur sigma along X before cilia segmentation"
        )
        cilia_min_size = st.number_input(
            "Min size (voxels)", min_value=0, step=1, key="cilia_min_size",
            help="🔄 Remove cilia smaller than this many voxels (noise filter). 0 = disabled."
        )
        cilia_max_size = st.number_input(
            "Max size (voxels)", min_value=0, step=10, key="cilia_max_size",
            help="🔄 Remove cilia larger than this many voxels (cluster filter). 0 = disabled."
        )

    with st.expander("🔵 Basal Bodies", expanded=True):
        bb_spot_sigma = st.slider(
            "Spot sigma", 0.5, 10.0, step=0.5, key="bb_spot_sigma",
            help="🔄 Voronoi-Otsu object-separation scale for basal body detection"
        )
        bb_outline_sigma = st.slider(
            "Outline sigma", 0.5, 10.0, step=0.5, key="bb_outline_sigma",
            help="🔄 Boundary-precision smoothing for basal body labeling"
        )
        bb_log = st.checkbox(
            "Log-normalize intensities", key="bb_log",
            help="🔄 Apply a log1p transform to the basal-body channel before "
                 "segmentation (compresses dynamic range)."
        )
        bb_gauss_on = st.checkbox(
            "Apply Gaussian blur", key="bb_gauss_on",
            help="🔄 Optional Gaussian pre-blur before Voronoi-Otsu"
        )
        bb_gauss_z = st.slider(
            "Gaussian σ_z", 0.0, 5.0, step=0.5, key="bb_gauss_z",
            disabled=not bb_gauss_on,
            help="🔄 Gaussian blur sigma along Z before basal body segmentation"
        )
        bb_gauss_y = st.slider(
            "Gaussian σ_y", 0.0, 5.0, step=0.5, key="bb_gauss_y",
            disabled=not bb_gauss_on,
            help="🔄 Gaussian blur sigma along Y before basal body segmentation"
        )
        bb_gauss_x = st.slider(
            "Gaussian σ_x", 0.0, 5.0, step=0.5, key="bb_gauss_x",
            disabled=not bb_gauss_on,
            help="🔄 Gaussian blur sigma along X before basal body segmentation"
        )
        bb_min_size = st.number_input(
            "Min size (voxels)", min_value=0, step=1, key="bb_min_size",
            help="🔄 Remove basal bodies smaller than this many voxels. 0 = disabled."
        )
        bb_max_size = st.number_input(
            "Max size (voxels)", min_value=0, step=10, key="bb_max_size",
            help="🔄 Remove basal bodies larger than this many voxels. 0 = disabled."
        )

    st.markdown("---")

    st.subheader(
        "📏 Distance Thresholds",
        help="🔄 Require re-running the pipeline."
    )
    max_cilia = st.slider(
        "Max cilia distance (µm)", 0.5, 10.0, step=0.1, key="max_cilia",
        help="🔄 Max centroid→neurite distance to include a cilium"
    )
    max_basal = st.slider(
        "Max basal body distance (µm)", 0.5, 10.0, step=0.1, key="max_basal",
        help="🔄 Max centroid→neurite distance to include a basal body"
    )
    require_bb = st.checkbox(
        "Require basal body (filter cilia by BB distance)", key="require_bb",
        help="🔄 Off = keep ALL cilia regardless of basal-body pairing distance."
    )
    ratio_from_bb = st.checkbox(
        "Ratio from basal body position", key="ratio_from_bb",
        help="🔄 Sample the soma/neurite ratio at the paired basal body (the "
             "anchor) instead of the cilium centroid. Unpaired cilia keep their "
             "own value."
    )
    ratio_epsilon = st.slider(
        "Ratio epsilon (µm)", 0.0, 10.0, step=0.1, key="ratio_epsilon",
        help="🔄 Constant ε added to both numerator and denominator: "
             "ratio = (dt_nuclei + ε) / (dt_neurite + ε). "
             "Prevents division by zero and log(0). Larger ε → smoother ratio."
    )

    # ── 🤖 AI cilia validation (optional) ─────────────────────────────────────
    _models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    _ai_models = ([f for f in sorted(os.listdir(_models_dir)) if f.endswith(".pt")]
                  if os.path.isdir(_models_dir) else [])
    ai_validate = st.checkbox(
        "🤖 AI-validate cilia after run", key="ai_validate",
        value=False, disabled=not _ai_models,
        help="Score each cilium's ROI image with a trained validator and write "
             "csv/human_validation.csv (loaded by the data app's Screening tab)."
        if _ai_models else "No models found in the codebase models/ folder."
    )
    ai_model_name = st.selectbox(
        "AI model", _ai_models or ["(no models found)"], key="ai_model_name",
        disabled=not (_ai_models and ai_validate))
    ai_threshold = st.slider(
        "Keep if AI score ≥", 0.50, 0.99, 0.50, 0.01, key="ai_threshold",
        disabled=not (_ai_models and ai_validate))

    st.markdown("---")

    # ── ⚡ Post-Segmentation Filters ──────────────────────────────────────────
    st.subheader(
        "⚡ Post-Segmentation Filters",
        help="Applied instantly to loaded data — no pipeline rerun needed. "
             "Adjusting any control here immediately updates all graphs, tables, and overlays."
    )

    with st.expander("🏷️ Classification Thresholds", expanded=True):
        pf_axon_thr = st.number_input(
            "Axon  (log_ratio >)", step=0.1, min_value=0.0, key="pf_axon_thr",
            help="⚡ log_ratio above this → 'axon'. Updates instantly."
        )
        pf_soma_thr = st.number_input(
            "Soma  (log_ratio <)", step=0.1, min_value=0.0, key="pf_soma_thr",
            help="⚡ log_ratio below this → 'soma'. Between thresholds → 'ambiguous'."
        )

    with st.expander("🔵 Nearest-Neighbour Filter", expanded=True):
        st.caption(
            "Excludes dense clusters (rosettes). "
            "Distances in **image pixels** (Y × X plane)."
        )
        pf_nn_min_dist = st.slider(
            "Min NN distance (px)", 0.0, 200.0, step=1.0, key="pf_nn_min_dist",
            help="⚡ Discard cilia whose nearest neighbour is closer than this — removes dense rosette clusters"
        )
        pf_nn_radius = st.slider(
            "Count-filter radius (px)", 1.0, 500.0, step=1.0, key="pf_nn_radius",
            help="⚡ Radius used for counting nearby neighbours"
        )
        pf_nn_max_count = st.number_input(
            "Max neighbours in radius", min_value=0, step=1, key="pf_nn_max_count",
            help="⚡ Discard cilia with more than N neighbours within the radius above"
        )

    # Placeholder filled with NN removal count after data is loaded
    _nn_count_slot = st.empty()

    if st.button("↩️ Reset Filters", width="stretch"):
        for _rk, _rv in _PF_DEFAULTS.items():
            st.session_state[_rk] = _rv
        st.rerun()

    st.markdown("---")

    run_clicked = st.button(
        "🚀 Run Pipeline",
        disabled=st.session_state.pipeline_running,
        width="stretch",
        type="primary",
    )

# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH ON CLICK
# ─────────────────────────────────────────────────────────────────────────────
if run_clicked:
    if not input_path:
        st.error("Please specify an input folder.")
    elif not os.path.exists(input_path):
        st.error(f"Input folder not found: `{input_path}`")
    else:
        _log_list: list = []
        st.session_state.logs = _log_list
        st.session_state.custom_plot_png = None
        _start = time.time()
        st.session_state.pipeline_start_time = _start
        st.session_state.pipeline_running = True
        st.session_state.pipeline_status = "running"
        st.session_state.pipeline_error = None
        st.session_state.pipeline_elapsed = None

        _result: dict = {
            "running": True,
            "status": "running",
            "error": None,
            "elapsed": None,
            "start_time": _start,
        }
        st.session_state.thread_result = _result

        _params = dict(
            input_path=input_path,
            output_path=output_path,
            max_cilia_dist_cutoff_um=max_cilia,
            max_basal_body_cutoff_um=max_basal,
            require_basal_body=require_bb,
            ratio_from_basal_body=ratio_from_bb,
            batch_ai_model=(os.path.join(_models_dir, ai_model_name)
                            if ai_validate and ai_model_name.endswith(".pt")
                            else None),
            batch_ai_threshold=float(ai_threshold),
            nuclei_spot_sigma=nuclei_sigma,
            tophat_radius=tophat_radius,
            neurite_spot_sigma=neurite_sigma,
            cilia_classifier_path=classifier_path,
            neurite_threshold=float(pf_axon_thr),
            soma_threshold=float(pf_soma_thr),
            p_low=p_low,
            p_high=p_high,
            outline_sigma=outline_sigma,
            gpu_device=gpu_device,
            cilia_channel=int(cilia_channel),
            neurites_channel=int(neurites_channel),
            basal_bodies_channel=int(basal_bodies_channel),
            nuclei_channel=int(nuclei_channel),
            use_mip=bool(use_mip),
            cilia_gaussian_sigma=(
                (float(cilia_gauss_z), float(cilia_gauss_y), float(cilia_gauss_x))
                if cilia_gauss_on else (0.0, 0.0, 0.0)
            ),
            nuclei_gaussian_sigma=(
                (float(nuclei_gauss_z), float(nuclei_gauss_y), float(nuclei_gauss_x))
                if nuclei_gauss_on else (0.0, 0.0, 0.0)
            ),
            neurite_gaussian_sigma=(
                (float(neurite_gauss_z), float(neurite_gauss_y), float(neurite_gauss_x))
                if neurite_gauss_on else (0.0, 0.0, 0.0)
            ),
            cilia_log=bool(cilia_log),
            nuclei_log=bool(nuclei_log),
            neurite_log=bool(neurite_log),
            bb_log=bool(bb_log),
            cilia_min_size=int(cilia_min_size),
            cilia_max_size=int(cilia_max_size),
            bb_spot_sigma=float(bb_spot_sigma),
            bb_outline_sigma=float(bb_outline_sigma),
            bb_gaussian_sigma=(
                (float(bb_gauss_z), float(bb_gauss_y), float(bb_gauss_x))
                if bb_gauss_on else (0.0, 0.0, 0.0)
            ),
            bb_min_size=int(bb_min_size),
            bb_max_size=int(bb_max_size),
            ratio_epsilon=float(ratio_epsilon),
        )

        _save_to_path_history("input", input_path)
        _save_to_path_history("output", output_path)
        _save_to_path_history("classifier", classifier_path)

        threading.Thread(
            target=_run_pipeline_thread,
            args=(_params, _log_list, _result),
            daemon=True,
        ).start()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED PATHS — resolve latest lc-analysis-* run directory automatically
# ─────────────────────────────────────────────────────────────────────────────


_sel_run_name = st.session_state.get("_sel_run")
_run_dir = (
    os.path.join(output_path, _sel_run_name)
    if _sel_run_name and os.path.isdir(os.path.join(output_path, str(_sel_run_name)))
    else _find_latest_run_dir(output_path) or output_path
)
_csv_dir = os.path.join(_run_dir, "csv")
_fig_dir = os.path.join(_run_dir, "figures")
_overlay_dir = os.path.join(_fig_dir, "overlays")
_mip_dir = os.path.join(_fig_dir, "mips")
_excel_path = os.path.join(_csv_dir, "all_cilia_features.xlsx")
_params_json_path = os.path.join(_csv_dir, "run_parameters.json")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA + APPLY POST-FILTERS (central — used by every result tab)
# ─────────────────────────────────────────────────────────────────────────────
_data_ready = os.path.exists(_excel_path)

if _data_ready:
    _mtime = os.path.getmtime(_excel_path)
    _df_all = _load_excel(_excel_path, _mtime, "all_data")
    _df_cil_raw = (
        _df_all[_df_all["object_type"] == "cilia"].copy()
        if "object_type" in _df_all.columns else _df_all.copy()
    )
    _df_kept, _df_removed = _apply_post_filters(
        _df_cil_raw,
        axon_thr=float(pf_axon_thr),
        soma_thr=float(pf_soma_thr),
        nn_min_dist=float(pf_nn_min_dist),
        nn_radius=float(pf_nn_radius),
        nn_max_count=int(pf_nn_max_count),
    )
    _nn_removed = len(_df_removed)
    _x_col = "file_short" if "file_short" in _df_kept.columns else "filename"
    _df_bb = (
        _df_all[_df_all["object_type"] == "basal_body"].copy()
        if "object_type" in _df_all.columns else pd.DataFrame()
    )
    if not _df_bb.empty and "log_ratio" in _df_bb.columns:
        _df_bb["class"] = _reclassify(
            _df_bb["log_ratio"], float(pf_axon_thr), float(pf_soma_thr)
        )
else:
    _df_all = _df_cil_raw = pd.DataFrame()
    _df_kept = _df_removed = _df_bb = pd.DataFrame()
    _nn_removed = 0
    _x_col = "filename"

# Fill the NN count badge in the sidebar (the placeholder was created above)
_nn_active = (float(pf_nn_min_dist) > 0.0) or (int(pf_nn_max_count) < 1000)
if _nn_active:
    _nn_count_slot.metric("🗑 Removed by NN filter", _nn_removed)
else:
    _nn_count_slot.caption("NN filter inactive (default settings)")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_run, tab_tables, tab_graphs, tab_overlays, tab_logs = st.tabs(
    ["▶ Run Pipeline", "📋 Data Tables", "📊 Graphs", "🔬 Overlays", "📋 Live Logs"]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
with tab_run:
    if st.session_state.pipeline_running:
        _tr = st.session_state.thread_result
        _n = _tr.get("n_files", 0)
        _idx = _tr.get("file_idx", 0)
        _curr = _tr.get("current_file", "")
        _prog = _tr.get("progress", 0.0)
        if _n > 0:
            st.progress(
                _prog,
                text=f"Processing file **{_idx + 1} / {_n}** — `{_curr}`",
            )
        else:
            st.progress(0.0, text="Initializing…")

    st.subheader("Output Summary")
    if _run_dir != output_path:
        st.caption(f"Active run: `{os.path.basename(_run_dir)}`")
    _m1, _m2, _m3 = st.columns(3)
    with _m1:
        if os.path.exists(_excel_path):
            _ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(_excel_path)))
            st.metric("Excel file", "Found", delta=_ts)
        else:
            st.metric("Excel file", "Not found")
    with _m2:
        _n_figs = len(glob.glob(os.path.join(_fig_dir, "*.png"))) if os.path.exists(_fig_dir) else 0
        st.metric("Figures", _n_figs)
    with _m3:
        _n_ovl = len(glob.glob(os.path.join(_overlay_dir, "*.png"))) if os.path.exists(_overlay_dir) else 0
        st.metric("Overlays", _n_ovl)

    if _data_ready:
        st.markdown("---")
        with open(_excel_path, "rb") as _fh:
            st.download_button(
                "⬇️ Download Full Results (Excel)", _fh.read(), "results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_run_excel",
            )
    else:
        st.info(
            "No data yet — configure parameters in the sidebar and click **Run Pipeline**. "
            "Results will appear in the **Data Tables**, **Graphs**, and **Overlays** tabs."
        )

    st.markdown("---")

    # ── Run Parameters JSON ───────────────────────────────────────────────────
    st.subheader("📄 Run Parameters")

    _json_exists = os.path.exists(_params_json_path)
    if _json_exists:
        with open(_params_json_path) as _jf:
            _json_data = json.load(_jf)
        _json_ts = _json_data.get("timestamp", "unknown")
        st.caption(f"Last run: **{_json_ts}**  ·  `{_params_json_path}`")

        with st.expander("View parameters", expanded=False):
            st.json(_json_data)

        _jr1, _jr2 = st.columns(2)
        with _jr1:
            st.download_button(
                "⬇️ Download parameters (JSON)",
                json.dumps(_json_data, indent=2).encode(),
                "run_parameters.json", "application/json",
                key="dl_params_json",
            )
        with _jr2:
            if st.button("↩️ Reload these parameters into sidebar", key="reload_params"):
                _apply_json_params(_json_data)
                st.success("Parameters loaded — sidebar controls updated.")
                st.rerun()
    else:
        st.info("No parameter file found yet. Run the pipeline to generate one.")

    st.markdown("---")
    st.subheader("📂 Load Parameters from File")
    st.caption("Load a previously saved `run_parameters.json` to restore any run's settings.")
    _load_json_path = st.text_input(
        "Path to run_parameters.json", placeholder="Paste path here…", key="_load_json_path"
    )
    if _load_json_path and os.path.exists(_load_json_path):
        if st.button("↩️ Load parameters", key="load_params_btn"):
            try:
                with open(_load_json_path) as _lf:
                    _load_data = json.load(_lf)
                _apply_json_params(_load_data)
                st.success(f"Loaded parameters from `{_load_json_path}`.")
                st.rerun()
            except Exception as _le:
                st.error(f"Could not load file: {_le}")
    elif _load_json_path:
        st.warning("File not found.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA TABLES
# ═════════════════════════════════════════════════════════════════════════════
with tab_tables:
    if not _data_ready:
        st.info("No results found — run the pipeline first.")
    else:
        if _nn_active:
            st.info(
                f"🔵 NN filter active — **{len(_df_kept):,}** cilia kept, "
                f"**{_nn_removed}** removed. Cilia counts and stats in both tables reflect this filter."
            )

        # ── Prepare combined per-object dataframe ─────────────────────────────
        # _df_bb is already computed and reclassified at the top-level data section
        _df_objects = (
            pd.concat([_df_kept, _df_bb], ignore_index=True)
            if not _df_bb.empty else _df_kept.copy()
        )

        # Promote key columns to the front
        _obj_priority = [
            "file_short", "filename", "object_type", "cilia_id", "class",
            "distance_to_neurite_um", "ratio", "log_ratio",
            "dt_neurite", "dt_nuclei", "log_dt_neurite", "log_dt_nuclei", "coords",
        ]
        _obj_cols = [c for c in _obj_priority if c in _df_objects.columns]
        _obj_extra = [c for c in _df_objects.columns if c not in _obj_cols]
        _df_objects = _df_objects[_obj_cols + _obj_extra]

        # ── Prepare per-sample dataframe ──────────────────────────────────────
        _sample_rows = []
        for _fname in sorted(_df_all["filename"].dropna().unique()):
            _short = (
                _df_all.loc[_df_all["filename"] == _fname, "file_short"].iloc[0]
                if "file_short" in _df_all.columns else _fname
            )
            _cilia_s = (
                _df_kept[_df_kept["filename"] == _fname]
                if not _df_kept.empty and "filename" in _df_kept.columns
                else pd.DataFrame()
            )
            _bb_s = (
                _df_bb[_df_bb["filename"] == _fname]
                if not _df_bb.empty and "filename" in _df_bb.columns
                else pd.DataFrame()
            )
            _row: dict = {
                "file_short": _short,
                "filename": _fname,
                "n_cilia": len(_cilia_s),
                "n_basal_bodies": len(_bb_s),
                "n_nuclei": "—",  # not tracked by current pipeline
            }
            if len(_cilia_s) > 0:
                for _col, _stat, _key in [
                    ("ratio",                  "mean", "mean_ratio"),
                    ("ratio",                  "std",  "std_ratio"),
                    ("log_ratio",              "mean", "mean_log_ratio"),
                    ("log_ratio",              "std",  "std_log_ratio"),
                    ("distance_to_neurite_um", "mean", "mean_distance_um"),
                    ("distance_to_neurite_um", "max",  "max_distance_um"),
                ]:
                    if _col in _cilia_s.columns:
                        _series = _cilia_s[_col].replace([np.inf, -np.inf], np.nan)
                        _val = getattr(_series, _stat)()
                        _row[_key] = round(float(_val), 4) if pd.notna(_val) else None
                    else:
                        _row[_key] = None
            else:
                for _key in ["mean_ratio", "std_ratio", "mean_log_ratio",
                             "std_log_ratio", "mean_distance_um", "max_distance_um"]:
                    _row[_key] = None
            _sample_rows.append(_row)
        _df_per_sample = pd.DataFrame(_sample_rows) if _sample_rows else pd.DataFrame()

        # ── Table 1: Per Object ───────────────────────────────────────────────
        st.subheader("🔬 Per-Object Table (Cilia & Basal Bodies)")
        st.caption(
            f"**{len(_df_kept):,}** cilia (post-filter) · "
            f"**{len(_df_bb):,}** basal bodies (unfiltered) · "
            f"**{_nn_removed}** cilia excluded by NN filter"
        )

        with st.expander("🔍 Filters", expanded=False):
            _fc1, _fc2, _fc3 = st.columns(3)
            with _fc1:
                _otypes = (
                    ["All"] + sorted(_df_objects["object_type"].dropna().unique().tolist())
                    if "object_type" in _df_objects.columns else ["All"]
                )
                _obj_f = st.selectbox("Object type", _otypes, key="t_obj")
            with _fc2:
                _files_t = (
                    ["All"] + sorted(_df_objects["filename"].dropna().unique().tolist())
                    if "filename" in _df_objects.columns else ["All"]
                )
                _file_f = st.selectbox("Sample", _files_t, key="t_file")
            with _fc3:
                _classes_t = (
                    ["All"] + sorted(_df_objects["class"].dropna().unique().tolist())
                    if "class" in _df_objects.columns else ["All"]
                )
                _class_f = st.selectbox("Class", _classes_t, key="t_class")

        _df_obj_filt = _df_objects.copy()
        if _obj_f != "All" and "object_type" in _df_obj_filt.columns:
            _df_obj_filt = _df_obj_filt[_df_obj_filt["object_type"] == _obj_f]
        if _file_f != "All" and "filename" in _df_obj_filt.columns:
            _df_obj_filt = _df_obj_filt[_df_obj_filt["filename"] == _file_f]
        if _class_f != "All" and "class" in _df_obj_filt.columns:
            _df_obj_filt = _df_obj_filt[_df_obj_filt["class"] == _class_f]

        st.caption(f"Showing {len(_df_obj_filt):,} of {len(_df_objects):,} objects")
        st.dataframe(_df_obj_filt, width="stretch", height=320)

        _dlo1, _dlo2 = st.columns(2)
        with _dlo1:
            st.download_button(
                "⬇️ Download filtered (CSV)",
                _df_obj_filt.to_csv(index=False).encode(),
                "objects_filtered.csv", "text/csv", key="dl_obj_filt",
            )
        with _dlo2:
            st.download_button(
                "⬇️ Download all objects (CSV)",
                _df_objects.to_csv(index=False).encode(),
                "objects_all.csv", "text/csv", key="dl_obj_all",
            )

        st.markdown("---")

        # ── Table 2: Per Sample ───────────────────────────────────────────────
        st.subheader("🧫 Per-Sample Summary")
        st.caption(
            "Cilia counts and stats reflect current ⚡ Post-Segmentation Filters. "
            "Basal body counts are unfiltered. "
            "Nuclei count (—) is not currently saved by the pipeline."
        )

        if _df_per_sample.empty:
            st.info("No sample data available.")
        else:
            st.dataframe(_df_per_sample, width="stretch")
            st.download_button(
                "⬇️ Download per-sample summary (CSV)",
                _df_per_sample.to_csv(index=False).encode(),
                "per_sample_summary.csv", "text/csv", key="dl_sample",
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — GRAPHS
# ═════════════════════════════════════════════════════════════════════════════
with tab_graphs:
    if not _data_ready:
        st.info("No results found — run the pipeline first.")
    else:
        if _nn_active:
            st.info(
                f"🔵 NN filter active — **{len(_df_kept):,}** cilia kept, "
                f"**{_nn_removed}** removed"
            )

        # ── Key Metrics 2×2 Grid ──────────────────────────────────────────────
        st.subheader("📊 Key Metrics Overview")

        if _df_kept.empty:
            st.info("No cilia rows remain after current filters.")
        else:
            _GRID = [
                ("Log Ratio Distribution", "hist_lr"),
                ("Ratio Distribution",     "hist_r"),
                ("Log Ratio per Sample",   "box_lr"),
                ("Class Distribution",     "count_class"),
            ]
            _gcols = st.columns(2)

            for _gi, (_gtitle, _gkey) in enumerate(_GRID):
                _fig_g, _ax_g = plt.subplots(figsize=(5, 3.8))
                _ok = True
                try:
                    if _gkey == "hist_lr" and "log_ratio" in _df_kept.columns:
                        sns.histplot(
                            _df_kept["log_ratio"].dropna(), kde=True,
                            ax=_ax_g, color="#2E6FA3", alpha=0.75,
                            line_kws={"linewidth": 1.5, "color": "#1A3F5C"},
                            edgecolor="white", linewidth=0.4,
                        )
                        _ax_g.set_xlabel("Log Ratio", fontsize=9, labelpad=6)
                        _ax_g.set_ylabel("Count", fontsize=9, labelpad=6)

                    elif _gkey == "hist_r" and "ratio" in _df_kept.columns:
                        sns.histplot(
                            _df_kept["ratio"].dropna(), kde=True,
                            ax=_ax_g, color="#C0622F", alpha=0.75,
                            line_kws={"linewidth": 1.5, "color": "#7A3A18"},
                            edgecolor="white", linewidth=0.4,
                            kde_kws={"cut": 0},  # ratio ≥ 0 — don't extend KDE below 0
                        )
                        _ax_g.set_xlim(left=0)
                        _ax_g.set_xlabel("Ratio", fontsize=9, labelpad=6)
                        _ax_g.set_ylabel("Count", fontsize=9, labelpad=6)

                    elif _gkey == "box_lr" and "log_ratio" in _df_kept.columns:
                        sns.violinplot(
                            data=_df_kept, x=_x_col, y="log_ratio",
                            hue=_x_col, legend=False, ax=_ax_g,
                            palette="GnBu",
                            width=0.55,
                            linewidth=0.8,
                            
                        )
                        _ax_g.set_xlabel(_x_col, fontsize=9, labelpad=6)
                        _ax_g.set_ylabel("Log Ratio", fontsize=9, labelpad=6)
                        _ax_g.tick_params(axis="x", rotation=45, labelsize=8)

                    elif _gkey == "count_class" and "class" in _df_kept.columns:
                        sns.countplot(
                            data=_df_kept, x=_x_col, hue="class",
                            ax=_ax_g, palette=_CLASS_PALETTE,
                            edgecolor="white", linewidth=0.4,
                        )
                        _ax_g.set_xlabel(_x_col, fontsize=9, labelpad=6)
                        _ax_g.set_ylabel("Count", fontsize=9, labelpad=6)
                        _ax_g.tick_params(axis="x", rotation=45, labelsize=8)
                        _ax_g.legend(
                            title="Class", title_fontsize=8, fontsize=8,
                            frameon=False,
                            loc="upper left", bbox_to_anchor=(1.0, 1.0),
                        )
                    else:
                        _ok = False

                    if _ok:
                        # ── spine cleanup (remove top + right) ──────────────────────
                        _ax_g.spines["top"].set_visible(False)
                        _ax_g.spines["right"].set_visible(False)
                        _ax_g.spines["left"].set_linewidth(0.7)
                        _ax_g.spines["bottom"].set_linewidth(0.7)

                        # ── tick aesthetics ──────────────────────────────────────────
                        _ax_g.tick_params(
                            axis="both", which="both",
                            labelsize=8, length=3, width=0.7,
                            direction="out",
                        )

                        # ── subtle grid (horizontal only) ───────────────────────────
                        _ax_g.yaxis.grid(True, linestyle="--", linewidth=0.5,
                                        color="#CCCCCC", alpha=0.7, zorder=0)
                        _ax_g.set_axisbelow(True)
                        _ax_g.xaxis.grid(False)

                        # ── title ────────────────────────────────────────────────────
                        _ax_g.set_title(
                            _gtitle, fontsize=10, fontweight="bold",
                            pad=8, loc="left",
                        )

                        plt.tight_layout()
                except Exception as _ge:
                    _ok = False
                    with _gcols[_gi % 2]:
                        st.warning(f"Could not render '{_gtitle}': {_ge}")

                if _ok:
                    _png_g = _fig_to_png(_fig_g)
                    with _gcols[_gi % 2]:
                        st.pyplot(_fig_g, width="stretch")
                        st.download_button(
                            f"⬇️ {_gtitle} (PNG)", _png_g,
                            f"{_gtitle.lower().replace(' ', '_')}.png",
                            "image/png", key=f"dl_grid_{_gi}",
                        )
                plt.close(_fig_g)

        # ── Ratio Distribution per Sample ─────────────────────────────────────
        if "ratio" in _df_kept.columns and not _df_kept.empty:
            st.markdown("---")
            st.subheader("📊 Ratio Distribution per Sample")
            st.caption(
                "Raw ratio = (dt_nuclei + ε) / (dt_neurite + ε) — no log transform applied."
            )
            _fig_ratio, _ax_ratio = plt.subplots(figsize=(10, 4))
            try:
                _ratio_data = _df_kept[["ratio", _x_col]].dropna()
                sns.violinplot(
                    data=_ratio_data, x=_x_col, y="ratio",
                    hue=_x_col, legend=False, ax=_ax_ratio,
                    palette="Set2", width=0.6, linewidth=0.8,
                    cut=0,  # clip KDE to data range — ratio is strictly ≥ 0
                )
                _ax_ratio.set_ylim(bottom=0)  # ratio is ≥ 0 — no negative axis margin
                _ax_ratio.set_xlabel(_x_col, fontsize=9, labelpad=6)
                _ax_ratio.set_ylabel("Ratio", fontsize=9, labelpad=6)
                _ax_ratio.tick_params(axis="x", rotation=45, labelsize=8)
                _ax_ratio.spines["top"].set_visible(False)
                _ax_ratio.spines["right"].set_visible(False)
                _ax_ratio.yaxis.grid(True, linestyle="--", linewidth=0.5,
                                     color="#CCCCCC", alpha=0.7, zorder=0)
                _ax_ratio.set_axisbelow(True)
                _ax_ratio.set_title(
                    "Ratio per Sample", fontsize=10, fontweight="bold", pad=8, loc="left"
                )
                plt.tight_layout()
                _png_ratio = _fig_to_png(_fig_ratio)
                st.pyplot(_fig_ratio, width="stretch")
                st.download_button(
                    "⬇️ Ratio Distribution per Sample (PNG)", _png_ratio,
                    "ratio_per_sample.png", "image/png", key="dl_ratio_violin",
                )
            except Exception as _re:
                st.warning(f"Could not render ratio distribution: {_re}")
            plt.close(_fig_ratio)

        # ── Log-Ratio KDE per Sample ──────────────────────────────────────────
        if "log_ratio" in _df_kept.columns and not _df_kept.empty:
            st.markdown("---")
            st.subheader("📊 Log-Ratio KDE per Sample")
            st.caption(
                "Kernel density estimate per sample — dashed lines show axon (red) "
                "and soma (blue) classification thresholds."
            )
            _kde_samples = sorted(_df_kept[_x_col].dropna().unique())
            _kde_many = len(_kde_samples) > 12
            # Wide figure when many samples so the plot area is not squeezed
            _kde_figw = max(10, min(16, 8 + len(_kde_samples) * 0.15)) if not _kde_many else 10
            _fig_kde, _ax_kde = plt.subplots(figsize=(_kde_figw, 4))
            try:
                _kde_palette = sns.color_palette("tab10", n_colors=len(_kde_samples))
                for _si, _sname in enumerate(_kde_samples):
                    _sdata = _df_kept[_df_kept[_x_col] == _sname]["log_ratio"].dropna()
                    if len(_sdata) > 1:
                        sns.kdeplot(
                            _sdata, ax=_ax_kde, color=_kde_palette[_si],
                            label=None if _kde_many else str(_sname),
                            linewidth=1.5, fill=False,
                        )
                _ax_kde.axvline(float(pf_axon_thr), color="#C0392B", linestyle="--",
                                linewidth=1.4, label=f"axon > {pf_axon_thr}")
                _ax_kde.axvline(float(pf_soma_thr), color="#2980B9", linestyle="--",
                                linewidth=1.4, label=f"soma < {pf_soma_thr}")
                _ax_kde.set_xlabel("log_ratio", fontsize=9, labelpad=6)
                _ax_kde.set_ylabel("Density", fontsize=9, labelpad=6)
                _ax_kde.spines["top"].set_visible(False)
                _ax_kde.spines["right"].set_visible(False)
                _ax_kde.yaxis.grid(True, linestyle="--", linewidth=0.5,
                                   color="#CCCCCC", alpha=0.7, zorder=0)
                _ax_kde.set_axisbelow(True)
                _n_lbl = f" ({len(_kde_samples)} samples, colors not labelled)" if _kde_many else ""
                _ax_kde.set_title(
                    f"Log-Ratio KDE per Sample{_n_lbl}",
                    fontsize=10, fontweight="bold", pad=8, loc="left",
                )
                if _kde_many:
                    # Only threshold lines in legend — placed inside the axes (top-right)
                    _ax_kde.legend(loc="upper right", fontsize=9, frameon=False)
                else:
                    _ax_kde.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0),
                                   fontsize=8, frameon=False)
                plt.tight_layout()
                _png_kde = _fig_to_png(_fig_kde)
                st.pyplot(_fig_kde, width="stretch")
                st.download_button(
                    "⬇️ Log-Ratio KDE per Sample (PNG)", _png_kde,
                    "logratio_kde_per_sample.png", "image/png", key="dl_kde",
                )
            except Exception as _kde_e:
                st.warning(f"Could not render KDE plot: {_kde_e}")
            plt.close(_fig_kde)

        st.markdown("---")

        # ── Custom Plot Builder ───────────────────────────────────────────────
        st.subheader("🎨 Custom Plot")

        _num_cols = (
            _df_kept.select_dtypes(include=[np.number]).columns.tolist()
            if not _df_kept.empty else []
        )
        _cat_cols = (
            _df_kept.select_dtypes(exclude=[np.number]).columns.tolist()
            if not _df_kept.empty else []
        )

        if not _num_cols:
            st.info("No numeric columns available for plotting.")
        else:
            _cp1, _cp2, _cp3, _cp4 = st.columns(4)
            with _cp1:
                _ptype = st.selectbox(
                    "Plot type",
                    ["histogram", "scatter", "box", "violin", "kde"],
                    key="cp_type",
                )
            with _cp2:
                _def_x = "log_ratio" if "log_ratio" in _num_cols else _num_cols[0]
                _x_axis = st.selectbox(
                    "X axis", _num_cols, index=_num_cols.index(_def_x), key="cp_x"
                )
            with _cp3:
                _y_axis = st.selectbox(
                    "Y axis (scatter)", ["None"] + _num_cols,
                    key="cp_y", disabled=_ptype != "scatter",
                )
            with _cp4:
                _color_by = st.selectbox(
                    "Color by", ["None"] + _cat_cols, key="cp_color"
                )

            _pp1, _pp2, _pp3 = st.columns(3)
            with _pp1:
                _bins = st.slider(
                    "Bins", 5, 200, 50, key="cp_bins",
                    disabled=_ptype != "histogram",
                )
                _show_points = st.checkbox(
                    "Overlay sample points", key="cp_points",
                    help="Overlay individual data points (strip plot) on box/violin plots.",
                    disabled=_ptype not in ("box", "violin"),
                )
            with _pp2:
                _log_x = st.checkbox("Log X scale", key="cp_log_x")
                _log_y = st.checkbox("Log Y scale", key="cp_log_y")
            with _pp3:
                _set_xlim = st.checkbox("Set X limits", key="cp_xlim_on")
                _xl_min = st.number_input("X min", value=0.0, key="cp_xl_min",
                                          disabled=not _set_xlim)
                _xl_max = st.number_input("X max", value=10.0, key="cp_xl_max",
                                          disabled=not _set_xlim)

            if st.button("📈 Generate Plot", key="cp_gen"):
                _hue = None if _color_by == "None" else _color_by
                _cat_x = _hue if _hue else _x_col
                _fig_cp, _ax_cp = plt.subplots(figsize=(9, 5))
                _plot_ok = True
                try:
                    if _ptype == "histogram":
                        sns.histplot(data=_df_kept, x=_x_axis, hue=_hue,
                                     bins=_bins, kde=True, ax=_ax_cp)
                    elif _ptype == "scatter":
                        if _y_axis == "None":
                            st.warning("Select a Y axis for the scatter plot.")
                            _plot_ok = False
                        else:
                            sns.scatterplot(data=_df_kept, x=_x_axis, y=_y_axis,
                                            hue=_hue, ax=_ax_cp, alpha=0.6, s=20)
                    elif _ptype == "box":
                        sns.boxplot(data=_df_kept, x=_cat_x, y=_x_axis,
                                    hue=_cat_x, legend=False, ax=_ax_cp, palette="tab10")
                        if _show_points:
                            sns.stripplot(data=_df_kept, x=_cat_x, y=_x_axis, ax=_ax_cp,
                                          size=3, color="black", alpha=0.45, jitter=True)
                        _ax_cp.tick_params(axis="x", rotation=45)
                    elif _ptype == "violin":
                        sns.violinplot(data=_df_kept, x=_cat_x, y=_x_axis,
                                       hue=_cat_x, legend=False, ax=_ax_cp, palette="tab10",
                                       cut=0)  # clip KDE tails to the data range
                        if _show_points:
                            sns.stripplot(data=_df_kept, x=_cat_x, y=_x_axis, ax=_ax_cp,
                                          size=3, color="black", alpha=0.45, jitter=True)
                        _ax_cp.tick_params(axis="x", rotation=45)
                    elif _ptype == "kde":
                        sns.kdeplot(data=_df_kept, x=_x_axis, hue=_hue,
                                    ax=_ax_cp, common_norm=False)

                    if _plot_ok:
                        if _log_x:
                            _ax_cp.set_xscale("log")
                        if _log_y:
                            _ax_cp.set_yscale("log")
                        if _set_xlim:
                            _ax_cp.set_xlim(_xl_min, _xl_max)
                        plt.tight_layout()
                        st.session_state.custom_plot_png = _fig_to_png(_fig_cp)
                except Exception as _cpe:
                    st.error(f"Plot error: {_cpe}")
                    st.session_state.custom_plot_png = None
                finally:
                    plt.close(_fig_cp)

            if st.session_state.custom_plot_png:
                st.image(st.session_state.custom_plot_png, width="stretch")
                st.download_button(
                    "⬇️ Download plot (PNG)",
                    st.session_state.custom_plot_png,
                    "custom_plot.png", "image/png", key="dl_custom",
                )

        st.markdown("---")

        # ── Interactive Explorer (Plotly) ─────────────────────────────────────
        st.subheader("⚡ Interactive Explorer")

        if not _PLOTLY_AVAILABLE:
            st.info("Install `plotly` to enable the interactive explorer: `pip install plotly`")
        elif _df_kept.empty:
            st.info("No cilia data available.")
        else:
            _num_cols_px = (
                _df_kept.select_dtypes(include=[np.number]).columns.tolist()
            )
            _cat_cols_px = (
                _df_kept.select_dtypes(exclude=[np.number]).columns.tolist()
            )
            _px_kind = st.radio(
                "Plot type",
                ["Scatter", "KDE histogram"],
                horizontal=True,
                key="px_kind",
            )

            _ip1, _ip2, _ip3 = st.columns(3)
            with _ip1:
                _px_x = st.selectbox(
                    "X axis" if _px_kind == "Scatter" else "Variable",
                    _num_cols_px,
                    index=_num_cols_px.index("log_ratio") if "log_ratio" in _num_cols_px else 0,
                    key="px_x",
                )
            with _ip2:
                _px_y_opts = _num_cols_px
                _px_y_def = (
                    _px_y_opts.index("distance_to_neurite_um")
                    if "distance_to_neurite_um" in _px_y_opts
                    else min(1, len(_px_y_opts) - 1)
                )
                _px_y = st.selectbox(
                    "Y axis", _px_y_opts, index=_px_y_def, key="px_y",
                    disabled=(_px_kind != "Scatter"),
                )
            with _ip3:
                _px_color_opts = ["class", _x_col] + [
                    c for c in _cat_cols_px if c not in ("class", _x_col)
                ]
                _px_color = st.selectbox(
                    "Color by" if _px_kind == "Scatter" else "Group by",
                    _px_color_opts, key="px_color",
                )

            if _px_kind == "KDE histogram":
                _n_grp = (
                    int(_df_kept[_px_color].nunique())
                    if _px_color in _df_kept.columns else 1
                )
                _px_hist = st.checkbox(
                    "Show histogram bars",
                    value=(_n_grp <= 2),
                    key="px_hist",
                    help="With many groups the bars overlap into noise — "
                         "turn this off to show only the smooth KDE curves.",
                )

            try:
                if _px_kind == "Scatter":
                    _px_hover = [
                        c for c in ["cilia_id", "filename", "log_ratio",
                                     "dt_neurite", "dt_nuclei", "coords"]
                        if c in _df_kept.columns
                    ]
                    _px_fig = _px.scatter(
                        _df_kept,
                        x=_px_x,
                        y=_px_y,
                        color=_px_color if _px_color in _df_kept.columns else None,
                        hover_data=_px_hover,
                        opacity=0.65,
                        color_discrete_map=(_CLASS_PALETTE if _px_color == "class" else None),
                        height=460,
                    )
                    _px_fig.update_traces(marker_size=5)
                else:
                    # KDE histogram: one density curve + histogram per group.
                    # Drop non-finite values (NaN/±inf) — gaussian_kde rejects them.
                    def _finite(series):
                        _a = series.to_numpy(dtype=float)
                        return _a[np.isfinite(_a)]

                    _grp_data, _grp_labels, _grp_colors = [], [], []
                    if _px_color in _df_kept.columns and _df_kept[_px_color].nunique() > 1:
                        for _g, _sub in _df_kept.groupby(_px_color):
                            _vals = _finite(_sub[_px_x])
                            if _vals.size > 1 and _vals.std() > 0:
                                _grp_data.append(_vals)
                                _grp_labels.append(str(_g))
                                _grp_colors.append(
                                    _CLASS_PALETTE.get(str(_g)) if _px_color == "class" else None
                                )
                    else:
                        _vals = _finite(_df_kept[_px_x])
                        if _vals.size > 1 and _vals.std() > 0:
                            _grp_data.append(_vals)
                            _grp_labels.append(_px_x)
                            _grp_colors.append(None)

                    if not _grp_data:
                        st.info(
                            "Not enough finite, varying data to build a KDE histogram "
                            f"for **{_px_x}**."
                        )
                        raise StopIteration
                    _colors_arg = _grp_colors if all(_grp_colors) else None
                    _px_fig = _ff.create_distplot(
                        _grp_data, _grp_labels,
                        colors=_colors_arg,
                        show_hist=_px_hist,
                        show_rug=False,
                    )
                    _px_fig.update_layout(height=460, xaxis_title=_px_x, yaxis_title="Density")

                _px_fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    margin=dict(l=50, r=160, t=40, b=50),
                    legend=dict(orientation="v", x=1.02, y=1.0),
                    xaxis=dict(showgrid=True, gridcolor="#E5E5E5", zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor="#E5E5E5", zeroline=False),
                )
                st.plotly_chart(_px_fig, width="stretch")
            except StopIteration:
                pass
            except Exception as _pxe:
                st.warning(f"Could not render interactive plot: {_pxe}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — OVERLAYS
# ═════════════════════════════════════════════════════════════════════════════
with tab_overlays:
    st.subheader("🔬 Overlays")
    st.caption(
        "Live overlay — updates with ⚡ Post-Segmentation Filters. "
        "Channel panels show raw MIPs with segmentation contours. "
        "Use the **Custom Overlay Builder** to compose and export a custom figure."
    )

    if not _data_ready or (_df_kept.empty and _df_removed.empty):
        st.info("Run the pipeline first to generate overlay data.")
    else:
        # ── QC PDF report (all samples) ───────────────────────────────────────
        with st.expander("📄 QC PDF Report (all samples)", expanded=False):
            st.caption(
                "One document with, per sample: original channels, segmentations, "
                "all detected cilia/basal bodies, kept cilia, close-up ROIs of every "
                "detected cilium with its outline (for visual QC of segmentation), "
                "plus a summary table of counts and mean areas. Requires a run that "
                "saved channel/label MIPs (older runs show only the channels they stored)."
            )
            _rep_files = sorted(
                set(
                    (_df_kept["filename"].dropna().unique().tolist()
                     if "filename" in _df_kept.columns else [])
                    + (_df_bb["filename"].dropna().unique().tolist()
                       if not _df_bb.empty and "filename" in _df_bb.columns else [])
                )
            )
            if st.button("🛠 Generate report", key="qc_report_btn"):
                if not _rep_files:
                    st.warning("No samples available to report.")
                else:
                    with st.spinner(f"Building QC report for {len(_rep_files)} sample(s)…"):
                        try:
                            _pdf_bytes = _build_qc_report_pdf(
                                _mip_dir, _rep_files, _df_kept, _df_bb, _x_col
                            )
                            st.session_state["_qc_report_pdf"] = _pdf_bytes
                        except Exception as _rep_e:
                            st.error(f"Could not build report: {_rep_e}")
            if st.session_state.get("_qc_report_pdf"):
                st.download_button(
                    "⬇ Download QC report (PDF)",
                    data=st.session_state["_qc_report_pdf"],
                    file_name=f"qc_report_{os.path.basename(_run_dir)}.pdf",
                    mime="application/pdf",
                    key="qc_report_dl",
                )

        # ── Controls ──────────────────────────────────────────────────────────
        _ovc1, _ovc2 = st.columns([3, 2])

        with _ovc1:
            _all_files = sorted(
                set(
                    (_df_kept["filename"].dropna().unique().tolist()
                     if "filename" in _df_kept.columns else [])
                    + (_df_removed["filename"].dropna().unique().tolist()
                       if "filename" in _df_removed.columns else [])
                )
            )
            _sel_file = st.selectbox("Sample", _all_files, key="ov_file") if _all_files else None

        with _ovc2:
            _dot_size = st.slider("Dot size (pts²)", 5, 150, 20, key="ov_dot_size")
            _show_removed = st.checkbox(
                f"Show NN-removed  ({_nn_removed} ×)", value=True, key="ov_removed"
            )

        if not _sel_file:
            st.info("No samples available.")
        else:
            _stem = os.path.splitext(os.path.basename(_sel_file))[0]

            def _sub(df, fname):
                if df.empty or "filename" not in df.columns:
                    return pd.DataFrame()
                return df[df["filename"] == fname].copy()

            _fk = _sub(_df_kept, _sel_file)
            _fr = _sub(_df_removed, _sel_file)

            # Optional Z-slice filter
            if "coords" in _fk.columns and len(_fk) > 0:
                try:
                    _z_arr = np.array([_parse_coords(c)[0] for c in _fk["coords"]], dtype=float)
                    _z_min_v, _z_max_v = int(_z_arr.min()), int(_z_arr.max())
                    if _z_max_v > _z_min_v:
                        _z_lo, _z_hi = st.slider(
                            "Z-slice range (pixels)", _z_min_v, _z_max_v,
                            (_z_min_v, _z_max_v), key="ov_z",
                            help="Show only cilia within this Z range"
                        )
                        _fk = _fk[(_z_arr >= _z_lo) & (_z_arr <= _z_hi)].copy()
                except Exception:
                    pass

            _mip_probe = os.path.join(_mip_dir, f"{_stem}_neurite_mip.npy")
            _mip_mtime = os.path.getmtime(_mip_probe) if os.path.exists(_mip_probe) else 0.0
            _mips = _load_mips(_mip_dir, _stem, _mip_mtime)

            if _mips is None:
                st.warning(
                    "MIP arrays not found — showing the pipeline-saved overlay PNG. "
                    "Re-run the pipeline to enable live overlay updates.",
                    icon="⚠️",
                )
                _bg_path = os.path.join(_overlay_dir, f"{_stem}_overlay.png")
                if not os.path.exists(_bg_path):
                    _cands = glob.glob(os.path.join(_overlay_dir, f"*{_stem}*.png"))
                    _bg_path = _cands[0] if _cands else None
                if _bg_path:
                    st.image(_bg_path, width="stretch")
                else:
                    st.info("No overlay PNG found either.")
            else:
                # ── Shared scatter data ────────────────────────────────────────
                _scores_ov = (
                    _fk["log_ratio"].values.astype(float) if "log_ratio" in _fk.columns
                    else np.array([])
                )
                _valid_ov = _scores_ov[np.isfinite(_scores_ov)] if len(_scores_ov) else np.array([])
                _vmin_ov, _vmax_ov = (
                    (float(np.percentile(_valid_ov, 5)), float(np.percentile(_valid_ov, 95)))
                    if len(_valid_ov) > 1 else (0.0, 1.0)
                )

                def _scatter_cilia(ax, dot_sz):
                    if not _fk.empty and "coords" in _fk.columns and len(_scores_ov) > 0:
                        try:
                            _ck = np.array([_parse_coords(c) for c in _fk["coords"]])
                            ax.scatter(_ck[:, 2], _ck[:, 1],
                                       c=_scores_ov, cmap="coolwarm",
                                       vmin=_vmin_ov, vmax=_vmax_ov,
                                       s=dot_sz, edgecolor="black", linewidth=0.3, zorder=4)
                        except Exception:
                            pass
                    if _show_removed and not _fr.empty and "coords" in _fr.columns:
                        try:
                            _cr = np.array([_parse_coords(c) for c in _fr["coords"]])
                            ax.scatter(_cr[:, 2], _cr[:, 1], c="red",
                                       s=dot_sz, marker="x", linewidth=0.8, alpha=0.7, zorder=4)
                        except Exception:
                            pass

                # ── Section A: Channel MIPs ────────────────────────────────────
                st.markdown("#### 📡 Channel MIPs")
                # alpha: fill opacity for the segmentation colour overlay
                # (higher = more visible; large structures like nuclei use low alpha)
                _ch_defs = [
                    ("Cilia",        "cilia",   "cilia_labels",  "#00FF88", 0.55),
                    ("Neurites",     "neurite",  None,            "#FF44FF", 0.28),
                    ("Basal Bodies", "bb",       "bb_labels",     "#FF8800", 0.55),
                    ("Nuclei",       "nuclei",   "nuclei_labels", "#FFFF00", 0.20),
                ]
                _ch_avail = [(n, mk, lk, oc, al) for n, mk, lk, oc, al in _ch_defs if mk in _mips]

                _missing_new = [n for n, mk, *_ in _ch_defs if mk not in _mips]
                if _missing_new:
                    st.caption(
                        f"ℹ️ **{', '.join(_missing_new)}** "
                        "channel(s) and segmentation outlines require a pipeline re-run "
                        "with the updated code to generate the extra MIP files."
                    )

                with st.expander("🔆 Brightness / contrast", expanded=False):
                    _ch_plo, _ch_phi = st.slider(
                        "Clip range (percentile)", 0, 100, (0, 100), step=1,
                        key="ch_br",
                        help="Clips the display range of all channel panels. "
                             "Drag left handle up to brighten dim channels.",
                    )

                def _seg_overlay(ax, mask, hex_color, alpha):
                    """Draw a coloured semi-transparent fill over segmented pixels."""
                    if not mask.any():
                        return
                    r = int(hex_color[1:3], 16) / 255
                    g = int(hex_color[3:5], 16) / 255
                    b = int(hex_color[5:7], 16) / 255
                    ov = np.zeros((*mask.shape, 4), dtype=np.float32)
                    ov[mask] = [r, g, b, alpha]
                    ax.imshow(ov, aspect="equal", zorder=2)

                if _ch_avail:
                    _ch_cols = st.columns(len(_ch_avail))
                    for _ci, (_ch_name, _ch_mkey, _lbl_key, _oc, _al) in enumerate(_ch_avail):
                        with _ch_cols[_ci]:
                            _arr = _mips[_ch_mkey]
                            _vlo = np.percentile(_arr, _ch_plo)
                            _vhi = np.percentile(_arr, _ch_phi)
                            _fig_ch, _ax_ch = plt.subplots(figsize=(4, 4))
                            _ax_ch.imshow(_arr, cmap="gray",
                                          vmin=_vlo, vmax=_vhi, aspect="equal")
                            _ax_ch.set_title(_ch_name, fontsize=9, pad=4)
                            _ax_ch.axis("off")
                            if _lbl_key and _lbl_key in _mips:
                                _seg_overlay(_ax_ch, _mips[_lbl_key] > 0, _oc, _al)
                            elif _ch_mkey == "neurite" and "neurite_mask" in _mips:
                                _seg_overlay(_ax_ch, _mips["neurite_mask"], _oc, _al)
                            _scatter_cilia(_ax_ch, max(4, _dot_size // 2))
                            plt.tight_layout(pad=0.2)
                            st.pyplot(_fig_ch, width="stretch")
                            plt.close(_fig_ch)
                else:
                    st.caption("No channel MIPs available — re-run the pipeline to generate them.")

                # ── Section B: log(ratio) Heatmap ─────────────────────────────
                st.markdown("#### 🌡️ log(ratio) Heatmap")
                with np.errstate(divide="ignore", invalid="ignore"):
                    _ratio_log_ov = np.where(
                        np.isfinite(_mips["ratio"]), np.log(_mips["ratio"]), np.nan
                    )
                _cmap_ratio = plt.cm.coolwarm.copy()
                _cmap_ratio.set_bad("black")
                _fig_rat, _ax_rat = plt.subplots(figsize=(8, 6))
                _im_rat = _ax_rat.imshow(_ratio_log_ov, cmap=_cmap_ratio, aspect="equal")
                plt.colorbar(_im_rat, ax=_ax_rat, label="log(ratio)", shrink=0.8)
                _ax_rat.set_title("log(ratio) — neurite masked", fontsize=10)
                _ax_rat.axis("off")
                _scatter_cilia(_ax_rat, _dot_size)
                if "neurite_mask" in _mips:
                    _ax_rat.contour(_mips["neurite_mask"].astype(float),
                                    levels=[0.5], colors=["#FF44FF"], linewidths=0.5, alpha=0.5)
                plt.tight_layout()
                _rat_png = _fig_to_png(_fig_rat)
                plt.close(_fig_rat)
                st.image(_rat_png, width="stretch")

                _ov_fname = (
                    f"{_stem}_overlay"
                    f"_nn{int(pf_nn_min_dist)}"
                    f"_r{int(pf_nn_radius)}"
                    f"_n{int(pf_nn_max_count)}.png"
                )
                _cov1, _cov2 = st.columns(2)
                with _cov1:
                    st.download_button(
                        "⬇️ Download Ratio Overlay (PNG)",
                        _rat_png, _ov_fname, "image/png", key="dl_overlay",
                    )
                with _cov2:
                    st.caption(f"{len(_fk)} kept  ·  {len(_fr)} NN-removed")

                st.markdown("---")

                # ── Section B2: Per-channel intensity histograms ──────────────
                with st.expander("📊 Per-channel intensity histograms", expanded=False):
                    st.caption(
                        "Pixel-intensity distribution of each channel's MIP for this "
                        "sample. Useful for spotting saturation, weak staining, or "
                        "background offsets, and for tuning normalisation percentiles."
                    )
                    _hist_chs = [(_n, _mk, _oc) for _n, _mk, _lk, _oc, _al in _ch_avail]
                    if not _hist_chs:
                        st.info("No channel MIPs available — re-run the pipeline.")
                    else:
                        _hlog = st.checkbox(
                            "Log Y (count) scale", value=True, key="ch_hist_log",
                            help="Log scale makes sparse bright pixels / tails visible.",
                        )
                        _ncols = len(_hist_chs)
                        _fh, _hax = plt.subplots(
                            1, _ncols, figsize=(3.2 * _ncols, 3.0), squeeze=False
                        )
                        for _ax_h, (_hn, _hk, _hc) in zip(_hax[0], _hist_chs):
                            _vals = np.asarray(_mips[_hk], dtype=float).ravel()
                            _vals = _vals[np.isfinite(_vals)]
                            if _vals.size:
                                _ax_h.hist(_vals, bins=80, color=_hc,
                                           alpha=0.85, edgecolor="none")
                            if _hlog:
                                _ax_h.set_yscale("log")
                            _ax_h.set_title(_hn, fontsize=9, fontweight="bold")
                            _ax_h.set_xlabel("intensity", fontsize=8)
                            _ax_h.tick_params(labelsize=7)
                            _ax_h.spines[["top", "right"]].set_visible(False)
                        _hax[0][0].set_ylabel("count", fontsize=8)
                        _fh.tight_layout()
                        st.pyplot(_fh, width="stretch")
                        st.download_button(
                            "⬇️ Intensity histograms (PNG)", _fig_to_png(_fh),
                            f"{_stem}_intensity_histograms.png", "image/png",
                            key="dl_ch_hist",
                        )
                        plt.close(_fh)

                # ── Section C: Custom Overlay Builder ─────────────────────────
                with st.expander("🎨 Custom Overlay Builder", expanded=False):
                    _b1, _b2, _b3 = st.columns(3)
                    with _b1:
                        st.markdown("**Layers**")
                        _bld_cilia_ch   = st.checkbox("Cilia channel",        value=True,  key="bld_cilia_ch")
                        _bld_neurite_ch = st.checkbox("Neurite channel",      value=True,  key="bld_neurite_ch")
                        _bld_nuclei_ch  = st.checkbox("Nuclei channel",       value=False, key="bld_nuclei_ch")
                        _bld_bb_ch      = st.checkbox("BB channel",           value=False, key="bld_bb_ch")
                        _bld_ratio_pan  = st.checkbox("Ratio heatmap panel",  value=True,  key="bld_ratio_pan")
                        _bld_nmask_lay  = st.checkbox("Neurite mask overlay", value=True,  key="bld_nmask_lay")
                        _bld_outlines   = st.checkbox("Seg. outlines",        value=True,  key="bld_outlines")
                        _bld_dots       = st.checkbox("Cilia dots (kept)",    value=True,  key="bld_dots")
                        _bld_rmvd       = st.checkbox("Cilia dots (removed)", value=True,  key="bld_rmvd")
                    with _b2:
                        st.markdown("**Colors**")
                        _c_nmask   = st.color_picker("Neurite mask",    "#FF44FF", key="bld_c_nmask")
                        _c_cil_ol  = st.color_picker("Cilia outlines",  "#00FF88", key="bld_c_cil")
                        _c_nuc_ol  = st.color_picker("Nuclei outlines", "#FFFF00", key="bld_c_nuc")
                        _c_bb_ol   = st.color_picker("BB outlines",     "#FF8800", key="bld_c_bb")
                        _c_removed = st.color_picker("Removed dots",    "#FF0000", key="bld_c_rmvd")
                        _bld_cmap  = st.selectbox(
                            "Ratio colormap",
                            ["coolwarm", "viridis", "plasma", "RdBu_r", "PiYG"],
                            key="bld_cmap",
                        )
                    with _b3:
                        st.markdown("**Display**")
                        _bld_dot_sz   = st.slider("Dot size", 5, 150, _dot_size,   key="bld_dsz")
                        _bld_nalpha   = st.slider("Mask alpha", 0.0, 1.0, 0.3, step=0.05, key="bld_nalpha")
                        _bld_vmin_pct = st.slider("Image vmin %", 0, 50, 0,        key="bld_vmin")
                        _bld_vmax_pct = st.slider("Image vmax %", 50, 100, 100,    key="bld_vmax")
                        _bld_figw     = st.slider("Fig width (in)", 6, 28, 14,     key="bld_figw")
                        _bld_figh     = st.slider("Fig height (in)", 4, 20, 8,     key="bld_figh")
                        _bld_fmt      = st.radio("Export format", ["PNG", "PDF"],  key="bld_fmt")

                    if st.button("🖼️ Build Custom Overlay", key="bld_run", type="primary"):
                        _bld_panels = []
                        if _bld_cilia_ch   and "cilia"   in _mips: _bld_panels.append(("Cilia",        "cilia"))
                        if _bld_neurite_ch and "neurite" in _mips: _bld_panels.append(("Neurites",     "neurite"))
                        if _bld_bb_ch      and "bb"      in _mips: _bld_panels.append(("Basal Bodies", "bb"))
                        if _bld_nuclei_ch  and "nuclei"  in _mips: _bld_panels.append(("Nuclei",       "nuclei"))
                        if _bld_ratio_pan:                          _bld_panels.append(("log(ratio)",   "ratio"))

                        _n_pan = max(1, len(_bld_panels))
                        _bld_fig, _bld_axs = plt.subplots(
                            1, _n_pan, figsize=(_bld_figw, _bld_figh), squeeze=False,
                        )
                        _bld_axs = _bld_axs[0]

                        def _norm_ch(arr):
                            lo = np.percentile(arr, _bld_vmin_pct)
                            hi = np.percentile(arr, _bld_vmax_pct)
                            return (arr - lo) / (hi - lo + 1e-8)

                        _outline_color_map = {
                            "cilia":   _c_cil_ol,
                            "neurite": _c_nmask,
                            "bb":      _c_bb_ol,
                            "nuclei":  _c_nuc_ol,
                        }

                        for _pi, (_ptitle, _pkey) in enumerate(_bld_panels):
                            _bax = _bld_axs[_pi]
                            _bax.set_title(_ptitle, fontsize=9, pad=4)
                            _bax.axis("off")

                            if _pkey == "ratio":
                                _r_log = np.where(
                                    np.isfinite(_mips["ratio"]), np.log(_mips["ratio"]), np.nan
                                )
                                _cm_r = plt.get_cmap(_bld_cmap).copy()
                                _cm_r.set_bad("black")
                                _im_b = _bax.imshow(_r_log, cmap=_cm_r, aspect="equal")
                                plt.colorbar(_im_b, ax=_bax, label="log(ratio)", shrink=0.8)
                            else:
                                _bax.imshow(_norm_ch(_mips[_pkey]),
                                            cmap="gray", aspect="equal", vmin=0, vmax=1)
                                if _bld_outlines:
                                    _lk = _pkey + "_labels"
                                    _oc_bld = _outline_color_map.get(_pkey, "#FFFFFF")
                                    _al_bld = 0.55 if _pkey in ("cilia", "bb") else 0.25
                                    if _lk in _mips and _mips[_lk] is not None:
                                        _lm = _mips[_lk] > 0
                                        if _lm.any():
                                            _seg_overlay(_bax, _lm, _oc_bld, _al_bld)
                                    elif _pkey == "neurite" and "neurite_mask" in _mips:
                                        _seg_overlay(_bax, _mips["neurite_mask"], _c_nmask, 0.28)
                                if _bld_nmask_lay and "neurite_mask" in _mips:
                                    _r_hex = int(_c_nmask[1:3], 16) / 255
                                    _g_hex = int(_c_nmask[3:5], 16) / 255
                                    _b_hex = int(_c_nmask[5:7], 16) / 255
                                    _mask_rgba = np.zeros(
                                        (*_mips["neurite_mask"].shape, 4), dtype=np.float32
                                    )
                                    _mask_rgba[_mips["neurite_mask"]] = [_r_hex, _g_hex, _b_hex, _bld_nalpha]
                                    _bax.imshow(_mask_rgba, aspect="equal")

                            if _bld_dots and not _fk.empty and "coords" in _fk.columns and len(_scores_ov) > 0:
                                try:
                                    _ck2 = np.array([_parse_coords(c) for c in _fk["coords"]])
                                    _bax.scatter(
                                        _ck2[:, 2], _ck2[:, 1],
                                        c=_scores_ov, cmap="coolwarm",
                                        vmin=_vmin_ov, vmax=_vmax_ov,
                                        s=_bld_dot_sz, edgecolor="black", linewidth=0.3, zorder=4,
                                    )
                                except Exception:
                                    pass
                            if _bld_rmvd and not _fr.empty and "coords" in _fr.columns:
                                try:
                                    _cr2 = np.array([_parse_coords(c) for c in _fr["coords"]])
                                    _bax.scatter(
                                        _cr2[:, 2], _cr2[:, 1],
                                        c=_c_removed, s=_bld_dot_sz,
                                        marker="x", linewidth=0.8, alpha=0.7, zorder=4,
                                    )
                                except Exception:
                                    pass

                        plt.tight_layout()
                        if _bld_fmt == "PDF":
                            _bld_bytes = _fig_to_pdf(_bld_fig)
                            _bld_ext   = "pdf"
                            _bld_mime  = "application/pdf"
                        else:
                            _bld_bytes = _fig_to_png(_bld_fig)
                            _bld_ext   = "png"
                            _bld_mime  = "image/png"
                        plt.close(_bld_fig)
                        st.session_state["_custom_ov_bytes"] = _bld_bytes
                        st.session_state["_custom_ov_ext"]   = _bld_ext
                        st.session_state["_custom_ov_mime"]  = _bld_mime

                    if st.session_state.get("_custom_ov_bytes"):
                        _ext_ov = st.session_state["_custom_ov_ext"]
                        if _ext_ov == "png":
                            st.image(st.session_state["_custom_ov_bytes"], width="stretch")
                        else:
                            st.info("PDF preview not available — use the download button below.")
                        st.download_button(
                            f"⬇️ Download Custom Overlay ({_ext_ov.upper()})",
                            st.session_state["_custom_ov_bytes"],
                            f"{_stem}_custom_overlay.{_ext_ov}",
                            st.session_state["_custom_ov_mime"],
                            key="dl_custom_ov",
                        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — LIVE LOGS
# ═════════════════════════════════════════════════════════════════════════════
with tab_logs:
    st.subheader("📋 Live Logs")

    _lc1, _lc2, _ = st.columns([1, 1, 5])
    with _lc1:
        if st.button("🗑️ Clear", key="log_clear"):
            st.session_state.logs = []
            st.rerun()
    with _lc2:
        _pause_label = "▶ Resume" if st.session_state.logs_paused else "⏸ Pause"
        if st.button(_pause_label, key="log_pause"):
            st.session_state.logs_paused = not st.session_state.logs_paused
            st.rerun()

    if st.session_state.logs:
        st.code("\n".join(st.session_state.logs[-500:]), language=None)
        st.caption(f"{len(st.session_state.logs)} lines total · showing last 500")
    else:
        st.info("No logs yet — run the pipeline to see output here.")

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-RERUN WHILE PIPELINE IS RUNNING (unless logs are paused)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.pipeline_running and not st.session_state.logs_paused:
    time.sleep(0.5)
    st.rerun()
