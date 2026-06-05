import os
os.environ["PYOPENCL_NO_CACHE"] = "1"

import ast
import json
import re
import sys
import time
import glob
import threading
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.spatial import KDTree

from limoncello.analysis.pipeline import run_pipeline3

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
                st.pyplot(_tfig, use_container_width=True)
                plt.close(_tfig)
            _rlbl = _ch_role.get(i, "—")
            st.caption(f"Ch {i}" + (f" · {_rlbl}" if _rlbl != "—" else ""))
            # Button click triggers a rerun WITHOUT st.rerun() so dialog stays open
            if st.button("Select", key=f"_dsel_{i}", use_container_width=True,
                         type="primary" if i == ch else "secondary"):
                st.session_state.dialog_ch_idx = i

    st.divider()

    # ── Large view + arrow navigation ─────────────────────────────────────────
    _nl, _nc, _nr = st.columns([1, 6, 1])
    with _nl:
        if st.button("◀", key="_dprev", disabled=(ch == 0), use_container_width=True):
            st.session_state.dialog_ch_idx = ch - 1
    with _nc:
        _current_role = _ch_role.get(ch, "unassigned")
        st.markdown(f"#### Channel {ch} — *{_current_role}*")
    with _nr:
        if st.button("▶", key="_dnext", disabled=(ch == n_channels - 1), use_container_width=True):
            st.session_state.dialog_ch_idx = ch + 1

    if previews is not None:
        _mfig, _max = plt.subplots(figsize=(9, 5))
        _max.imshow(previews[ch], cmap="gray", aspect="equal")
        _max.axis("off")
        plt.tight_layout(pad=0.1)
        st.pyplot(_mfig, use_container_width=True)
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

    if st.button("✅ Apply Assignments", type="primary", use_container_width=True):
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
    # Neurite mask MIP is optional (saved by newer pipeline runs)
    _mask_path = os.path.join(mip_dir, f"{stem}_neurite_mask_mip.npy")
    if os.path.exists(_mask_path):
        result["neurite_mask"] = np.load(_mask_path).astype(bool)
    return result


def _fig_to_png(fig: plt.Figure) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
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
    (("basal_bodies", "spot_sigma"),                       "bb_spot_sigma",        float),
    (("basal_bodies", "outline_sigma"),                    "bb_outline_sigma",     float),
    (("basal_bodies", "min_size"),                         "bb_min_size",          int),
    (("basal_bodies", "max_size"),                         "bb_max_size",          int),
    (("distance_thresholds", "max_cilia_um"),              "max_cilia",            float),
    (("distance_thresholds", "max_basal_body_um"),         "max_basal",            float),
    (("distance_thresholds", "ratio_epsilon"),             "ratio_epsilon",        float),
    (("classification", "axon_threshold"),                 "pf_axon_thr",          float),
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
    # Segmentation parameters (keyed so JSON reload works)
    "p_low":               2,
    "p_high":              98,
    "nuclei_sigma":        15,
    "tophat_radius":       12,
    "nuclei_outline_sigma": 3,
    "neurite_sigma":       5,
    "cilia_min_size":      20,
    "cilia_max_size":      0,
    "bb_spot_sigma":       2.0,
    "bb_outline_sigma":    2.0,
    "bb_gauss_z":          1.0,
    "bb_gauss_y":          1.0,
    "bb_gauss_x":          0.0,
    "bb_min_size":         5,
    "bb_max_size":         0,
    "max_cilia":           2.0,
    "max_basal":           2.0,
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
    _gs = _qd.get("basal_bodies", {}).get("gaussian_sigma")
    if _gs and len(_gs) >= 3:
        st.session_state["bb_gauss_z"] = float(_gs[0])
        st.session_state["bb_gauss_y"] = float(_gs[1])
        st.session_state["bb_gauss_x"] = float(_gs[2])

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
                     use_container_width=True, type="primary"):
            # Clear dialog assignment state so it initialises from current sidebar values
            for _, _sk, _ in _DIALOG_ROLES:
                st.session_state.pop(f"_dassign_{_sk}", None)
            _channel_assignment_dialog(input_path, _file_info["filename"], _n_ch)
    else:
        st.info("Enter a valid input folder to open the channel viewer.")

    # Fallback: compact selectboxes for quick edits without opening the dialog
    with st.expander("⚙️ Manual override", expanded=False):
        cilia_channel = st.selectbox(
            "Cilia", _ch_opts, index=min(0, _n_ch - 1), key="ch_cilia",
            help="🔄 Raw channel fed to the ML cilia segmenter",
        )
        neurites_channel = st.selectbox(
            "Neurites", _ch_opts, index=min(1, _n_ch - 1), key="ch_neurites",
            help="🔄 Normalised channel used for neurite tracing",
        )
        basal_bodies_channel = st.selectbox(
            "Basal bodies", _ch_opts, index=min(2, _n_ch - 1), key="ch_bb",
            help="🔄 Raw channel used for basal body detection",
        )
        nuclei_channel = st.selectbox(
            "Nuclei (DAPI)", _ch_opts, index=min(3, _n_ch - 1), key="ch_nuclei",
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

    with st.expander("🧵 Neurites", expanded=True):
        neurite_sigma = st.slider(
            "Spot sigma", 1, 20, step=1, key="neurite_sigma",
            help="🔄 Voronoi-Otsu object-separation scale for neurite detection"
        )

    with st.expander("🎯 Cilia", expanded=True):
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
        st.caption("Gaussian pre-blur (applied before Voronoi-Otsu)")
        bb_gauss_z = st.slider(
            "Gaussian σ_z", 0.0, 5.0, step=0.5, key="bb_gauss_z",
            help="🔄 Gaussian blur sigma along Z before basal body segmentation"
        )
        bb_gauss_y = st.slider(
            "Gaussian σ_y", 0.0, 5.0, step=0.5, key="bb_gauss_y",
            help="🔄 Gaussian blur sigma along Y before basal body segmentation"
        )
        bb_gauss_x = st.slider(
            "Gaussian σ_x", 0.0, 5.0, step=0.5, key="bb_gauss_x",
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
    ratio_epsilon = st.slider(
        "Ratio epsilon (µm)", 0.0, 10.0, step=0.1, key="ratio_epsilon",
        help="🔄 Constant ε added to both numerator and denominator: "
             "ratio = (dt_nuclei + ε) / (dt_neurite + ε). "
             "Prevents division by zero and log(0). Larger ε → smoother ratio."
    )

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
            nuclei_spot_sigma=nuclei_sigma,
            tophat_radius=tophat_radius,
            neurite_spot_sigma=neurite_sigma,
            cilia_classifier_path=classifier_path,
            axon_threshold=float(pf_axon_thr),
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
            cilia_min_size=int(cilia_min_size),
            cilia_max_size=int(cilia_max_size),
            bb_spot_sigma=float(bb_spot_sigma),
            bb_outline_sigma=float(bb_outline_sigma),
            bb_gaussian_sigma=(float(bb_gauss_z), float(bb_gauss_y), float(bb_gauss_x)),
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
# DERIVED PATHS
# ─────────────────────────────────────────────────────────────────────────────
# DERIVED PATHS — resolve latest lc-analysis-* run directory automatically
# ─────────────────────────────────────────────────────────────────────────────
def _find_latest_run_dir(base: str) -> str | None:
    """Return the most recent lc-analysis-YYYY-MM-DD_HH-MM-SS subfolder."""
    if not base or not os.path.isdir(base):
        return None
    runs = sorted(
        d for d in os.listdir(base)
        if d.startswith("lc-analysis-") and os.path.isdir(os.path.join(base, d))
    )
    return os.path.join(base, runs[-1]) if runs else None


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
                        )
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
                            frameon=False, loc="upper right",
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
                )
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
                        _ax_cp.tick_params(axis="x", rotation=45)
                    elif _ptype == "violin":
                        sns.violinplot(data=_df_kept, x=_cat_x, y=_x_axis,
                                       hue=_cat_x, legend=False, ax=_ax_cp, palette="tab10")
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


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — OVERLAYS
# ═════════════════════════════════════════════════════════════════════════════
with tab_overlays:
    st.subheader("🔬 Overlays")
    st.caption(
        "2×2 MIP overlay — updates live as you adjust ⚡ Post-Segmentation Filters. "
        "The pipeline-saved PNG (in the output folder) shows **all** detected cilia before "
        "any NN filter; the interactive overlay below reflects your current filter settings."
    )

    if not _data_ready or (_df_kept.empty and _df_removed.empty):
        st.info("Run the pipeline first to generate overlay data.")
    else:
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

            # Try to load MIP arrays saved by pipeline; fall back to saved PNG
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
                # ── 2×2 MIP-backed figure (identical layout to pipeline.py) ──
                _fig_ov, _axes_ov = plt.subplots(2, 2, figsize=(14, 10))

                with np.errstate(divide="ignore", invalid="ignore"):
                    _ratio_log = np.where(
                        np.isfinite(_mips["ratio"]), np.log(_mips["ratio"]), np.nan
                    )

                # Ratio panel only: NaN outside neurites → black background
                _cmap_ratio = plt.cm.coolwarm.copy()
                _cmap_ratio.set_bad("black")

                # [0,0] Cilia MIP  [0,1] log(ratio) masked
                # [1,0] Nuclei MIP [1,1] Cilia MIP
                _axes_ov[0, 0].imshow(_mips["cilia"], cmap="gray")
                _axes_ov[0, 0].set_title("Cilia MIP")
                _axes_ov[0, 1].imshow(_ratio_log, cmap=_cmap_ratio)
                _axes_ov[0, 1].set_title("log(ratio) — neurite masked")
                _axes_ov[1, 0].imshow(_mips["nuclei"], cmap="gray")
                _axes_ov[1, 0].set_title("Nuclei MIP")
                _axes_ov[1, 1].imshow(_mips["cilia"], cmap="gray")
                _axes_ov[1, 1].set_title("Cilia MIP")

                # Colour scale from kept-cilia log_ratio (5th–95th percentile of finite values)
                _scores_ov = (
                    _fk["log_ratio"].values.astype(float) if "log_ratio" in _fk.columns
                    else np.array([])
                )
                _valid_ov = _scores_ov[np.isfinite(_scores_ov)] if len(_scores_ov) else np.array([])
                _vmin_ov, _vmax_ov = (
                    (float(np.percentile(_valid_ov, 5)), float(np.percentile(_valid_ov, 95)))
                    if len(_valid_ov) > 0 else (0.0, 1.0)
                )

                # Kept cilia — dots on all four panels, no score clamping
                if not _fk.empty and "coords" in _fk.columns and len(_scores_ov) > 0:
                    try:
                        _coords_k = np.array([_parse_coords(c) for c in _fk["coords"]])
                        _ys_k, _xs_k = _coords_k[:, 1], _coords_k[:, 2]
                        for _ax_ov in _axes_ov.flat:
                            _ax_ov.scatter(
                                _xs_k, _ys_k,
                                c=_scores_ov, cmap="coolwarm",
                                vmin=_vmin_ov, vmax=_vmax_ov,
                                s=_dot_size, edgecolor="black", linewidth=0.3,
                                zorder=3,
                            )
                    except Exception as _esc:
                        st.warning(f"Could not plot kept cilia: {_esc}")

                # NN-removed cilia — red × on all 4 panels
                if _show_removed and not _fr.empty and "coords" in _fr.columns:
                    try:
                        _coords_r = np.array([_parse_coords(c) for c in _fr["coords"]])
                        _ys_r, _xs_r = _coords_r[:, 1], _coords_r[:, 2]
                        for _ax_ov in _axes_ov.flat:
                            _ax_ov.scatter(
                                _xs_r, _ys_r, c="red",
                                s=_dot_size, marker="x", linewidth=0.8, alpha=0.7,
                                zorder=3, label="NN-removed",
                            )
                    except Exception as _esr:
                        st.warning(f"Could not plot NN-removed cilia: {_esr}")

                # Colorbars on all 4 panels
                _cmap_ov = plt.cm.coolwarm
                _norm_ov = plt.Normalize(vmin=_vmin_ov, vmax=_vmax_ov)
                _sm_ov = plt.cm.ScalarMappable(cmap=_cmap_ov, norm=_norm_ov)
                _sm_ov.set_array([])
                for _ax_ov in _axes_ov.flat:
                    plt.colorbar(_sm_ov, ax=_ax_ov, label="log_ratio")

                plt.tight_layout()
                _ov_png = _fig_to_png(_fig_ov)
                plt.close(_fig_ov)

                st.image(_ov_png, width="stretch")

                _ov_fname = (
                    f"{_stem}_overlay"
                    f"_nn{int(pf_nn_min_dist)}"
                    f"_r{int(pf_nn_radius)}"
                    f"_n{int(pf_nn_max_count)}.png"
                )
                _cov1, _cov2 = st.columns(2)
                with _cov1:
                    st.download_button(
                        "⬇️ Download Overlay (PNG)",
                        _ov_png, _ov_fname, "image/png", key="dl_overlay",
                    )
                with _cov2:
                    st.caption(f"{len(_fk)} kept  ·  {len(_fr)} NN-removed")


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
