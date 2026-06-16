import os
os.environ.setdefault("PYOPENCL_NO_CACHE", "1")

import sys
import glob
import warnings
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore", message=r".*PyOpenCL compiler caching failed.*")

# Allow imports from the project root (same working dir as app.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from limoncello.utils.reader import load_image
from limoncello.preprocessing.preprocessing import percentile_minmax_normalize
from limoncello.segmentation.cilia import segment_cilia_ml
from limoncello.segmentation.nuclei import segment_nuclei
from limoncello.segmentation.neurites import segment_neurites
from limoncello.segmentation.basal_bodies import segment_basal_bodies

st.set_page_config(page_title="Preview — Limoncello 🍋", layout="wide")

st.markdown(
    "<h1 style='margin-bottom:2px'>🔬 Preview & Settings Tuning</h1>"
    "<p style='color:#888;margin-top:0'>Test the analysis settings on a single image — "
    "channels, segmentations, and intensity histograms.</p>",
    unsafe_allow_html=True,
)
st.caption(
    "Settings are read from the **main page sidebar** (channel assignment, "
    "normalisation, blur, log, sigmas …). Adjust them there, then re-run the preview."
)


def _p(key, default):
    return st.session_state.get(key, default)


def _int(key, default):
    try:
        return int(_p(key, default))
    except (TypeError, ValueError):
        return default


# ── Inputs ────────────────────────────────────────────────────────────────────
_input_path = st.text_input(
    "Input folder",
    value=str(_p("_resolved_input", "") or _p("_new_input", "")),
    placeholder="Folder containing .ims files…",
)
_classifier = st.text_input(
    "Cilia classifier (.cl)",
    value=str(_p("_resolved_classifier", r"segmenters\cilia-segmenter.cl")),
)

_files = sorted(glob.glob(os.path.join(_input_path, "*.ims"))) if _input_path else []
if not _files:
    st.info("Enter a valid input folder containing .ims files.")
    st.stop()

_sel = st.selectbox(
    "Image to preview", _files, index=0,
    format_func=lambda p: os.path.basename(p),
    help="Defaults to the first image in the dataset.",
)

# Channel assignment + per-channel options (read from shared session state)
_ch = {
    "cilia":   _int("ch_cilia", 0),
    "neurites": _int("ch_neurites", 1),
    "bb":      _int("ch_bb", 2),
    "nuclei":  _int("ch_nuclei", 3),
}
_use_mip = bool(_p("use_mip", False))
_p_low, _p_high = _int("p_low", 2), _int("p_high", 98)

with st.expander("Effective settings being previewed", expanded=False):
    st.json({
        "channels": _ch, "use_mip": _use_mip,
        "p_low": _p_low, "p_high": _p_high,
        "nuclei": {"spot_sigma": _int("nuclei_sigma", 15),
                   "tophat_radius": _int("tophat_radius", 12),
                   "outline_sigma": _int("nuclei_outline_sigma", 3),
                   "log": bool(_p("nuclei_log", False))},
        "neurites": {"spot_sigma": _int("neurite_sigma", 5),
                     "log": bool(_p("neurite_log", False))},
        "cilia": {"log": bool(_p("cilia_log", False)),
                  "min_size": _int("cilia_min_size", 20),
                  "max_size": _int("cilia_max_size", 0)},
        "basal_bodies": {"spot_sigma": float(_p("bb_spot_sigma", 2.0)),
                         "log": bool(_p("bb_log", False))},
    })


def _gauss(pre, default_on):
    on = bool(_p(f"{pre}_gauss_on", default_on))
    if not on:
        return (0.0, 0.0, 0.0)
    return (float(_p(f"{pre}_gauss_z", 0.0)),
            float(_p(f"{pre}_gauss_y", 0.0)),
            float(_p(f"{pre}_gauss_x", 0.0)))


def _disp_mip(vol):
    """(Z,Y,X) → 2-D max projection (or the slice itself if already 2-D)."""
    a = np.asarray(vol, dtype=np.float32)
    return a if a.ndim == 2 else a.max(axis=0)


def _run_preview():
    img, meta = load_image(_sel)
    n_ch = img.shape[1]
    out = {}

    def _chan(ch):
        ch = max(0, min(ch, n_ch - 1))
        raw = np.asarray(img[0, ch], dtype=np.float32)            # (Z,Y,X)
        norm = percentile_minmax_normalize(raw, p_low=_p_low, p_high=_p_high)
        if _use_mip:
            raw = raw.max(axis=0, keepdims=True)
            norm = norm.max(axis=0, keepdims=True)
        return raw, norm

    # Cilia (raw input, APOC)
    c_raw, _ = _chan(_ch["cilia"])
    cilia_lab = np.asarray(segment_cilia_ml(
        c_raw, classifier_path=_classifier,
        gaussian_sigma=_gauss("cilia", True), log_transform=bool(_p("cilia_log", False)),
        min_size=_int("cilia_min_size", 20), max_size=_int("cilia_max_size", 0),
    )).astype(np.int32)
    out["Cilia"] = (c_raw, cilia_lab)

    # Neurites (normalised)
    _, n_norm = _chan(_ch["neurites"])
    _sk, neur_gpu = segment_neurites(
        n_norm, spot_sigma=_int("neurite_sigma", 5),
        gaussian_sigma=_gauss("neurite", False), log_transform=bool(_p("neurite_log", False)),
    )
    out["Neurites"] = (n_norm, np.asarray(neur_gpu).astype(np.int32))

    # Basal bodies (raw)
    b_raw, _ = _chan(_ch["bb"])
    bb_lab = np.asarray(segment_basal_bodies(
        b_raw, spot_sigma=float(_p("bb_spot_sigma", 2.0)),
        outline_sigma=float(_p("bb_outline_sigma", 2.0)),
        gaussian_sigma=_gauss("bb", True), log_transform=bool(_p("bb_log", False)),
        min_size=_int("bb_min_size", 5), max_size=_int("bb_max_size", 0),
    )).astype(np.int32)
    out["Basal Bodies"] = (b_raw, bb_lab)

    # Nuclei (normalised)
    _, u_norm = _chan(_ch["nuclei"])
    tr = _int("tophat_radius", 12)
    nuc_lab = segment_nuclei(
        u_norm, tophat_radius=(tr, tr, tr), spot_sigma=_int("nuclei_sigma", 15),
        outline_sigma=_int("nuclei_outline_sigma", 3),
        gaussian_sigma=_gauss("nuclei", False), log_transform=bool(_p("nuclei_log", False)),
    ).astype(np.int32)
    out["Nuclei"] = (u_norm, nuc_lab)
    return out


if st.button("🔬 Run preview", type="primary"):
    with st.spinner(f"Segmenting {os.path.basename(_sel)} …"):
        try:
            st.session_state["_preview_result"] = _run_preview()
            st.session_state["_preview_file"] = _sel
        except Exception as exc:
            st.error(f"Preview failed: {exc}")
            st.session_state["_preview_result"] = None

# ── Render ─────────────────────────────────────────────────────────────────────
_res = st.session_state.get("_preview_result")
if _res:
    st.caption(f"Showing: **{os.path.basename(st.session_state.get('_preview_file', _sel))}**")
    _colors = {"Cilia": "#00FF88", "Neurites": "#FF44FF",
               "Basal Bodies": "#FF8800", "Nuclei": "#FFFF00"}
    _hlog = st.checkbox("Log Y on histograms", value=True, key="prev_hlog")

    _n = len(_res)
    _fig, _ax = plt.subplots(_n, 3, figsize=(13, 3.2 * _n), squeeze=False)
    for _r, (_name, (_chan_vol, _lab)) in enumerate(_res.items()):
        _mip = _disp_mip(_chan_vol)
        _lmip = _disp_mip(_lab).astype(np.int32)
        _nobj = int(np.asarray(_lab).max())

        # Channel MIP
        _lo, _hi = np.percentile(_mip, (1, 99.5))
        _ax[_r][0].imshow(_mip, cmap="gray", vmin=_lo, vmax=_hi + 1e-8)
        _ax[_r][0].set_ylabel(_name, fontsize=11, fontweight="bold")
        _ax[_r][0].set_title("channel MIP" if _r == 0 else "", fontsize=10)
        _ax[_r][0].set_xticks([]); _ax[_r][0].set_yticks([])

        # Segmentation
        _disp = np.ma.masked_where(_lmip == 0, _lmip)
        _ax[_r][1].imshow(_mip, cmap="gray", vmin=_lo, vmax=_hi + 1e-8)
        _ax[_r][1].imshow(_disp, cmap="nipy_spectral", interpolation="nearest", alpha=0.55)
        _ax[_r][1].set_title(f"segmentation (n={_nobj})" if _r == 0
                             else f"n={_nobj}", fontsize=10)
        _ax[_r][1].axis("off")

        # Histogram
        _vals = np.asarray(_chan_vol, dtype=float).ravel()
        _vals = _vals[np.isfinite(_vals)]
        if _vals.size:
            _ax[_r][2].hist(_vals, bins=80, color=_colors.get(_name, "#888"),
                            alpha=0.85, edgecolor="none")
        if _hlog:
            _ax[_r][2].set_yscale("log")
        _ax[_r][2].set_title("intensity histogram" if _r == 0 else "", fontsize=10)
        _ax[_r][2].tick_params(labelsize=7)
        _ax[_r][2].spines[["top", "right"]].set_visible(False)

    _fig.tight_layout()
    st.pyplot(_fig, width="stretch")

    _buf = BytesIO()
    _fig.savefig(_buf, format="png", dpi=130, bbox_inches="tight")
    st.download_button(
        "⬇️ Download preview (PNG)", _buf.getvalue(),
        f"{os.path.splitext(os.path.basename(_sel))[0]}_preview.png", "image/png",
    )
    plt.close(_fig)
else:
    st.info("Set your parameters on the main page, then click **Run preview**.")
