"""LimonCELLo · CiliaQ — a CiliaQ-style 3-D cilia quantification app.

Reimplements the CiliaQ methodology (intensity-threshold segmentation +
per-cilium morphometrics: volume, surface, skeleton length, intensity,
colocalisation, bending) and reproduces the LimonCELLo base analysis
(dt-ratio axon/soma classification, basal-body pairing). Reads Imaris .ims.

Run:  streamlit run ciliaq_app.py
"""
import os
os.environ.setdefault("PYOPENCL_NO_CACHE", "1")

import sys
import glob
import time
import warnings
import threading
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore", message=r".*PyOpenCL compiler caching failed.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from limoncello.ciliaq.pipeline import run_ciliaq
from limoncello.ciliaq.segment import THRESHOLD_METHODS

st.set_page_config(page_title="LimonCELLo · CiliaQ 🍋", layout="wide")

_CLASS_PALETTE = {"axon": "#2ecc71", "soma": "#e74c3c", "ambiguous": "#f39c12"}

# ── session defaults ────────────────────────────────────────────────────────────
for _k, _v in {
    "ciliaq_running": False, "ciliaq_result": None, "ciliaq_logs": [],
}.items():
    st.session_state.setdefault(_k, _v)


def _parse_coords(val):
    if isinstance(val, (list, np.ndarray)):
        return np.array(val, dtype=float)
    import ast, re
    s = re.sub(r"np\.\w+\(([^)]+)\)", r"\1", str(val))
    try:
        return np.array(ast.literal_eval(s), dtype=float)
    except Exception:
        return np.zeros(3)


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — parameters
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🍋 CiliaQ settings")

    st.subheader("📁 Paths")
    input_path = st.text_input("Input folder (.ims)", key="cq_input",
                               placeholder="Folder containing .ims files…")
    output_path = st.text_input("Output folder", value="tutorial/output", key="cq_output")

    st.subheader("🔬 Channels")
    st.caption("Channel indices (0-based) in the .ims file.")
    c1, c2 = st.columns(2)
    with c1:
        ch_cilia = st.number_input("Cilia", 0, 31, 0, key="cq_ch_cilia")
        ch_signal_a = st.number_input("Signal A (e.g. ARL13B, -1=off)", -1, 31, -1, key="cq_ch_a")
        ch_coloc = st.number_input("Coloc channel (-1=off)", -1, 31, -1, key="cq_ch_coloc")
    with c2:
        ch_signal_b = st.number_input("Signal B (-1=off)", -1, 31, -1, key="cq_ch_b")

    st.subheader("✂️ Cilia segmentation (CiliaQ)")
    threshold_method = st.selectbox("Threshold method", THRESHOLD_METHODS, key="cq_method")
    threshold_factor = st.slider("Threshold factor", 0.2, 3.0, 1.0, 0.05, key="cq_tf",
                                 help=">1 = stricter (fewer/smaller objects)")
    hysteresis_low_factor = st.slider("Hysteresis low factor", 0.1, 0.95, 0.5, 0.05,
                                      key="cq_hlf", disabled=threshold_method != "hysteresis")
    background_radius = st.slider("Background subtraction radius (vox, 0=off)",
                                  0.0, 20.0, 0.0, 1.0, key="cq_bg")
    g1, g2, g3 = st.columns(3)
    cg_z = g1.slider("Gauss σz", 0.0, 5.0, 1.0, 0.5, key="cq_gz")
    cg_y = g2.slider("Gauss σy", 0.0, 5.0, 1.0, 0.5, key="cq_gy")
    cg_x = g3.slider("Gauss σx", 0.0, 5.0, 1.0, 0.5, key="cq_gx")
    min_voxels = st.number_input("Min size (voxels)", 0, 100000, 10, key="cq_minv")
    max_voxels = st.number_input("Max size (voxels, 0=off)", 0, 10_000_000, 0, key="cq_maxv")
    coloc_threshold = st.number_input("Coloc threshold (0=auto)", 0.0, 1e6, 0.0, key="cq_ct")

    st.subheader("🧠 Base analysis (LimonCELLo)")
    base_analysis = st.checkbox("Run dt-ratio + classification + BB pairing",
                                value=True, key="cq_base")
    with st.expander("Base channels & params", expanded=False):
        ch_neurites = st.number_input("Neurites ch", 0, 31, 1, key="cq_ch_neur")
        ch_bb = st.number_input("Basal bodies ch", 0, 31, 2, key="cq_ch_bb")
        ch_nuclei = st.number_input("Nuclei ch", 0, 31, 3, key="cq_ch_nuc")
        nuclei_sigma = st.slider("Nuclei spot σ", 1, 50, 15, key="cq_ns")
        tophat_radius = st.slider("Nuclei tophat r", 1, 50, 12, key="cq_th")
        neurite_sigma = st.slider("Neurite spot σ", 1, 20, 5, key="cq_nes")
        bb_spot_sigma = st.slider("BB spot σ", 0.5, 10.0, 2.0, 0.5, key="cq_bbs")
        max_cilia_dist_um = st.slider("Max cilia→neurite (µm)", 0.5, 10.0, 2.0, 0.1, key="cq_mcd")
        ratio_epsilon = st.slider("Ratio ε", 0.0, 10.0, 1.0, 0.1, key="cq_eps")
        axon_threshold = st.number_input("Axon log-ratio >", 0.0, 10.0, 2.5, 0.1, key="cq_axt")
        soma_threshold = st.number_input("Soma log-ratio <", 0.0, 10.0, 1.0, 0.1, key="cq_sot")

    p_low, p_high = 2, 98


def _collect_params():
    return dict(
        ch_cilia=int(ch_cilia),
        ch_signal_a=int(ch_signal_a), ch_signal_b=int(ch_signal_b),
        ch_coloc=int(ch_coloc),
        threshold_method=threshold_method, threshold_factor=float(threshold_factor),
        hysteresis_low_factor=float(hysteresis_low_factor),
        background_radius=float(background_radius),
        cilia_gaussian_sigma=(float(cg_z), float(cg_y), float(cg_x)),
        min_voxels=int(min_voxels), max_voxels=int(max_voxels),
        coloc_threshold=(None if coloc_threshold == 0 else float(coloc_threshold)),
        base_analysis=bool(base_analysis),
        ch_neurites=int(ch_neurites), ch_bb=int(ch_bb), ch_nuclei=int(ch_nuclei),
        nuclei_sigma=int(nuclei_sigma), tophat_radius=int(tophat_radius),
        neurite_sigma=int(neurite_sigma), bb_spot_sigma=float(bb_spot_sigma),
        max_cilia_dist_um=float(max_cilia_dist_um), ratio_epsilon=float(ratio_epsilon),
        axon_threshold=float(axon_threshold), soma_threshold=float(soma_threshold),
        p_low=p_low, p_high=p_high,
    )


# ── header ──────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0'>🍋 LimonCELLo · CiliaQ</h1>"
    "<p style='color:#888;margin-top:2px'>CiliaQ-style 3-D cilia quantification "
    "with .ims input and the LimonCELLo dt-ratio analysis</p>",
    unsafe_allow_html=True,
)

tab_run, tab_table, tab_graphs, tab_overlay = st.tabs(
    ["▶ Run", "📋 Per-cilium table", "📊 Graphs", "🔬 Overlays"])

# ════════════════════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════════════════════
with tab_run:
    st.subheader("Run the CiliaQ pipeline")
    _files = sorted(glob.glob(os.path.join(input_path, "*.ims"))) if input_path else []
    st.caption(f"{len(_files)} .ims file(s) found." if input_path
               else "Enter an input folder in the sidebar.")

    if st.button("▶ Run analysis", type="primary", disabled=not _files):
        _prog = st.progress(0.0, text="Starting…")
        _logbuf = StringIO()
        _orig = sys.stdout

        def _cb(i, n, name):
            _prog.progress((i) / max(n, 1), text=f"[{i + 1}/{n}] {name}")

        try:
            sys.stdout = _logbuf
            with st.spinner("Processing…"):
                result = run_ciliaq(input_path, output_path, _collect_params(), _cb)
            st.session_state["ciliaq_result"] = result
            st.session_state["ciliaq_logs"] = _logbuf.getvalue().splitlines()
        except Exception as exc:
            st.session_state["ciliaq_logs"] = _logbuf.getvalue().splitlines()
            st.error(f"Run failed: {exc}")
        finally:
            sys.stdout = _orig
            _prog.empty()

    _res = st.session_state.get("ciliaq_result")
    if _res and _res.get("n_cilia"):
        st.success(f"✓ {_res['n_cilia']} cilia quantified → {_res['out_dir']}")
        if os.path.exists(_res.get("excel", "")):
            with open(_res["excel"], "rb") as fh:
                st.download_button("⬇️ Download Excel", fh.read(),
                                   "ciliaq_features.xlsx", key="cq_dl_xl")

    if st.session_state["ciliaq_logs"]:
        with st.expander("📋 Logs", expanded=False):
            st.code("\n".join(st.session_state["ciliaq_logs"][-300:]), language=None)


# ── locate latest results to explore ────────────────────────────────────────────
def _latest_excel(base):
    runs = sorted(glob.glob(os.path.join(base, "ciliaq-analysis-*")), reverse=True)
    for r in runs:
        x = os.path.join(r, "csv", "ciliaq_features.xlsx")
        if os.path.exists(x):
            return r, x
    return None, None


_run_dir, _excel = _latest_excel(output_path)


@st.cache_data(show_spinner=False)
def _load(excel, mtime):
    return pd.read_excel(excel, sheet_name="all_data")


_df = pd.DataFrame()
if _excel:
    _df = _load(_excel, os.path.getmtime(_excel))

_NUMERIC_METRICS = [
    "volume_um3", "surface_um2", "length_um", "max_span_um", "sphere_radius_um",
    "shape_complexity", "bending_index", "n_branches", "n_endpoints",
    "cilia_mean", "channelA_mean", "coloc_pct", "log_ratio",
    "distance_to_neurite_um", "pair_distance_um",
]

# ════════════════════════════════════════════════════════════════════════════════
# TABLE
# ════════════════════════════════════════════════════════════════════════════════
with tab_table:
    if _df.empty:
        st.info("No results yet — run the pipeline.")
    else:
        st.caption(f"{len(_df)} cilia · run: {os.path.basename(_run_dir)}")
        _samples = ["(all)"] + sorted(_df["file_short"].dropna().unique().tolist())
        _sel = st.selectbox("Sample", _samples, key="cq_tbl_sample")
        _view = _df if _sel == "(all)" else _df[_df["file_short"] == _sel]
        st.dataframe(_view, width="stretch", height=460)
        st.download_button("⬇️ CSV", _view.to_csv(index=False).encode(),
                           "ciliaq_table.csv", key="cq_dl_csv")

# ════════════════════════════════════════════════════════════════════════════════
# GRAPHS
# ════════════════════════════════════════════════════════════════════════════════
with tab_graphs:
    if _df.empty:
        st.info("No results yet — run the pipeline.")
    else:
        _avail = [m for m in _NUMERIC_METRICS if m in _df.columns]
        g1, g2 = st.columns(2)
        with g1:
            _metric = st.selectbox("Metric", _avail, key="cq_g_metric")
            _by_sample = st.checkbox("Split by sample (box + points)", key="cq_g_box")
            _fig, _ax = plt.subplots(figsize=(7, 4))
            _x = "file_short"
            if _by_sample and _x in _df.columns:
                sns.boxplot(data=_df, x=_x, y=_metric, ax=_ax, color="#cfe8ff")
                sns.stripplot(data=_df, x=_x, y=_metric, ax=_ax, size=3,
                              color="black", alpha=0.45, jitter=True)
                _ax.tick_params(axis="x", rotation=45)
            else:
                sns.histplot(_df[_metric].dropna(), kde=True, ax=_ax, color="#2E6FA3")
            _ax.set_title(_metric)
            plt.tight_layout(); st.pyplot(_fig, width="stretch"); plt.close(_fig)
        with g2:
            _mx = st.selectbox("X", _avail, key="cq_g_x")
            _my = st.selectbox("Y", _avail, index=min(1, len(_avail) - 1), key="cq_g_y")
            _hue = "class" if "class" in _df.columns else None
            _fig2, _ax2 = plt.subplots(figsize=(7, 4))
            sns.scatterplot(data=_df, x=_mx, y=_my, hue=_hue, ax=_ax2, s=22,
                            alpha=0.7, palette=(_CLASS_PALETTE if _hue else None))
            _ax2.set_title(f"{_my} vs {_mx}")
            plt.tight_layout(); st.pyplot(_fig2, width="stretch"); plt.close(_fig2)

        st.markdown("**Per-sample summary**")
        _agg = {m: "mean" for m in _avail}
        st.dataframe(_df.groupby("file_short").agg({"cilia_id": "count", **_agg})
                     .rename(columns={"cilia_id": "n_cilia"}).round(3),
                     width="stretch")

# ════════════════════════════════════════════════════════════════════════════════
# OVERLAYS
# ════════════════════════════════════════════════════════════════════════════════
with tab_overlay:
    if _df.empty or not _run_dir:
        st.info("No results yet — run the pipeline.")
    else:
        _mip_dir = os.path.join(_run_dir, "figures", "mips")
        _files_ov = sorted(_df["filename"].dropna().unique().tolist())
        _selov = st.selectbox("Sample", _files_ov, key="cq_ov_file",
                              format_func=lambda f: os.path.basename(f))
        _stem = os.path.splitext(os.path.basename(_selov))[0]
        _cmip = os.path.join(_mip_dir, f"{_stem}_cilia_mip.npy")
        _clab = os.path.join(_mip_dir, f"{_stem}_cilia_labels_mip.npy")
        if not os.path.exists(_cmip):
            st.warning("MIP arrays not found for this sample.")
        else:
            _metric_ov = st.selectbox(
                "Colour cilia by", [m for m in _NUMERIC_METRICS if m in _df.columns],
                key="cq_ov_metric")
            _sub = _df[_df["filename"] == _selov]
            _chan = np.load(_cmip)
            _lab = np.load(_clab) if os.path.exists(_clab) else None
            _fig, _ax = plt.subplots(figsize=(9, 9))
            _lo, _hi = np.percentile(_chan, (1, 99.5))
            _ax.imshow(_chan, cmap="gray", vmin=_lo, vmax=_hi + 1e-8)
            if _lab is not None:
                _ax.contour((_lab > 0).astype(float), levels=[0.5],
                            colors="#00FF88", linewidths=0.6)
            if "coords" in _sub.columns and not _sub.empty:
                _c = np.vstack([_parse_coords(c) for c in _sub["coords"]])
                _vals = _sub[_metric_ov].to_numpy(dtype=float)
                _sc = _ax.scatter(_c[:, 2], _c[:, 1], c=_vals, cmap="viridis",
                                  s=30, edgecolor="black", linewidth=0.4)
                plt.colorbar(_sc, ax=_ax, label=_metric_ov, shrink=0.7)
            _ax.set_title(f"{os.path.basename(_selov)} — cilia coloured by {_metric_ov}")
            _ax.axis("off")
            plt.tight_layout(); st.pyplot(_fig, width="stretch"); plt.close(_fig)
