import json
import os
import sys
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import streamlit as st

# Allow imports from the project root (same working dir as app.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from limoncello.utils.app_helpers import find_latest_run_dir, load_run_json

st.set_page_config(page_title="Pipeline Info — Limoncello 🍋", layout="wide")

st.markdown(
    "<h1 style='margin-bottom:2px'>📖 Pipeline Info</h1>"
    "<p style='color:#888;margin-top:0'>Analysis details, parameters, and flowchart for a selected run</p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# RUN SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📁 Select a Run")
_info_output_path = st.text_input(
    "Output folder",
    value=st.session_state.get("_new_output", "tutorial/output"),
    placeholder="Path containing lc-analysis-* folders…",
    key="info_output_path",
)

_info_runs = sorted(
    (d for d in (os.listdir(_info_output_path)
                 if _info_output_path and os.path.isdir(_info_output_path) else [])
     if d.startswith("lc-analysis-") and os.path.isdir(os.path.join(_info_output_path, d))),
    reverse=True,
)

if not _info_runs:
    st.info("No runs found in the specified folder. Run the pipeline first.")
    st.stop()

_info_sel_run = st.selectbox(
    "Analysis run",
    options=_info_runs,
    format_func=lambda d: d.replace("lc-analysis-", "").replace("_", "  "),
)
_info_run_dir = os.path.join(_info_output_path, _info_sel_run)
_info_params = load_run_json(_info_run_dir)

if _info_params is None:
    st.error("Could not load `run_parameters.json` from this run directory.")
    st.stop()

_info_ts = _info_params.get("timestamp", "unknown")
st.success(f"Loaded run: **{_info_sel_run}**  ·  {_info_ts}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER DETAILS GRID
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("⚙️ Run Parameters")

_p_ch   = _info_params.get("channels", {})
_p_norm = _info_params.get("intensity_normalization", {})
_p_nuc  = _info_params.get("nuclei", {})
_p_neur = _info_params.get("neurites", {})
_p_cil  = _info_params.get("cilia", {})
_p_bb   = _info_params.get("basal_bodies", {})
_p_dist = _info_params.get("distance_thresholds", {})
_p_cls  = _info_params.get("classification", {})

_pc1, _pc2, _pc3, _pc4 = st.columns(4)
with _pc1:
    st.markdown("**Channels**")
    for _lbl, _key, _def in [
        ("Cilia",        "cilia",        0),
        ("Neurites",     "neurites",     1),
        ("Basal bodies", "basal_bodies", 2),
        ("Nuclei",       "nuclei",       3),
    ]:
        st.metric(_lbl, f"Ch {_p_ch.get(_key, _def)}")
with _pc2:
    st.markdown("**Normalization & Nuclei**")
    st.metric("p_low",          _p_norm.get("p_low",  2))
    st.metric("p_high",         _p_norm.get("p_high", 98))
    st.metric("spot_sigma",     _p_nuc.get("spot_sigma",   15))
    st.metric("tophat_radius",  _p_nuc.get("tophat_radius", 12))
    st.metric("outline_sigma",  _p_nuc.get("outline_sigma",  3))
with _pc3:
    st.markdown("**Neurites / Cilia / BB**")
    st.metric("neurite σ",      _p_neur.get("spot_sigma",    5))
    st.metric("cilia min size", _p_cil.get("min_size",       20))
    st.metric("cilia max size", _p_cil.get("max_size",        0))
    st.metric("BB spot σ",      _p_bb.get("spot_sigma",     2.0))
    st.metric("BB outline σ",   _p_bb.get("outline_sigma",  2.0))
    st.metric("BB min size",    _p_bb.get("min_size",          5))
with _pc4:
    st.markdown("**Thresholds & Classification**")
    st.metric("max_cilia_um",   _p_dist.get("max_cilia_um",      2.0))
    st.metric("max_bb_um",      _p_dist.get("max_basal_body_um", 2.0))
    st.metric("ratio_epsilon",  _p_dist.get("ratio_epsilon",     1.0))
    st.metric("axon_threshold", _p_cls.get("axon_threshold",     2.5))
    st.metric("soma_threshold", _p_cls.get("soma_threshold",     1.0))
    st.metric("use_mip",        str(_info_params.get("use_mip", False)))

with st.expander("🔍 Full JSON", expanded=False):
    st.json(_info_params)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE FLOWCHART
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🗺️ Pipeline Flowchart")
st.caption("From raw .ims through segmentation and classification to per-cilium "
           "ROI export and the human + CNN screening/validation step. Generated "
           "from this run's parameters. Download as PNG.")


def _draw_pipeline_diagram(params: dict) -> plt.Figure:
    ch    = params.get("channels", {})
    norm  = params.get("intensity_normalization", {})
    nuc   = params.get("nuclei", {})
    neur  = params.get("neurites", {})
    cil   = params.get("cilia", {})
    bb    = params.get("basal_bodies", {})
    dist  = params.get("distance_thresholds", {})
    cls   = params.get("classification", {})

    fig, ax = plt.subplots(figsize=(16, 12.8))
    ax.set_xlim(0, 16)
    ax.set_ylim(-3.4, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    # Palette
    C_IO    = "#2C3E50"   # input / output
    C_PREP  = "#2980B9"   # preprocessing
    C_SEG   = "#27AE60"   # segmentation
    C_DIST  = "#E67E22"   # distance maps
    C_FEAT  = "#8E44AD"   # feature assignment
    C_CLS   = "#C0392B"   # classification
    C_ROI   = "#0FB9B1"   # ROI export
    C_AI    = "#D6336C"   # screening + AI validation

    def _box(cx, cy, w, h, title, subtitle="", color="#2980B9"):
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=color, edgecolor="white",
            linewidth=1.5, alpha=0.92, zorder=2,
        )
        ax.add_patch(rect)
        y_title = cy + (0.18 if subtitle else 0)
        ax.text(cx, y_title, title,
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white", zorder=3)
        if subtitle:
            ax.text(cx, cy - 0.28, subtitle,
                    ha="center", va="center", fontsize=6.5,
                    color="white", alpha=0.9, zorder=3)

    def _arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color="#555555",
                lw=1.3, mutation_scale=12,
                connectionstyle="arc3,rad=0",
            ),
            zorder=1,
        )

    # Row positions (y)
    Y_IN   = 9.3
    Y_NORM = 8.0
    Y_SEG  = 6.3
    Y_DIST = 4.7
    Y_FEAT = 3.3
    Y_CLS  = 2.0
    Y_OUT  = 0.7

    # Input
    _box(8, Y_IN, 5, 0.75,
         "Input (.ims files)",
         f"Ch cilia:{ch.get('cilia',0)}  neurite:{ch.get('neurites',1)}  "
         f"bb:{ch.get('basal_bodies',2)}  nuclei:{ch.get('nuclei',3)}  "
         f"| MIP={params.get('use_mip', False)}",
         color=C_IO)

    _arrow(8, Y_IN - 0.38, 8, Y_NORM + 0.38)

    # Normalisation
    _box(8, Y_NORM, 5, 0.75,
         "Intensity Normalisation",
         f"p_low={norm.get('p_low',2)}%  p_high={norm.get('p_high',98)}%",
         color=C_PREP)

    # Segmentation — 4 boxes
    _seg_xs = [2.5, 6.0, 10.0, 13.5]
    _seg_labels = [
        ("Cilia segmentation",
         f"APOC classifier\nmin={cil.get('min_size',20)} max={cil.get('max_size',0)}"),
        ("Nuclei segmentation",
         f"Voronoi-Otsu\nσ={nuc.get('spot_sigma',15)} tophat={nuc.get('tophat_radius',12)}"),
        ("Neurite tracing",
         f"Voronoi-Otsu + skeleton\nσ={neur.get('spot_sigma',5)}"),
        ("Basal body detection",
         f"Gauss+Voronoi-Otsu\nspot σ={bb.get('spot_sigma',2.0)} min={bb.get('min_size',5)}"),
    ]
    for _sx, (_st, _ss) in zip(_seg_xs, _seg_labels):
        # fan-out arrow from norm
        _arrow(8, Y_NORM - 0.38, _sx, Y_SEG + 0.38)
        _box(_sx, Y_SEG, 3.0, 0.75, _st, _ss, color=C_SEG)

    # Distance maps
    _box(8, Y_DIST, 8, 0.75,
         "Distance Maps",
         f"dt_nuclei · dt_neurite · ratio=(dt_nuclei+ε)/(dt_neurite+ε)  ε={dist.get('ratio_epsilon',1.0)}",
         color=C_DIST)
    for _sx in _seg_xs:
        _arrow(_sx, Y_SEG - 0.38, 8, Y_DIST + 0.38)

    _arrow(8, Y_DIST - 0.38, 8, Y_FEAT + 0.38)

    # Feature assignment
    _box(8, Y_FEAT, 8, 0.75,
         "Feature Assignment & Distance Filtering",
         f"max_cilia={dist.get('max_cilia_um',2.0)} µm  "
         f"max_bb={dist.get('max_basal_body_um',2.0)} µm  "
         "· log_ratio · dt_neurite · dt_nuclei per centroid",
         color=C_FEAT)

    _arrow(8, Y_FEAT - 0.38, 8, Y_CLS + 0.38)

    # Classification
    _box(8, Y_CLS, 8, 0.75,
         "Classification",
         f"axon: log_ratio > {cls.get('axon_threshold',2.5)}  |  "
         f"soma: log_ratio < {cls.get('soma_threshold',1.0)}  |  ambiguous: in between",
         color=C_CLS)

    _arrow(8, Y_CLS - 0.38, 8, Y_OUT + 0.38)

    # Output
    _box(8, Y_OUT, 8, 0.6,
         "Output  →  Excel (all_cilia_features.xlsx) · Overlay PNGs · Summary figures",
         color=C_IO)

    # ── ROI export + screening / AI validation (post-classification) ───────────
    Y_ROI = -0.7
    Y_SCR = -2.0

    _arrow(8, Y_OUT - 0.30, 8, Y_ROI + 0.38)
    _box(8, Y_ROI, 9, 0.75,
         "Per-cilium ROI Export",
         "figures/cilia_rois/ — uniform XY-MIP PNG (cilia + basal body) "
         "+ raw .npz crop (GPU isotropic resample)",
         color=C_ROI)

    _arrow(8, Y_ROI - 0.38, 8, Y_SCR + 0.38)
    _box(8, Y_SCR, 11, 0.95,
         "Screening & AI Validation  (data app)",
         "Human keep/reject  →  trains CNN on ROI images "
         "(tiny CPU / big GPU · augmentation · early-stopping)\n"
         "→  predicts P(real cilium) on the run  →  sets human_validated "
         "(false-positive QC; rejected cilia drop from all tabs)",
         color=C_AI)

    # Dashed feedback: human_validated re-filters the analysis (straight line to
    # the right of every box, so it never crosses them).
    ax.annotate(
        "", xy=(13.9, Y_FEAT), xytext=(13.9, Y_SCR),
        arrowprops=dict(arrowstyle="-|>", color=C_AI, lw=1.2, ls="--",
                        mutation_scale=11, connectionstyle="arc3,rad=0"),
        zorder=1,
    )
    ax.text(14.15, (Y_FEAT + Y_SCR) / 2, "human_validated\nre-filters\nanalysis",
            fontsize=6, color=C_AI, style="italic", ha="left", va="center")

    # Legend
    _legend_items = [
        (C_IO,   "I/O"),
        (C_PREP, "Preprocessing"),
        (C_SEG,  "Segmentation"),
        (C_DIST, "Distance maps"),
        (C_FEAT, "Feature assignment"),
        (C_CLS,  "Classification"),
        (C_ROI,  "ROI export"),
        (C_AI,   "Screening + AI"),
    ]
    for _li, (_lc, _ll) in enumerate(_legend_items):
        ax.text(0.1 + _li * 1.95, -3.1, _ll, fontsize=7, color=_lc,
                fontweight="bold", va="bottom", ha="left", zorder=3)

    plt.tight_layout(pad=0.5)
    return fig


_diag_fig = _draw_pipeline_diagram(_info_params)
_buf = BytesIO()
_diag_fig.savefig(_buf, format="png", dpi=180, bbox_inches="tight")
_diag_png = _buf.getvalue()
plt.close(_diag_fig)

st.image(_diag_png, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOADS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
_dl1, _dl2 = st.columns(2)
with _dl1:
    st.download_button(
        "⬇️ Download Flowchart (PNG)",
        _diag_png,
        f"{_info_sel_run}_pipeline_diagram.png",
        "image/png",
        key="dl_diag",
    )
with _dl2:
    st.download_button(
        "⬇️ Download Parameters (JSON)",
        json.dumps(_info_params, indent=2).encode(),
        "run_parameters.json",
        "application/json",
        key="dl_params_info",
    )
