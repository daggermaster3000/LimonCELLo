"""
Limoncello — Data Analysis app 🍋📊

A *data-focused* companion to the main pipeline app (``app.py``). It does **not**
run the pipeline; it loads the CSV/Excel results of one or more finished
``lc-analysis-*`` runs and lets you explore and **compare** them:

  • Overlays  — the 3-D napari screenshots saved during a batch run
                (``figures/napari_overlays/<stem>_{logratio,channels,labels}.png``)
                plus the per-cilium ROI close-ups (``figures/cilia_rois/``).
  • Hist/KDE  — distribution of any metric, grouped/compared across runs.
  • Scatter   — any two numeric columns, coloured by run / class.
  • Table     — the combined per-object table with filters + CSV export.
  • Report    — a PDF bundling the 3-D images + comparison plots.

Run with:  streamlit run data_app.py
"""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from limoncello.utils.app_helpers import find_latest_run_dir

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG / CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Limoncello Data 🍋📊", layout="wide",
                   initial_sidebar_state="expanded")

_CLASS_PALETTE = {"axon": "#2ecc71", "soma": "#e74c3c", "ambiguous": "#f39c12"}
_OVERVIEW_TAGS = ["logratio", "channels", "labels"]
_OVERVIEW_LABEL = {"logratio": "log(ratio) + cilia",
                   "channels": "all channels + cilia",
                   "labels":   "cilia + BB labels + cilia"}

# GraphPad-Prism-like qualitative palette (bold, saturated, colour-blind friendly).
_PRISM_PALETTE = ["#0C5DA5", "#E8000B", "#00B945", "#FF9500",
                  "#845B97", "#474747", "#9E3FFF", "#00C2C7"]


def _set_prism_style():
    """Publication style à la GraphPad Prism: white background, no grid,
    detached bold axes, outward ticks, heavy sans-serif labels."""
    sns.set_theme(style="ticks")
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.4,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.labelcolor": "#111111",
        "axes.prop_cycle": matplotlib.cycler(color=_PRISM_PALETTE),
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.major.width": 1.4, "ytick.major.width": 1.4,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.dpi": 110,
    })


_set_prism_style()


# ─────────────────────────────────────────────────────────────────────────────
# RUN DISCOVERY / DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _is_run_dir(path: str) -> bool:
    return bool(path) and os.path.exists(
        os.path.join(path, "csv", "all_cilia_features.xlsx")
    )


def discover_runs(base: str) -> dict[str, str]:
    """Map ``{run_name: run_dir}`` for a base folder.

    Accepts either an output base that *contains* ``lc-analysis-*`` run folders,
    or a single run directory itself.
    """
    out: dict[str, str] = {}
    if not base or not os.path.isdir(base):
        return out
    if _is_run_dir(base):
        out[os.path.basename(base.rstrip("/\\"))] = base
        return out
    for d in sorted(os.listdir(base)):
        full = os.path.join(base, d)
        if d.startswith("lc-analysis-") and _is_run_dir(full):
            out[d] = full
    return out


def _excel_mtime(run_dir: str) -> float:
    p = os.path.join(run_dir, "csv", "all_cilia_features.xlsx")
    return os.path.getmtime(p) if os.path.exists(p) else 0.0


@st.cache_data(show_spinner=False)
def load_run_df(run_dir: str, mtime: float) -> pd.DataFrame:
    """Load the per-object ``all_data`` sheet for a run (cached on mtime)."""
    _ = mtime
    path = os.path.join(run_dir, "csv", "all_cilia_features.xlsx")
    try:
        return pd.read_excel(path, sheet_name="all_data")
    except Exception as exc:                              # noqa: BLE001
        st.warning(f"Could not read {path}: {exc}")
        return pd.DataFrame()


def combined_dataframe(runs: list[dict]) -> pd.DataFrame:
    """Concatenate every added run's table, tagged with a ``run`` column."""
    frames = []
    for r in runs:
        df = load_run_df(r["dir"], _excel_mtime(r["dir"]))
        if df is None or df.empty:
            continue
        df = df.copy()
        df["run"] = r["label"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def sample_stems(df: pd.DataFrame) -> list[str]:
    """Distinct image stems (filename without extension) present in a df."""
    if "filename" not in df.columns:
        return []
    return sorted({Path(str(f)).stem for f in df["filename"].dropna().unique()})


def overview_images(run_dir: str, stem: str) -> dict[str, str]:
    """Existing napari overview screenshots for one sample: ``{tag: path}``."""
    d = os.path.join(run_dir, "figures", "napari_overlays")
    out = {}
    for tag in _OVERVIEW_TAGS:
        p = os.path.join(d, f"{stem}_{tag}.png")
        if os.path.exists(p):
            out[tag] = p
    return out


def roi_images(run_dir: str, stem: str) -> list[str]:
    """Per-cilium ROI close-up screenshots for one sample, sorted by id."""
    d = os.path.join(run_dir, "figures", "cilia_rois")
    if not os.path.isdir(d):
        return []
    paths = [os.path.join(d, f) for f in os.listdir(d)
             if f.startswith(f"{stem}_cilia") and f.lower().endswith(".png")]

    def _cid(p):
        m = os.path.splitext(os.path.basename(p))[0].split("_cilia")[-1]
        try:
            return int(m)
        except ValueError:
            return 0
    return sorted(paths, key=_cid)


def reclassify(log_ratio: pd.Series, axon_thr: float, soma_thr: float) -> pd.Series:
    """Class from log_ratio: 'axon' if > axon_thr, 'soma' if < soma_thr, else
    'ambiguous' (soma wins in any overlap, applied last)."""
    lr = pd.to_numeric(log_ratio, errors="coerce")
    cls = pd.Series("ambiguous", index=lr.index, dtype=object)
    cls[lr > axon_thr] = "axon"
    cls[lr < soma_thr] = "soma"
    return cls


def normalize_axis(s: pd.Series, method: str) -> pd.Series:
    """Normalize the values of a numeric axis. Methods:
    'none', 'z-score' ((x-mean)/std), 'min–max' (0..1), 'robust' ((x-median)/IQR)."""
    s = pd.to_numeric(s, errors="coerce")
    if method == "z-score":
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd else s * 0.0
    if method == "min–max":
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else s * 0.0
    if method == "robust":
        med = s.median()
        iqr = s.quantile(0.75) - s.quantile(0.25)
        return (s - med) / iqr if iqr else s * 0.0
    return s


def _tidy_legend(ax, title=None, max_items: int = 16):
    """Move the legend outside the axes so it can't squash the plot; hide it
    (with a short note) when there are too many groups to be readable."""
    leg = ax.get_legend()
    if leg is None:
        return
    if title is None and leg.get_title() is not None:
        title = leg.get_title().get_text()
    handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
    labels = [t.get_text() for t in leg.get_texts()]
    leg.remove()
    n = len(labels)
    if n == 0:
        return
    if n > max_items:
        ax.text(1.02, 1.0, f"{n} groups —\nlegend hidden", transform=ax.transAxes,
                va="top", ha="left", fontsize=8, color="#666")
        return
    ax.legend(handles, labels, title=title, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), frameon=False,
              fontsize=8, title_fontsize=9, ncol=2 if n > 12 else 1)


def _prism_despine(fig, trim: bool = False, offset: int = 6):
    """Detach the axes spines for the Prism look. ``trim=False`` keeps each
    spine spanning the full plotting area (only offset/detached, not clipped to
    the first/last tick)."""
    try:
        sns.despine(fig=fig, trim=trim, offset=offset)
    except Exception:                                     # noqa: BLE001
        for ax in fig.axes:
            ax.spines[["top", "right"]].set_visible(False)


def _show_and_export(fig, basename: str, key: str, *, despine: bool = True):
    """Render a figure in the page and offer publication PNG + vector SVG export."""
    if despine:
        _prism_despine(fig)
    fig.tight_layout()
    png = BytesIO(); fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    svg = BytesIO(); fig.savefig(svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    st.image(png.getvalue(), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ PNG (300 dpi)", png.getvalue(),
                       f"{basename}.png", "image/png", key=f"{key}_png")
    c2.download_button("⬇️ SVG (vector)", svg.getvalue(),
                       f"{basename}.svg", "image/svg+xml", key=f"{key}_svg")


def apply_size_filter(df: pd.DataFrame, has_otype: bool, vol_rng, len_rng,
                      drop_na: bool) -> pd.DataFrame:
    """Drop cilia rows whose volume / length fall outside the given ranges.

    Only ``object_type == 'cilia'`` rows are filtered (basal bodies have no
    volume/length and are always kept). NaN sizes are kept unless ``drop_na``.
    """
    if df.empty:
        return df
    is_cil = (df["object_type"] == "cilia") if has_otype else pd.Series(True, index=df.index)
    keep = pd.Series(True, index=df.index)
    for rng, col in ((vol_rng, "volume_um3"), (len_rng, "length_um")):
        if rng is None or col not in df.columns:
            continue
        within = df[col].between(rng[0], rng[1])
        na = df[col].isna()
        col_keep = within | (na & (not drop_na))
        keep &= (~is_cil) | col_keep
    return df[keep]


@st.cache_data(show_spinner=False)
def load_qc_sheets(run_dir: str, mtime: float) -> dict[str, pd.DataFrame]:
    """The pipeline's own QC sheets, if present (``qc_per_sample``/``qc_global``)."""
    _ = mtime
    path = os.path.join(run_dir, "csv", "all_cilia_features.xlsx")
    out = {}
    for sh in ("qc_per_sample", "qc_global"):
        try:
            out[sh] = pd.read_excel(path, sheet_name=sh)
        except Exception:                                 # noqa: BLE001
            out[sh] = pd.DataFrame()
    return out


# QC flags on detection counts (not log_ratio): a sample is suspicious if it has
# unusually many/few detected cilia or basal bodies relative to its run.
_QC_METRICS = [("n_cilia", "cilia count"),
               ("n_bb", "basal body count")]


def per_sample_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Per (run, sample) detection counts for cilia and basal bodies (plus a few
    informational stats). Built from the full per-object table."""
    if df.empty:
        return pd.DataFrame()
    sample_col = "file_short" if "file_short" in df.columns else "filename"
    keys = [c for c in ("run", sample_col) if c in df.columns]
    has_ot = "object_type" in df.columns
    rows = []
    for kv, g in df.groupby(keys):
        kv = kv if isinstance(kv, tuple) else (kv,)
        rec = dict(zip(keys, kv))
        if has_ot:
            rec["n_cilia"] = int((g["object_type"] == "cilia").sum())
            rec["n_bb"] = int((g["object_type"] == "basal_body").sum())
            cil = g[g["object_type"] == "cilia"]
        else:
            rec["n_cilia"] = int(len(g))
            rec["n_bb"] = 0
            cil = g
        if "log_ratio" in cil.columns and len(cil):
            rec["mean_log_ratio"] = float(cil["log_ratio"].dropna().mean())
        if "class" in cil.columns:
            vc = cil["class"].value_counts()
            for cls in ("axon", "soma", "ambiguous"):
                rec[f"n_{cls}"] = int(vc.get(cls, 0))
        rows.append(rec)
    return pd.DataFrame(rows)


def flag_deviant(qc: pd.DataFrame, group: str = "run", z_thresh: float = 3.5) -> pd.DataFrame:
    """Add a ``flags`` column marking robust outliers (modified z-score on
    median/MAD) **within each run** (needs >= 4 samples in a run to judge)."""
    qc = qc.copy()
    if qc.empty:
        qc["flags"] = []
        return qc
    flags = {i: [] for i in qc.index}
    grouper = qc.groupby(group) if group in qc.columns else [("all", qc)]
    for _, g in grouper:
        if len(g) < 4:
            continue
        for key, nice in _QC_METRICS:
            if key not in g.columns:
                continue
            vals = g[key].to_numpy(dtype=float)
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
            for pos, idx in enumerate(g.index):
                if fin[pos] and abs(score[pos]) > thr:
                    flags[idx].append(f"{nice} {'high' if vals[pos] > med else 'low'}")
    qc["flags"] = [", ".join(flags[i]) for i in qc.index]
    return qc


def _qc_bar_fig(qc: pd.DataFrame):
    """Grouped bar chart of detected cilia & basal bodies per sample; flagged
    samples get a red, bold x-axis label."""
    sample_col = "file_short" if "file_short" in qc.columns else "filename"
    labels = [
        (f"{r['run']}:{r[sample_col]}" if "run" in qc.columns else str(r[sample_col]))
        for _, r in qc.iterrows()
    ]
    x = np.arange(len(qc))
    has_bb = "n_bb" in qc.columns
    w = 0.4 if has_bb else 0.7
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(qc)), 4.5))
    ax.bar(x - (w / 2 if has_bb else 0), qc["n_cilia"], width=w,
           color="#0C5DA5", label="cilia")
    if has_bb:
        ax.bar(x + w / 2, qc["n_bb"], width=w, color="#FF9500", label="basal bodies")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    for tick, (_, r) in zip(ax.get_xticklabels(), qc.iterrows()):
        if r["flags"]:
            tick.set_color("#e74c3c")
            tick.set_fontweight("bold")
    ax.set_ylabel("detected per sample")
    ax.set_title("Detected cilia & basal bodies per sample  (red label = flagged)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — run management (add / label / compare / remove)
# ─────────────────────────────────────────────────────────────────────────────
st.session_state.setdefault("runs", [])          # [{"dir":..., "label":...}]

with st.sidebar:
    st.header("🍋 Runs")
    st.caption("Add one or more finished `lc-analysis-*` runs to explore and compare.")

    base = st.text_input(
        "Output folder or run directory",
        value=st.session_state.get("_base", ""),
        help="A folder containing lc-analysis-* runs, or a single run folder.",
    )
    st.session_state["_base"] = base

    found = discover_runs(base)
    if base and not found:
        st.info("No `lc-analysis-*` runs with results found at that path.")

    if found:
        added_dirs = {r["dir"] for r in st.session_state["runs"]}
        choices = [name for name, d in found.items() if d not in added_dirs]
        c1, c2 = st.columns([3, 1])
        with c1:
            pick = st.multiselect("Available runs", choices, key="pick_runs")
        with c2:
            st.write("")
            st.write("")
            if st.button("➕ Add", use_container_width=True):
                for name in pick:
                    st.session_state["runs"].append(
                        {"dir": found[name], "label": name}
                    )
                st.rerun()
        if st.button("➕ Add latest run", use_container_width=True):
            latest = find_latest_run_dir(base) if not _is_run_dir(base) else base
            if latest and latest not in added_dirs:
                st.session_state["runs"].append(
                    {"dir": latest, "label": os.path.basename(latest)}
                )
                st.rerun()

    st.markdown("---")
    if not st.session_state["runs"]:
        st.info("No runs added yet.")
    else:
        st.subheader("Loaded runs")
        for i, r in enumerate(list(st.session_state["runs"])):
            with st.container(border=True):
                r["label"] = st.text_input(
                    "Label", value=r["label"], key=f"lbl_{i}",
                )
                st.caption(os.path.basename(r["dir"]))
                if st.button("🗑️ Remove", key=f"rm_{i}", use_container_width=True):
                    st.session_state["runs"].pop(i)
                    st.rerun()
        if st.button("Clear all runs", use_container_width=True):
            st.session_state["runs"] = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREP
# ─────────────────────────────────────────────────────────────────────────────
_runs = st.session_state["runs"]
_df = combined_dataframe(_runs)
_ready = not _df.empty

st.title("Limoncello — Data Analysis 🍋📊")

if not _ready:
    st.info(
        "👈 Add one or more runs from the sidebar to begin. "
        "Point it at the **Output folder** you used in the pipeline app "
        "(it contains the `lc-analysis-*` run folders)."
    )
    st.stop()

# dt_neurite / dt_nuclei are already physical µm (the pipeline's distance
# transforms use sampling=voxel_size). Rename them with an explicit _um suffix so
# they read as micrometres everywhere, like distance_to_neurite_um.
_df = _df.rename(columns={"dt_neurite": "dt_neurite_um", "dt_nuclei": "dt_nuclei_um"})

# ── Reclassify axon / soma on the fly from log_ratio thresholds ──────────────
with st.sidebar:
    st.markdown("---")
    st.header("🏷️ Classification")
    st.caption("Re-threshold log_ratio → axon / soma / ambiguous, live across all tabs.")
    _axon_thr = st.number_input("Axon if log_ratio >", value=2.5, step=0.1,
                                format="%.2f", key="cls_axon")
    _soma_thr = st.number_input("Soma if log_ratio <", value=1.0, step=0.1,
                                format="%.2f", key="cls_soma")
    if _soma_thr > _axon_thr:
        st.warning("Soma threshold is above the axon threshold — nothing will be "
                   "classified 'ambiguous'.")

if "log_ratio" in _df.columns:
    _df["class"] = reclassify(_df["log_ratio"], _axon_thr, _soma_thr)
    _cls_counts = _df.loc[_df.get("object_type", "cilia").eq("cilia")
                          if "object_type" in _df.columns else slice(None), "class"]
    st.sidebar.caption("Cilia: " + "  ".join(
        f"{k}={int((_cls_counts == k).sum())}" for k in ("axon", "soma", "ambiguous")))

_has_otype = "object_type" in _df.columns

# ── Cilia size filter — remove false positives by 3-D volume / length ─────────
# Slider bounds come from the *unfiltered* data so they stay stable across reruns.
_cil_raw = _df[_df["object_type"] == "cilia"] if _has_otype else _df
with st.sidebar:
    st.markdown("---")
    st.header("🔧 Cilia size filter")
    st.caption("Remove false-positive cilia by 3-D volume / length. "
               "Applies to every tab and the report.")

    def _size_slider(col, label, unit):
        if col not in _cil_raw.columns:
            st.caption(f"{label}: column `{col}` not in data.")
            return None
        vals = _cil_raw[col].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            st.caption(f"{label}: no values.")
            return None
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            st.caption(f"{label}: single value ({lo:.3g} {unit}).")
            return None
        return st.slider(f"{label} ({unit})", lo, hi, (lo, hi),
                         step=(hi - lo) / 100.0, key=f"flt_{col}")

    _vol_rng = _size_slider("volume_um3", "Volume", "µm³")
    _len_rng = _size_slider("length_um", "Length", "µm")
    _drop_na = st.checkbox("Drop cilia with missing size", value=False, key="flt_dropna")

_n_cil_before = int((_df["object_type"] == "cilia").sum()) if _has_otype else len(_df)
_df = apply_size_filter(_df, _has_otype, _vol_rng, _len_rng, _drop_na)
_n_cil_after = int((_df["object_type"] == "cilia").sum()) if _has_otype else len(_df)
if _n_cil_after < _n_cil_before:
    st.sidebar.success(
        f"🔧 Size filter kept {_n_cil_after} / {_n_cil_before} cilia "
        f"({_n_cil_before - _n_cil_after} removed)."
    )

# Convenience masks/columns (computed AFTER the size filter)
_cilia = _df[_df["object_type"] == "cilia"] if _has_otype else _df
_num_cols = _df.select_dtypes(include=[np.number]).columns.tolist()
_cat_cols = [c for c in _df.columns if c not in _num_cols]
_group_opts = [c for c in ("run", "class", "file_short", "filename") if c in _df.columns]

# Compact comparison header
_mcols = st.columns(len(_runs) + 1)
with _mcols[0]:
    st.metric("Runs", len(_runs))
for _i, _r in enumerate(_runs, 1):
    _sub = _df[_df["run"] == _r["label"]]
    _nc = int((_sub["object_type"] == "cilia").sum()) if _has_otype else len(_sub)
    if _i < len(_mcols):
        with _mcols[_i]:
            st.metric(_r["label"][:18], f"{_nc} cilia")


tab_over, tab_dist, tab_scatter, tab_corr, tab_table, tab_qc, tab_report = st.tabs(
    ["🔬 Overlays", "📊 Hist / KDE", "🟢 Scatter", "🔗 Correlation",
     "📋 Data table", "🩺 QC", "📄 Report"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB — OVERLAYS (3-D napari screenshots + ROI gallery)
# ─────────────────────────────────────────────────────────────────────────────
with tab_over:
    st.subheader("🔬 3-D napari overlays")
    _run_labels = [r["label"] for r in _runs]
    oc1, oc2 = st.columns(2)
    with oc1:
        _sel_label = st.selectbox("Run", _run_labels, key="ov_run")
    _sel_run = next(r for r in _runs if r["label"] == _sel_label)
    _stems = sample_stems(_df[_df["run"] == _sel_label])
    with oc2:
        _stem = st.selectbox("Sample", _stems, key="ov_sample") if _stems else None

    if not _stem:
        st.info("No samples found for this run.")
    else:
        imgs = overview_images(_sel_run["dir"], _stem)
        if not imgs:
            st.warning(
                "No napari overview screenshots for this sample. They are produced "
                "by batch runs with **Capture screenshots** enabled."
            )
        else:
            cols = st.columns(len(imgs))
            for col, tag in zip(cols, [t for t in _OVERVIEW_TAGS if t in imgs]):
                with col:
                    st.caption(_OVERVIEW_LABEL[tag])
                    st.image(imgs[tag], use_container_width=True)

        _roi_dir = os.path.join(_sel_run["dir"], "figures", "cilia_rois")
        _all_rois = (sorted(f for f in os.listdir(_roi_dir) if f.lower().endswith(".png"))
                     if os.path.isdir(_roi_dir) else [])
        rois = roi_images(_sel_run["dir"], _stem)
        # Fallback: if filename-stem matching finds nothing but the folder has
        # ROIs, show them all (helps when the on-disk stem differs slightly).
        _fallback = not rois and bool(_all_rois)
        if _fallback:
            rois = [os.path.join(_roi_dir, f) for f in _all_rois]

        st.markdown("---")
        st.subheader(f"🔎 Per-cilium ROIs ({len(rois)})")

        with st.expander("🛠️ ROI diagnostics", expanded=not rois):
            st.write(f"Folder: `{_roi_dir}`")
            st.write(f"Exists: **{os.path.isdir(_roi_dir)}** · "
                     f"PNG files in folder: **{len(_all_rois)}**")
            st.write(f"Looking for sample stem: `{_stem}`")
            if _all_rois:
                _stems_here = sorted({f.split('_cilia')[0] for f in _all_rois if '_cilia' in f})
                st.write("Distinct stems present in folder:")
                st.code("\n".join(_stems_here) or "(none parsed)")
            if _fallback:
                st.warning("No exact stem match — showing **all** ROIs in this run's "
                           "folder instead. The sample selector above may use a "
                           "different name than the saved files.")

        if not rois:
            st.caption(
                "No per-cilium ROI screenshots found. They are only produced by a "
                "**batch** run with both *Capture screenshots* **and** *also save "
                "per-cilium 3-D ROIs* enabled."
            )
        else:
            _per_row = st.slider("Thumbnails per row", 2, 8, 4, key="roi_cols")
            _max = st.slider("Max ROIs to show", 4, max(4, len(rois)),
                             min(24, len(rois)), key="roi_max")
            grid = rois[:_max]
            for start in range(0, len(grid), _per_row):
                row = grid[start:start + _per_row]
                for col, p in zip(st.columns(_per_row), row):
                    with col:
                        st.image(p, caption=os.path.basename(p),
                                 use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB — HIST / KDE
# ─────────────────────────────────────────────────────────────────────────────
with tab_dist:
    st.subheader("📊 Distribution (compare runs)")
    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1:
        _def_m = "log_ratio" if "log_ratio" in _num_cols else (_num_cols[0] if _num_cols else None)
        _metric = st.selectbox("Metric", _num_cols,
                               index=_num_cols.index(_def_m) if _def_m else 0,
                               key="d_metric") if _num_cols else None
    with dc2:
        _grp = st.selectbox("Compare by", _group_opts,
                            index=_group_opts.index("run") if "run" in _group_opts else 0,
                            key="d_group")
    with dc3:
        _kind = st.radio("Style", ["hist + kde", "kde only", "violin + points"],
                         key="d_kind")
    with dc4:
        _only_cilia = st.checkbox("Cilia only", value=True, key="d_cilia")

    # ── Axis normalization ───────────────────────────────────────────────────
    nc1, nc2, nc3 = st.columns(3)
    with nc1:
        _norm_method = st.selectbox("Normalize values",
                                    ["none", "z-score", "min–max", "robust"],
                                    key="d_norm",
                                    help="Rescale the metric so runs on different "
                                         "scales are comparable.")
    with nc2:
        _norm_pergroup = st.checkbox("Normalize per group", value=False, key="d_norm_grp",
                                     help="Normalize within each compared group "
                                          "(compare distribution shapes).")
    with nc3:
        _stat = st.selectbox("Frequency axis", ["count", "density", "probability"],
                             key="d_stat",
                             help="Normalize the y-axis (density/probability) so "
                                  "groups with different N are comparable.")

    _data = _cilia if (_only_cilia and _has_otype) else _df
    if _metric is None or _data.empty:
        st.info("Nothing to plot.")
    else:
        # Apply value normalization to the plotted metric (global or per group).
        _xlabel = _metric
        if _norm_method != "none":
            _data = _data.copy()
            if _norm_pergroup and _grp in _data.columns:
                _data[_metric] = _data.groupby(_grp)[_metric].transform(
                    lambda x: normalize_axis(x, _norm_method))
            else:
                _data[_metric] = normalize_axis(_data[_metric], _norm_method)
            _xlabel = f"{_metric} [{_norm_method}{' per group' if _norm_pergroup else ''}]"
        fig, ax = plt.subplots(figsize=(7.5, 5))
        try:
            if _kind == "hist + kde":
                sns.histplot(data=_data, x=_metric, hue=_grp, kde=True,
                             common_norm=False, stat=_stat, ax=ax,
                             palette=_PRISM_PALETTE, alpha=0.45, element="step",
                             line_kws={"linewidth": 2})
                ax.set_xlabel(_xlabel)
                _tidy_legend(ax, title=_grp)
            elif _kind == "kde only":
                sns.kdeplot(data=_data, x=_metric, hue=_grp, common_norm=False,
                            fill=True, alpha=0.3, ax=ax, palette=_PRISM_PALETTE,
                            linewidth=2)
                ax.set_xlabel(_xlabel)
                _tidy_legend(ax, title=_grp)
            else:  # violin + points (GraphPad-style: violin with jittered raw points)
                sns.violinplot(data=_data, x=_grp, y=_metric, hue=_grp,
                               legend=False, palette=_PRISM_PALETTE, cut=0,
                               inner=None, linewidth=1.2, saturation=0.85, ax=ax)
                # de-saturate the violins so the overlaid points stand out
                for _coll in ax.collections:
                    _coll.set_alpha(0.35)
                sns.stripplot(data=_data, x=_grp, y=_metric, ax=ax,
                              color="#222222", size=2.5, alpha=0.5, jitter=0.25)
                ax.set_xlabel(_grp)
                ax.set_ylabel(_xlabel)
                ax.tick_params(axis="x", rotation=30)
            _show_and_export(fig, f"dist_{_metric}", "dl_dist")
        except Exception as exc:                          # noqa: BLE001
            st.error(f"Plot error: {exc}")
            plt.close(fig)

        # Per-group summary stats for the chosen metric
        if _grp in _data.columns:
            _stats = (_data.groupby(_grp)[_metric]
                      .agg(["count", "mean", "median", "std"])
                      .reset_index())
            st.dataframe(_stats, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB — SCATTER
# ─────────────────────────────────────────────────────────────────────────────
with tab_scatter:
    st.subheader("🟢 Scatter (compare runs)")
    if len(_num_cols) < 2:
        st.info("Need at least two numeric columns.")
    else:
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            _xi = _num_cols.index("dt_neurite_um") if "dt_neurite_um" in _num_cols else 0
            _sx = st.selectbox("X", _num_cols, index=_xi, key="s_x")
        with sc2:
            _yi = _num_cols.index("dt_nuclei_um") if "dt_nuclei_um" in _num_cols else 1
            _sy = st.selectbox("Y", _num_cols, index=_yi, key="s_y")
        with sc3:
            _color_opts = [c for c in ("run", "class", "file_short") if c in _df.columns] or ["run"]
            _sc_color = st.selectbox("Color by", _color_opts, key="s_color")
        with sc4:
            _sc_cilia = st.checkbox("Cilia only", value=True, key="s_cilia")

        # ── Per-axis normalization (independent X and Y) ──────────────────────
        sn1, sn2, sn3 = st.columns(3)
        with sn1:
            _sx_norm = st.selectbox("Normalize X", ["none", "z-score", "min–max", "robust"],
                                    key="s_xnorm")
        with sn2:
            _sy_norm = st.selectbox("Normalize Y", ["none", "z-score", "min–max", "robust"],
                                    key="s_ynorm")
        with sn3:
            _sn_pergroup = st.checkbox("Normalize per group", value=False, key="s_norm_grp",
                                       help=f"Normalize within each '{_sc_color}' group.")

        _sdata = _cilia if (_sc_cilia and _has_otype) else _df
        _sx_lab, _sy_lab = _sx, _sy
        if _sx_norm != "none" or _sy_norm != "none":
            _sdata = _sdata.copy()
            for _col, _m, _setlab in ((_sx, _sx_norm, "x"), (_sy, _sy_norm, "y")):
                if _m == "none":
                    continue
                if _sn_pergroup and _sc_color in _sdata.columns:
                    _sdata[_col] = _sdata.groupby(_sc_color)[_col].transform(
                        lambda v: normalize_axis(v, _m))
                else:
                    _sdata[_col] = normalize_axis(_sdata[_col], _m)
                _lab = f"{_col} [{_m}{' per group' if _sn_pergroup else ''}]"
                if _setlab == "x":
                    _sx_lab = _lab
                else:
                    _sy_lab = _lab

        _pal = _CLASS_PALETTE if _sc_color == "class" else _PRISM_PALETTE
        fig, ax = plt.subplots(figsize=(7.5, 6))
        try:
            sns.scatterplot(data=_sdata, x=_sx, y=_sy, hue=_sc_color,
                            palette=_pal, alpha=0.7, s=28, ax=ax,
                            edgecolor="white", linewidth=0.3)
            ax.set_xlabel(_sx_lab)
            ax.set_ylabel(_sy_lab)
            _tidy_legend(ax, title=_sc_color)
            _show_and_export(fig, f"scatter_{_sx}_{_sy}", "dl_scatter")
        except Exception as exc:                          # noqa: BLE001
            st.error(f"Plot error: {exc}")
            plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# TAB — CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
with tab_corr:
    st.subheader("🔗 Correlation heatmap")
    _default_corr = [c for c in ("log_ratio", "ratio", "distance_to_neurite_um",
                                 "dt_neurite_um", "dt_nuclei_um", "volume_um3",
                                 "length_um", "pair_distance_um") if c in _num_cols]
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        _corr_cols = st.multiselect("Variables", _num_cols,
                                    default=_default_corr or _num_cols[:6],
                                    key="corr_cols")
    with cc2:
        _method = st.selectbox("Method", ["pearson", "spearman"], key="corr_method")
    with cc3:
        _corr_run = st.selectbox("Run", ["All"] + [r["label"] for r in _runs],
                                 key="corr_run")

    _cdata = _cilia if _has_otype else _df
    if _corr_run != "All":
        _cdata = _cdata[_cdata["run"] == _corr_run]

    if len(_corr_cols) < 2:
        st.info("Pick at least two variables.")
    elif _cdata.empty:
        st.info("No data for this selection.")
    else:
        _corr = _cdata[_corr_cols].corr(method=_method)
        _sz = 0.8 * len(_corr_cols)
        fig, ax = plt.subplots(figsize=(_sz + 2.2, _sz + 1.6))
        _mask = np.triu(np.ones_like(_corr, dtype=bool), k=1)  # lower triangle only
        sns.heatmap(_corr, mask=_mask, annot=True, fmt=".2f", cmap="vlag",
                    center=0, vmin=-1, vmax=1, square=True,
                    linewidths=0.6, linecolor="white",
                    cbar_kws={"shrink": 0.75, "label": f"{_method} r"},
                    annot_kws={"size": 8}, ax=ax)
        ax.set_title(f"Correlation ({_method}) — n = {len(_cdata):,}", fontweight="bold")
        ax.tick_params(length=0)
        _show_and_export(fig, f"correlation_{_method}", "dl_corr", despine=False)


# ─────────────────────────────────────────────────────────────────────────────
# TAB — DATA TABLE
# ─────────────────────────────────────────────────────────────────────────────
with tab_table:
    st.subheader("📋 Per-object table")
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        _f_run = st.multiselect("Run", [r["label"] for r in _runs],
                                default=[r["label"] for r in _runs], key="t_run")
    with fc2:
        _otypes = ["All"] + (sorted(_df["object_type"].dropna().unique()) if _has_otype else [])
        _f_ot = st.selectbox("Object type", _otypes, key="t_ot")
    with fc3:
        _classes = ["All"] + (sorted(_df["class"].dropna().unique()) if "class" in _df.columns else [])
        _f_cls = st.selectbox("Class", _classes, key="t_cls")
    with fc4:
        _files = ["All"] + (sorted(_df["filename"].dropna().unique()) if "filename" in _df.columns else [])
        _f_file = st.selectbox("Sample", _files, key="t_file")

    _view = _df[_df["run"].isin(_f_run)] if _f_run else _df.iloc[0:0]
    if _f_ot != "All" and _has_otype:
        _view = _view[_view["object_type"] == _f_ot]
    if _f_cls != "All" and "class" in _view.columns:
        _view = _view[_view["class"] == _f_cls]
    if _f_file != "All" and "filename" in _view.columns:
        _view = _view[_view["filename"] == _f_file]

    # Put the most useful columns first
    _priority = ["run", "file_short", "filename", "object_type", "cilia_id", "class",
                 "log_ratio", "ratio", "distance_to_neurite_um", "dt_neurite_um",
                 "dt_nuclei_um", "volume_um3", "length_um", "paired_id",
                 "pair_distance_um", "pairing_status"]
    _order = [c for c in _priority if c in _view.columns]
    _order += [c for c in _view.columns if c not in _order]
    _view = _view[_order]

    st.caption(f"Showing {len(_view):,} of {len(_df):,} objects")
    st.dataframe(_view, use_container_width=True, height=420)
    st.download_button("⬇️ Download filtered (CSV)",
                       _view.to_csv(index=False).encode(),
                       "objects_filtered.csv", "text/csv", key="dl_tbl")

    # Per-run / per-sample summary
    st.markdown("---")
    st.subheader("🧫 Summary")
    if _has_otype:
        _cil = _df[_df["object_type"] == "cilia"]
    else:
        _cil = _df
    if not _cil.empty and "log_ratio" in _cil.columns:
        _agg = {"n_cilia": ("cilia_id", "count") if "cilia_id" in _cil.columns else ("log_ratio", "count"),
                "mean_log_ratio": ("log_ratio", "mean"),
                "std_log_ratio": ("log_ratio", "std")}
        _by = [c for c in ("run", "file_short") if c in _cil.columns] or ["run"]
        _summary = _cil.groupby(_by).agg(**_agg).reset_index()
        st.dataframe(_summary, use_container_width=True)
        st.download_button("⬇️ Download summary (CSV)",
                           _summary.to_csv(index=False).encode(),
                           "summary.csv", "text/csv", key="dl_sum")


# ─────────────────────────────────────────────────────────────────────────────
# TAB — QC
# ─────────────────────────────────────────────────────────────────────────────
with tab_qc:
    st.subheader("🩺 Quality control")
    st.caption("Samples are flagged on **detected counts** of cilia and basal "
               "bodies (robust modified z-score within each run).")
    _qc = flag_deviant(per_sample_qc(_df))
    if _qc.empty:
        st.info("No data available for QC.")
    else:
        _n_flagged = int((_qc["flags"] != "").sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Samples", len(_qc))
        m2.metric("Flagged", _n_flagged)
        m3.metric("Total cilia", int(_qc["n_cilia"].sum()))
        if "n_bb" in _qc.columns:
            m4.metric("Total basal bodies", int(_qc["n_bb"].sum()))

        if len(_qc) < 4:
            st.caption("⚠ Fewer than 4 samples per run — too few to flag statistical outliers.")
        elif _n_flagged:
            st.warning(
                "Deviant samples (robust modified z-score > 3.5 within their run): "
                + ", ".join(
                    f"**{r.get('file_short', r.get('filename'))}** ({r['flags']})"
                    for _, r in _qc[_qc["flags"] != ""].iterrows()
                )
            )
        else:
            st.success("✓ No deviant samples — all metrics within robust range.")

        st.pyplot(_qc_bar_fig(_qc), use_container_width=True)

        # Per-sample QC table, flagged rows highlighted
        def _hl(row):
            return ["background-color: #fdecea" if row["flags"] else "" for _ in row]
        st.dataframe(_qc.style.apply(_hl, axis=1), use_container_width=True)
        st.download_button("⬇️ Download QC table (CSV)",
                           _qc.to_csv(index=False).encode(),
                           "qc_per_sample.csv", "text/csv", key="dl_qc")

        # The pipeline's own QC sheets (per run), shown verbatim
        with st.expander("📑 Pipeline QC sheets (per run)", expanded=False):
            for r in _runs:
                sheets = load_qc_sheets(r["dir"], _excel_mtime(r["dir"]))
                st.markdown(f"**{r['label']}**")
                if not sheets["qc_global"].empty:
                    st.caption("qc_global")
                    st.dataframe(sheets["qc_global"], use_container_width=True)
                if not sheets["qc_per_sample"].empty:
                    st.caption("qc_per_sample")
                    st.dataframe(sheets["qc_per_sample"], use_container_width=True)
                if sheets["qc_global"].empty and sheets["qc_per_sample"].empty:
                    st.caption("— no QC sheets saved for this run —")


# ─────────────────────────────────────────────────────────────────────────────
# TAB — PDF REPORT (3-D images + comparison plots)
# ─────────────────────────────────────────────────────────────────────────────
def _qc_overview_page(pdf, qc: pd.DataFrame):
    """One QC page: aggregate text + cilia-per-sample bar + flagged list."""
    if qc.empty:
        return
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("QC overview", fontsize=15, fontweight="bold")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.35)

    ax_txt = fig.add_subplot(gs[0]); ax_txt.axis("off")
    flagged = qc[qc["flags"] != ""]
    lines = [
        f"Samples            : {len(qc)}",
        f"Flagged samples    : {len(flagged)}",
        f"Total cilia        : {int(qc['n_cilia'].sum())}",
    ]
    if "n_bb" in qc.columns:
        lines.append(f"Total basal bodies : {int(qc['n_bb'].sum())}")
    lines.append("")
    if len(qc) < 4:
        lines.append("Too few samples (<4) to flag statistical outliers.")
    elif len(flagged):
        lines.append("Deviant on detection counts (robust z > 3.5):")
        sample_col = "file_short" if "file_short" in qc.columns else "filename"
        for _, r in flagged.iterrows():
            lines.append(f"  • {r[sample_col]}  →  {r['flags']}")
    else:
        lines.append("No deviant samples detected.")
    ax_txt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=11, transform=ax_txt.transAxes)

    sample_col = "file_short" if "file_short" in qc.columns else "filename"
    labels = [(f"{r['run']}:{r[sample_col]}" if "run" in qc.columns else str(r[sample_col]))
              for _, r in qc.iterrows()]
    x = np.arange(len(qc))
    has_bb = "n_bb" in qc.columns
    w = 0.4 if has_bb else 0.7
    ax_bar = fig.add_subplot(gs[1])
    ax_bar.bar(x - (w / 2 if has_bb else 0), qc["n_cilia"], width=w,
               color="#0C5DA5", label="cilia")
    if has_bb:
        ax_bar.bar(x + w / 2, qc["n_bb"], width=w, color="#FF9500", label="basal bodies")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=90, fontsize=7)
    for tick, (_, r) in zip(ax_bar.get_xticklabels(), qc.iterrows()):
        if r["flags"]:
            tick.set_color("#e74c3c")
            tick.set_fontweight("bold")
    ax_bar.set_ylabel("detected per sample")
    ax_bar.set_title("Detected cilia & basal bodies per sample  (red label = flagged)",
                     fontsize=10)
    ax_bar.legend(fontsize=8)
    ax_bar.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def _img_page(pdf, title: str, image_paths: list[str], subtitles: list[str],
              ncols: int = 3):
    if not image_paths:
        return
    nrows = int(np.ceil(len(image_paths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.4 * nrows + 0.5))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(title, fontsize=11, fontweight="bold")
    for ax, p, sub in zip(axes, image_paths, subtitles):
        try:
            ax.imshow(plt.imread(p))
        except Exception:                                 # noqa: BLE001
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.set_title(sub, fontsize=8)
        ax.axis("off")
    for ax in axes[len(image_paths):]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig)
    plt.close(fig)


def build_report_pdf(runs, df, *, include_qc, include_overviews, include_rois,
                     roi_cap, metric) -> bytes:
    buf = BytesIO()
    cilia = df[df["object_type"] == "cilia"] if "object_type" in df.columns else df
    with PdfPages(buf) as pdf:
        # ── Title page ──────────────────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.92, "Limoncello — Data Report 🍋", ha="center",
                 fontsize=18, fontweight="bold")
        lines = [f"Runs compared: {len(runs)}", ""]
        for r in runs:
            sub = cilia[cilia["run"] == r["label"]] if "run" in cilia.columns else cilia
            lines.append(f"• {r['label']} — {len(sub)} cilia")
        fig.text(0.1, 0.78, "\n".join(lines), fontsize=11, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        # ── QC overview (detection counts) ──────────────────────────────────
        if include_qc:
            _qc_overview_page(pdf, flag_deviant(per_sample_qc(df)))

        # ── Comparison plots ────────────────────────────────────────────────
        if metric in cilia.columns and "run" in cilia.columns and not cilia.empty:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            sns.kdeplot(data=cilia, x=metric, hue="run", common_norm=False,
                        fill=True, alpha=0.3, ax=axes[0], palette=_PRISM_PALETTE,
                        linewidth=2)
            axes[0].set_title(f"{metric} — KDE per run")
            sns.boxplot(data=cilia, x="run", y=metric, ax=axes[1],
                        palette=_PRISM_PALETTE, linewidth=1.2, fliersize=2)
            axes[1].set_title(f"{metric} per run")
            axes[1].tick_params(axis="x", rotation=30)
            _prism_despine(fig, trim=False)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        if {"dt_neurite_um", "dt_nuclei_um", "run"}.issubset(cilia.columns) and not cilia.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=cilia, x="dt_neurite_um", y="dt_nuclei_um", hue="run",
                            alpha=0.7, s=22, ax=ax, palette=_PRISM_PALETTE,
                            edgecolor="white", linewidth=0.3)
            ax.set_title("dt_neurite_um vs dt_nuclei_um")
            _prism_despine(fig)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ── Per-run / per-sample 3-D images ─────────────────────────────────
        for r in runs:
            rdf = df[df["run"] == r["label"]] if "run" in df.columns else df
            if include_overviews:
                for stem in sample_stems(rdf):
                    ov = overview_images(r["dir"], stem)
                    if ov:
                        paths = [ov[t] for t in _OVERVIEW_TAGS if t in ov]
                        subs = [_OVERVIEW_LABEL[t] for t in _OVERVIEW_TAGS if t in ov]
                        _img_page(pdf, f"{r['label']} — {stem}", paths, subs,
                                  ncols=max(1, len(paths)))
            if include_rois:
                # Scan the ROI folder directly (robust to stem-naming differences):
                # group every saved ROI png by its on-disk sample stem.
                roi_dir = os.path.join(r["dir"], "figures", "cilia_rois")
                if os.path.isdir(roi_dir):
                    by_stem: dict[str, list[str]] = {}
                    for f in sorted(os.listdir(roi_dir)):
                        if "_cilia" in f and f.lower().endswith(".png"):
                            by_stem.setdefault(f.split("_cilia")[0], []).append(
                                os.path.join(roi_dir, f))
                    for s, paths in by_stem.items():
                        paths = paths[:roi_cap]
                        _img_page(pdf, f"{r['label']} — {s} — ROIs", paths,
                                  [os.path.basename(p) for p in paths], ncols=3)
    return buf.getvalue()


with tab_report:
    st.subheader("📄 PDF report")
    st.caption("Bundles the 3-D napari images and run-comparison plots into one PDF.")
    rc0, rc1, rc2, rc3 = st.columns(4)
    with rc0:
        _inc_qc = st.checkbox("Include QC overview", value=True, key="r_qc")
    with rc1:
        _inc_ov = st.checkbox("Include 3-D overviews", value=True, key="r_ov")
    with rc2:
        _inc_roi = st.checkbox("Include per-cilium ROIs", value=True, key="r_roi")
    with rc3:
        _roi_cap = st.number_input("Max ROIs per sample", 1, 200, 12, key="r_cap")
    _rmetric = st.selectbox(
        "Comparison metric",
        [c for c in ("log_ratio", "ratio", "distance_to_neurite_um") if c in _num_cols]
        or _num_cols,
        key="r_metric",
    )

    if st.button("🖨️ Build PDF report", type="primary", key="r_build"):
        with st.spinner("Rendering report …"):
            try:
                pdf_bytes = build_report_pdf(
                    _runs, _df,
                    include_qc=_inc_qc,
                    include_overviews=_inc_ov, include_rois=_inc_roi,
                    roi_cap=int(_roi_cap), metric=_rmetric,
                )
                st.session_state["_report_pdf"] = pdf_bytes
                st.success("Report ready.")
            except Exception as exc:                       # noqa: BLE001
                st.error(f"Report failed: {exc}")
                st.session_state["_report_pdf"] = None

    if st.session_state.get("_report_pdf"):
        st.download_button("⬇️ Download report (PDF)",
                           st.session_state["_report_pdf"],
                           "limoncello_report.pdf", "application/pdf", key="dl_report")
