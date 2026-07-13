"""
Limoncello — Data Analysis app

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
import tempfile
import hashlib
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

try:
    import plotly.express as px
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG / CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Limoncello Data", layout="wide",
                   initial_sidebar_state="expanded")

_CLASS_PALETTE = {"neurite": "#2ecc71", "soma": "#e74c3c", "ambiguous": "#f39c12"}
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


def load_nuclei_summary(run_dir: str) -> pd.DataFrame | None:
    """Per-sample nuclei volume/count written by the pipeline, or None."""
    p = os.path.join(run_dir, "csv", "nuclei_summary.csv")
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception:                                 # noqa: BLE001
            return None
    return None


def combined_nuclei(runs: list[dict]) -> pd.DataFrame:
    """Concatenate each run's nuclei summary, tagged with a ``run`` column."""
    frames = []
    for r in runs:
        n = load_nuclei_summary(r["dir"])
        if n is not None and not n.empty:
            n = n.copy()
            n["run"] = r["label"]
            frames.append(n)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


_IMG_EXTS = (".ims", ".tif", ".tiff", ".czi", ".nd2", ".lif", ".png")


def _img_stem(f) -> str:
    """Filename without its image extension. Unlike ``Path.stem`` this keeps
    internal dots (e.g. ``E6_gCl.5_…``) intact — only a known image suffix is
    stripped — so hand (no extension) and pipeline (.ims) names match."""
    s = str(f).strip()
    low = s.lower()
    for ext in _IMG_EXTS:
        if low.endswith(ext):
            return s[: -len(ext)]
    return s


def parse_hand_analysis(src) -> pd.DataFrame:
    """Parse a per-sample hand-count Excel (path or uploaded file) into tidy
    columns. Finds the ``Filename`` header row (a stray merged row may precede
    it); counts may contain stray '?' marks which are coerced to numbers."""
    raw = pd.read_excel(src, sheet_name=0, header=None)
    if raw.empty:
        return pd.DataFrame()
    cols = ["filename", "hand_nuclei", "hand_centrosomes", "hand_ciliated_cells",
            "hand_pct_on_soma", "hand_cilia", "hand_on_soma",
            "hand_on_neurite_close", "hand_on_neurite_far",
            "hand_centrosomes_on_neurites", "comments"]
    # Locate the header row ("Filename" in the first column); data starts below.
    _hdr = next((i for i in range(min(6, len(raw)))
                 if str(raw.iloc[i, 0]).strip().lower() == "filename"), None)
    _start = (_hdr + 1) if _hdr is not None else 1
    body = raw.iloc[_start:, :len(cols)].copy()
    body.columns = cols[:body.shape[1]]
    body = body[body["filename"].notna()].copy()
    body = body[body["filename"].astype(str).str.strip().str.lower() != "filename"]
    for c in body.columns:
        if c not in ("filename", "comments"):
            body[c] = pd.to_numeric(
                body[c].astype(str).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
    body["stem"] = body["filename"].apply(_img_stem)
    body["hand_on_neurite"] = body[["hand_on_neurite_close",
                                    "hand_on_neurite_far"]].sum(axis=1, min_count=1)
    return body.reset_index(drop=True)


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


def _load_sample_mip(run_dir: str, stem: str, tag: str):
    """One saved XY-MIP array (``figures/mips/<stem>_<tag>_mip.npy``) or None."""
    p = os.path.join(run_dir, "figures", "mips", f"{stem}_{tag}_mip.npy")
    try:
        return np.load(p) if os.path.exists(p) else None
    except Exception:                                         # noqa: BLE001
        return None


def _parse_coords_yx(series):
    """(y, x) pixel positions from a cilia df ``coords`` column (lists or the
    ``"[z, y, x]"`` strings Excel round-trips to)."""
    from limoncello.analysis.pair_cilia_to_bb import parse_coords_string
    arr = parse_coords_string(series)                         # (n, 3) z,y,x
    return [(float(r[1]), float(r[2])) for r in arr]


# ── Human validation (manual cilia screening) ───────────────────────────────
# Decisions live in st.session_state (keyed by run/filename/cilia_id) and are
# persisted to ``<run_dir>/csv/human_validation.csv`` so they survive restarts.
def _val_file(run_dir: str) -> str:
    return os.path.join(run_dir, "csv", "human_validation.csv")


def load_validation_file(run_dir: str) -> dict[tuple[str, int], bool]:
    """``{(filename, cilia_id): human_validated}`` read from disk, or ``{}``."""
    p = _val_file(run_dir)
    out: dict[tuple[str, int], bool] = {}
    if os.path.exists(p):
        try:
            v = pd.read_csv(p)
            for _, r in v.iterrows():
                out[(str(r["filename"]), int(r["cilia_id"]))] = bool(r["human_validated"])
        except Exception:                                 # noqa: BLE001
            pass
    return out


def save_validation_file(run_dir: str, rows: list[dict]) -> str:
    """Write a run's decisions to ``human_validation.csv`` and return the path."""
    p = _val_file(run_dir)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    pd.DataFrame(rows, columns=["filename", "cilia_id", "human_validated"]).to_csv(
        p, index=False)
    return p


# ── Run label persisted as metadata, so a custom name survives across sessions ──
def _label_file(run_dir: str) -> str:
    return os.path.join(run_dir, "csv", "run_label.txt")


def load_run_label(run_dir: str) -> str | None:
    """Return the user's saved label for a run, or None if never set."""
    p = _label_file(run_dir)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return f.read().strip() or None
        except Exception:                                 # noqa: BLE001
            return None
    return None


def save_run_label(run_dir: str, label: str) -> None:
    try:
        os.makedirs(os.path.dirname(_label_file(run_dir)), exist_ok=True)
        with open(_label_file(run_dir), "w", encoding="utf-8") as f:
            f.write(str(label).strip())
    except Exception:                                     # noqa: BLE001
        pass


def reclassify(log_ratio: pd.Series, neurite_thr: float, soma_thr: float) -> pd.Series:
    """Class from log_ratio: 'neurite' if > neurite_thr, 'soma' if < soma_thr,
    else 'ambiguous' (soma wins in any overlap, applied last)."""
    lr = pd.to_numeric(log_ratio, errors="coerce")
    cls = pd.Series("ambiguous", index=lr.index, dtype=object)
    cls[lr > neurite_thr] = "neurite"
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


def _draw_mean_bars(ax, data, grp, metric, width: float = 0.3,
                    color: str = "#000000"):
    """Short black horizontal bar at each group's mean (GraphPad style)."""
    means = data.groupby(grp)[metric].mean()
    lut = {str(k): v for k, v in means.items()}
    for tick, lab in zip(ax.get_xticks(), ax.get_xticklabels()):
        key = lab.get_text()
        if key in lut and pd.notna(lut[key]):
            ax.hlines(lut[key], tick - width / 2, tick + width / 2,
                      color=color, linewidth=2.5, zorder=10)


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
    c1.download_button(":material/download: PNG (300 dpi)", png.getvalue(),
                       f"{basename}.png", "image/png", key=f"{key}_png")
    c2.download_button(":material/download: SVG (vector)", svg.getvalue(),
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
            for cls in ("neurite", "soma", "ambiguous"):
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
    st.header(":material/nutrition: Runs")
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
        # Show the saved label (if any) next to the folder so a run can be found
        # by the name you gave it in a previous session.
        _disp_to_name = {}
        for name, d in found.items():
            if d in added_dirs:
                continue
            _saved = load_run_label(d)
            _disp_to_name[f":material/label: {_saved}  ({name})" if _saved else name] = name
        c1, c2 = st.columns([3, 1])
        with c1:
            pick = st.multiselect("Available runs", list(_disp_to_name), key="pick_runs")
        with c2:
            st.write("")
            st.write("")
            if st.button(":material/add: Add", use_container_width=True):
                for disp in pick:
                    name = _disp_to_name[disp]
                    _dir = found[name]
                    st.session_state["runs"].append(
                        {"dir": _dir, "label": load_run_label(_dir) or name}
                    )
                st.rerun()
        if st.button(":material/add: Add latest run", use_container_width=True):
            latest = find_latest_run_dir(base) if not _is_run_dir(base) else base
            if latest and latest not in added_dirs:
                st.session_state["runs"].append(
                    {"dir": latest,
                     "label": load_run_label(latest) or os.path.basename(latest)}
                )
                st.rerun()

    st.markdown("---")
    if not st.session_state["runs"]:
        st.info("No runs added yet.")
    else:
        st.subheader("Loaded runs")
        for r in list(st.session_state["runs"]):
            # Key widgets by the stable run directory (not the list index): with
            # index keys, deleting a run shifts the indices and the surviving run
            # inherits the deleted run's stored label.
            _wkey = f"lbl_{r['dir']}"
            with st.container(border=True):
                _new_label = st.text_input("Label", value=r["label"], key=_wkey)
                # Relabeling keeps all screening data (decisions are stored on
                # disk per run dir and re-seeded under the new label) and saves
                # the new name as metadata so it persists across sessions.
                if _new_label and _new_label != r["label"]:
                    r["label"] = _new_label
                    save_run_label(r["dir"], _new_label)
                    st.rerun()
                st.caption(os.path.basename(r["dir"]))
                if st.button(":material/delete: Remove", key=f"rm_{r['dir']}",
                             use_container_width=True):
                    st.session_state["runs"] = [
                        x for x in st.session_state["runs"] if x["dir"] != r["dir"]]
                    try:                                  # drop the orphaned widget
                        del st.session_state[_wkey]
                    except Exception:                     # noqa: BLE001
                        pass
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

st.title(":material/nutrition: Limoncello — Data Analysis :material/bar_chart:")

if not _ready:
    st.info(
        ":material/keyboard_arrow_left: Add one or more runs from the sidebar to begin. "
        "Point it at the **Output folder** you used in the pipeline app "
        "(it contains the `lc-analysis-*` run folders)."
    )
    st.stop()

# dt_neurite / dt_nuclei are already physical µm (the pipeline's distance
# transforms use sampling=voxel_size). Rename them with an explicit _um suffix so
# they read as micrometres everywhere, like distance_to_neurite_um.
_df = _df.rename(columns={"dt_neurite": "dt_neurite_um", "dt_nuclei": "dt_nuclei_um"})

# ── Reclassify neurite / soma on the fly from log_ratio thresholds ───────────
with st.sidebar:
    st.markdown("---")
    st.header(":material/label: Classification")
    st.caption("Re-threshold log_ratio → neurite / soma / ambiguous, live across all tabs.")
    # Apply thresholds requested by the Validation optimizer (set before the
    # widgets instantiate, so it's allowed to write their keys).
    if "_apply_thr" in st.session_state:
        _sv, _nv = st.session_state.pop("_apply_thr")
        st.session_state["cls_soma"] = float(_sv)
        st.session_state["cls_neurite"] = float(_nv)
    _neurite_thr = st.number_input("Neurite if log_ratio >", value=0.1, step=0.1,
                                   format="%.2f", key="cls_neurite")
    _soma_thr = st.number_input("Soma if log_ratio <", value=-0.1, step=0.1,
                                format="%.2f", key="cls_soma")
    if _soma_thr > _neurite_thr:
        st.warning("Soma threshold is above the neurite threshold — nothing will be "
                   "classified 'ambiguous'.")

if "log_ratio" in _df.columns:
    _df["class"] = reclassify(_df["log_ratio"], _neurite_thr, _soma_thr)
    _cls_counts = _df.loc[_df.get("object_type", "cilia").eq("cilia")
                          if "object_type" in _df.columns else slice(None), "class"]
    st.sidebar.caption("Cilia: " + "  ".join(
        f"{k}={int((_cls_counts == k).sum())}" for k in ("neurite", "soma", "ambiguous")))

_has_otype = "object_type" in _df.columns

# ── Cilia size filter — remove false positives by 3-D volume / length ─────────
# Slider bounds come from the *unfiltered* data so they stay stable across reruns.
_cil_raw = _df[_df["object_type"] == "cilia"] if _has_otype else _df
with st.sidebar:
    st.markdown("---")
    st.header(":material/build: Cilia size filter")
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
# Keep a full copy for the Screening tab — the size filter is analysis-only; you
# still want to review *every* detected ROI (including odd-sized junk).
_df_presize = _df.copy()
_df = apply_size_filter(_df, _has_otype, _vol_rng, _len_rng, _drop_na)
_n_cil_after = int((_df["object_type"] == "cilia").sum()) if _has_otype else len(_df)
if _n_cil_after < _n_cil_before:
    st.sidebar.success(
        f":material/build: Size filter kept {_n_cil_after} / {_n_cil_before} cilia "
        f"({_n_cil_before - _n_cil_after} removed)."
    )

# ── Human validation column (manual screening, see the Screening tab) ─────────
st.session_state.setdefault("validation", {})        # {(run, filename, cid): bool}
st.session_state.setdefault("_val_seeded", set())    # (dir, label) pairs already seeded
_VAL = st.session_state["validation"]
# A cilium is "decided/screened" exactly when it has an entry in ``_VAL`` — which
# is persisted to disk and re-seeded on load, so review progress survives
# restarts (unlike a session-only "touched this run" set).


def _val_key(run_label, filename, cid):
    return (str(run_label), str(filename), int(cid))


# Seed each run's saved decisions from disk under its current label (never
# clobbers live edits). Keyed by (dir, label) so a relabel re-seeds the same
# disk decisions under the new label — relabeling never loses screening data.
for _r in _runs:
    _seed_key = (_r["dir"], _r["label"])
    if _seed_key in st.session_state["_val_seeded"]:
        continue
    for (_fn, _cid), _ok in load_validation_file(_r["dir"]).items():
        _VAL.setdefault(_val_key(_r["label"], _fn, _cid), _ok)
    st.session_state["_val_seeded"].add(_seed_key)


def _lookup_validated(run_label, filename, cid):
    if cid is None or (isinstance(cid, float) and np.isnan(cid)):
        return True
    try:
        return _VAL.get(_val_key(run_label, filename, int(cid)), True)
    except (ValueError, TypeError):
        return True


if "cilia_id" in _df.columns and {"run", "filename"}.issubset(_df.columns):
    _is_cil = (_df["object_type"] == "cilia") if _has_otype else pd.Series(True, index=_df.index)
    _df["human_validated"] = True
    _df.loc[_is_cil, "human_validated"] = [
        _lookup_validated(rl, fn, cid)
        for rl, fn, cid in zip(_df.loc[_is_cil, "run"],
                               _df.loc[_is_cil, "filename"],
                               _df.loc[_is_cil, "cilia_id"])
    ]
else:
    _df["human_validated"] = True

# Snapshot of *all* detected cilia for the Screening tab — built from the
# pre-size-filter frame so you can review every ROI, not just the ones the
# analysis size filter keeps. Decisions still apply everywhere.
_cilia_full = (_df_presize[_df_presize["object_type"] == "cilia"]
               if _has_otype else _df_presize).copy()
if {"run", "filename", "cilia_id"}.issubset(_cilia_full.columns):
    _cilia_full["human_validated"] = [
        _lookup_validated(rl, fn, cid)
        for rl, fn, cid in zip(_cilia_full["run"], _cilia_full["filename"],
                               _cilia_full["cilia_id"])
    ]
else:
    _cilia_full["human_validated"] = True

with st.sidebar:
    st.markdown("---")
    st.header(":material/person_search: Screening")
    _n_rejected = int((~_df["human_validated"]).sum())
    _excl_rejected = st.checkbox(
        "Exclude human-rejected cilia from all tabs", value=True, key="excl_rej",
        help="Decisions are made in the :material/person_search: Screening tab and saved per run.")
    st.caption(f"{_n_rejected} cilia manually rejected so far.")

if _excl_rejected and "human_validated" in _df.columns:
    _df = _df[_df["human_validated"]]

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


(tab_over, tab_screen, tab_dist, tab_scatter, tab_inter, tab_corr,
 tab_table, tab_qc, tab_validate, tab_boxes, tab_report) = st.tabs(
    [":material/biotech: Overlays", ":material/person_search: Screening", ":material/bar_chart: Hist / KDE", ":material/scatter_plot: Scatter", ":material/bolt: Interactive",
     ":material/link: Correlation", ":material/list_alt: Data table", ":material/monitor_heart: QC", ":material/handshake: Validation",
     ":material/select_all: Box GT", ":material/description: Report"]
)

_run_dir_by_label = {r["label"]: r["dir"] for r in _runs}


def _roi_png_path(run_label: str, filename, cid):
    """Path to a cilium's ROI screenshot, or None if it doesn't exist."""
    d = _run_dir_by_label.get(run_label)
    if not d or cid is None:
        return None
    stem = Path(str(filename)).stem
    p = os.path.join(d, "figures", "cilia_rois", f"{stem}_cilia{int(cid)}.png")
    return p if os.path.exists(p) else None


def _roi_npz_path(run_label: str, filename, cid):
    """Path to a cilium's raw .npz crop (all channels), or None."""
    d = _run_dir_by_label.get(run_label)
    if not d or cid is None:
        return None
    stem = Path(str(filename)).stem
    p = os.path.join(d, "figures", "cilia_rois", f"{stem}_cilia{int(cid)}.npz")
    return p if os.path.exists(p) else None


_AI_THUMB_DIR = os.path.join(tempfile.gettempdir(), "lc_ai_thumbs")


def _roi_model_png(run_label: str, filename, cid, correct: bool):
    """Path to the **model-input** thumbnail for one cilium, re-rendered from its
    raw ``.npz`` crop with display correction on/off — so what the user sees in
    the gallery is exactly what the model is trained / scored on. Disk-cached in
    the temp dir; falls back to the saved PNG when no crop is available.

    Not ``st.cache_data``-decorated on purpose: it does file IO and the cache
    key would have to hash the (possibly numpy) filename — caching on disk by a
    content hash is simpler and avoids hashing surprises.
    """
    npz = _roi_npz_path(run_label, filename, cid)
    if npz:
        try:
            from limoncello.visualization.cilia_rois import render_npz_thumb
            from PIL import Image
            os.makedirs(_AI_THUMB_DIR, exist_ok=True)
            _sig = f"{npz}|{os.path.getmtime(npz)}|{int(bool(correct))}"
            _key = hashlib.md5(_sig.encode()).hexdigest()
            outp = os.path.join(_AI_THUMB_DIR, f"{_key}.png")
            if os.path.exists(outp):
                return outp
            arr = render_npz_thumb(npz, correct_display=bool(correct))
            if arr is not None:
                Image.fromarray(arr).save(outp)
                return outp
        except Exception:                                 # noqa: BLE001
            pass
    return _roi_png_path(run_label, filename, cid)


@st.cache_data(show_spinner=False)
def _roi_multichannel_png(npz_path: str):
    """All-channel ROI render (cilia=green, BB=magenta, neurite=cyan,
    nuclei=blue) from a saved .npz crop, cached on path. None if unavailable."""
    if not npz_path:
        return None
    try:
        from limoncello.visualization.cilia_rois import render_npz_rgb
        arr = render_npz_rgb(npz_path)
        return None if arr is None else arr
    except Exception:                                     # noqa: BLE001
        return None


def _raw_roi_thumb(png_path: str):
    """Raw-intensity ROI thumbnail re-rendered from the cilium's ``.npz`` crop
    with display correction OFF — same image the Screening gallery shows. Shares
    the ``_AI_THUMB_DIR`` disk cache (sig ``npz|mtime|0``). Falls back to the
    saved PNG when no crop is on disk."""
    npz = os.path.splitext(png_path)[0] + ".npz"
    if os.path.exists(npz):
        try:
            from limoncello.visualization.cilia_rois import render_npz_thumb
            from PIL import Image
            os.makedirs(_AI_THUMB_DIR, exist_ok=True)
            _sig = f"{npz}|{os.path.getmtime(npz)}|0"
            _key = hashlib.md5(_sig.encode()).hexdigest()
            outp = os.path.join(_AI_THUMB_DIR, f"{_key}.png")
            if os.path.exists(outp):
                return outp
            arr = render_npz_thumb(npz, correct_display=False)
            if arr is not None:
                Image.fromarray(arr).save(outp)
                return outp
        except Exception:                                 # noqa: BLE001
            pass
    return png_path


# ─────────────────────────────────────────────────────────────────────────────
# TAB — OVERLAYS (3-D napari screenshots + ROI gallery)
# ─────────────────────────────────────────────────────────────────────────────
with tab_over:
    st.subheader(":material/biotech: 3-D napari overlays")
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

        # ── Validation rings on the XY MIP (pixel-accurate, from coords) ───────
        _cil_mip = _load_sample_mip(_sel_run["dir"], _stem, "cilia")
        if _cil_mip is not None:
            with st.expander(":green[:material/circle:]:red[:material/circle:] Validation rings on overview MIP", expanded=True):
                _samp = _cilia_full[(_cilia_full["run"] == _sel_label)
                                    & (_cilia_full["filename"].apply(
                                        lambda f: Path(str(f)).stem == _stem))].copy()
                if _samp.empty or "coords" not in _samp.columns:
                    st.caption("No cilia coordinates available for this sample.")
                else:
                    rc1, rc2, rc3 = st.columns(3)
                    with rc1:
                        _ring_by = st.selectbox(
                            "Colour rings by",
                            ["Human/AI validation", "Class"], key="ov_ring_by")
                    with rc2:
                        _ring_r = st.slider("Ring radius (px)", 4, 40, 14,
                                            key="ov_ring_r")
                    with rc3:
                        _show_ids = st.checkbox("Show cilia IDs", value=True,
                                                key="ov_ring_ids")
                    # Per-channel display min/max (range sliders, 0–1 on the
                    # min-max-normalised MIP).
                    st.caption("Display range per channel (min, max)")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    with mc1:
                        _clim_cil = st.slider("Cilia", 0.0, 1.0, (0.0, 1.0),
                                              0.01, key="ov_clim_cil")
                    with mc2:
                        _clim_bb = st.slider("Basal body", 0.0, 1.0, (0.0, 1.0),
                                             0.01, key="ov_clim_bb")
                    with mc3:
                        _clim_neu = st.slider("Neurite", 0.0, 1.0, (0.0, 1.0),
                                              0.01, key="ov_clim_neu")
                    with mc4:
                        _clim_nuc = st.slider("Nuclei", 0.0, 1.0, (0.0, 1.0),
                                              0.01, key="ov_clim_nuc")
                    from limoncello.visualization.ring_overlay import (
                        mip_rgb, ring_overlay_figure)
                    _bg = mip_rgb(
                        cilia=_cil_mip,
                        neurite=_load_sample_mip(_sel_run["dir"], _stem, "neurite"),
                        nuclei=_load_sample_mip(_sel_run["dir"], _stem, "nuclei"),
                        bb=_load_sample_mip(_sel_run["dir"], _stem, "bb"),
                        cilia_clim=_clim_cil, bb_clim=_clim_bb,
                        neurite_clim=_clim_neu, nuclei_clim=_clim_nuc)
                    _yx = _parse_coords_yx(_samp["coords"])
                    if _ring_by == "Class" and "class" in _samp.columns:
                        _cmap = {"neurite": True, "soma": False}
                        _keep = [_cmap.get(str(c), None) for c in _samp["class"]]
                        _cap = ":green[:material/circle:] neurite · :red[:material/circle:] soma · :orange[:material/circle:] ambiguous"
                    else:
                        _keep = ([bool(v) for v in _samp["human_validated"]]
                                 if "human_validated" in _samp.columns
                                 else [None] * len(_samp))
                        _cap = ":green[:material/circle:] kept/validated · :red[:material/circle:] rejected"
                    _ids = ([int(c) for c in _samp["cilia_id"]]
                            if "cilia_id" in _samp.columns else None)
                    _fig = ring_overlay_figure(
                        _bg, _yx, _keep, None, ids=_ids, show_ids=_show_ids,
                        radius=float(_ring_r), title=f"{_stem}")
                    st.pyplot(_fig, use_container_width=True)
                    plt.close(_fig)
                    st.caption(_cap + ". Positions from each cilium's centroid "
                               "coordinates, projected onto the XY MIP.")

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
        st.subheader(f":material/search: Per-cilium ROIs ({len(rois)})")

        with st.expander(":material/construction: ROI diagnostics", expanded=not rois):
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
                        st.image(_raw_roi_thumb(p), caption=os.path.basename(p),
                                 use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB — SCREENING (manual cilia review: keep / reject each ROI)
# ─────────────────────────────────────────────────────────────────────────────
def _cilia_stat_caption(row) -> str:
    """A compact one-line stat string for a cilium card."""
    bits = []
    if "log_ratio" in row and pd.notna(row["log_ratio"]):
        bits.append(f"logR {row['log_ratio']:.2f}")
    if "volume_um3" in row and pd.notna(row["volume_um3"]):
        bits.append(f"{row['volume_um3']:.1f} µm³")
    if "length_um" in row and pd.notna(row["length_um"]):
        bits.append(f"{row['length_um']:.1f} µm")
    return " · ".join(bits)


def _persist_run(run_label):
    """Auto-save one run's decisions to its `csv/human_validation.csv`."""
    rd = _run_dir_by_label.get(run_label)
    if not rd:
        return
    rows = [{"filename": fn, "cilia_id": cid, "human_validated": ok}
            for (rl, fn, cid), ok in _VAL.items() if rl == run_label]
    try:
        save_validation_file(rd, rows)
    except Exception:                                     # noqa: BLE001
        pass


def _commit_decision(tk, keep, persist=True):
    """Record a decision in the store (= marks it screened) + sync the toggle."""
    _VAL[tk] = keep
    st.session_state[f"sg_{tk}"] = keep
    if persist:
        _persist_run(tk[0])                               # tk = (run_label, fn, cid)


def _gallery_toggle(tk):
    """on_change callback: commit *only* this card's toggle to its own key.

    Using a callback bound to ``tk`` (instead of comparing every card's widget
    against the store on each rerun) guarantees the click affects the right
    cilium — not a neighbour in the grid.
    """
    _commit_decision(tk, bool(st.session_state.get(f"sg_{tk}", True)))


@st.cache_data(show_spinner=False)
def _arch_diagram_png(arch: str) -> bytes:
    """Rendered architecture diagram (cached per arch) for the AI Train tab."""
    from limoncello.visualization.model_diagram import model_architecture_figure
    fig = model_architecture_figure(arch)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ROI-validator models live in the codebase, not in per-run output folders.
try:
    _MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
except NameError:                                         # __file__ may be unset
    _MODELS_DIR = os.path.abspath("models")


def _discover_roi_models():
    """``{filename: path}`` of trained ROI-validator models in the codebase."""
    found = {}
    if os.path.isdir(_MODELS_DIR):
        for f in sorted(os.listdir(_MODELS_DIR)):
            if f.endswith(".pt"):
                found[f] = os.path.join(_MODELS_DIR, f)
    return found


with tab_screen:
    st.subheader(":material/person_search: Cilia screening — keep the good, toss the junk")
    st.caption("Review every detected cilium's 3-D ROI and decide whether it's a "
               "real cilium. Rejected ones get `human_validated = False` and can be "
               "dropped from every other tab via the sidebar toggle.")

    if _cilia_full.empty or "cilia_id" not in _cilia_full.columns:
        st.info("No cilia with IDs to screen. Run a batch with ROI capture enabled.")
    else:
        # Honour a deferred "review unsure" jump from the AI panel (must run
        # before the Show/Mode widgets below are instantiated).
        if st.session_state.pop("_jump_unsure", False):
            st.session_state["sc_show"] = ":material/smart_toy: AI unsure"
            st.session_state["sc_mode"] = ":material/image: Gallery"

        # Run is the shared context for the AI assistant and the gallery below.
        _sc_run = st.selectbox("Run", [r["label"] for r in _runs], key="sc_run")
        _run_dir = _run_dir_by_label.get(_sc_run)

        _NONE_MODEL = "(none — manual screening)"
        _ai_set = st.session_state.setdefault("ai_validated", set())

        # ── :material/smart_toy: AI assistant: TRAIN a model, then PREDICT on the run ───────────
        with st.expander(f"## :material/smart_toy: AI screening assistant"):
            st.markdown("## :material/smart_toy: AI screening assistant")
            _tab_train, _tab_predict, _tab_acts = st.tabs(
                [":material/psychology: Train a model", ":material/online_prediction: Predict on this run",
                 ":material/biotech: Activation maps"])

            # ===== TRAIN =====
            with _tab_train:
                st.caption("Learns from the cilia you've already kept / rejected "
                           "(their ROI images) to suggest decisions for the rest.")
                # Labelled examples = saved decisions with an ROI image (skip the
                # model's own past suggestions to avoid a feedback loop).
                # Train on RAW-intensity thumbnails (correct=False) — the same
                # images shown in the gallery, so the user sees what the model sees.
                _labelled, _n_keep_ex, _n_rej_ex = [], 0, 0
                for _tk, _ok in _VAL.items():
                    if _tk in _ai_set:
                        continue
                    _pp = _roi_model_png(*_tk, False)
                    if _pp:
                        _labelled.append((_pp, 1 if _ok else 0))
                        _n_keep_ex += int(bool(_ok))
                        _n_rej_ex += int(not _ok)

                a1, a2, a3 = st.columns(3)
                a1.metric("Training images", len(_labelled))
                a2.metric(":material/check_circle: keep examples", _n_keep_ex)
                a3.metric(":material/cancel: reject examples", _n_rej_ex)

                tn1, tn2 = st.columns([2, 1])
                with tn1:
                    _model_name = st.text_input(
                        "Model name", value="roi_validator", key="ai_model_name",
                        help="Saved as <name>.pt — give versions different names "
                             "so you can choose between them in the Predict tab.")
                with tn2:
                    _arch_label = st.selectbox(
                        "Model size", ["Big (GPU)", "Tiny (fast)"], key="ai_arch",
                        help="Big = a larger BatchNorm CNN (96², ~2M params); uses "
                             "your CUDA GPU automatically. Tiny = small & fast on "
                             "CPU. Both use augmentation + early stopping.")
                _arch = "big" if _arch_label.startswith("Big") else "tiny"
                _epochs = st.slider("Max epochs", 10, 120, 40, key="ai_epochs",
                                    help="Early stopping ends training once the "
                                         "validation loss stops improving, so this "
                                         "is just an upper bound.")
                _safe_name = "".join(c for c in _model_name.strip()
                                     if c.isalnum() or c in "-_") or "roi_validator"
                _model_path = os.path.join(_MODELS_DIR, f"{_safe_name}.pt")

                with st.expander(f":material/schema: {_arch_label} architecture", expanded=False):
                    _apng = _arch_diagram_png(_arch)
                    st.image(_apng, width=540)
                    st.download_button(
                        ":material/download: Download architecture (PNG)", _apng,
                        f"roi_validator_{_arch}_architecture.png", "image/png",
                        key="dl_arch")

                _ready = len(_labelled) >= 8 and _n_keep_ex > 0 and _n_rej_ex > 0
                if not _ready:
                    st.info("Screen at least **8 cilia including some of each** "
                            "(keep *and* reject, with ROI images) to train.")
                if st.button(":material/psychology: Train / retrain model", type="primary",
                             disabled=not _ready, key="ai_train"):
                    try:
                        from limoncello.ml.roi_validator import (train_validator,
                                                                 save_bundle)
                        _bar = st.progress(0.0, text="starting…")
                        _res = train_validator(
                            _labelled, arch=_arch, epochs=_epochs,
                            progress=lambda f, m: _bar.progress(f, text=m))
                        save_bundle(_model_path, _res["model"], _res)
                        st.session_state["roi_clf"] = _res
                        st.session_state["_select_model"] = f"{_safe_name}.pt"
                        st.session_state["_roi_clf_loaded"] = f"{_safe_name}.pt"
                        _bar.empty()
                        st.success(
                            f"Trained on {_res['n_train']} images "
                            f"({_res.get('device', 'cpu').upper()}, {_arch}) · "
                            f"{_res['val_acc'] * 100:.0f}% val accuracy · saved as "
                            f"**{_safe_name}.pt**. Use it in the **:material/online_prediction: Predict** tab.")
                    except Exception as exc:              # noqa: BLE001
                        st.error(f"Training failed: {exc}")

                # Performance + curves of the most recently trained / loaded model.
                _clf = st.session_state.get("roi_clf")
                if _clf:
                    st.markdown("**Model performance** (held-out cilia)")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Accuracy", f"{_clf['val_acc'] * 100:.0f}%")
                    _auc = _clf.get("val_auc")
                    mc2.metric("AUC", "—" if _auc is None or
                               (isinstance(_auc, float) and np.isnan(_auc))
                               else f"{_auc:.2f}")
                    mc3.metric("Trained on", f"{_clf.get('n_train', '?')} imgs")
                    _cf = _clf.get("confusion")
                    if _cf:
                        st.caption(
                            f"Validation: :material/check_circle: {_cf['tp']} correct keeps · "
                            f":material/check_circle: {_cf['tn']} correct rejects · "
                            f":material/warning: {_cf['fp']} wrongly kept · {_cf['fn']} missed.")
                    _hist = _clf.get("history")
                    if _hist and isinstance(_hist[0], dict):
                        _hdf = pd.DataFrame(_hist).set_index("epoch")
                        g1, g2 = st.columns(2)
                        with g1:
                            st.caption("Loss per epoch (lower = better)")
                            st.line_chart(_hdf[["train_loss", "val_loss"]])
                        with g2:
                            st.caption("Validation accuracy per epoch")
                            st.line_chart(_hdf[["val_acc"]])

            # ===== PREDICT =====
            with _tab_predict:
                _roi_models = _discover_roi_models()
                if not _roi_models:
                    st.info("No trained model yet — train one in the **:material/psychology: Train** "
                            "tab first.")
                else:
                    _want = st.session_state.pop("_select_model", None)
                    if _want in _roi_models:
                        st.session_state["ai_model_pick"] = _want
                    _opts = [_NONE_MODEL] + list(_roi_models)
                    if "ai_model_pick" not in st.session_state:   # default: newest
                        st.session_state["ai_model_pick"] = max(
                            _roi_models, key=lambda n: os.path.getmtime(_roi_models[n]))
                    _pick = st.selectbox(
                        "Model", _opts, key="ai_model_pick",
                        help="Which trained model scores the cilia.")
                    if _pick == _NONE_MODEL:
                        st.session_state.pop("roi_clf", None)
                        st.session_state["_roi_clf_loaded"] = None
                    elif st.session_state.get("_roi_clf_loaded") != _pick:
                        try:
                            from limoncello.ml.roi_validator import load_bundle
                            _m, _meta = load_bundle(_roi_models[_pick])
                            st.session_state["roi_clf"] = {"model": _m, **_meta}
                            st.session_state["_roi_clf_loaded"] = _pick
                        except Exception as _exc:         # noqa: BLE001
                            st.error(f"Could not load model '{_pick}': {_exc}")

                    _clf = st.session_state.get("roi_clf")
                    if not _clf or _pick == _NONE_MODEL:
                        st.caption("Pick a model above to score this run.")
                    else:
                        _a0 = _clf.get("val_acc")
                        st.success(f"Using **{_pick}**" +
                                   (f" · {_a0 * 100:.0f}% val accuracy"
                                    if _a0 is not None else ""))
                        t1, t2 = st.columns(2)
                        with t1:
                            _keep_thr = st.slider("Keep if AI score ≥", 0.50, 0.99,
                                                  0.60, 0.01, key="ai_keepthr")
                        with t2:
                            _rej_thr = st.slider("Reject if AI score ≤", 0.01, 0.50,
                                                 0.40, 0.01, key="ai_rejthr")

                        # Score ALL cilia of the selected run that have an image.
                        _cand = []
                        _runscope = _cilia_full[_cilia_full["run"] == _sc_run]
                        for _, _rw in _runscope.iterrows():
                            if pd.isna(_rw.get("cilia_id")):
                                continue
                            _tk = _val_key(_rw["run"], _rw["filename"],
                                           _rw["cilia_id"])
                            _pp = _roi_model_png(_rw["run"], _rw["filename"],
                                                 _rw["cilia_id"], False)
                            if _pp:
                                _cand.append((_tk, _pp))

                        st.caption(f"Scores all **{len(_cand)}** cilia with an ROI "
                                   f"image in run **{_sc_run}**.")
                        if _cand and st.button(f":material/online_prediction: Predict on '{_sc_run}'",
                                               type="primary", key="ai_predict"):
                            from limoncello.ml.roi_validator import predict_proba
                            _probs = predict_proba(
                                _clf["model"], [p for _, p in _cand],
                                size=_clf.get("size", 64),
                                normalize=_clf.get("normalize"))
                            st.session_state["ai_preds"] = {
                                "tks": [tk for tk, _ in _cand],
                                "paths": [p for _, p in _cand],
                                "probs": _probs.tolist(),
                            }
                            st.session_state["ai_scores"] = {
                                tk: float(pr) for (tk, _), pr in zip(_cand, _probs)
                                if np.isfinite(pr)}

                        _preds = st.session_state.get("ai_preds")
                        if _preds:
                            _pa = np.array(_preds["probs"], dtype=float)
                            # Apply only affects *undecided* cilia (human calls kept).
                            _und = [(_tk, _pb) for _tk, _pb
                                    in zip(_preds["tks"], _pa)
                                    if np.isfinite(_pb) and _tk not in _VAL]
                            _n_ks = sum(1 for _, p in _und if p >= _keep_thr)
                            _n_rs = sum(1 for _, p in _und if p <= _rej_thr)
                            _n_un = sum(1 for _, p in _und
                                        if _rej_thr < p < _keep_thr)
                            s1, s2, s3 = st.columns(3)
                            s1.metric("→ suggest keep", _n_ks)
                            s2.metric("→ suggest reject", _n_rs)
                            s3.metric("→ needs you", _n_un)

                            if _n_un and st.button(
                                    f":material/visibility: Review the {_n_un} unsure in the gallery",
                                    key="ai_review_unsure"):
                                st.session_state["_jump_unsure"] = True
                                st.rerun()

                            _order = [i for i in np.argsort(np.abs(_pa - 0.5))
                                      if np.isfinite(_pa[i])][:6]
                            if _order:
                                st.caption("Most uncertain — review yourself:")
                                for _col, _i in zip(st.columns(len(_order)), _order):
                                    with _col:
                                        st.image(_preds["paths"][_i],
                                                 use_container_width=True)
                                        _tk_i = _preds["tks"][_i]
                                        _samp_i = Path(str(_tk_i[1])).stem
                                        st.caption(f"score {_pa[_i]:.2f}")
                                        st.markdown(
                                            f"<div title='{_samp_i}' "
                                            f"style='color:#888;font-size:0.75em;"
                                            f"overflow:hidden;text-overflow:ellipsis;"
                                            f"white-space:nowrap'>cilia "
                                            f"{int(_tk_i[2])} · {_samp_i}</div>",
                                            unsafe_allow_html=True)

                            st.warning("Applies to **undecided** cilia only — your "
                                       "manual decisions are preserved. ‘Needs "
                                       "you’ cilia are left for review.")
                            if st.button(":material/check_circle: Apply keep/reject suggestions",
                                         type="primary", key="ai_apply"):
                                _changed, _applied = set(), 0
                                for _tk, _pb in zip(_preds["tks"], _pa):
                                    if not np.isfinite(_pb) or _tk in _VAL:
                                        continue
                                    if _pb >= _keep_thr:
                                        _decision = True
                                    elif _pb <= _rej_thr:
                                        _decision = False
                                    else:
                                        continue
                                    _VAL[_tk] = _decision
                                    _ai_set.add(_tk)
                                    _changed.add(_tk[0])
                                    _applied += 1
                                for _r in _changed:
                                    _persist_run(_r)
                                st.session_state.pop("ai_preds", None)
                                st.success(f"Applied {_applied} AI suggestions.")
                                st.rerun()

            # ===== ACTIVATION MAPS =====
            with _tab_acts:
                st.caption("Peek inside the CNN: see which features each conv "
                           "layer fires on, and the Grad-CAM region that drove "
                           "the keep / reject decision for one ROI.")
                _clf = st.session_state.get("roi_clf")
                if not _clf:
                    st.info("Load or train a model first (in the **:material/psychology: Train** or "
                            "**:material/online_prediction: Predict** tab).")
                else:
                    # Pick an ROI from the current run that has an image.
                    _av_scope = _cilia_full[_cilia_full["run"] == _sc_run]
                    _av_choices = []
                    for _, _rw in _av_scope.iterrows():
                        if pd.isna(_rw.get("cilia_id")):
                            continue
                        _pp = _roi_png_path(_rw["run"], _rw["filename"],
                                            _rw["cilia_id"])
                        if _pp:
                            _lbl = f"{Path(str(_rw['filename'])).stem} · " \
                                   f"cilia {int(_rw['cilia_id'])}"
                            _av_choices.append((_lbl, _pp))
                    if not _av_choices:
                        st.info("No ROI images found for this run.")
                    else:
                        _pick_lbl = st.selectbox(
                            "ROI to inspect", [c[0] for c in _av_choices],
                            key="av_roi")
                        _av_path = dict(_av_choices)[_pick_lbl]
                        _size = int(_clf.get("size", 64))

                        _vc1, _vc2 = st.columns(2)
                        with _vc1:
                            _show_cam = st.checkbox("Grad-CAM (decision region)",
                                                    value=True, key="av_cam")
                        with _vc2:
                            _show_layers = st.checkbox("Per-layer activations",
                                                       value=True, key="av_layers")

                        if st.button(":material/biotech: Visualize", type="primary",
                                     key="av_run"):
                            try:
                                from limoncello.ml.roi_validator import (
                                    collect_activations, grad_cam)
                                from limoncello.visualization.activations import (
                                    layer_overview_figure, gradcam_figure)
                                _res = collect_activations(
                                    _clf["model"], _av_path, size=_size)
                                st.session_state["av_result"] = {
                                    "path": _av_path, "size": _size}
                                if _res is None:
                                    st.error("Could not read that ROI image.")
                                else:
                                    if _show_cam:
                                        _cam = grad_cam(_clf["model"], _av_path,
                                                        size=_size)
                                        if _cam is not None:
                                            st.pyplot(gradcam_figure(_cam),
                                                      use_container_width=True)
                                    if _show_layers:
                                        st.pyplot(layer_overview_figure(_res),
                                                  use_container_width=True)
                                        # Per-filter detail for a chosen layer.
                                        st.session_state["av_layers_cache"] = [
                                            {"name": l["name"],
                                             "n_channels": l["n_channels"]}
                                            for l in _res["layers"]]
                                        st.session_state["av_full"] = _res
                            except Exception as _exc:     # noqa: BLE001
                                st.error(f"Activation visualization failed: {_exc}")

                        # Per-layer channel grid (kept after the run above).
                        _full = st.session_state.get("av_full")
                        if _full and _full.get("layers"):
                            st.markdown("**Individual channel maps**")
                            _lnames = [l["name"] for l in _full["layers"]]
                            _lsel = st.selectbox("Layer", _lnames, key="av_layer_sel")
                            _maxc = st.slider("Max channels", 4, 64, 32, 4,
                                              key="av_maxc")
                            _layer = next(l for l in _full["layers"]
                                          if l["name"] == _lsel)
                            from limoncello.visualization.activations import (
                                feature_maps_figure as _fmf)
                            st.pyplot(_fmf(_layer, max_channels=_maxc),
                                      use_container_width=True)

        # ── Gallery controls ──────────────────────────────────────────────────
        gc1, gc2, gc3 = st.columns(3)
        _scope = _cilia_full[_cilia_full["run"] == _sc_run]
        _sc_stems = ["All"] + sample_stems(_scope)
        with gc1:
            _sc_sample = st.selectbox("Sample", _sc_stems, key="sc_sample")
        if _sc_sample != "All" and "filename" in _scope.columns:
            _scope = _scope[_scope["filename"].apply(
                lambda f: Path(str(f)).stem == _sc_sample)]
        with gc2:
            _sc_show = st.selectbox("Show", ["All", "Kept", "Rejected",
                                             "Unscreened", "Ambiguous only",
                                             ":material/smart_toy: AI unsure"],
                                    key="sc_show")
        with gc3:
            _sort_opts = [c for c in ("cilia_id", "log_ratio", "volume_um3",
                                      "length_um", "class") if c in _scope.columns]
            _sc_sort = st.selectbox("Sort by", _sort_opts or ["cilia_id"], key="sc_sort")

        # When a single sample is filtered, show its overview MIP with the
        # kept / rejected cilia highlighted (live decisions), so you can screen
        # each ROI against the full-image context.
        if (_sc_sample != "All" and _run_dir and "coords" in _scope.columns
                and "cilia_id" in _scope.columns):
            _cil_mip = _load_sample_mip(_run_dir, _sc_sample, "cilia")
            _samp_ov = _scope[_scope["coords"].notna() & _scope["cilia_id"].notna()]
            if _cil_mip is not None and not _samp_ov.empty:
                with st.expander(
                        f":material/biotech: Overview — kept / rejected on "
                        f"{_sc_sample}", expanded=True):
                    from limoncello.visualization.ring_overlay import (
                        mip_rgb, ring_overlay_figure)
                    sr1, sr2 = st.columns(2)
                    with sr1:
                        _sc_ring_r = st.slider("Ring radius (px)", 4, 40, 14,
                                               key="sc_ring_r")
                    with sr2:
                        _sc_ring_ids = st.checkbox("Show cilia IDs", value=True,
                                                   key="sc_ring_ids")
                    # Per-channel display window (min, max) on the min-max-normalised MIP.
                    st.caption("Display range per channel (min, max)")
                    sm1, sm2, sm3, sm4 = st.columns(4)
                    with sm1:
                        _sc_clim_cil = st.slider("Cilia", 0.0, 1.0, (0.0, 1.0),
                                                 0.01, key="sc_clim_cil")
                    with sm2:
                        _sc_clim_bb = st.slider("Basal body", 0.0, 1.0, (0.0, 1.0),
                                                0.01, key="sc_clim_bb")
                    with sm3:
                        _sc_clim_neu = st.slider("Neurite", 0.0, 1.0, (0.0, 1.0),
                                                 0.01, key="sc_clim_neu")
                    with sm4:
                        _sc_clim_nuc = st.slider("Nuclei", 0.0, 1.0, (0.0, 1.0),
                                                 0.01, key="sc_clim_nuc")
                    _bg = mip_rgb(
                        cilia=_cil_mip,
                        neurite=_load_sample_mip(_run_dir, _sc_sample, "neurite"),
                        nuclei=_load_sample_mip(_run_dir, _sc_sample, "nuclei"),
                        bb=_load_sample_mip(_run_dir, _sc_sample, "bb"),
                        cilia_clim=_sc_clim_cil, bb_clim=_sc_clim_bb,
                        neurite_clim=_sc_clim_neu, nuclei_clim=_sc_clim_nuc)
                    _yx = _parse_coords_yx(_samp_ov["coords"])
                    _keep = [_VAL.get(_val_key(_sc_run, fn, cid), None)
                             for fn, cid in zip(_samp_ov["filename"],
                                                _samp_ov["cilia_id"])]
                    _ids = [int(c) for c in _samp_ov["cilia_id"]]
                    _fig = ring_overlay_figure(
                        _bg, _yx, _keep, None, ids=_ids,
                        show_ids=_sc_ring_ids, radius=float(_sc_ring_r),
                        title=_sc_sample)
                    st.pyplot(_fig, use_container_width=True)
                    plt.close(_fig)
                    st.caption(":green[:material/circle:] kept · "
                               ":red[:material/circle:] rejected · "
                               ":orange[:material/circle:] undecided")

        # Resolve each row's store key + current decision, then apply the filter.
        _scope = _scope[_scope["cilia_id"].notna()].copy()
        _scope["_tk"] = [_val_key(_sc_run, fn, cid)
                         for fn, cid in zip(_scope["filename"], _scope["cilia_id"])]
        _scope["_keep"] = [_VAL.get(t, True) for t in _scope["_tk"]]
        _scope["_done"] = [t in _VAL for t in _scope["_tk"]]   # decided = has a saved call
        # AI score per card (from the last "Predict & preview"), NaN if none.
        _ai_scores = st.session_state.get("ai_scores", {})
        _scope["_ai"] = [_ai_scores.get(t, np.nan) for t in _scope["_tk"]]
        # Uncertain band = between the current reject/keep thresholds.
        _kt = float(st.session_state.get("ai_keepthr", 0.60))
        _rt = float(st.session_state.get("ai_rejthr", 0.40))
        _unsure_mask = (_scope["_ai"].notna() & (_scope["_ai"] > _rt)
                        & (_scope["_ai"] < _kt) & ~_scope["_done"])

        if _sc_show == "Kept":
            _view_c = _scope[_scope["_keep"]]
        elif _sc_show == "Rejected":
            _view_c = _scope[~_scope["_keep"]]
        elif _sc_show == "Unscreened":
            _view_c = _scope[~_scope["_done"]]
        elif _sc_show == "Ambiguous only" and "class" in _scope.columns:
            _view_c = _scope[_scope["class"] == "ambiguous"]
        elif _sc_show == ":material/smart_toy: AI unsure":
            _view_c = _scope[_unsure_mask]
            if _view_c.empty:
                st.info("No AI-uncertain cilia to review. Run **:material/online_prediction: Predict** in "
                        "the AI assistant above first.")
        else:
            _view_c = _scope
        if _sc_sort in _view_c.columns:
            _view_c = _view_c.sort_values(_sc_sort, kind="stable")

        # ── Progress + scoreboard ─────────────────────────────────────────────
        _tot = len(_scope)
        _kept = int(_scope["_keep"].sum())
        _rej = _tot - _kept
        _seen = int(_scope["_done"].sum())
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("In view", f"{len(_view_c)} / {_tot}")
        p2.metric(":material/check_circle: Kept", _kept)
        p3.metric(":material/cancel: Rejected", _rej)
        p4.metric(":material/visibility: Screened", f"{_seen}/{_tot}")
        st.progress(_seen / _tot if _tot else 0.0,
                    text=(":material/celebration: All screened!" if _seen >= _tot and _tot
                          else f"{_seen} of {_tot} reviewed in this run"))

        _mode = st.radio("Mode", [":material/image: Gallery", ":material/center_focus_strong: Focus (one at a time)"],
                         horizontal=True, key="sc_mode")

        # ── Bulk actions (must mutate state BEFORE the toggles are built) ──────
        st.caption("Reject-only workflow? Toggle off the bad cilia, then click "
                   "**:material/check: Mark all reviewed** so the kept ones also count as "
                   "screened (and train the AI). Your rejections are preserved.")
        b1, b2, b3, b4, b5 = st.columns(5)
        _shown_tks = list(_view_c["_tk"])
        if b1.button(":material/check: Mark all reviewed", type="primary",
                     use_container_width=True, key="sc_markrev",
                     help="Marks every still-undecided cilium shown as KEPT, "
                          "leaving your rejections untouched — so the whole set "
                          "counts as screened and is used for training."):
            for t in _shown_tks:
                if t not in _VAL:                          # implicit keep → make it explicit
                    _commit_decision(t, True, persist=False)
            _persist_run(_sc_run)
            st.rerun()
        if b2.button(":material/check_circle: Keep all shown", use_container_width=True, key="sc_keepall",
                     help="Force-keep everything shown, overriding any rejections."):
            for t in _shown_tks:
                _commit_decision(t, True, persist=False)
            _persist_run(_sc_run)
            st.rerun()
        if b3.button(":material/cancel: Reject all shown", use_container_width=True, key="sc_rejall"):
            for t in _shown_tks:
                _commit_decision(t, False, persist=False)
            _persist_run(_sc_run)
            st.rerun()
        if b4.button(":material/undo: Reset shown", use_container_width=True, key="sc_resetall"):
            for t in _shown_tks:
                _VAL.pop(t, None)                          # back to undecided (defaults to keep)
                st.session_state.setdefault("ai_validated", set()).discard(t)
            _persist_run(_sc_run)
            st.rerun()
        if b5.button(":material/save: Save decisions", type="secondary",
                     use_container_width=True, key="sc_save"):
            _saved = 0
            for r in _runs:
                rows = [{"filename": fn, "cilia_id": cid, "human_validated": ok}
                        for (rl, fn, cid), ok in _VAL.items() if rl == r["label"]]
                if rows:
                    save_validation_file(r["dir"], rows)
                    _saved += len(rows)
            st.success(f":material/save: Saved {_saved} decisions to each run's "
                       "`csv/human_validation.csv`.")

        if _view_c.empty:
            st.info("No cilia match this filter.")

        # ── FOCUS mode: one big card, keep/reject advances ────────────────────
        elif _mode.startswith(":material/center_focus_strong:"):
            _sig = f"{_sc_run}|{_sc_sample}|{_sc_show}|{_sc_sort}"
            if st.session_state.get("_focus_sig") != _sig:
                st.session_state["_focus_sig"] = _sig
                st.session_state["_focus_i"] = 0
            _n = len(_view_c)
            _i = int(np.clip(st.session_state.get("_focus_i", 0), 0, _n - 1))
            row = _view_c.iloc[_i]
            tk = row["_tk"]
            cid = int(row["cilia_id"])
            cls = str(row["class"]) if "class" in row else "?"
            keep = bool(row["_keep"])
            colr = "#2ecc71" if keep else "#e74c3c"

            st.markdown(
                f"<div style='border:3px solid {colr};border-radius:10px;"
                f"padding:6px 12px;display:inline-block'>"
                f"<b>cilia {cid}</b> &nbsp;·&nbsp; "
                f"<span style='color:{_CLASS_PALETTE.get(cls,'#888')}'>●</span> {cls} "
                f"&nbsp;·&nbsp; {'KEPT' if keep else 'REJECTED'}</div>",
                unsafe_allow_html=True)
            ic, sc = st.columns([2, 1])
            with ic:
                # Gallery shows RAW intensities (re-rendered from the .npz crop).
                _p = _roi_model_png(_sc_run, row["filename"], cid, False)
                if _p:
                    st.image(_p, use_container_width=True)
                else:
                    st.warning("No ROI screenshot saved for this cilium.")
            with sc:
                st.markdown(f"**{Path(str(row['filename'])).stem}**")
                st.markdown(_cilia_stat_caption(row) or "_no metrics_")
                st.caption(f"{_i + 1} / {_n}")

            n1, n2, n3, n4 = st.columns(4)
            if n1.button(":material/arrow_back: Prev", use_container_width=True, key="fx_prev"):
                st.session_state["_focus_i"] = max(0, _i - 1)
                st.rerun()
            if n2.button(":material/check_circle: Keep ▶", use_container_width=True, key="fx_keep"):
                _commit_decision(tk, True)
                st.session_state["_focus_i"] = min(_n - 1, _i + 1)
                st.rerun()
            if n3.button(":material/cancel: Reject ▶", use_container_width=True, key="fx_rej"):
                _commit_decision(tk, False)
                st.session_state["_focus_i"] = min(_n - 1, _i + 1)
                st.rerun()
            if n4.button("Next :material/arrow_forward:", use_container_width=True, key="fx_next"):
                st.session_state["_focus_i"] = min(_n - 1, _i + 1)
                st.rerun()

        # ── GALLERY mode: grid of cards with a keep/reject toggle each ────────
        else:
            _per_row = 6                # square thumbnails → compact uniform grid
            grid = list(_view_c.iterrows())

            @st.fragment
            def _screen_card(row, run_label):
                """One card. As a fragment, toggling it reruns *only this card* —
                the rest of the page (other cards, the AI panel) is left untouched,
                so screening no longer reloads the whole UI on every click."""
                tk = row["_tk"]
                cid = int(row["cilia_id"])
                cls = str(row["class"]) if "class" in row else "?"
                wk = f"sg_{tk}"
                # Sync the toggle to the store *before* the widget is made.
                st.session_state[wk] = bool(_VAL.get(tk, True))
                keep = bool(st.session_state[wk])
                colr = "#2ecc71" if keep else "#e74c3c"
                dot = _CLASS_PALETTE.get(cls, "#888")
                _samp = (str(row["file_short"])
                         if "file_short" in row and pd.notna(row.get("file_short"))
                         else Path(str(row["filename"])).stem)
                # Sample name shown muted + truncated; full name on hover (title attr).
                st.markdown(
                    f"<div style='border-top:7px solid {colr};"
                    f"border-radius:5px;margin-bottom:2px'></div>"
                    f"<span title='sample: {_samp}'>"
                    f"<span style='color:{dot}'>●</span> "
                    f"<b>cilia {cid}</b> · {cls}</span>"
                    f"<div title='{_samp}' style='color:#888;font-size:0.75em;"
                    f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>"
                    f"{_samp}</div>", unsafe_allow_html=True)
                # Gallery shows RAW intensities (re-rendered from the .npz crop).
                _p = _roi_model_png(run_label, row["filename"], cid, False)
                if _p:
                    st.image(_p, use_container_width=True)
                else:
                    st.markdown(
                        "<div style='height:120px;display:flex;"
                        "align-items:center;justify-content:center;"
                        "background:#f3f3f3;border-radius:5px;color:#999'>"
                        "no ROI image</div>", unsafe_allow_html=True)
                cap = _cilia_stat_caption(row)
                if cap:
                    st.caption(cap)
                _aiv = row.get("_ai", np.nan)
                if pd.notna(_aiv):
                    st.caption(f":material/smart_toy: score {_aiv:.2f}")
                st.toggle("keep this cilium", key=wk,
                          on_change=_gallery_toggle, args=(tk,))

            for start in range(0, len(grid), _per_row):
                chunk = grid[start:start + _per_row]
                for col, (_, row) in zip(st.columns(_per_row), chunk):
                    with col:
                        _screen_card(row, _sc_run)



# ─────────────────────────────────────────────────────────────────────────────
# TAB — HIST / KDE
# ─────────────────────────────────────────────────────────────────────────────
with tab_dist:
    st.subheader(":material/bar_chart: Distribution (compare runs)")
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
        _kind = st.radio("Style",
                         ["hist + kde", "kde only", "violin + points",
                          "beeswarm", "stacked bar"],
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

    _dots_only = False
    _dot_hue = None
    if _kind in ("violin + points", "beeswarm"):
        vc1, vc2 = st.columns(2)
        with vc1:
            if _kind == "violin + points":
                _dots_only = st.checkbox("Dots + mean only (hide violin)",
                                         value=False, key="d_dots_only")
        with vc2:
            _hue_opts = ["(none)"] + [c for c in _cat_cols if c != _grp]
            _dot_hue = st.selectbox("Color dots by", _hue_opts, key="d_dot_hue")
            _dot_hue = None if _dot_hue == "(none)" else _dot_hue

    _color_by = "metric bins"
    _cmap_name = "viridis"
    _custom_colors = None
    if _kind == "stacked bar":
        sc1, sc2 = st.columns(2)
        with sc1:
            _color_opts = ["metric bins"] + [c for c in _cat_cols if c != _grp]
            _color_by = st.selectbox("Color by", _color_opts, key="d_stack_color",
                                     help="Stack each bar by metric bins, or by a "
                                          "category like class / sample name.")
        with sc2:
            _scale_opts = ["viridis", "plasma", "magma", "cividis", "coolwarm",
                           "tab10", "Set2", "Prism palette", "custom"]
            _cmap_name = st.selectbox("Color scale", _scale_opts, key="d_stack_scale")

        # Segment labels — needed to render the per-segment custom pickers.
        _seg_labels = []
        if _color_by == "metric bins":
            if _metric and not _data.empty:
                try:
                    _seg_labels = [str(b) for b in
                                   pd.cut(_data[_metric], bins=10).cat.categories]
                except Exception:                              # noqa: BLE001
                    _seg_labels = []
        else:
            _seg_labels = [str(v) for v in
                           sorted(_data[_color_by].dropna().unique().tolist())]

        if _cmap_name == "custom" and _seg_labels:
            _custom_colors = {}
            with st.expander("Pick a color per segment", expanded=True):
                _pcols = st.columns(min(4, len(_seg_labels)) or 1)
                for _i, _lab in enumerate(_seg_labels):
                    _base = _PRISM_PALETTE[_i % len(_PRISM_PALETTE)]
                    with _pcols[_i % len(_pcols)]:
                        _custom_colors[_lab] = st.color_picker(
                            _lab, _base, key=f"d_stack_col_{_i}")
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
            elif _kind == "stacked bar":
                # x-axis = chosen group; each bar split into stacked segments,
                # either by metric bins or by a chosen category (class / sample).
                if _color_by == "metric bins":
                    _seg = pd.cut(_data[_metric], bins=10)
                    _seg_title = _xlabel
                else:
                    _seg = _data[_color_by]
                    _seg_title = _color_by
                _ct = (_data.assign(_seg=_seg)
                       .groupby([_grp, "_seg"], observed=True).size()
                       .unstack("_seg").fillna(0))
                if _stat in ("density", "probability"):
                    _ct = _ct.div(_ct.sum(axis=1).replace(0, 1), axis=0)
                _n = _ct.shape[1]
                _labels = [str(c) for c in _ct.columns]
                if _custom_colors is not None:
                    _colors = [_custom_colors.get(_l, "#888888") for _l in _labels]
                elif _cmap_name == "Prism palette":
                    _colors = [_PRISM_PALETTE[i % len(_PRISM_PALETTE)]
                               for i in range(_n)]
                else:
                    _cmap = plt.get_cmap(_cmap_name)
                    if getattr(_cmap, "N", 256) <= 20:        # qualitative map
                        _colors = [_cmap(i % _cmap.N) for i in range(_n)]
                    else:                                      # continuous map
                        _colors = [_cmap(i / max(_n - 1, 1)) for i in range(_n)]
                _ct.plot(kind="bar", stacked=True, ax=ax, width=0.85, color=_colors)
                ax.set_xlabel(_grp)
                ax.set_ylabel(_stat if _stat != "count" else "count")
                ax.tick_params(axis="x", rotation=30)
                _tidy_legend(ax, title=_seg_title)
            elif _kind == "beeswarm":  # GraphPad bee-swarm: non-overlapping dots
                _dot_kw = dict(data=_data, x=_grp, y=_metric, ax=ax, size=3,
                               alpha=0.8)
                if _dot_hue:
                    sns.swarmplot(hue=_dot_hue, palette=_PRISM_PALETTE, **_dot_kw)
                    _tidy_legend(ax, title=_dot_hue)
                else:
                    sns.swarmplot(color="#222222", **_dot_kw)
                _draw_mean_bars(ax, _data, _grp, _metric)
                ax.set_xlabel(_grp)
                ax.set_ylabel(_xlabel)
                ax.tick_params(axis="x", rotation=30)
            else:  # violin + points (GraphPad-style: violin with jittered raw points)
                if not _dots_only:
                    sns.violinplot(data=_data, x=_grp, y=_metric, hue=_grp,
                                   legend=False, palette=_PRISM_PALETTE, cut=0,
                                   inner=None, linewidth=1.2, saturation=0.85, ax=ax)
                    # de-saturate the violins so the overlaid points stand out
                    for _coll in ax.collections:
                        _coll.set_alpha(0.35)
                _dot_kw = dict(data=_data, x=_grp, y=_metric, ax=ax, size=2.5,
                               alpha=0.5, jitter=0.25)
                if _dot_hue:
                    sns.stripplot(hue=_dot_hue, palette=_PRISM_PALETTE, dodge=False,
                                  **_dot_kw)
                    _tidy_legend(ax, title=_dot_hue)
                else:
                    sns.stripplot(color="#222222", **_dot_kw)
                if _dots_only:
                    _draw_mean_bars(ax, _data, _grp, _metric)
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
    st.subheader(":material/scatter_plot: Scatter (compare runs)")
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
            _equal_axes = st.checkbox("Equal axes (same min/max)", value=False,
                                      key="s_equal",
                                      help="Force X and Y to share limits + draw y=x.")
            _show_bounds = st.checkbox("Class boundaries", value=False, key="s_bounds",
                                       help="Draw the neurite/soma log_ratio threshold "
                                            "lines. Needs dt_neurite vs dt_nuclei (or "
                                            "their log) on the axes, no normalization.")

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
            if _show_bounds:
                _NEU = {"dt_neurite_um", "dt_neurite", "log_dt_neurite_um",
                        "log_dt_neurite"}
                _NUC = {"dt_nuclei_um", "dt_nuclei", "log_dt_nuclei_um",
                        "log_dt_nuclei"}

                def _role(c):
                    if c in _NEU:
                        return ("neurite", "log" in c)
                    if c in _NUC:
                        return ("nuclei", "log" in c)
                    return (None, None)

                _rx, _ry = _role(_sx), _role(_sy)
                if _sx_norm != "none" or _sy_norm != "none":
                    st.caption(":material/warning: Class boundaries hidden — turn off normalization.")
                elif None in (_rx[0], _ry[0]) or _rx[0] == _ry[0] or _rx[1] != _ry[1]:
                    st.caption(":material/warning: Class boundaries need dt_neurite vs dt_nuclei "
                               "(or their log) on the two axes.")
                else:
                    _islog = _rx[1]
                    _x_neu = _rx[0] == "neurite"   # neurite is the denominator
                    _xlo, _xhi = ax.get_xlim()
                    _ylim0 = ax.get_ylim()         # keep the data-driven y view
                    if not _islog:
                        _xlo = max(_xlo, 0.0)
                    _xs = np.linspace(_xlo, _xhi, 50)
                    for _thr, _nm, _clr in ((_neurite_thr, "neurite ≥", "#0C5DA5"),
                                            (_soma_thr, "soma ≤", "#E8000B")):
                        if _islog:
                            _ys = _xs + _thr if _x_neu else _xs - _thr
                        else:
                            _m = np.exp(_thr) if _x_neu else np.exp(-_thr)
                            _ys = _m * _xs
                        ax.plot(_xs, _ys, ls="--", lw=1.4, color=_clr,
                                label=f"{_nm} {_thr:g}", zorder=1)
                    ax.set_ylim(_ylim0)            # lines must not rescale the view
            if _equal_axes:
                _vals = pd.concat([pd.to_numeric(_sdata[_sx], errors="coerce"),
                                   pd.to_numeric(_sdata[_sy], errors="coerce")])
                _lo, _hi = _vals.min(), _vals.max()
                if pd.notna(_lo) and pd.notna(_hi) and _hi > _lo:
                    _pad = 0.03 * (_hi - _lo)
                    _lim = (_lo - _pad, _hi + _pad)
                    ax.set_xlim(_lim)
                    ax.set_ylim(_lim)
                    ax.plot(_lim, _lim, ls="--", lw=1, color="#888", zorder=0)
                    ax.set_aspect("equal", adjustable="box")
            _tidy_legend(ax, title=_sc_color)
            _show_and_export(fig, f"scatter_{_sx}_{_sy}", "dl_scatter")
        except Exception as exc:                          # noqa: BLE001
            st.error(f"Plot error: {exc}")
            plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# TAB — INTERACTIVE SCATTER (hover = sample/ratio/log_ratio, click = ROI image)
# ─────────────────────────────────────────────────────────────────────────────
with tab_inter:
    st.subheader(":material/bolt: Interactive scatter")
    if not _PLOTLY:
        st.info("Install plotly for the interactive explorer:  `pip install plotly`")
    elif len(_num_cols) < 2:
        st.info("Need at least two numeric columns.")
    else:
        _i_mode = st.radio("Plot", ["Scatter", "PCA"], horizontal=True, key="i_mode")
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            _ixi = _num_cols.index("dt_neurite_um") if "dt_neurite_um" in _num_cols else 0
            _ix = st.selectbox("X", _num_cols, index=_ixi, key="i_x",
                               disabled=_i_mode == "PCA")
        with ic2:
            _iyi = _num_cols.index("dt_nuclei_um") if "dt_nuclei_um" in _num_cols else 1
            _iy = st.selectbox("Y", _num_cols, index=_iyi, key="i_y",
                               disabled=_i_mode == "PCA")
        with ic3:
            _icol_opts = [c for c in ("class", "run", "file_short",
                                      "human_validated") if c in _df.columns] or ["run"]
            _icolor = st.selectbox("Color by", _icol_opts, key="i_color")

        _idata = (_cilia if _has_otype else _df).copy()
        _sample_col = "file_short" if "file_short" in _idata.columns else "filename"
        # Fields shown on hover (only those present).
        _hover = {c: True for c in ("ratio", "log_ratio", "cilia_id") if c in _idata.columns}
        _hover[_sample_col] = True
        _cd = [c for c in ("run", "filename", "cilia_id") if c in _idata.columns]
        _cmap = _CLASS_PALETTE if _icolor == "class" else None

        fig = None
        if _i_mode == "PCA":
            # PCA over all numeric vars except human_validated (z-scored, via SVD).
            _pca_cols = [c for c in _num_cols if c != "human_validated"]
            _mat = (_idata[_pca_cols].apply(pd.to_numeric, errors="coerce")
                    .dropna())
            if _mat.shape[0] < 3 or _mat.shape[1] < 2:
                st.info("Not enough complete numeric rows for PCA.")
            else:
                _std = _mat.std(ddof=0).replace(0, 1)
                _z = (_mat - _mat.mean()) / _std
                _u, _s, _ = np.linalg.svd(_z.values, full_matrices=False)
                _scores = _u[:, :2] * _s[:2]
                _ev = (_s ** 2) / (_s ** 2).sum()
                _pf = _idata.loc[_mat.index].copy()
                _pf["PC1"], _pf["PC2"] = _scores[:, 0], _scores[:, 1]
                fig = px.scatter(
                    _pf, x="PC1", y="PC2", color=_icolor,
                    color_discrete_map=_cmap, custom_data=_cd, hover_data=_hover,
                    opacity=0.8, template="simple_white",
                    labels={"PC1": f"PC1 ({_ev[0] * 100:.1f}%)",
                            "PC2": f"PC2 ({_ev[1] * 100:.1f}%)"},
                )
        else:
            fig = px.scatter(
                _idata, x=_ix, y=_iy, color=_icolor,
                color_discrete_map=_cmap, custom_data=_cd, hover_data=_hover,
                opacity=0.8, template="simple_white",
            )

        if fig is not None:
            fig.update_traces(marker=dict(size=8, line=dict(width=0.4, color="white")))
            fig.update_layout(height=560, legend_title_text=_icolor,
                              margin=dict(l=10, r=10, t=30, b=10))

            st.caption("Hover for sample / ratio / log_ratio. **Click a point** to see "
                       "its cilium ROI image below.")
            _plot_col, _img_col = st.columns([3, 1])
            with _plot_col:
                try:
                    _event = st.plotly_chart(fig, use_container_width=True,
                                             on_select="rerun", selection_mode="points",
                                             key="iplot")
                except TypeError:
                    # Older Streamlit without selection events — hover still works.
                    st.plotly_chart(fig, use_container_width=True)
                    _event = None

            # Resolve the clicked point → its ROI screenshot.
            _sel = []
            if _event is not None:
                try:
                    _sel = _event.selection["points"]
                except Exception:                             # noqa: BLE001
                    try:
                        _sel = _event["selection"]["points"]
                    except Exception:                         # noqa: BLE001
                        _sel = []
            with _img_col:
                if not _cd:
                    st.caption("No run/filename columns to locate ROI images.")
                elif _sel:
                    cd = _sel[0].get("customdata") or []
                    info = dict(zip(_cd, cd))
                    rl = info.get("run", _runs[0]["label"])
                    st.markdown(f"**cilia {info.get('cilia_id')}**  ·  {info.get('filename', '')}")
                    # Prefer the all-channel render (cilia=green, BB=magenta,
                    # neurite=cyan, nuclei=blue); fall back to the saved thumbnail.
                    _npz = _roi_npz_path(rl, info.get("filename"), info.get("cilia_id"))
                    _multi = _roi_multichannel_png(_npz)
                    if _multi is not None:
                        st.image(_multi, use_container_width=True)
                        st.caption(":green[:material/circle:] cilia · :violet[:material/circle:] basal body · :blue[:material/circle:] nuclei · cyan neurite")
                    else:
                        path = _roi_png_path(rl, info.get("filename"),
                                             info.get("cilia_id"))
                        if path:
                            st.image(path, use_container_width=True)
                        else:
                            st.caption("No ROI saved for this cilium.")
                else:
                    st.caption("Click a point to preview its ROI.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB — CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
with tab_corr:
    st.subheader(":material/link: Correlation heatmap")
    _default_corr = [c for c in ("log_ratio", "ratio", "distance_to_neurite_um",
                                 "dt_neurite_um", "dt_nuclei_um", "volume_um3",
                                 "length_um", "pair_distance_um") if c in _num_cols]
    _HV = "human_validated"
    _corr_opts = _num_cols + ([_HV] if _HV in _df.columns else [])
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        _corr_cols = st.multiselect("Variables", _corr_opts,
                                    default=_default_corr or _num_cols[:6],
                                    key="corr_cols")
    with cc2:
        _method = st.selectbox("Method", ["pearson", "spearman"], key="corr_method")
    with cc3:
        _corr_run = st.selectbox("Run", ["All"] + [r["label"] for r in _runs],
                                 key="corr_run")

    # If human_validated is in play, correlate over ALL cilia (incl. rejected)
    # so the column actually has variance even when the exclude toggle is on.
    if _HV in _corr_cols:
        _cdata = _cilia_full.copy()
        _cdata[_HV] = _cdata[_HV].astype(float)
    else:
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
    st.subheader(":material/list_alt: Per-object table")
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
                 "human_validated", "log_ratio", "ratio", "distance_to_neurite_um",
                 "dt_neurite_um", "dt_nuclei_um", "volume_um3", "length_um",
                 "paired_id", "pair_distance_um", "pairing_status"]
    _order = [c for c in _priority if c in _view.columns]
    _order += [c for c in _view.columns if c not in _order]
    _view = _view[_order]

    st.caption(f"Showing {len(_view):,} of {len(_df):,} objects")
    st.dataframe(_view, use_container_width=True, height=420)
    st.download_button(":material/download: Download filtered (CSV)",
                       _view.to_csv(index=False).encode(),
                       "objects_filtered.csv", "text/csv", key="dl_tbl")

    # Per-run / per-sample summary
    st.markdown("---")
    st.subheader(":material/science: Summary")
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
        st.download_button(":material/download: Download summary (CSV)",
                           _summary.to_csv(index=False).encode(),
                           "summary.csv", "text/csv", key="dl_sum")


# ─────────────────────────────────────────────────────────────────────────────
# TAB — QC
# ─────────────────────────────────────────────────────────────────────────────
with tab_qc:
    st.subheader(":material/monitor_heart: Quality control")
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
            st.caption(":material/warning: Fewer than 4 samples per run — too few to flag statistical outliers.")
        elif _n_flagged:
            st.warning(
                "Deviant samples (robust modified z-score > 3.5 within their run): "
                + ", ".join(
                    f"**{r.get('file_short', r.get('filename'))}** ({r['flags']})"
                    for _, r in _qc[_qc["flags"] != ""].iterrows()
                )
            )
        else:
            st.success(":material/check: No deviant samples — all metrics within robust range.")

        st.pyplot(_qc_bar_fig(_qc), use_container_width=True)

        # Per-sample QC table, flagged rows highlighted
        def _hl(row):
            return ["background-color: #fdecea" if row["flags"] else "" for _ in row]
        st.dataframe(_qc.style.apply(_hl, axis=1), use_container_width=True)
        st.download_button(":material/download: Download QC table (CSV)",
                           _qc.to_csv(index=False).encode(),
                           "qc_per_sample.csv", "text/csv", key="dl_qc")

        # ── Cilia-count comparison across runs ─────────────────────────────────
        if _has_otype and {"run"}.issubset(_cilia.columns) and not _cilia.empty:
            st.markdown("#### :material/pin: Cilia count comparison")
            _per_run = (_cilia.groupby("run").size()
                        .rename("n_cilia").reset_index())
            _per_samp = (_cilia.groupby(["run", "filename"]).size()
                         .rename("n_cilia").reset_index())
            cc1, cc2 = st.columns(2)
            with cc1:
                st.caption("Total cilia per run")
                _frun = plt.figure(figsize=(5, 3.2))
                _ax = _frun.gca()
                _ax.bar(_per_run["run"].astype(str), _per_run["n_cilia"],
                        color=_PRISM_PALETTE[:len(_per_run)] or "#0C5DA5")
                _ax.set_ylabel("cilia")
                _ax.tick_params(axis="x", rotation=30)
                st.pyplot(_frun, use_container_width=True)
                plt.close(_frun)
            with cc2:
                st.caption("Per sample (cilia per image)")
                _fsm = plt.figure(figsize=(5, 3.2))
                _axs = _fsm.gca()
                sns.boxplot(data=_per_samp, x="run", y="n_cilia", ax=_axs,
                            hue="run", legend=False, palette=_PRISM_PALETTE)
                sns.stripplot(data=_per_samp, x="run", y="n_cilia", ax=_axs,
                              color="#222", size=3, alpha=0.6)
                _axs.set_ylabel("cilia / image")
                _axs.tick_params(axis="x", rotation=30)
                st.pyplot(_fsm, use_container_width=True)
                plt.close(_fsm)
            st.dataframe(_per_run, use_container_width=True, hide_index=True)

        # ── Ciliation rate (cilia per estimated nucleus) ───────────────────────
        st.markdown("#### :material/science: Ciliation rate")
        _nuc = combined_nuclei(_runs)
        if _nuc.empty:
            st.info("No `nuclei_summary.csv` found for the loaded runs. Re-run the "
                    "pipeline to export nuclei volumes, then ciliation rate "
                    "(cilia ÷ estimated nuclei) will appear here.")
        else:
            st.caption("Nuclei often merge in segmentation, so the nucleus count is "
                       "**estimated** as total nuclei volume ÷ a typical single-"
                       "nucleus volume. Ciliation rate = cilia ÷ estimated nuclei.")
            _single_vol = st.number_input(
                "Typical single-nucleus volume (µm³)", min_value=1.0,
                value=500.0, step=10.0, key="cil_single_vol",
                help="Set from your data (e.g. median single-nucleus volume). The "
                     "estimated nucleus count = nuclei_volume_um3 / this.")
            # Cilia per (run, filename).
            _cpf = (_cilia.groupby(["run", "filename"]).size()
                    .rename("n_cilia").reset_index())
            _merged = _nuc.merge(_cpf, on=["run", "filename"], how="left")
            _merged["n_cilia"] = _merged["n_cilia"].fillna(0).astype(int)
            _merged["est_nuclei"] = _merged["nuclei_volume_um3"] / _single_vol
            _merged["ciliation_rate"] = (
                _merged["n_cilia"] / _merged["est_nuclei"].replace(0, np.nan))
            _disp_cols = [c for c in ("run", "file_short", "filename", "n_cilia",
                                      "nuclei_volume_um3", "est_nuclei",
                                      "ciliation_rate") if c in _merged.columns]
            gc1, gc2 = st.columns([1, 1])
            with gc1:
                st.caption("Mean ciliation rate per run")
                _cr_run = (_merged.groupby("run")["ciliation_rate"]
                           .mean().reset_index())
                _fcr = plt.figure(figsize=(5, 3.2))
                _axc = _fcr.gca()
                _axc.bar(_cr_run["run"].astype(str), _cr_run["ciliation_rate"],
                         color=_PRISM_PALETTE[:len(_cr_run)] or "#00B945")
                _axc.set_ylabel("cilia / nucleus")
                _axc.tick_params(axis="x", rotation=30)
                st.pyplot(_fcr, use_container_width=True)
                plt.close(_fcr)
            with gc2:
                st.caption("Per sample")
                _fcs = plt.figure(figsize=(5, 3.2))
                _axcs = _fcs.gca()
                sns.boxplot(data=_merged, x="run", y="ciliation_rate", ax=_axcs,
                            hue="run", legend=False, palette=_PRISM_PALETTE)
                sns.stripplot(data=_merged, x="run", y="ciliation_rate", ax=_axcs,
                              color="#222", size=3, alpha=0.6)
                _axcs.set_ylabel("cilia / nucleus")
                _axcs.tick_params(axis="x", rotation=30)
                st.pyplot(_fcs, use_container_width=True)
                plt.close(_fcs)
            st.dataframe(_merged[_disp_cols].round(3), use_container_width=True,
                         hide_index=True)
            st.download_button(":material/download: Download ciliation table (CSV)",
                               _merged[_disp_cols].to_csv(index=False).encode(),
                               "ciliation_rate.csv", "text/csv", key="dl_cil")

        # ── False positives from human screening ──────────────────────────────
        st.markdown("#### :material/block: False positives (human-rejected detections)")
        st.caption("A false positive = a detected cilium the reviewer rejected "
                   "in the :material/person_search: Screening tab. Rate is over **screened** cilia "
                   "(unreviewed ones are ignored).")
        if "human_validated" not in _cilia_full.columns or _cilia_full.empty:
            st.info("No cilia to evaluate.")
        else:
            _fpd = _cilia_full.copy()
            _fpd["_fp"] = ~_fpd["human_validated"].astype(bool)
            # "Screened" = cilia with a saved decision (in _VAL); count only the
            # *human* ones (exclude AI-applied) so the rate stays grounded in
            # human validation. (The AI exclusion is session-scoped.)
            _human_seen = set(_VAL) - st.session_state.get("ai_validated", set())

            def _is_seen(rl, fn, cid):
                if cid is None or (isinstance(cid, float) and np.isnan(cid)):
                    return False
                try:
                    return _val_key(rl, fn, cid) in _human_seen
                except (ValueError, TypeError):
                    return False

            _fpd["_seen"] = [
                _is_seen(rl, fn, cid)
                for rl, fn, cid in zip(_fpd["run"], _fpd["filename"],
                                       _fpd["cilia_id"])
            ] if "cilia_id" in _fpd.columns else False
            _det = int(len(_fpd))
            _scr = int(_fpd["_seen"].sum())
            _fp_n = int(_fpd["_fp"].sum())
            _rate = 100 * _fp_n / _scr if _scr else 0.0
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Detected", _det)
            f2.metric("Screened", _scr)
            f3.metric("False positives", _fp_n)
            f4.metric("FP rate", f"{_rate:.1f}%")
            if _scr == 0:
                st.warning("Nothing screened yet — review cilia in the :material/person_search: "
                           "Screening tab to compute false positives.")
            else:
                _fp_tbl = (_fpd.groupby("run")
                           .agg(detected=("_fp", "size"),
                                screened=("_seen", "sum"),
                                false_positives=("_fp", "sum"))
                           .reset_index())
                _fp_tbl["FP_rate_%"] = (
                    100 * _fp_tbl["false_positives"]
                    / _fp_tbl["screened"].replace(0, np.nan)).round(1)
                st.dataframe(_fp_tbl, use_container_width=True)
                st.download_button(":material/download: Download false-positive table (CSV)",
                                   _fp_tbl.to_csv(index=False).encode(),
                                   "false_positives.csv", "text/csv", key="dl_fp")

        # The pipeline's own QC sheets (per run), shown verbatim
        with st.expander(":material/article: Pipeline QC sheets (per run)", expanded=False):
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
# TAB — VALIDATION (pipeline vs hand analysis)
# ─────────────────────────────────────────────────────────────────────────────
def _agreement(hand, pipe):
    """(pearson_r, mean_abs_error, bias) for two aligned numeric series."""
    h = pd.to_numeric(hand, errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(pipe, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(h) & np.isfinite(p)
    h, p = h[ok], p[ok]
    if len(h) < 2:
        return float("nan"), float("nan"), float("nan")
    r = float(np.corrcoef(h, p)[0, 1]) if h.std() and p.std() else float("nan")
    return r, float(np.mean(np.abs(p - h))), float(np.mean(p - h))


def _validation_compare(label, merged, hand_col, pipe_col, unit=""):
    """Compact agreement view for one quantity: a one-line stats summary plus a
    correlation scatter (y=x) and a Bland–Altman plot, side by side."""
    if hand_col not in merged or pipe_col not in merged:
        return
    sub = merged[[hand_col, pipe_col, "stem"]].dropna(subset=[hand_col, pipe_col])
    if sub.empty:
        return
    h = sub[hand_col].to_numpy(float)
    p = sub[pipe_col].to_numpy(float)
    r, mae, bias = _agreement(sub[hand_col], sub[pipe_col])
    _mean = (h + p) / 2.0
    _diff = p - h
    _md, _sd = float(np.mean(_diff)), float(np.std(_diff, ddof=1) if len(_diff) > 1 else 0.0)
    _loa = 1.96 * _sd

    st.markdown(
        f"**{label}** — r={'—' if np.isnan(r) else f'{r:.2f}'} · "
        f"MAE={mae:.2f}{unit} · bias={bias:+.2f}{unit} · "
        f"Σpipe/Σhand={p.sum():.0f}/{h.sum():.0f}")
    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(3.5, 3.1))
        ax.scatter(h, p, s=26, color="#0C5DA5", edgecolor="white",
                   linewidth=0.4, zorder=3)
        _lim = [0, float(max(h.max(), p.max())) * 1.1 + 1]
        ax.plot(_lim, _lim, ls="--", lw=1, color="#888", zorder=1)
        ax.set_xlim(_lim); ax.set_ylim(_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("hand"); ax.set_ylabel("pipeline")
        _r2 = "—" if np.isnan(r) else f"{r**2:.2f}"
        ax.text(0.05, 0.95, f"$R^2$ = {_r2}", transform=ax.transAxes,
                va="top", ha="left", fontsize=9, fontweight="bold")
        ax.set_title("correlation", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with g2:
        fig2, ax2 = plt.subplots(figsize=(3.5, 3.1))
        ax2.scatter(_mean, _diff, s=26, color="#E8000B", edgecolor="white",
                    linewidth=0.4, zorder=3)
        ax2.axhline(_md, color="#222", lw=1.2)
        ax2.axhline(_md + _loa, color="#888", ls="--", lw=1)
        ax2.axhline(_md - _loa, color="#888", ls="--", lw=1)
        ax2.set_xlabel("mean(hand, pipeline)"); ax2.set_ylabel("pipeline − hand")
        ax2.set_title("Bland–Altman", fontsize=9)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


def parse_boxes_as_hand(src) -> pd.DataFrame:
    """Aggregate the annotator's per-cilia box Excel into the same per-sample
    columns ``parse_hand_analysis`` yields, so the box file can drive this tab:
    ``hand_cilia`` = boxes/sample, ``hand_on_soma``/``hand_on_neurite`` from the
    box class. Nucleus counts aren't available from boxes (left NaN)."""
    try:
        b = pd.read_excel(src, sheet_name="cilia_boxes")
    except Exception:                                     # noqa: BLE001
        b = pd.read_excel(src, sheet_name=0)
    if b.empty or "filename" not in b.columns:
        return pd.DataFrame()
    b = b.copy()
    b["stem"] = b["filename"].apply(_img_stem)
    b["class"] = (b["class"].astype(str).str.strip().str.lower()
                  if "class" in b.columns else "uncertain")
    g = b.groupby("stem")
    out = pd.DataFrame({
        "hand_cilia": g.size(),
        "hand_on_soma": g["class"].apply(lambda s: int((s == "soma").sum())),
        "hand_on_neurite": g["class"].apply(lambda s: int((s == "neurite").sum())),
    }).reset_index()
    out["filename"] = out["stem"]
    out["hand_nuclei"] = np.nan
    return out


with tab_validate:
    st.subheader(":material/handshake: Validate pipeline vs hand analysis")
    st.caption("Compare the pipeline to ground truth (matched by filename). Use a "
               "per-sample hand-count Excel, or the per-cilia box Excel from the "
               "napari annotator (aggregated to per-sample counts).")

    _val_mode = st.radio(
        "Ground-truth source",
        ["Per-sample hand counts", "Per-cilia boxes (annotator)"],
        horizontal=True, key="val_mode")
    _box_mode = _val_mode.startswith("Per-cilia")

    _up = st.file_uploader(
        "Per-cilia box Excel (.xlsx)" if _box_mode else "Hand-analysis Excel (.xlsx)",
        type=["xlsx"], key="val_upload")
    _src = _up
    if _box_mode:
        if _up is None and st.session_state.get("_box_path"):
            _src = st.session_state["_box_path"]
            st.caption(f"No file uploaded — using the Box-GT tab path `{_src}`.")
    else:
        _default_hand = os.path.join("hand-analysis", "d38-hand-analysis.xlsx")
        if _up is None and os.path.exists(_default_hand):
            st.caption(f"No file uploaded — using `{_default_hand}`.")
            _src = _default_hand

    if _src is None:
        st.info("Upload a ground-truth Excel to begin.")
    else:
        try:
            _hand = (parse_boxes_as_hand(_src) if _box_mode
                     else parse_hand_analysis(_src))
        except Exception as _exc:                         # noqa: BLE001
            st.error(f"Could not read the ground-truth file: {_exc}")
            _hand = pd.DataFrame()

        if _box_mode and not _hand.empty:
            st.caption("Box mode: `hand_cilia` = boxes/sample; on-soma / on-neurite "
                       "from the box class; nucleus counts unavailable.")

        if _hand.empty:
            st.warning("No rows parsed from the hand-analysis file.")
        else:
            vc1, vc2 = st.columns(2)
            with vc1:
                _val_run = st.selectbox("Pipeline run to compare",
                                        [r["label"] for r in _runs], key="val_run")
            with vc2:
                _val_single_vol = st.number_input(
                    "Single-nucleus volume (µm³)", min_value=1.0, value=500.0,
                    step=10.0, key="val_single_vol",
                    help="Used to estimate the pipeline nucleus count "
                         "(nuclei_volume ÷ this) for the comparison.")

            # Pipeline per-sample counts (validated/kept cilia) keyed by stem.
            _pc = _cilia[_cilia["run"] == _val_run].copy()
            _pc["stem"] = _pc["filename"].apply(_img_stem)
            _agg = _pc.groupby("stem").agg(pipe_cilia=("cilia_id", "count"))
            if "class" in _pc.columns:
                _agg["pipe_on_soma"] = _pc[_pc["class"] == "soma"].groupby("stem").size()
                _agg["pipe_on_neurite"] = _pc[_pc["class"].isin(
                    ["neurite", "axon"])].groupby("stem").size()
            _agg = _agg.reset_index()

            # Estimated nuclei from the per-sample nuclei volume.
            _nuc = combined_nuclei(_runs)
            if not _nuc.empty:
                _nr = _nuc[_nuc["run"] == _val_run].copy()
                _nr["stem"] = _nr["filename"].apply(_img_stem)
                _nr["pipe_est_nuclei"] = _nr["nuclei_volume_um3"] / _val_single_vol
                _agg = _agg.merge(_nr[["stem", "pipe_est_nuclei"]], on="stem",
                                  how="outer")

            _m = _hand.merge(_agg, on="stem", how="inner").fillna(
                {"pipe_cilia": 0, "pipe_on_soma": 0, "pipe_on_neurite": 0})
            st.caption(f"**{len(_m)}** samples matched by filename "
                       f"(hand: {len(_hand)}, pipeline run: {_agg['stem'].nunique()}).")

            if _m.empty:
                st.warning("No filenames matched between the hand analysis and this "
                           "run. Check that the names line up (extension aside).")
                with st.expander("Filenames on each side"):
                    cA, cB = st.columns(2)
                    cA.caption("Hand analysis"); cA.write(sorted(_hand["stem"]))
                    cB.caption("Pipeline run"); cB.write(sorted(_agg["stem"]))
            else:
                _validation_compare("cilia count", _m, "hand_cilia", "pipe_cilia")
                if "pipe_est_nuclei" in _m.columns:
                    _validation_compare("nucleus count", _m, "hand_nuclei",
                                        "pipe_est_nuclei")
                else:
                    st.info("No `nuclei_summary.csv` for this run — re-run the "
                            "pipeline to validate nucleus counts.")
                if "pipe_on_soma" in _m.columns:
                    _validation_compare("cilia on soma", _m, "hand_on_soma",
                                        "pipe_on_soma")
                    _validation_compare("cilia on neurite", _m, "hand_on_neurite",
                                        "pipe_on_neurite")

                # ── Optimize the soma/neurite thresholds to match the hand split ──
                with st.expander(":material/build: Optimize neurite / soma thresholds to match "
                                 "hand", expanded=False):
                    st.caption("Grid-searches the two log_ratio thresholds to best "
                               "reproduce the hand soma/neurite split (matches each "
                               "sample's **soma fraction**, so it's robust to total "
                               "count differences).")
                    _matched = set(_m["stem"])
                    _lr_by = {s: g["log_ratio"].dropna().to_numpy(float)
                              for s, g in _pc[_pc["stem"].isin(_matched)
                                              & _pc["log_ratio"].notna()].groupby("stem")
                              } if "log_ratio" in _pc.columns else {}
                    _hs = dict(zip(_m["stem"], _m["hand_on_soma"]))
                    _hn = dict(zip(_m["stem"], _m["hand_on_neurite"]))
                    _usable = [s for s in _lr_by
                               if len(_lr_by[s]) and (_hs.get(s, 0) or 0) +
                               (_hn.get(s, 0) or 0) > 0]
                    if len(_usable) < 3:
                        st.info("Need ≥3 matched samples with hand soma/neurite "
                                "counts and pipeline log_ratio.")
                    else:
                        def _loss(s_thr, n_thr):
                            errs = []
                            for st_ in _usable:
                                lr = _lr_by[st_]
                                ps = int((lr < s_thr).sum())
                                pn = int((lr > n_thr).sum())
                                hf = _hs[st_] / (_hs[st_] + _hn[st_])
                                pf = ps / (ps + pn) if (ps + pn) else 1.0
                                errs.append(abs(pf - hf))
                            return float(np.mean(errs))

                        _all = np.concatenate([_lr_by[s] for s in _usable])
                        _lo, _hi = np.percentile(_all, [2, 98])
                        _grid = np.linspace(float(_lo), float(_hi), 25)
                        _bs = _bn = None
                        _bl = np.inf
                        _M = np.full((25, 25), np.nan)
                        for _i, _sv in enumerate(_grid):
                            for _j, _nv in enumerate(_grid):
                                if _nv < _sv:
                                    continue
                                _L = _loss(_sv, _nv)
                                _M[_i, _j] = _L
                                if _L < _bl:
                                    _bs, _bn, _bl = _sv, _nv, _L
                        _cur = _loss(_soma_thr, _neurite_thr)
                        o1, o2, o3 = st.columns(3)
                        o1.metric("Best soma thr", f"{_bs:.2f}")
                        o2.metric("Best neurite thr", f"{_bn:.2f}")
                        o3.metric("Soma-fraction error", f"{_bl:.3f}",
                                  delta=f"{_bl - _cur:+.3f} vs current",
                                  delta_color="inverse")
                        st.caption(f"Current (soma<{_soma_thr:.2f}, "
                                   f"neurite>{_neurite_thr:.2f}) error = {_cur:.3f} "
                                   f"over {len(_usable)} samples.")
                        _ofig, _oax = plt.subplots(figsize=(4.2, 3.6))
                        _im = _oax.imshow(_M, origin="lower", aspect="auto",
                                          cmap="viridis_r",
                                          extent=[_grid[0], _grid[-1],
                                                  _grid[0], _grid[-1]])
                        _oax.scatter([_bn], [_bs], color="red", marker="*", s=90,
                                     edgecolor="white", zorder=3)
                        _oax.set_xlabel("neurite thr"); _oax.set_ylabel("soma thr")
                        _ofig.colorbar(_im, ax=_oax, label="soma-fraction error")
                        _ofig.tight_layout()
                        st.pyplot(_ofig, use_container_width=False)
                        plt.close(_ofig)
                        if st.button(":material/check_circle: Apply optimized thresholds",
                                     type="primary", key="apply_opt_thr"):
                            st.session_state["_apply_thr"] = (round(float(_bs), 2),
                                                              round(float(_bn), 2))
                            st.rerun()

                # Side-by-side table with per-sample deltas.
                _show_cols = [c for c in (
                    "stem", "hand_cilia", "pipe_cilia", "hand_nuclei",
                    "pipe_est_nuclei", "hand_on_soma", "pipe_on_soma",
                    "hand_on_neurite", "pipe_on_neurite") if c in _m.columns]
                _tbl = _m[_show_cols].copy()
                if {"hand_cilia", "pipe_cilia"}.issubset(_tbl.columns):
                    _tbl["Δ cilia"] = _tbl["pipe_cilia"] - _tbl["hand_cilia"]
                st.markdown("**Per-sample comparison**")
                st.dataframe(_tbl.round(1), use_container_width=True, hide_index=True)
                st.download_button(
                    ":material/download: Download comparison (CSV)",
                    _tbl.to_csv(index=False).encode(),
                    "pipeline_vs_hand.csv", "text/csv", key="dl_validation")


# ─────────────────────────────────────────────────────────────────────────────
# TAB — BOX GROUND TRUTH (per-cilia comparison vs hand-drawn bounding boxes)
# ─────────────────────────────────────────────────────────────────────────────
# Boxes come from ``napari_annotator_app.py`` (sheet ``cilia_boxes``): one row
# per hand-drawn box with pixel y/x bounds + a manual class. A detected cilium
# (its centroid projected onto the XY MIP) is scored against these boxes so we
# can measure which cilia the pipeline misses and which it misclassifies.
_ANNOT_SHEET = "cilia_boxes"
_ANNOT_CLASSES = ["neurite", "soma", "uncertain"]

# The pipeline's "ambiguous" class is the same thing as the annotator's
# "uncertain" — treat them as equal when scoring agreement.
_CLS_ALIAS = {"ambiguous": "uncertain"}


def _cls_norm(c) -> str:
    c = str(c).strip().lower()
    return _CLS_ALIAS.get(c, c)


def _cls_eq(a, b) -> bool:
    return _cls_norm(a) == _cls_norm(b)


@st.cache_data(show_spinner=False)
def parse_annotation_boxes(src) -> pd.DataFrame:
    """Read the annotator Excel into tidy per-box rows (adds a ``stem`` column)."""
    try:
        df = pd.read_excel(src, sheet_name=_ANNOT_SHEET)
    except Exception:                                         # noqa: BLE001
        df = pd.read_excel(src, sheet_name=0)                 # sheet-name fallback
    if df.empty or "filename" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["stem"] = df["filename"].apply(_img_stem)
    if "class" in df.columns:
        df["class"] = df["class"].astype(str).str.strip().str.lower()
    return df


def _box_scale(run_dir: str, stem: str, annot_h, annot_w):
    """(scale_y, scale_x) mapping annotation pixels → the detected cilia MIP grid.

    The annotator and pipeline may resample XY differently; rescale by the saved
    MIP shape when available. Returns (1, 1) if the shape can't be recovered."""
    mip = _load_sample_mip(run_dir, stem, "cilia")
    if mip is None or not annot_h or not annot_w:
        return 1.0, 1.0, (mip.shape if mip is not None else None)
    try:
        return (mip.shape[0] / float(annot_h),
                mip.shape[1] / float(annot_w), mip.shape)
    except Exception:                                         # noqa: BLE001
        return 1.0, 1.0, None


def match_detections_to_boxes(boxes: pd.DataFrame, yx, pred_cls, ids,
                              scale=(1.0, 1.0)):
    """Score detected cilia against hand-drawn boxes for one sample.

    Returns (box_records, det_records):
      • box_records — one per GT box: matched?, predicted class of the cilium
        inside (nearest to box centre if several), so a missed box (no detection
        inside) is a false negative for its class.
      • det_records — one per detected cilium: gt_class of the box it falls in
        (None → detection outside every box, i.e. a possible false positive).
    """
    sy, sx = scale
    b = boxes.copy()
    for c, s in (("y_min", sy), ("y_max", sy), ("x_min", sx), ("x_max", sx)):
        if c in b.columns:
            b[c] = pd.to_numeric(b[c], errors="coerce") * s
    box_rows = b.to_dict("records")

    det_records = []
    contained = {i: [] for i in range(len(box_rows))}    # box idx → [det idx]
    for di, ((y, x), pc, cid) in enumerate(zip(yx, pred_cls, ids)):
        hit = None
        for bi, br in enumerate(box_rows):
            if (br["y_min"] <= y <= br["y_max"]) and (br["x_min"] <= x <= br["x_max"]):
                hit = bi
                contained[bi].append(di)
                break
        det_records.append(dict(
            cilia_id=cid, y=y, x=x, pred_class=str(pc),
            gt_class=(str(box_rows[hit].get("class", "uncertain")) if hit is not None else None),
            in_box=hit is not None))

    box_records = []
    for bi, br in enumerate(box_rows):
        dets = contained[bi]
        pred = None
        if dets:
            # Nearest detection to the box centre decides the predicted class.
            cy = (br["y_min"] + br["y_max"]) / 2.0
            cx = (br["x_min"] + br["x_max"]) / 2.0
            nearest = min(dets, key=lambda di: (yx[di][0] - cy) ** 2
                          + (yx[di][1] - cx) ** 2)
            pred = str(pred_cls[nearest])
        box_records.append(dict(
            box_id=br.get("box_id", bi + 1),
            gt_class=str(br.get("class", "uncertain")),
            matched=bool(dets), n_inside=len(dets), pred_class=pred))
    return box_records, det_records


with tab_boxes:
    st.subheader(":material/select_all: Per-cilia comparison vs hand-drawn boxes")
    st.caption("Load the Excel written by the napari **annotator** app. Each "
               "detected cilium (centroid on the XY MIP) is matched to your "
               "boxes — so you can see which cilia the pipeline **misses** and "
               "which it **misclassifies**. Matched by filename stem.")

    _bup = st.file_uploader("Annotation Excel (.xlsx)", type=["xlsx"],
                            key="box_upload")
    _box_path = st.text_input(
        "…or path to annotation Excel",
        value=st.session_state.get("_box_path", ""), key="box_path_in")
    st.session_state["_box_path"] = _box_path
    _bsrc = _bup if _bup is not None else (_box_path or None)

    if not _bsrc:
        st.info("Upload or point to a `cilia_annotations.xlsx` to begin.")
    elif "coords" not in _cilia_full.columns:
        st.warning("Detected cilia have no `coords` column — cannot match boxes.")
    else:
        _boxes_all = parse_annotation_boxes(_bsrc)
        if _boxes_all.empty:
            st.error("No boxes found in that file (expected sheet "
                     f"`{_ANNOT_SHEET}` with a `filename` column).")
        else:
            _run_labels = [r["label"] for r in _runs]
            _sel_label = st.selectbox("Run", _run_labels, key="box_run")
            _sel_dir = _run_dir_by_label[_sel_label]

            # Compare only cilia the run actually keeps: AI-validated or manually
            # confirmed (human_validated == True). Rejected detections are not
            # the pipeline's output, so scoring them against GT is misleading.
            _only_conf = st.checkbox(
                "Only compare AI/manually confirmed cilia", value=True,
                key="box_only_conf",
                help="Keep detections with human_validated = True (AI-validated or "
                     "manually kept in the Screening tab). Rejected cilia are dropped.")

            # Only samples that appear both in the boxes and in this run.
            _det_run = _cilia_full[_cilia_full["run"] == _sel_label].copy()
            if _only_conf and "human_validated" in _det_run.columns:
                _n_all = len(_det_run)
                _det_run = _det_run[_det_run["human_validated"].astype(bool)]
                st.caption(f"Kept {len(_det_run)} / {_n_all} confirmed detections "
                           "in this run.")
            _det_run["stem"] = _det_run["filename"].apply(lambda f: Path(str(f)).stem)
            _common = sorted(set(_boxes_all["stem"]) & set(_det_run["stem"]))
            _only_box = sorted(set(_boxes_all["stem"]) - set(_det_run["stem"]))
            if _only_box:
                st.caption(f"{len(_only_box)} annotated sample(s) have no detections "
                           f"in this run (skipped): {', '.join(_only_box[:6])}"
                           + (" …" if len(_only_box) > 6 else ""))
            if not _common:
                st.warning("No filename-stem overlap between the boxes and this "
                           "run's detected cilia.")
            else:
                _all_box_rec, _all_det_rec = [], []
                _shape_warned = False
                for _stem in _common:
                    _bx = _boxes_all[_boxes_all["stem"] == _stem]
                    _dt = _det_run[_det_run["stem"] == _stem]
                    _yx = _parse_coords_yx(_dt["coords"])
                    _pred = list(_dt["class"]) if "class" in _dt.columns else \
                        ["ambiguous"] * len(_dt)
                    _ids = list(_dt["cilia_id"]) if "cilia_id" in _dt.columns else \
                        list(range(len(_dt)))
                    _ah = _bx.iloc[0].get("img_height")
                    _aw = _bx.iloc[0].get("img_width")
                    _sy, _sx, _mshape = _box_scale(_sel_dir, _stem, _ah, _aw)
                    if _mshape is None and not _shape_warned:
                        st.caption(":material/warning: No saved cilia MIP found — "
                                   "assuming boxes and detections share the same "
                                   "pixel grid (no rescaling).")
                        _shape_warned = True
                    _brec, _drec = match_detections_to_boxes(
                        _bx, _yx, _pred, _ids, scale=(_sy, _sx))
                    for _r in _brec:
                        _r["stem"] = _stem
                    for _r in _drec:
                        _r["stem"] = _stem
                    _all_box_rec += _brec
                    _all_det_rec += _drec

                _bdf = pd.DataFrame(_all_box_rec)
                _ddf = pd.DataFrame(_all_det_rec)

                # ── Headline detection metrics ─────────────────────────────────
                _n_gt = len(_bdf)
                _n_matched = int(_bdf["matched"].sum()) if _n_gt else 0
                _n_missed = _n_gt - _n_matched
                _n_det = len(_ddf)
                _n_fp = int((~_ddf["in_box"]).sum()) if _n_det else 0
                _recall = _n_matched / _n_gt if _n_gt else float("nan")
                _prec = (_n_det - _n_fp) / _n_det if _n_det else float("nan")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("GT boxes", _n_gt)
                m2.metric("Missed (FN)", _n_missed,
                          help="Hand-drawn boxes with no detected cilium inside.")
                m3.metric("Recall", "—" if np.isnan(_recall) else f"{_recall:.0%}")
                m4.metric("False positives", _n_fp,
                          help="Detected cilia falling outside every box.")

                # ── Misses by GT class (what type do we miss?) ─────────────────
                st.markdown("#### What do we miss?")
                _miss = _bdf.groupby("gt_class")["matched"].agg(
                    n="count", found="sum")
                _miss["missed"] = _miss["n"] - _miss["found"]
                _miss["recall"] = (_miss["found"] / _miss["n"]).map(
                    lambda v: f"{v:.0%}")
                st.dataframe(_miss.reset_index()[["gt_class", "n", "found",
                             "missed", "recall"]], use_container_width=True,
                             hide_index=True)

                # ── Confusion: GT class × predicted class (matched boxes only) ─
                st.markdown("#### Misclassification (matched cilia)")
                _mm = _bdf[_bdf["matched"]]
                if _mm.empty:
                    st.caption("No matched boxes to build a confusion matrix.")
                else:
                    _conf = pd.crosstab(_mm["gt_class"], _mm["pred_class"])
                    st.dataframe(_conf, use_container_width=True)
                    _correct = int(sum(_cls_eq(g, p) for g, p in
                                       zip(_mm["gt_class"], _mm["pred_class"])))
                    _acc = _correct / len(_mm)
                    st.caption(f"Class accuracy on matched cilia: **{_acc:.0%}** "
                               f"({_correct}/{len(_mm)}). Rows = your label, "
                               "columns = pipeline class. Pipeline *ambiguous* is "
                               "counted as manual *uncertain*.")

                # ── Per-sample breakdown + downloads ───────────────────────────
                with st.expander(":material/table: Per-sample breakdown", expanded=False):
                    _ps = _bdf.groupby("stem")["matched"].agg(
                        boxes="count", found="sum").reset_index()
                    _ps["missed"] = _ps["boxes"] - _ps["found"]
                    _fp_ps = _ddf.groupby("stem")["in_box"].agg(
                        detections="count",
                        fp=lambda s: int((~s).sum())).reset_index()
                    _ps = _ps.merge(_fp_ps, on="stem", how="outer")
                    st.dataframe(_ps, use_container_width=True, hide_index=True)

                c1, c2 = st.columns(2)
                c1.download_button(
                    ":material/download: Per-box results (CSV)",
                    _bdf.to_csv(index=False).encode(),
                    "box_gt_per_box.csv", "text/csv", key="dl_box_gt")
                c2.download_button(
                    ":material/download: Per-detection results (CSV)",
                    _ddf.to_csv(index=False).encode(),
                    "box_gt_per_detection.csv", "text/csv", key="dl_det_gt")

                # ── Per-cilia plots (one point = one detected cilium) ──────────
                st.markdown("#### Per-cilia plots")
                # Attach each detection's pipeline metrics to its match outcome.
                _mcols = [c for c in _det_run.select_dtypes(include=[np.number]).columns
                          if c != "cilia_id"]
                _join = ["stem", "cilia_id"] + _mcols + (
                    ["filename"] if "filename" in _det_run.columns else [])
                _pcm = _ddf.merge(_det_run[_join], on=["stem", "cilia_id"], how="left")
                _pcm["gt_class"] = _pcm["gt_class"].fillna("(no box)")
                _pcm["match"] = [
                    ("outside box" if not ib else
                     ("correct" if _cls_eq(g, p) else "misclassified"))
                    for ib, g, p in zip(_pcm["in_box"], _pcm["gt_class"],
                                        _pcm["pred_class"])]
                if _pcm.empty or not _mcols:
                    st.caption("No per-cilia metrics available to plot.")
                else:
                    pc1, pc2, pc3 = st.columns(3)
                    _sub = pc1.selectbox(
                        "Cilia", ["matched (in box)", "all detected", "outside boxes"],
                        key="pcp_sub")
                    _colby = pc2.selectbox("Colour by",
                                           ["gt_class", "pred_class", "match"],
                                           key="pcp_col")
                    _kind = pc3.selectbox("Plot",
                                          ["Strip by class", "Scatter (2 metrics)"],
                                          key="pcp_kind")
                    _d = _pcm.copy()
                    if _sub.startswith("matched"):
                        _d = _d[_d["in_box"]]
                    elif _sub.startswith("outside"):
                        _d = _d[~_d["in_box"]]

                    def _mi(name, default=0):
                        return _mcols.index(name) if name in _mcols else default

                    if _d.empty:
                        st.caption("No cilia in this subset.")
                    elif _kind.startswith("Strip"):
                        _y = st.selectbox("Metric (y)", _mcols,
                                          index=_mi("log_ratio"), key="pcp_y1")
                        _xcat = st.selectbox("Group on x",
                                             ["gt_class", "pred_class", "match"],
                                             key="pcp_xcat")
                        fig, ax = plt.subplots(figsize=(5.5, 3.6))
                        sns.stripplot(data=_d, x=_xcat, y=_y, hue=_colby, ax=ax,
                                      size=4, alpha=0.75, jitter=0.25,
                                      dodge=_colby != _xcat, legend=_colby != _xcat)
                        _draw_mean_bars(ax, _d, _xcat, _y)
                        ax.set_title(f"{_y} per {_xcat}  (n={len(_d)} cilia)", fontsize=9)
                        _tidy_legend(ax)
                        _show_and_export(fig, f"per_cilia_{_y}_by_{_xcat}", "pcp_strip")
                    else:
                        sc1, sc2 = st.columns(2)
                        _x = sc1.selectbox("X", _mcols, index=_mi("dt_neurite_um"),
                                           key="pcp_x2")
                        _y = sc2.selectbox("Y", _mcols, index=_mi("log_ratio"),
                                           key="pcp_y2")
                        fig, ax = plt.subplots(figsize=(5.5, 4.2))
                        sns.scatterplot(data=_d, x=_x, y=_y, hue=_colby, ax=ax,
                                        s=32, alpha=0.8, edgecolor="white",
                                        linewidth=0.3)
                        _xv = pd.to_numeric(_d[_x], errors="coerce").to_numpy(float)
                        _yv = pd.to_numeric(_d[_y], errors="coerce").to_numpy(float)
                        _ok = np.isfinite(_xv) & np.isfinite(_yv)
                        if _ok.sum() > 1 and _xv[_ok].std() and _yv[_ok].std():
                            _rr = np.corrcoef(_xv[_ok], _yv[_ok])[0, 1]
                            ax.text(0.05, 0.95, f"$R^2$ = {_rr**2:.2f}",
                                    transform=ax.transAxes, va="top", ha="left",
                                    fontsize=9, fontweight="bold")
                        ax.set_title(f"{_y} vs {_x}  (n={len(_d)} cilia)", fontsize=9)
                        _tidy_legend(ax)
                        _show_and_export(fig, f"per_cilia_{_y}_vs_{_x}", "pcp_scatter")

                # ── Interactive: manual class × pipeline class (click → ROI) ───
                st.markdown("#### Interactive: manual vs pipeline class")
                if not _PLOTLY:
                    st.info("Install plotly for the interactive view: `pip install plotly`")
                elif _pcm.empty:
                    st.caption("No detections to plot.")
                else:
                    st.caption("Each point is one cilium, placed at (manual box class, "
                               "pipeline class) with jitter. Off-diagonal = "
                               "misclassified. **Click a point** to see its ROI.")
                    _gt_order = ["neurite", "soma", "uncertain", "(no box)"]
                    _pd_order = ["neurite", "soma", "ambiguous"]
                    _gt_cats = _gt_order + sorted(
                        set(_pcm["gt_class"].astype(str)) - set(_gt_order))
                    _pd_cats = _pd_order + sorted(
                        set(_pcm["pred_class"].astype(str)) - set(_pd_order))
                    _gi = {c: i for i, c in enumerate(_gt_cats)}
                    _pi = {c: i for i, c in enumerate(_pd_cats)}
                    _pf = _pcm.copy()
                    _pf["_run"] = _sel_label
                    if "filename" not in _pf.columns:
                        _pf["filename"] = _pf["stem"]
                    _rng = np.random.default_rng(0)
                    _pf["manual_class"] = _pf["gt_class"].astype(str)
                    _pf["pipeline_class"] = _pf["pred_class"].astype(str)
                    _pf["_gx"] = [_gi[c] + _rng.uniform(-0.18, 0.18)
                                  for c in _pf["manual_class"]]
                    _pf["_py"] = [_pi[c] + _rng.uniform(-0.18, 0.18)
                                  for c in _pf["pipeline_class"]]

                    _color_opts = ["match", "stem", "manual_class", "pipeline_class"]
                    _icolor = st.selectbox("Colour by", _color_opts, key="box_i_color")
                    _match_cmap = {"correct": "#32CD32", "misclassified": "#e74c3c",
                                   "outside box": "#95a5a6"}
                    _cmap = (_CLASS_PALETTE if _icolor in
                             ("manual_class", "pipeline_class")
                             else _match_cmap if _icolor == "match" else None)
                    _cd = ["_run", "filename", "cilia_id"]
                    _hover = {c: True for c in
                              ("stem", "match", "log_ratio", "cilia_id",
                               "dt_nuclei_um", "dt_neurite_um")
                              if c in _pf.columns}
                    _hover["_gx"] = False
                    _hover["_py"] = False
                    fig = px.scatter(
                        _pf, x="_gx", y="_py", color=_icolor,
                        color_discrete_map=_cmap, custom_data=_cd, hover_data=_hover,
                        opacity=0.8, template="simple_white")
                    fig.update_traces(marker=dict(size=9, line=dict(width=0.4,
                                                                    color="white")))
                    fig.update_xaxes(tickvals=list(range(len(_gt_cats))),
                                     ticktext=_gt_cats, title="manual (box) class",
                                     range=[-0.5, len(_gt_cats) - 0.5])
                    fig.update_yaxes(tickvals=list(range(len(_pd_cats))),
                                     ticktext=_pd_cats, title="pipeline class",
                                     range=[-0.5, len(_pd_cats) - 0.5])
                    fig.update_layout(height=520, legend_title_text=_icolor,
                                      margin=dict(l=10, r=10, t=30, b=10))

                    _pcol, _rcol = st.columns([3, 1])
                    with _pcol:
                        try:
                            _ev = st.plotly_chart(
                                fig, use_container_width=True, on_select="rerun",
                                selection_mode="points", key="box_iplot")
                        except TypeError:
                            st.plotly_chart(fig, use_container_width=True)
                            _ev = None
                    _sel = []
                    if _ev is not None:
                        try:
                            _sel = _ev.selection["points"]
                        except Exception:                         # noqa: BLE001
                            _sel = []
                    with _rcol:
                        if _sel:
                            info = dict(zip(_cd, _sel[0].get("customdata") or []))
                            st.markdown(f"**cilia {info.get('cilia_id')}**  ·  "
                                        f"{info.get('filename', '')}")
                            _npz = _roi_npz_path(_sel_label, info.get("filename"),
                                                 info.get("cilia_id"))
                            _multi = _roi_multichannel_png(_npz)
                            if _multi is not None:
                                st.image(_multi, use_container_width=True)
                                st.caption(":green[:material/circle:] cilia · "
                                           ":violet[:material/circle:] basal body · "
                                           ":blue[:material/circle:] nuclei · cyan neurite")
                            else:
                                _p = _roi_png_path(_sel_label, info.get("filename"),
                                                   info.get("cilia_id"))
                                st.image(_p, use_container_width=True) if _p else \
                                    st.caption("No ROI saved for this cilium.")
                        else:
                            st.caption("Click a point to preview its ROI.")

                # ── Overlay boxes + detections on the sample MIP ───────────────
                st.markdown("#### Visual check")
                _vstem = st.selectbox("Sample", _common, key="box_vstem")
                _cil_mip = _load_sample_mip(_sel_dir, _vstem, "cilia")
                if _cil_mip is None:
                    st.caption("No saved cilia MIP for this sample to draw on.")
                else:
                    _bx = _boxes_all[_boxes_all["stem"] == _vstem]
                    _ah = _bx.iloc[0].get("img_height")
                    _aw = _bx.iloc[0].get("img_width")
                    _sy, _sx, _ = _box_scale(_sel_dir, _vstem, _ah, _aw)
                    _dt = _det_run[_det_run["stem"] == _vstem]
                    _yx = _parse_coords_yx(_dt["coords"])
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(_cil_mip, cmap="gray",
                              vmax=np.percentile(_cil_mip, 99.5) or 1)
                    for _, _r in _bx.iterrows():
                        y0 = _r["y_min"] * _sy; x0 = _r["x_min"] * _sx
                        h = (_r["y_max"] - _r["y_min"]) * _sy
                        w = (_r["x_max"] - _r["x_min"]) * _sx
                        _col = _CLASS_PALETTE.get(str(_r.get("class")), "#f39c12")
                        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False,
                                     edgecolor=_col, linewidth=1.6))
                    if _yx:
                        _ys = [p[0] for p in _yx]; _xs = [p[1] for p in _yx]
                        ax.scatter(_xs, _ys, s=18, facecolor="none",
                                   edgecolor="cyan", linewidth=1.0)
                    ax.set_axis_off()
                    ax.set_title(f"{_vstem} — boxes (class colour) + detections (cyan)",
                                 fontsize=9)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)


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
              ncols: int = 3, subtitle_colors: list | None = None):
    if not image_paths:
        return
    nrows = int(np.ceil(len(image_paths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.4 * nrows + 0.5))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(title, fontsize=11, fontweight="bold")
    colors = subtitle_colors or ["#222"] * len(image_paths)
    for ax, p, sub, col in zip(axes, image_paths, subtitles, colors):
        try:
            ax.imshow(plt.imread(p))
        except Exception:                                 # noqa: BLE001
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.set_title(sub, fontsize=8, color=col, fontweight="bold")
        ax.axis("off")
    for ax in axes[len(image_paths):]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig)
    plt.close(fig)


_CLASS_ORDER = {"neurite": 0, "soma": 1, "ambiguous": 2}


def _roi_class_lookup(rdf: pd.DataFrame) -> dict[tuple[str, int], str]:
    """Map ``{(sample_stem, cilia_id): class}`` for a run's cilia rows."""
    lut: dict[tuple[str, int], str] = {}
    if not {"filename", "cilia_id", "class"}.issubset(rdf.columns):
        return lut
    sub = rdf[rdf["object_type"] == "cilia"] if "object_type" in rdf.columns else rdf
    for _, row in sub.iterrows():
        try:
            lut[(Path(str(row["filename"])).stem, int(row["cilia_id"]))] = str(row["class"])
        except Exception:                                 # noqa: BLE001
            continue
    return lut


def _parse_roi_name(fname: str):
    """('<stem>', <cilia_id or None>) from '<stem>_cilia<id>.png'."""
    base = os.path.splitext(fname)[0]
    stem, _, cid = base.partition("_cilia")
    try:
        return stem, int(cid)
    except ValueError:
        return stem, None


def build_report_pdf(runs, df, *, include_qc, include_overviews, include_rois,
                     roi_cap, metric) -> bytes:
    buf = BytesIO()
    cilia = df[df["object_type"] == "cilia"] if "object_type" in df.columns else df
    with PdfPages(buf) as pdf:
        # ── Title page ──────────────────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.92, "Limoncello — Data Report", ha="center",
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
                # Scan the ROI folder directly (robust to stem-naming differences),
                # group by sample, and label/sort each ROI by its classifier class.
                roi_dir = os.path.join(r["dir"], "figures", "cilia_rois")
                if os.path.isdir(roi_dir):
                    cls_lut = _roi_class_lookup(rdf)
                    by_stem: dict[str, list[str]] = {}
                    for f in sorted(os.listdir(roi_dir)):
                        if "_cilia" in f and f.lower().endswith(".png"):
                            by_stem.setdefault(_parse_roi_name(f)[0], []).append(
                                os.path.join(roi_dir, f))
                    for s, paths in by_stem.items():
                        # sort by class (neurite, soma, ambiguous, then unknown)
                        def _key(p):
                            _stem, cid = _parse_roi_name(os.path.basename(p))
                            cls = cls_lut.get((_stem, cid), "?")
                            return (_CLASS_ORDER.get(cls, 9), cid if cid is not None else 0)
                        paths = sorted(paths, key=_key)[:roi_cap]
                        subs, cols = [], []
                        for p in paths:
                            _stem, cid = _parse_roi_name(os.path.basename(p))
                            cls = cls_lut.get((_stem, cid), "?")
                            subs.append(f"cilia {cid} · {cls}")
                            cols.append(_CLASS_PALETTE.get(cls, "#222"))
                        _img_page(pdf, f"{r['label']} — {s} — ROIs (by class)",
                                  paths, subs, ncols=3, subtitle_colors=cols)
    return buf.getvalue()


with tab_report:
    st.subheader(":material/description: PDF report")
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

    if st.button(":material/print: Build PDF report", type="primary", key="r_build"):
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
        st.download_button(":material/download: Download report (PDF)",
                           st.session_state["_report_pdf"],
                           "limoncello_report.pdf", "application/pdf", key="dl_report")
