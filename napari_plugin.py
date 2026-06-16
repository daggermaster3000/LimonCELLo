"""
LimonCELLo napari plugin — interactive single-image parameter testing.
Run with:  python napari_plugin.py
"""


from pathlib import Path

import numpy as np
import napari
from magicgui import magicgui
from napari.qt.threading import thread_worker
from scipy.ndimage import distance_transform_edt, center_of_mass

from limoncello.utils.reader import load_image
from limoncello.preprocessing.preprocessing import percentile_minmax_normalize
from limoncello.segmentation.cilia import segment_cilia_ml
from limoncello.segmentation.nuclei import segment_nuclei
from limoncello.segmentation.neurites import segment_neurites
from limoncello.segmentation.basal_bodies import segment_basal_bodies
from limoncello.utils.assign_label_features import assign_label_features


# ── Core pipeline (no file I/O) ────────────────────────────────────────────────

def run_single_image(params: dict) -> dict:
    """Segment a single .ims file and return all intermediate arrays."""
    p = params

    print(f"[LC] Loading {p['ims_path']} …")
    img, meta = load_image(p["ims_path"])
    voxel_size = meta["voxel_size"] or (1.0, 1.0, 1.0)
    n_ch = img.shape[1]

    # Raw channel stack (original dtype) and normalised stack (float32 [0,1])
    print("[LC] Building channel arrays …")
    raw = np.stack([np.asarray(img[0, c]) for c in range(n_ch)])   # (C, Z, Y, X)
    norm = np.stack([
        percentile_minmax_normalize(raw[c], p_low=p["p_low"], p_high=p["p_high"])
        for c in range(n_ch)
    ])

    if p["use_mip"]:
        print("[LC] Applying MIP projection …")
        raw  = np.max(raw,  axis=1, keepdims=True)   # (C, 1, Y, X)
        norm = np.max(norm, axis=1, keepdims=True)

    # ── Segmentation ───────────────────────────────────────────────────────────
    print("[LC] Segmenting cilia …")
    cilia_labels = np.asarray(segment_cilia_ml(
        raw[p["ch_cilia"]],
        classifier_path=p["classifier_path"],
        gaussian_sigma=(p["cilia_gauss_z"], p["cilia_gauss_y"], p["cilia_gauss_x"]),
        log_transform=p["cilia_log"],
        min_size=p["cilia_min_size"],
        max_size=p["cilia_max_size"],
    )).astype(np.int32)

    print("[LC] Segmenting nuclei …")
    tr = p["tophat_radius"]
    nuclei_labels = segment_nuclei(
        norm[p["ch_nuclei"]],
        tophat_radius=(tr, tr, tr),
        spot_sigma=p["nuclei_sigma"],
        outline_sigma=p["nuclei_outline_sigma"],
        gaussian_sigma=(p["nuclei_gauss_z"], p["nuclei_gauss_y"], p["nuclei_gauss_x"]),
        log_transform=p["nuclei_log"],
    ).astype(np.int32)

    print("[LC] Segmenting neurites …")
    skeleton, neurites_gpu = segment_neurites(
        norm[p["ch_neurites"]],
        spot_sigma=p["neurite_sigma"],
        gaussian_sigma=(p["neurite_gauss_z"], p["neurite_gauss_y"], p["neurite_gauss_x"]),
        log_transform=p["neurite_log"],
    )
    neurite_labels = np.asarray(neurites_gpu).astype(np.int32)
    skeleton_mask  = np.asarray(skeleton) > 0

    print("[LC] Segmenting basal bodies …")
    bb_labels = np.asarray(segment_basal_bodies(
        raw[p["ch_bb"]],
        spot_sigma=p["bb_spot_sigma"],
        outline_sigma=p["bb_outline_sigma"],
        gaussian_sigma=(p["bb_gauss_z"], p["bb_gauss_y"], p["bb_gauss_x"]),
        log_transform=p["bb_log"],
        min_size=p["bb_min_size"],
        max_size=p["bb_max_size"],
    )).astype(np.int32)

    # ── Distance maps ──────────────────────────────────────────────────────────
    print("[LC] Computing distance maps …")
    neurite_mask = neurite_labels > 0
    eps          = p["ratio_epsilon"]

    dt_nuclei  = distance_transform_edt(~(nuclei_labels > 0), sampling=voxel_size)
    dt_neurite = distance_transform_edt(neurite_mask,          sampling=voxel_size)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_map = np.where(
            neurite_mask,
            (dt_nuclei + eps) / (dt_neurite + eps),
            np.nan,
        )
    ratio_map[~np.isfinite(ratio_map)] = np.nan
    log_ratio_map = np.where(np.isfinite(ratio_map), np.log(ratio_map), np.nan)

    dist_to_neurite = distance_transform_edt(~neurite_mask, sampling=voxel_size)
    _, nearest_skel_idx = distance_transform_edt(
        ~skeleton_mask, return_indices=True, sampling=voxel_size
    )

    # ── Feature assignment ─────────────────────────────────────────────────────
    print("[LC] Assigning features …")
    fname = Path(p["ims_path"]).name

    def _features(labels, dist_cutoff, obj_type):
        import pandas as pd
        ids = np.unique(labels)
        ids = ids[ids != 0]
        if ids.size == 0:
            return pd.DataFrame()
        centroids = center_of_mass(labels > 0, labels=labels, index=ids)
        return assign_label_features(
            labels, centroids, ids,
            dist_to_neurite, nearest_skel_idx,
            dt_neurite, dt_nuclei, ratio_map,
            dist_cutoff, fname,
            object_type=obj_type, ratio_epsilon=eps,
        )

    cilia_df = _features(cilia_labels, p["max_cilia_dist_um"], "cilia")
    bb_df    = _features(bb_labels,    p["max_basal_dist_um"], "basal_body")

    for df in (cilia_df, bb_df):
        if not df.empty:
            df["class"] = "ambiguous"
            df.loc[df["log_ratio"] > p["axon_threshold"], "class"] = "axon"
            df.loc[df["log_ratio"] < p["soma_threshold"], "class"] = "soma"

    n_cilia = len(cilia_df)
    n_bb    = len(bb_df)
    print(f"[LC] Done — {n_cilia} cilia, {n_bb} basal bodies assigned.")

    return dict(
        raw=raw, norm=norm,
        cilia_labels=cilia_labels, nuclei_labels=nuclei_labels,
        neurite_labels=neurite_labels, skeleton_mask=skeleton_mask,
        bb_labels=bb_labels,
        dt_neurite=dt_neurite, dt_nuclei=dt_nuclei,
        log_ratio_map=log_ratio_map,
        cilia_df=cilia_df, bb_df=bb_df,
        voxel_size=voxel_size,
        ch_cilia=p["ch_cilia"], ch_neurites=p["ch_neurites"],
        ch_bb=p["ch_bb"], ch_nuclei=p["ch_nuclei"],
    )


# ── Layer management ───────────────────────────────────────────────────────────

_CH_COLORMAPS = ["green", "cyan", "magenta", "blue"]
_CH_NAMES     = ["Cilia", "Neurites", "Basal Bodies", "Nuclei"]


def _clear_lc_layers(viewer: napari.Viewer) -> None:
    to_remove = [la for la in viewer.layers if la.name.startswith("LC: ")]
    for la in to_remove:
        viewer.layers.remove(la)


def _update_layers(viewer: napari.Viewer, r: dict) -> None:
    _clear_lc_layers(viewer)
    vs = r["voxel_size"]

    ch_keys = [r["ch_cilia"], r["ch_neurites"], r["ch_bb"], r["ch_nuclei"]]

    # Raw channels
    for name, ch, cmap in zip(_CH_NAMES, ch_keys, _CH_COLORMAPS):
        viewer.add_image(
            r["raw"][ch], name=f"LC: Raw {name}",
            scale=vs, colormap=cmap,
            blending="additive",
        )

    # Log-ratio map (NaN → clamp to global min so layer renders cleanly)
    lr = r["log_ratio_map"]
    finite_mask = np.isfinite(lr)
    if finite_mask.any():
        fill = float(np.nanmin(lr[finite_mask]))
        viewer.add_image(
            np.where(finite_mask, lr, fill),
            name="LC: Log Ratio",
            scale=vs, colormap="bwr",
            opacity=0.65, blending="translucent",
        )

    # Segmentation label maps
    viewer.add_labels(r["cilia_labels"],   name="LC: Cilia Labels",      scale=vs)
    viewer.add_labels(r["nuclei_labels"],  name="LC: Nuclei Labels",     scale=vs)
    viewer.add_labels(r["neurite_labels"], name="LC: Neurite Labels",    scale=vs)
    viewer.add_labels(r["bb_labels"],      name="LC: Basal Body Labels", scale=vs)

    # Cilia centroids coloured by log_ratio
    cdf = r["cilia_df"]
    if not cdf.empty:
        coords   = np.array(cdf["coords"].tolist())
        lr_vals  = cdf["log_ratio"].values.astype(float)
        vmin, vmax = (
            (float(np.nanpercentile(lr_vals, 5)), float(np.nanpercentile(lr_vals, 95)))
            if len(lr_vals) > 1 else (float(lr_vals[0]) - 1, float(lr_vals[0]) + 1)
        )
        pts = viewer.add_points(
            coords, name="LC: Cilia Centroids",
            scale=vs, size=5,
            properties={"log_ratio": lr_vals},
            face_color="log_ratio", face_colormap="bwr",
        )
        pts.face_contrast_limits = (vmin, vmax)

    # Basal body centroids
    bdf = r["bb_df"]
    if not bdf.empty:
        bb_coords = np.array(bdf["coords"].tolist())
        viewer.add_points(
            bb_coords, name="LC: BB Centroids",
            scale=vs, size=4,
            face_color="yellow",
        )

    print("[LC] Viewer updated.")


# ── Widget ─────────────────────────────────────────────────────────────────────

_DEFAULT_CLASSIFIER = str(
    (Path(__file__).parent / "segmenters" / "cilia-segmenter.cl").resolve()
)


def build_widget(viewer: napari.Viewer):

    @magicgui(
        call_button="Run Segmentation",
        ims_path={
            "label": "Image (.ims)",
            "widget_type": "FileEdit",
            "filter": "Imaris files (*.ims)",
        },
        classifier_path={
            "label": "Classifier (.cl)",
            "widget_type": "FileEdit",
            "filter": "APOC classifier (*.cl)",
        },
        ch_cilia={"label": "Ch: Cilia",        "min": 0, "max": 9},
        ch_neurites={"label": "Ch: Neurites",   "min": 0, "max": 9},
        ch_bb={"label": "Ch: Basal Bodies",     "min": 0, "max": 9},
        ch_nuclei={"label": "Ch: Nuclei",       "min": 0, "max": 9},
        p_low={"label":  "Norm p_low  (%)",     "min": 0,  "max": 49},
        p_high={"label": "Norm p_high (%)",     "min": 51, "max": 100},
        nuclei_sigma={"label": "Nuclei spot σ",       "min": 1, "max": 50},
        tophat_radius={"label": "Nuclei tophat r",    "min": 1, "max": 50},
        nuclei_outline_sigma={"label": "Nuclei outline σ", "min": 0, "max": 10},
        nuclei_gauss_z={"label": "Nuclei Gauss Z (0=off)", "min": 0.0, "max": 5.0, "step": 0.5},
        nuclei_gauss_y={"label": "Nuclei Gauss Y", "min": 0.0, "max": 5.0, "step": 0.5},
        nuclei_gauss_x={"label": "Nuclei Gauss X", "min": 0.0, "max": 5.0, "step": 0.5},
        neurite_sigma={"label": "Neurite spot σ",     "min": 1, "max": 20},
        neurite_gauss_z={"label": "Neurite Gauss Z (0=off)", "min": 0.0, "max": 5.0, "step": 0.5},
        neurite_gauss_y={"label": "Neurite Gauss Y", "min": 0.0, "max": 5.0, "step": 0.5},
        neurite_gauss_x={"label": "Neurite Gauss X", "min": 0.0, "max": 5.0, "step": 0.5},
        cilia_gauss_z={"label": "Cilia Gauss Z", "min": 0.0, "max": 5.0, "step": 0.5},
        cilia_gauss_y={"label": "Cilia Gauss Y", "min": 0.0, "max": 5.0, "step": 0.5},
        cilia_gauss_x={"label": "Cilia Gauss X", "min": 0.0, "max": 5.0, "step": 0.5},
        cilia_min_size={"label": "Cilia min voxels",       "min": 0},
        cilia_max_size={"label": "Cilia max voxels (0=off)", "min": 0},
        bb_spot_sigma={"label": "BB spot σ",    "min": 0.5, "max": 10.0, "step": 0.5},
        bb_outline_sigma={"label": "BB outline σ", "min": 0.5, "max": 10.0, "step": 0.5},
        bb_gauss_z={"label": "BB Gauss Z",      "min": 0.0, "max": 5.0, "step": 0.5},
        bb_gauss_y={"label": "BB Gauss Y",      "min": 0.0, "max": 5.0, "step": 0.5},
        bb_gauss_x={"label": "BB Gauss X",      "min": 0.0, "max": 5.0, "step": 0.5},
        bb_min_size={"label": "BB min voxels",       "min": 0},
        bb_max_size={"label": "BB max voxels (0=off)", "min": 0},
        max_cilia_dist_um={"label": "Max cilia dist (µm)", "min": 0.1, "max": 30.0, "step": 0.5},
        max_basal_dist_um={"label": "Max BB dist (µm)",    "min": 0.1, "max": 30.0, "step": 0.5},
        ratio_epsilon={"label": "Ratio ε",       "min": 0.0, "max": 10.0, "step": 0.1},
        axon_threshold={"label": "Axon log-ratio thr",  "min": 0.0, "max": 10.0, "step": 0.1},
        soma_threshold={"label": "Soma log-ratio thr",  "min": 0.0, "max": 10.0, "step": 0.1},
    )
    def widget(
        ims_path:         Path = Path("."),
        classifier_path:  Path = Path(_DEFAULT_CLASSIFIER),
        # ─ channels ─────────────────────────────────────────────────────────
        ch_cilia:    int  = 0,
        ch_neurites: int  = 1,
        ch_bb:       int  = 2,
        ch_nuclei:   int  = 3,
        use_mip:     bool = False,
        # ─ intensity normalisation ───────────────────────────────────────────
        p_low:  int = 2,
        p_high: int = 98,
        # ─ nuclei ────────────────────────────────────────────────────────────
        nuclei_sigma:         int = 15,
        tophat_radius:        int = 12,
        nuclei_outline_sigma: int = 3,
        nuclei_log: bool = False,
        nuclei_gauss_z: float = 0.0,
        nuclei_gauss_y: float = 0.0,
        nuclei_gauss_x: float = 0.0,
        # ─ neurites ──────────────────────────────────────────────────────────
        neurite_sigma: int = 5,
        neurite_log: bool = False,
        neurite_gauss_z: float = 0.0,
        neurite_gauss_y: float = 0.0,
        neurite_gauss_x: float = 0.0,
        # ─ cilia ─────────────────────────────────────────────────────────────
        cilia_log: bool = False,
        cilia_gauss_z: float = 1.0,
        cilia_gauss_y: float = 1.0,
        cilia_gauss_x: float = 1.0,
        cilia_min_size: int = 20,
        cilia_max_size: int = 0,
        # ─ basal bodies ──────────────────────────────────────────────────────
        bb_spot_sigma:    float = 2.0,
        bb_outline_sigma: float = 2.0,
        bb_log: bool = False,
        bb_gauss_z:       float = 1.0,
        bb_gauss_y:       float = 1.0,
        bb_gauss_x:       float = 0.0,
        bb_min_size: int = 5,
        bb_max_size: int = 0,
        # ─ distance thresholds ───────────────────────────────────────────────
        max_cilia_dist_um: float = 2.0,
        max_basal_dist_um: float = 2.0,
        ratio_epsilon:     float = 1.0,
        # ─ classification ────────────────────────────────────────────────────
        axon_threshold: float = 2.5,
        soma_threshold: float = 1.0,
    ):
        path_str = str(ims_path)
        if not Path(path_str).is_file() or not path_str.lower().endswith(".ims"):
            print("[LC] Please select a valid .ims file first.")
            return

        params = dict(
            ims_path=path_str,
            classifier_path=str(classifier_path),
            ch_cilia=ch_cilia, ch_neurites=ch_neurites,
            ch_bb=ch_bb, ch_nuclei=ch_nuclei,
            use_mip=use_mip, p_low=p_low, p_high=p_high,
            nuclei_sigma=nuclei_sigma, tophat_radius=tophat_radius,
            nuclei_outline_sigma=nuclei_outline_sigma, nuclei_log=nuclei_log,
            nuclei_gauss_z=nuclei_gauss_z, nuclei_gauss_y=nuclei_gauss_y, nuclei_gauss_x=nuclei_gauss_x,
            neurite_sigma=neurite_sigma, neurite_log=neurite_log,
            neurite_gauss_z=neurite_gauss_z, neurite_gauss_y=neurite_gauss_y, neurite_gauss_x=neurite_gauss_x,
            cilia_log=cilia_log,
            cilia_gauss_z=cilia_gauss_z, cilia_gauss_y=cilia_gauss_y, cilia_gauss_x=cilia_gauss_x,
            cilia_min_size=cilia_min_size, cilia_max_size=cilia_max_size,
            bb_spot_sigma=bb_spot_sigma, bb_outline_sigma=bb_outline_sigma, bb_log=bb_log,
            bb_gauss_z=bb_gauss_z, bb_gauss_y=bb_gauss_y, bb_gauss_x=bb_gauss_x,
            bb_min_size=bb_min_size, bb_max_size=bb_max_size,
            max_cilia_dist_um=max_cilia_dist_um, max_basal_dist_um=max_basal_dist_um,
            ratio_epsilon=ratio_epsilon,
            axon_threshold=axon_threshold, soma_threshold=soma_threshold,
        )

        @thread_worker
        def _run():
            return run_single_image(params)

        worker = _run()
        worker.returned.connect(lambda r: _update_layers(viewer, r))
        worker.errored.connect(lambda e: print(f"[LC] Error: {e}"))
        worker.start()

    return widget


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    viewer = napari.Viewer()
    widget = build_widget(viewer)
    viewer.window.add_dock_widget(widget, area="right", name="LimonCELLo")
    napari.run()
