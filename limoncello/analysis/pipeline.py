# pipeline.py
import gc
import json
import os
from datetime import datetime
from scipy.ndimage import center_of_mass
from ..utils.gpu_distance import distance_transform_edt
from skimage.measure import regionprops_table
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ..utils.reader import load_image, load_ims_metadata
from ..preprocessing.preprocessing import (
    normalize_intensity, percentile_minmax_normalize,
    make_isotropic as resample_isotropic,   # aliased: ``make_isotropic`` is a bool param of run_pipeline3
)
from ..segmentation.nuclei import segment_nuclei
from ..segmentation.neurites import segment_neurites
from ..segmentation.cilia import segment_cilia_ml
from ..segmentation.basal_bodies import segment_basal_bodies, segment_basal_bodies_ml
from ..qc.qc import run_qc, save_qc_excel, scatter_plot
from ..analysis.pair_cilia_to_bb import pair_from_mixed_df
from ..utils.assign_label_features import assign_label_features
import pyclesperanto_prototype as cle


# Max per-channel voxel count (Z*Y*X) attempted in 3-D GPU mode. Larger volumes
# would exhaust typical GPU VRAM during segmentation; they are skipped in batch
# with guidance to downsample or use MIP. Tune for your GPU if needed.
_MAX_3D_VOXELS = 4e8


def _flush_gpu():
    """Finish the OpenCL queue and force a collection so GPU buffers released in
    the current batch iteration are actually freed before the next file starts."""
    try:
        cle.get_device().queue.finish()
    except Exception:
        pass
    gc.collect()


def _region_props_3d(labels, voxel_size):
    """Per-label 3-D region properties from a label volume (Z, Y, X).

    Returns {label_id: (volume_voxels, volume_um3, length_um)} where the
    physical quantities use the anisotropic ``voxel_size`` (z, y, x in µm).
    """
    lab = np.asarray(labels).astype(np.int32)
    if lab.size == 0 or lab.max() == 0:
        return {}
    spacing = tuple(float(s) for s in voxel_size)
    vox_um3 = float(np.prod(spacing))
    out = {}
    try:
        rp = regionprops_table(
            lab, spacing=spacing,
            properties=("label", "num_pixels", "axis_major_length"),
        )
        for i, lid in enumerate(rp["label"]):
            n_vox = int(rp["num_pixels"][i])
            out[int(lid)] = (n_vox, n_vox * vox_um3, float(rp["axis_major_length"][i]))
    except Exception as exc:
        # Fall back to voxel counts only (no physical length) if regionprops fails
        print(f"  ⚠ 3-D region props failed ({exc}); using voxel counts only.")
        counts = np.bincount(lab.ravel())
        for lid in range(1, len(counts)):
            if counts[lid]:
                out[int(lid)] = (int(counts[lid]), counts[lid] * vox_um3, float("nan"))
    return out


def run_pipeline3(
    input_path,
    output_path,
    max_cilia_dist_cutoff_um=2.0,
    max_basal_body_cutoff_um=2.0,
    nuclei_spot_sigma=15,
    tophat_radius=12,
    neurite_spot_sigma=5,
    cilia_classifier_path=r'segmenters\cilia-segmenter.cl',
    axon_threshold=2.5,
    soma_threshold=1.0,
    p_low=2,
    p_high=98,
    outline_sigma=3,
    gpu_device=None,
    cilia_channel: int = 0,
    neurites_channel: int = 1,
    basal_bodies_channel: int = 2,
    nuclei_channel: int = 3,
    use_mip: bool = False,
    make_isotropic: bool = True,
    bb_method: str = "Voronoi-Otsu",
    bb_classifier_path: str | None = None,
    bb_spot_sigma: float = 2.0,
    bb_outline_sigma: float = 2.0,
    bb_gaussian_sigma: tuple = (1.0, 1.0, 0.0),
    cilia_gaussian_sigma: tuple = (1.0, 1.0, 1.0),
    nuclei_gaussian_sigma: tuple = (0.0, 0.0, 0.0),
    neurite_gaussian_sigma: tuple = (0.0, 0.0, 0.0),
    cilia_log: bool = False,
    nuclei_log: bool = False,
    neurite_log: bool = False,
    bb_log: bool = False,
    cilia_min_size: int = 20,
    cilia_max_size: int = 0,
    bb_min_size: int = 5,
    bb_max_size: int = 0,
    ratio_epsilon: float = 1.0,
    per_channel_norm: dict | None = None,
    progress_callback=None,
    per_file_callback=None,
):
    if gpu_device:
        cle.select_device(gpu_device)
    _dev = cle.get_device()
    print("═" * 60)
    print("  LimonCELLo pipeline")
    print("═" * 60)
    print(f"  GPU device : {getattr(_dev, 'name', _dev)}")
    print(f"  Mode       : {'MIP (2-D projection)' if use_mip else '3-D volume'}")

    # Each run gets its own timestamped subfolder inside output_path
    _run_stamp = datetime.now().strftime("lc-analysis-%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_path, _run_stamp)
    os.makedirs(output_path, exist_ok=True)
    print(f"  Run folder : {output_path}")

    csv_dir = os.path.join(output_path, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # ── Save run parameters for reproducibility / GUI reload ──────────────────
    params_log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "use_mip": use_mip,
        "make_isotropic": make_isotropic,
        "channels": {
            "cilia": cilia_channel,
            "neurites": neurites_channel,
            "basal_bodies": basal_bodies_channel,
            "nuclei": nuclei_channel,
        },
        "intensity_normalization": {
            "p_low": p_low, "p_high": p_high,
            "per_channel": (
                {str(k): list(v) for k, v in per_channel_norm.items()}
                if per_channel_norm else None
            ),
        },
        "nuclei": {
            "spot_sigma": nuclei_spot_sigma,
            "tophat_radius": tophat_radius,
            "outline_sigma": outline_sigma,
            "gaussian_sigma": list(nuclei_gaussian_sigma),
            "log_transform": nuclei_log,
        },
        "neurites": {
            "spot_sigma": neurite_spot_sigma,
            "gaussian_sigma": list(neurite_gaussian_sigma),
            "log_transform": neurite_log,
        },
        "cilia": {
            "gaussian_sigma": list(cilia_gaussian_sigma),
            "log_transform": cilia_log,
            "min_size": cilia_min_size,
            "max_size": cilia_max_size,
        },
        "basal_bodies": {
            "method": bb_method,
            "classifier_path": str(bb_classifier_path) if bb_classifier_path else None,
            "spot_sigma": bb_spot_sigma,
            "outline_sigma": bb_outline_sigma,
            "gaussian_sigma": list(bb_gaussian_sigma),
            "log_transform": bb_log,
            "min_size": bb_min_size,
            "max_size": bb_max_size,
        },
        "distance_thresholds": {
            "max_cilia_um": max_cilia_dist_cutoff_um,
            "max_basal_body_um": max_basal_body_cutoff_um,
            "ratio_epsilon": ratio_epsilon,
        },
        "classification": {
            "axon_threshold": axon_threshold,
            "soma_threshold": soma_threshold,
        },
    }
    _json_path = os.path.join(csv_dir, "run_parameters.json")
    with open(_json_path, "w") as _f:
        json.dump(params_log, _f, indent=2)

    all_dfs = []

    all_files = sorted(f for f in os.listdir(input_path) if f.endswith(".ims"))
    n_files = len(all_files)
    print(f"  Input      : {input_path}")
    print(f"  Files      : {n_files} .ims file(s)")
    print("─" * 60)

    for file_idx, file in enumerate(all_files):
        if progress_callback is not None:
            progress_callback(file_idx, n_files, file)
        print(f"[{file_idx + 1}/{n_files}] {file}")

        # Load data
        try:
            a, meta = load_image(os.path.join(input_path, file))
        except Exception as exc:
            print(f"  ✗ Could not open {file}: {exc} — skipping.")
            continue
        voxel_size = meta["voxel_size"]

        # ── Guard against volumes too large for the GPU in 3-D mode ─────────────
        # (T, C, Z, Y, X) → per-channel voxel count Z*Y*X. Large stitched images
        # would otherwise trigger a GPU MEM_OBJECT_ALLOCATION_FAILURE and abort
        # the whole batch; skip them with guidance and continue.
        _z, _y, _x = int(a.shape[2]), int(a.shape[3]), int(a.shape[4])
        _n_vox = _z * _y * _x
        if not use_mip and _n_vox > _MAX_3D_VOXELS:
            print(f"  ✗ {file}: volume {_z}×{_y}×{_x} = {_n_vox / 1e9:.2f} Gvox is "
                  f"too large for 3-D GPU segmentation and would exhaust VRAM. "
                  f"Downsample the image or enable MIP mode. Skipping.")
            continue

        if per_channel_norm:
            # Per-channel percentile normalisation (override or global default)
            a_norm = {}
            for c in range(a.shape[1]):
                _lo, _hi = per_channel_norm.get(c, (p_low, p_high))
                a_norm[0, c] = percentile_minmax_normalize(a[0, c], p_low=_lo, p_high=_hi)
        else:
            a_norm = normalize_intensity(a, p_low=p_low, p_high=p_high)

        # ── MIP mode: project each channel to 2-D then treat as single-Z 3-D ─
        if use_mip:
            print("  [MIP mode] projecting channels along Z…")
            n_ch = a.shape[1]

            def _z_project(vol3d):
                """(Z,Y,X) → GPU MIP → (1,Y,X) numpy."""
                proj = np.asarray(cle.maximum_z_projection(cle.push(vol3d)))
                return proj[np.newaxis] if proj.ndim == 2 else proj  # (1,Y,X)

            a = np.stack(
                [_z_project(a[0, ch]) for ch in range(n_ch)], axis=0
            )[np.newaxis]                     # (1, C, 1, Y, X)
            a_norm = np.stack(
                [_z_project(a_norm[0, ch]) for ch in range(n_ch)], axis=0
            )[np.newaxis]

        # ── Isotropic resampling (3-D only): downsample finer axes to coarsest ──
        elif make_isotropic and voxel_size and not all(
            abs(v - max(voxel_size)) < 1e-6 for v in voxel_size
        ):
            iso = max(voxel_size)
            print(f"  Anisotropic voxels {tuple(round(v, 3) for v in voxel_size)} — "
                  f"resampling to isotropic {iso:.3f} µm")
            n_ch = a.shape[1]
            a = np.stack(
                [resample_isotropic(np.asarray(a[0, ch]), voxel_size, iso)[0]
                 for ch in range(n_ch)], axis=0
            )[np.newaxis]                     # (1, C, Z', Y', X')
            a_norm = np.stack(
                [resample_isotropic(np.asarray(a_norm[0, ch]), voxel_size, iso)[0]
                 for ch in range(n_ch)], axis=0
            )[np.newaxis]
            voxel_size = (iso, iso, iso)

        # ── Segmentation ──────────────────────────────────────────────────────
        cilia_labels = segment_cilia_ml(
            a[0, cilia_channel],
            classifier_path=cilia_classifier_path,
            gaussian_sigma=cilia_gaussian_sigma,
            log_transform=cilia_log,
            min_size=cilia_min_size,
            max_size=cilia_max_size,
        )

        nuclei_labels_otsu = segment_nuclei(
            a_norm[0, nuclei_channel],
            tophat_radius=(tophat_radius, tophat_radius, tophat_radius),
            spot_sigma=nuclei_spot_sigma,
            outline_sigma=outline_sigma,
            gaussian_sigma=nuclei_gaussian_sigma,
            log_transform=nuclei_log,
        )

        skeleton, neurites_label = segment_neurites(
            a_norm[0, neurites_channel],
            spot_sigma=neurite_spot_sigma,
            gaussian_sigma=neurite_gaussian_sigma,
            log_transform=neurite_log,
        )

        if bb_method == "APOC classifier":
            basal_bodies_labels = segment_basal_bodies_ml(
                a[0, basal_bodies_channel],
                classifier_path=bb_classifier_path,
                gaussian_sigma=bb_gaussian_sigma,
                log_transform=bb_log,
                min_size=bb_min_size,
                max_size=bb_max_size,
            )
        else:
            basal_bodies_labels = segment_basal_bodies(
                a[0, basal_bodies_channel],
                spot_sigma=bb_spot_sigma,
                outline_sigma=bb_outline_sigma,
                gaussian_sigma=bb_gaussian_sigma,
                log_transform=bb_log,
                min_size=bb_min_size,
                max_size=bb_max_size,
            )

        # get neurites masks
        neurite_mask = np.asarray(neurites_label) > 0
        skeleton_mask = np.asarray(skeleton) > 0

        # Nuclei distance map
        distance_map_nuclei = distance_transform_edt(
            ~(nuclei_labels_otsu > 0), sampling=voxel_size
        )

        # Neurite thickness proxy (radius)
        distance_map_neurites = distance_transform_edt(
            neurite_mask, sampling=voxel_size
        )

        # Regularised ratio: (dt_nuclei + ε) / (dt_neurite + ε)
        # NaN outside the neurite mask so overlays show true background
        with np.errstate(divide="ignore", invalid="ignore"):
            map_ratio = np.where(
                neurite_mask,
                (distance_map_nuclei + ratio_epsilon) / (distance_map_neurites + ratio_epsilon),
                np.nan,
            )
        map_ratio[~np.isfinite(map_ratio)] = np.nan

        # Distance to neurites
        dist_to_neurite = distance_transform_edt(
            ~neurite_mask,
            sampling=voxel_size
        )

        # Nearest SKELETON mapping
        _, nearest_skel_idx = distance_transform_edt(
            ~skeleton_mask,
            return_indices=True,
            sampling=voxel_size,
        )

        # Cilia centroids
        cilia_ids = np.unique(cilia_labels)
        cilia_ids = cilia_ids[cilia_ids != 0]

        cilia_centroids = center_of_mass(
            cilia_labels > 0, labels=cilia_labels, index=cilia_ids
        )

        # Basal bodies centroids
        basal_bodies_ids = np.unique(basal_bodies_labels)
        basal_bodies_ids = basal_bodies_ids[basal_bodies_ids != 0]
        basal_bodies_centroids = center_of_mass(
            basal_bodies_labels > 0, labels=basal_bodies_labels, index=basal_bodies_ids
        )

        df_cilia = assign_label_features(
            cilia_labels,
            cilia_centroids,
            cilia_ids,
            dist_to_neurite,
            nearest_skel_idx,
            distance_map_neurites,
            distance_map_nuclei,
            map_ratio,
            max_cilia_dist_cutoff_um,
            file,
            ratio_epsilon=ratio_epsilon,
        )

        if df_cilia.empty:
            print(f"No valid cilia found in {file}")
            try:
                del cilia_labels, nuclei_labels_otsu, skeleton, neurites_label, \
                    basal_bodies_labels, neurite_mask, skeleton_mask, \
                    distance_map_nuclei, distance_map_neurites, map_ratio, \
                    dist_to_neurite, nearest_skel_idx, a_norm, a
            except NameError:
                pass
            _flush_gpu()
            continue

        # 3-D region properties per cilium (volume, length) from the label volume
        _cprops = _region_props_3d(np.asarray(cilia_labels), voxel_size)
        df_cilia["volume_voxels"] = [
            _cprops.get(int(i), (np.nan, np.nan, np.nan))[0] for i in df_cilia["cilia_id"]
        ]
        df_cilia["volume_um3"] = [
            _cprops.get(int(i), (np.nan, np.nan, np.nan))[1] for i in df_cilia["cilia_id"]
        ]
        df_cilia["length_um"] = [
            _cprops.get(int(i), (np.nan, np.nan, np.nan))[2] for i in df_cilia["cilia_id"]
        ]

        df_basal_bodies = assign_label_features(
            basal_bodies_labels,
            basal_bodies_centroids,
            basal_bodies_ids,
            dist_to_neurite,
            nearest_skel_idx,
            distance_map_neurites,
            distance_map_nuclei,
            map_ratio,
            max_basal_body_cutoff_um,
            file,
            "basal_body",
            ratio_epsilon=ratio_epsilon,
        )

        # ── Pair cilia ↔ basal bodies (closest, strict 1:1, µm cutoff) ──────────
        # Only cilia paired to a basal body within the cutoff are "validated";
        # everything downstream (CSV, overlays, label MIPs) keeps validated only.
        combined = pd.concat([df_cilia, df_basal_bodies], ignore_index=True)
        paired = pair_from_mixed_df(
            combined, voxel_size=voxel_size,
            max_pair_distance_um=max_basal_body_cutoff_um,
        )
        df_cilia = paired[(paired["object_type"] == "cilia") & paired["validated"]].reset_index(drop=True)
        df_basal_bodies = paired[(paired["object_type"] == "basal_body") & paired["validated"]].reset_index(drop=True)

        if df_cilia.empty:
            print(f"  No validated cilia (paired within {max_basal_body_cutoff_um} µm) in {file}")
            try:
                del cilia_labels, nuclei_labels_otsu, skeleton, neurites_label, \
                    basal_bodies_labels, neurite_mask, skeleton_mask, \
                    distance_map_nuclei, distance_map_neurites, map_ratio, \
                    dist_to_neurite, nearest_skel_idx, a_norm, a
            except NameError:
                pass
            _flush_gpu()
            continue

        all_dfs.append(df_basal_bodies)
        all_dfs.append(df_cilia)

        # Validated-only label volumes (everything but the paired cilia/BBs zeroed)
        _valid_cilia_ids = [int(i) for i in df_cilia["cilia_id"]]
        _valid_bb_ids    = [int(i) for i in df_basal_bodies["cilia_id"]]
        cilia_labels_v = np.where(
            np.isin(np.asarray(cilia_labels), _valid_cilia_ids), np.asarray(cilia_labels), 0
        ).astype(np.int32)
        bb_labels_v = np.where(
            np.isin(np.asarray(basal_bodies_labels), _valid_bb_ids), np.asarray(basal_bodies_labels), 0
        ).astype(np.int32)

        # ── Overlay visualization (MIP) ───────────────────────────────────────
        overlay_dir = os.path.join(output_path, "figures", "overlays")
        os.makedirs(overlay_dir, exist_ok=True)

        # Display MIPs: computed from raw data with gentle per-MIP normalisation
        # (avoids double-clipping caused by using the already-normalised a_norm)
        def _disp_mip(vol):
            mip = np.max(np.asarray(vol).astype(np.float32), axis=0)
            lo, hi = np.percentile(mip, 0), np.percentile(mip, 99.9)
            return (mip - lo) / (hi - lo + 1e-8)  # no hard clip

        neurite_mip = _disp_mip(a[0, neurites_channel])
        cilia_mip   = _disp_mip(a[0, cilia_channel])
        nuclei_mip  = _disp_mip(a[0, nuclei_channel])

        # Persist MIPs so the app overlay tab can regenerate figures without
        # reloading raw .ims data.
        _mip_out = os.path.join(output_path, "figures", "mips")
        os.makedirs(_mip_out, exist_ok=True)
        _fstem = os.path.splitext(file)[0]
        _neurite_mask_mip = np.max(neurite_mask, axis=0).astype(np.uint8)   # (Y,X) binary
        np.save(os.path.join(_mip_out, f"{_fstem}_neurite_mip.npy"),      neurite_mip)
        np.save(os.path.join(_mip_out, f"{_fstem}_cilia_mip.npy"),        cilia_mip)
        np.save(os.path.join(_mip_out, f"{_fstem}_nuclei_mip.npy"),       nuclei_mip)
        np.save(os.path.join(_mip_out, f"{_fstem}_neurite_mask_mip.npy"), _neurite_mask_mip)
        np.save(os.path.join(_mip_out, f"{_fstem}_ratio_mid.npy"),
                map_ratio[map_ratio.shape[0] // 2])
        # Additional MIPs for interactive overlay in app (saved by new runs only)
        np.save(os.path.join(_mip_out, f"{_fstem}_bb_mip.npy"),
                _disp_mip(a[0, basal_bodies_channel]))
        np.save(os.path.join(_mip_out, f"{_fstem}_cilia_labels_mip.npy"),
                np.max(cilia_labels_v.astype(np.uint16), axis=0))
        np.save(os.path.join(_mip_out, f"{_fstem}_nuclei_labels_mip.npy"),
                np.max(nuclei_labels_otsu.astype(np.uint16), axis=0))
        np.save(os.path.join(_mip_out, f"{_fstem}_bb_labels_mip.npy"),
                np.max(bb_labels_v.astype(np.uint16), axis=0))
        np.save(os.path.join(_mip_out, f"{_fstem}_neurite_labels_mip.npy"),
                np.max(np.asarray(neurites_label).astype(np.uint16), axis=0))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Ratio panel: NaN outside neurites → black background
        _cmap_ratio = plt.cm.coolwarm.copy()
        _cmap_ratio.set_bad("black")

        with np.errstate(divide="ignore", invalid="ignore"):
            _ratio_log_mid = np.log(map_ratio[map_ratio.shape[0] // 2])

        # [0,0] Cilia MIP  [0,1] log(ratio) masked
        # [1,0] Nuclei MIP [1,1] Cilia MIP
        axes[0, 0].imshow(cilia_mip, cmap="gray")
        axes[0, 0].set_title("Cilia MIP")
        axes[0, 1].imshow(_ratio_log_mid, cmap=_cmap_ratio)
        axes[0, 1].set_title("log(ratio) — neurite masked")
        axes[1, 0].imshow(nuclei_mip, cmap="gray")
        axes[1, 0].set_title("Nuclei MIP")
        axes[1, 1].imshow(cilia_mip, cmap="gray")
        axes[1, 1].set_title("Cilia MIP")

        scores = df_cilia["log_ratio"].values
        valid_scores = scores[np.isfinite(scores)]
        vmin, vmax = (np.percentile(valid_scores, [5, 95]) if len(valid_scores) > 0 else (0, 1))

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap_dots = plt.cm.coolwarm
        coords = np.vstack(df_cilia["coords"].values)
        ys = coords[:, 1]
        xs = coords[:, 2]
        colors = cmap_dots(norm(scores))  # no clipping

        for ax in axes.flat:
            ax.scatter(xs, ys, c=colors, s=10, edgecolor="black", linewidth=0.3)

        sm = plt.cm.ScalarMappable(cmap=cmap_dots, norm=norm)
        sm.set_array([])
        for ax in axes.flat:
            plt.colorbar(sm, ax=ax, label="log_ratio")

        plt.tight_layout()
        plt.savefig(
            os.path.join(overlay_dir, f"{_fstem}_overlay.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close("all")

        # ── Live napari visualisation hook (GUI thread, blocks until captured) ──
        if per_file_callback is not None:
            napari_overlay_dir = os.path.join(output_path, "figures", "napari_overlays")
            roi_dir = os.path.join(output_path, "figures", "cilia_rois")
            os.makedirs(napari_overlay_dir, exist_ok=True)
            os.makedirs(roi_dir, exist_ok=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                _log_ratio_map = np.log(map_ratio)                    # NaN outside neurites
            try:
                per_file_callback(dict(
                    stem=_fstem,
                    screenshots_dir=napari_overlay_dir,
                    rois_dir=roi_dir,
                    voxel_size=tuple(float(v) for v in voxel_size),
                    raw=np.asarray(a[0]),                              # (C, Z, Y, X)
                    channels=dict(ch_cilia=cilia_channel, ch_neurites=neurites_channel,
                                  ch_bb=basal_bodies_channel, ch_nuclei=nuclei_channel),
                    cilia_labels=cilia_labels_v,
                    bb_labels=bb_labels_v,
                    nuclei_labels=np.asarray(nuclei_labels_otsu).astype(np.int32),
                    neurite_labels=np.asarray(neurites_label).astype(np.int32),
                    skeleton_mask=np.asarray(skeleton_mask),
                    nearest_skel_idx=np.asarray(nearest_skel_idx),
                    log_ratio_map=_log_ratio_map,
                    cilia_df=df_cilia,
                    bb_df=df_basal_bodies,
                ))
            except Exception as _exc:
                print(f"  ⚠ napari capture failed for {file}: {_exc}")

        # ── Release GPU label buffers + large CPU arrays before the next file ───
        # Without this the previous file's buffers stay alive while the next
        # file's (GPU-heavy) segmentation allocates, exhausting VRAM after a few
        # files even for normal-sized images.
        try:
            del cilia_labels, nuclei_labels_otsu, skeleton, neurites_label, \
                basal_bodies_labels, neurite_mask, skeleton_mask, \
                distance_map_nuclei, distance_map_neurites, map_ratio, \
                dist_to_neurite, nearest_skel_idx, a_norm, a
        except NameError:
            pass
        _flush_gpu()

    if len(all_dfs) == 0:
        print("No data processed.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)

    # Short names
    unique_files = final_df["filename"].unique()
    file_map = {f: f"S{i+1}" for i, f in enumerate(unique_files)}
    final_df["file_short"] = final_df["filename"].map(file_map)

    # Classification
    def classify(score):
        if score > axon_threshold:
            return "axon"
        elif score < soma_threshold:
            return "soma"
        else:
            return "ambiguous"

    final_df["class"] = final_df["log_ratio"].apply(classify)

    # ── Summary figures ───────────────────────────────────────────────────────
    fig_dir = os.path.join(output_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    final_cilia_df = final_df[final_df["object_type"] == "cilia"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(final_cilia_df["ratio"], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Raw Ratio Distribution")
    sns.histplot(final_cilia_df["log_ratio"], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Log Ratio Distribution")
    sns.boxplot(data=final_cilia_df, x="file_short", y="log_ratio", ax=axes[1, 0])
    axes[1, 0].set_title("Log Ratio per Sample")
    axes[1, 0].tick_params(axis="x", rotation=45)
    sns.countplot(data=final_cilia_df, x="file_short", hue="class", ax=axes[1, 1])
    axes[1, 1].set_title("Class Distribution per Sample")
    axes[1, 1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "overview_panels.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=final_cilia_df, x="log_ratio", hue="file_short", common_norm=False)
    plt.title("Log Ratio Distribution per Sample")
    plt.xlabel("Log Ratio")
    plt.savefig(os.path.join(fig_dir, "logratio_kde_per_sample.png"), dpi=300)
    plt.close()

    # ── QC + Excel ────────────────────────────────────────────────────────────
    excel_path = os.path.join(csv_dir, "all_cilia_features.xlsx")

    qc_sample = final_cilia_df.groupby("file_short").agg(
        n_cilia=("cilia_id", "count"),
        mean_ratio=("ratio", "mean"),
        std_ratio=("ratio", "std"),
        mean_log_ratio=("log_ratio", "mean"),
        std_log_ratio=("log_ratio", "std"),
        mean_distance=("distance_to_neurite_um", "mean"),
        max_distance=("distance_to_neurite_um", "max"),
    ).reset_index()

    qc_global = pd.DataFrame({
        "metric": ["n_total_cilia", "mean_ratio", "std_ratio", "mean_log_ratio", "std_log_ratio"],
        "value": [
            len(final_cilia_df),
            final_cilia_df["ratio"].mean(),
            final_cilia_df["ratio"].std(),
            final_cilia_df["log_ratio"].mean(),
            final_cilia_df["log_ratio"].std(),
        ],
    })

    with pd.ExcelWriter(excel_path) as writer:
        final_df.to_excel(writer, sheet_name="all_data", index=False)
        qc_sample.to_excel(writer, sheet_name="qc_per_sample", index=False)
        qc_global.to_excel(writer, sheet_name="qc_global", index=False)

    print("─" * 60)
    print(f"  QC Excel   : {excel_path}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="dt_neurite", y="dt_nuclei", hue="file_short",
                    data=final_cilia_df, palette="tab10")
    plt.xlabel("Assigned Neurite Voxel thickness (um)")
    plt.ylabel("Assigned Neurite Voxel Distance to nuclei (um)")
    plt.title("DT Comparison per Sample")
    plt.legend(bbox_to_anchor=(0.5, 1), loc="upper left")
    plt.savefig(os.path.join(fig_dir, "dt_scatterplot.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="log_dt_neurite", y="log_dt_nuclei", hue="file_short",
                    data=final_cilia_df, palette="tab10")
    plt.xlabel("Log of Assigned Neurite Voxel thickness (um)")
    plt.ylabel("Log of Assigned Neurite Voxel Distance to nuclei (um)")
    plt.title("Log DT Comparison per Sample")
    plt.legend(bbox_to_anchor=(0.5, 1), loc="upper left")
    plt.savefig(os.path.join(fig_dir, "log_dt_scatterplot.png"), dpi=300)
    plt.close()

    print(f"  Figures    : {fig_dir}")
    print("═" * 60)
    print("  ✓ Pipeline complete")
    print("═" * 60)
