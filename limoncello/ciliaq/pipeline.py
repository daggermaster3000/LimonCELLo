"""Batch CiliaQ-style pipeline over a folder of .ims files.

For each image:
  1. threshold-segment the cilia channel (CiliaQ_Preparator style)
  2. quantify every cilium (volume, surface, length, intensity, coloc, bending)
  3. (optional) reproduce the LimonCELLo base analysis — segment nuclei &
     neurites, build the (dt_nuclei+ε)/(dt_neurite+ε) ratio, assign it per
     cilium, classify axon/soma/ambiguous, and pair cilia to basal bodies.

Results from all files are merged into one per-cilium table and saved to Excel,
with display MIPs persisted for the app overlays.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, center_of_mass

from ..utils.reader import load_image
from ..preprocessing.preprocessing import percentile_minmax_normalize
from ..segmentation.nuclei import segment_nuclei
from ..segmentation.neurites import segment_neurites
from ..segmentation.basal_bodies import segment_basal_bodies
from ..utils.assign_label_features import assign_label_features
from ..analysis.pair_cilia_to_bb import pair_from_mixed_df
from .segment import segment_cilia_threshold
from .quantify import quantify_cilia


def _classify(series, axon_thr, soma_thr):
    def _c(r):
        if pd.isna(r):
            return "ambiguous"
        return "axon" if r > axon_thr else ("soma" if r < soma_thr else "ambiguous")
    return series.apply(_c)


def run_ciliaq(input_path, output_path, params: dict, progress_callback=None):
    """Run the CiliaQ-style batch pipeline. ``params`` keys mirror the app UI."""
    p = params
    stamp = datetime.now().strftime("ciliaq-analysis-%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(output_path, stamp)
    csv_dir = os.path.join(out_dir, "csv")
    mip_dir = os.path.join(out_dir, "figures", "mips")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(mip_dir, exist_ok=True)

    print("═" * 60)
    print("  LimonCELLo · CiliaQ-style pipeline")
    print("═" * 60)
    with open(os.path.join(csv_dir, "run_parameters.json"), "w") as fh:
        json.dump({"timestamp": stamp, **p}, fh, indent=2, default=str)

    files = sorted(f for f in os.listdir(input_path) if f.endswith(".ims"))
    print(f"  Input : {input_path}")
    print(f"  Files : {len(files)} .ims")
    print("─" * 60)

    base = bool(p.get("base_analysis", True))
    eps = float(p.get("ratio_epsilon", 1.0))
    all_dfs = []

    for idx, fname in enumerate(files):
        if progress_callback:
            progress_callback(idx, len(files), fname)
        print(f"[{idx + 1}/{len(files)}] {fname}")
        try:
            img, meta = load_image(os.path.join(input_path, fname))
        except Exception as exc:
            print(f"  ✗ could not open: {exc}")
            continue
        voxel_size = meta.get("voxel_size") or (1.0, 1.0, 1.0)
        n_ch = img.shape[1]

        def _chan(ci):
            return np.asarray(img[0, int(min(ci, n_ch - 1))], dtype=np.float32)

        cilia_raw = _chan(p["ch_cilia"])

        # 1) CiliaQ segmentation
        cilia_labels = segment_cilia_threshold(
            cilia_raw,
            method=p.get("threshold_method", "otsu"),
            gaussian_sigma=tuple(p.get("cilia_gaussian_sigma", (1.0, 1.0, 1.0))),
            background_radius=float(p.get("background_radius", 0.0)),
            threshold_factor=float(p.get("threshold_factor", 1.0)),
            hysteresis_low_factor=float(p.get("hysteresis_low_factor", 0.5)),
            min_voxels=int(p.get("min_voxels", 10)),
            max_voxels=int(p.get("max_voxels", 0)),
        )
        n_cil = int(cilia_labels.max())
        print(f"    cilia segmented: {n_cil}")
        if n_cil == 0:
            continue

        # intensity channels for measurement
        intensity_images = {"cilia": cilia_raw}
        for label_key, ch_key in (("channelA", "ch_signal_a"),
                                   ("channelB", "ch_signal_b")):
            ci = p.get(ch_key)
            if ci is not None and ci != "" and int(ci) >= 0:
                intensity_images[label_key] = _chan(int(ci))
        coloc_img = None
        if p.get("ch_coloc") not in (None, "", -1):
            coloc_img = _chan(int(p["ch_coloc"]))

        # basal-body reference points (for base→tip orientation + pairing)
        bb_labels = None
        bb_centroids = None
        if base and p.get("ch_bb") is not None:
            bb_norm = percentile_minmax_normalize(
                _chan(p["ch_bb"]), p_low=p.get("p_low", 2), p_high=p.get("p_high", 98))
            bb_labels = np.asarray(segment_basal_bodies(
                _chan(p["ch_bb"]),
                spot_sigma=float(p.get("bb_spot_sigma", 2.0)),
                outline_sigma=float(p.get("bb_outline_sigma", 2.0)),
                gaussian_sigma=tuple(p.get("bb_gaussian_sigma", (1.0, 1.0, 0.0))),
                min_size=int(p.get("bb_min_size", 5)),
            )).astype(np.int32)
            _bb_ids = np.unique(bb_labels); _bb_ids = _bb_ids[_bb_ids != 0]
            if _bb_ids.size:
                bb_centroids = np.array(center_of_mass(
                    bb_labels > 0, labels=bb_labels, index=_bb_ids))

        # 2) CiliaQ quantification
        profiles = {}
        df = quantify_cilia(
            cilia_labels, voxel_size=voxel_size,
            intensity_images=intensity_images,
            coloc_image=coloc_img,
            coloc_threshold=p.get("coloc_threshold"),
            base_ref_points=bb_centroids,
            min_voxels=int(p.get("min_voxels", 10)),
            file=fname, profiles_out=profiles,
        )
        df["file_short"] = f"S{idx + 1}"
        df["object_type"] = "cilia"

        # 3) base LimonCELLo ratio / classification
        if base:
            try:
                df = _augment_base_analysis(df, img, meta, p, cilia_labels,
                                            bb_labels, eps, fname, idx)
            except Exception as exc:
                print(f"    ⚠ base analysis skipped: {exc}")

        all_dfs.append(df)

        # persist display MIPs for overlays
        _save_mips(mip_dir, os.path.splitext(fname)[0], cilia_raw, cilia_labels, bb_labels)

    if not all_dfs:
        print("No cilia found in any file.")
        return {"out_dir": out_dir, "n_cilia": 0}

    final = pd.concat(all_dfs, ignore_index=True)
    excel_path = os.path.join(csv_dir, "ciliaq_features.xlsx")
    with pd.ExcelWriter(excel_path) as xl:
        final.to_excel(xl, sheet_name="all_data", index=False)
        _per_sample_summary(final).to_excel(xl, sheet_name="qc_per_sample", index=False)
    print("─" * 60)
    print(f"  Saved : {excel_path}")
    print(f"  ✓ {len(final)} cilia across {final['file_short'].nunique()} sample(s)")
    print("═" * 60)
    return {"out_dir": out_dir, "excel": excel_path, "n_cilia": len(final)}


def _augment_base_analysis(df, img, meta, p, cilia_labels, bb_labels, eps, fname, idx):
    """Add distance_to_neurite / ratio / log_ratio / class + BB pairing."""
    voxel_size = meta.get("voxel_size") or (1.0, 1.0, 1.0)

    def _norm(ci):
        return percentile_minmax_normalize(
            np.asarray(img[0, int(ci)], dtype=np.float32),
            p_low=p.get("p_low", 2), p_high=p.get("p_high", 98))

    nuclei_labels = segment_nuclei(
        _norm(p["ch_nuclei"]),
        tophat_radius=(p.get("tophat_radius", 12),) * 3,
        spot_sigma=p.get("nuclei_sigma", 15),
        outline_sigma=p.get("nuclei_outline_sigma", 3),
    ).astype(np.int32)
    skeleton, neur_gpu = segment_neurites(
        _norm(p["ch_neurites"]), spot_sigma=p.get("neurite_sigma", 5))
    neurite_mask = np.asarray(neur_gpu) > 0
    skeleton_mask = np.asarray(skeleton) > 0

    dt_nuclei = distance_transform_edt(~(nuclei_labels > 0), sampling=voxel_size)
    dt_neurite = distance_transform_edt(neurite_mask, sampling=voxel_size)
    with np.errstate(divide="ignore", invalid="ignore"):
        map_ratio = np.where(neurite_mask,
                             (dt_nuclei + eps) / (dt_neurite + eps), np.nan)
    map_ratio[~np.isfinite(map_ratio)] = np.nan
    dist_to_neurite = distance_transform_edt(~neurite_mask, sampling=voxel_size)
    _, nearest_skel = distance_transform_edt(
        ~skeleton_mask, return_indices=True, sampling=voxel_size)

    ids = np.unique(cilia_labels); ids = ids[ids != 0]
    cents = center_of_mass(cilia_labels > 0, labels=cilia_labels, index=ids)
    feat = assign_label_features(
        cilia_labels, cents, ids, dist_to_neurite, nearest_skel,
        dt_neurite, dt_nuclei, map_ratio,
        float(p.get("max_cilia_dist_um", 2.0)), fname, ratio_epsilon=eps)

    if not feat.empty:
        keep = ["cilia_id", "distance_to_neurite_um", "ratio", "log_ratio",
                "dt_neurite", "dt_nuclei"]
        df = df.merge(feat[keep], on="cilia_id", how="left")
        df["class"] = _classify(df["log_ratio"],
                                float(p.get("axon_threshold", 2.5)),
                                float(p.get("soma_threshold", 1.0)))

    # cilia ↔ basal-body pairing (Hungarian)
    if bb_labels is not None and bb_labels.max() > 0:
        bb_ids = np.unique(bb_labels); bb_ids = bb_ids[bb_ids != 0]
        bb_cents = center_of_mass(bb_labels > 0, labels=bb_labels, index=bb_ids)
        bb_df = pd.DataFrame({
            "filename": fname, "cilia_id": bb_ids, "object_type": "basal_body",
            "coords": [list(map(float, c)) for c in np.atleast_2d(bb_cents)],
        })
        mixed = pd.concat([df[["filename", "cilia_id", "object_type", "coords"]],
                           bb_df], ignore_index=True)
        paired = pair_from_mixed_df(mixed)
        pc = paired[paired["object_type"] == "cilia"][
            ["cilia_id", "paired_id", "pair_distance_um", "pairing_status"]]
        df = df.merge(pc, on="cilia_id", how="left")
    return df


def _save_mips(mip_dir, stem, cilia_raw, cilia_labels, bb_labels):
    def _disp(vol):
        m = np.max(np.asarray(vol, np.float32), axis=0)
        lo, hi = np.percentile(m, 0), np.percentile(m, 99.9)
        return (m - lo) / (hi - lo + 1e-8)
    np.save(os.path.join(mip_dir, f"{stem}_cilia_mip.npy"), _disp(cilia_raw))
    np.save(os.path.join(mip_dir, f"{stem}_cilia_labels_mip.npy"),
            np.max(cilia_labels.astype(np.uint16), axis=0))
    if bb_labels is not None:
        np.save(os.path.join(mip_dir, f"{stem}_bb_labels_mip.npy"),
                np.max(bb_labels.astype(np.uint16), axis=0))


def _per_sample_summary(df):
    g = df.groupby("file_short")
    agg = {"cilia_id": "count", "volume_um3": "mean", "length_um": "mean",
           "surface_um2": "mean"}
    agg = {k: v for k, v in agg.items() if k in df.columns}
    out = g.agg(agg).rename(columns={"cilia_id": "n_cilia"}).reset_index()
    return out
