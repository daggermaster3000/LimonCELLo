# pipeline.py
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, center_of_mass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ..utils.reader import load_image, load_ims_metadata
from ..preprocessing.preprocessing import normalize_intensity,percentile_minmax_normalize
from ..segmentation.nuclei import segment_nuclei
from ..segmentation.neurites import segment_neurites
from ..segmentation.cilia import segment_cilia_ml
from ..segmentation.basal_bodies import segment_basal_bodies
from ..utils.reader import load_image,load_ims_metadata
from ..qc.qc import run_qc,save_qc_excel,scatter_plot
from ..analysis.pair_cilia_to_bb import pair_from_mixed_df
from ..utils.assign_label_features import assign_label_features
import pyclesperanto_prototype as cle
import os

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
):
    print(cle.available_device_names(dev_type="gpu"))
    if gpu_device:
        cle.select_device(gpu_device)
    print(cle.get_device())
    os.makedirs(output_path, exist_ok=True)
    csv_dir = os.path.join(output_path, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    all_dfs = []

    for file in tqdm(os.listdir(input_path)):
        if not file.endswith(".ims"):
            continue

        print(f"Processing {file}...")

        # Load data
        try:
            a, meta = load_image(os.path.join(input_path, file))
        except Exception as exc:
            print(f"  ✗ Could not open {file}: {exc} — skipping.")
            continue
        voxel_size = meta["voxel_size"]
        
        a_norm = normalize_intensity(a, p_low=p_low, p_high=p_high)
        
        # Segmentation
        cilia_labels = segment_cilia_ml(
            a[0, cilia_channel],
            classifier_path=cilia_classifier_path,
        )

        nuclei_labels_otsu = segment_nuclei(
            a_norm[0, nuclei_channel],
            tophat_radius=(tophat_radius, tophat_radius, tophat_radius),
            spot_sigma=nuclei_spot_sigma,
            outline_sigma=outline_sigma,
        )

        skeleton, neurites_label = segment_neurites(
            a_norm[0, neurites_channel],
            spot_sigma=neurite_spot_sigma,
        )

        basal_bodies_labels = segment_basal_bodies(
            a[0, basal_bodies_channel]
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

        # Ratio
        map_ratio = np.where(distance_map_neurites > 0,
                     distance_map_nuclei / distance_map_neurites,
                     0)  # or np.nan
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
        basal_bodies_ids = basal_bodies_ids[basal_bodies_ids != 0] # remove background
        basal_bodies_centroids = center_of_mass(
            basal_bodies_labels > 0, labels = basal_bodies_labels, index=basal_bodies_ids
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
            )
        
        if df_cilia.empty:
            print(f"No valid cilia found in {file}")
            continue

        

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
            "basal_body"
        )

        #paired_df = pair_cilia_basal_bidirectional(df_cilia,df_basal_bodies)
        all_dfs.append(df_basal_bodies)
        all_dfs.append(df_cilia)

        # Overlay visualization (MIP)
        overlay_dir = os.path.join(output_path, "figures", "overlays")
        os.makedirs(overlay_dir, exist_ok=True)

        # MIP of neurite + cilia channels
        neurite_mip = np.max(a_norm[0, neurites_channel], axis=0)   # (Y, X)
        cilia_mip   = np.max(a_norm[0, cilia_channel],   axis=0)    # (Y, X)
        nuclei_mip  = np.max(a_norm[0, nuclei_channel],  axis=0)    # (Y, X)

        # Persist MIPs so the app overlay tab can regenerate figures without
        # reloading raw .ims data.
        _mip_out = os.path.join(output_path, "figures", "mips")
        os.makedirs(_mip_out, exist_ok=True)
        _fstem = os.path.splitext(file)[0]
        np.save(os.path.join(_mip_out, f"{_fstem}_neurite_mip.npy"), neurite_mip)
        np.save(os.path.join(_mip_out, f"{_fstem}_cilia_mip.npy"),   cilia_mip)
        np.save(os.path.join(_mip_out, f"{_fstem}_nuclei_mip.npy"),  nuclei_mip)
        np.save(os.path.join(_mip_out, f"{_fstem}_ratio_mid.npy"),
                map_ratio[map_ratio.shape[0] // 2])

        fig, axes = plt.subplots(2,2, figsize=(14,10))

        # Base layer: neurites
        axes[0,0].imshow(neurite_mip, cmap="gray")
        axes[0,0].set_title('Neurites MIP')
        axes[1,1].imshow(cilia_mip,cmap='gray')
        axes[1,1].set_title('Cilia MIP')
        with np.errstate(divide="ignore", invalid="ignore"):
            _ratio_log_mid = np.log(map_ratio[map_ratio.shape[0] // 2])
        axes[0,1].imshow(_ratio_log_mid, cmap='coolwarm')
        axes[0,1].set_title('log(ratio) overlay')
        axes[1,0].imshow(nuclei_mip,cmap='gray')
        axes[1,0].set_title('Nuclei MIP')

        # Normalize scores for coloring (robust)
        scores = df_cilia["log_ratio"].values
        valid_scores = scores[np.isfinite(scores)]

        if len(valid_scores) > 0:
            vmin, vmax = np.percentile(valid_scores, [5, 95])
        else:
            vmin, vmax = 0, 1

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.coolwarm

        # Overlay cilia centroids
        coords = np.vstack(df_cilia["coords"].values)  # shape (n, 3)

        
        ys = coords[:, 1]
        xs = coords[:, 2]
        colors = cmap(norm(scores))

        axes[0,0].scatter(
            xs,
            ys,
            c=colors,
            s=10,
            edgecolor="black",
            linewidth=0.3,
            
        )
        axes[1,1].scatter(
            xs,
            ys,
            c=colors,
            s=10,
            edgecolor="black",
            linewidth=0.3,
            
        )
        axes[0,1].scatter(
            xs,
            ys,
            c=colors,
            s=10,
            edgecolor="black",
            linewidth=0.3,
            
        )
        axes[1,0].scatter(
            xs,
            ys,
            c=colors,
            s=10,
            edgecolor="black",
            linewidth=0.3,
            
        )

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=axes[1,0], label="log_ratio")
        plt.colorbar(sm, ax=axes[1,1], label="log_ratio")
        plt.colorbar(sm, ax=axes[0,0], label="log_ratio")
        plt.colorbar(sm, ax=axes[0,1], label="log_ratio")

        plt.axis("off")

        save_path = os.path.join(
            overlay_dir,
            f"{os.path.splitext(file)[0]}_overlay.png"
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

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

    # Figures 
    fig_dir = os.path.join(output_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    
    # 1. Multi-panel (ratio + log-ratio)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Ratio
    final_cilia_df = final_df[final_df["object_type"] == "cilia"]
    sns.histplot(final_cilia_df["ratio"], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Raw Ratio Distribution")

    # Log ratio
    sns.histplot(final_cilia_df["log_ratio"], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Log Ratio Distribution")

    # Log ratio per sample (boxplot)
    sns.boxplot(data=final_cilia_df, x="file_short", y="log_ratio", ax=axes[1, 0])
    axes[1, 0].set_title("Log Ratio per Sample")
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Class distribution per sample
    sns.countplot(data=final_cilia_df, x="file_short", hue="class", ax=axes[1, 1])
    axes[1, 1].set_title("Class Distribution per Sample")
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "overview_panels.png"), dpi=300)
    plt.close()

    # 2. KDE per sample
    plt.figure(figsize=(10, 6))

    sns.kdeplot(
        data=final_cilia_df,
        x="log_ratio",
        hue="file_short",
        common_norm=False,
    )

    plt.title("Log Ratio Distribution per Sample")
    plt.xlabel("Log Ratio")

    plt.savefig(os.path.join(fig_dir, "logratio_kde_per_sample.png"), dpi=300)
    plt.close()

    # QC + Excel
    excel_path = os.path.join(csv_dir, "all_cilia_features.xlsx")

    qc_sample = final_cilia_df.groupby("file_short").agg(
        n_cilia=("cilia_id", "count"),
        #n_basal_bodies=("basal_body_id", "count"),
        mean_ratio=("ratio", "mean"),
        std_ratio=("ratio", "std"),
        mean_log_ratio=("log_ratio", "mean"),
        std_log_ratio=("log_ratio", "std"),
        mean_distance=("distance_to_neurite_um", "mean"),
        max_distance=("distance_to_neurite_um", "max"),
    ).reset_index()

    qc_global = pd.DataFrame({
        "metric": [
            "n_total_cilia",
            "mean_ratio",
            "std_ratio",
            "mean_log_ratio",
            "std_log_ratio"
        ],
        "value": [
            len(final_cilia_df),
            final_cilia_df["ratio"].mean(),
            final_cilia_df["ratio"].std(),
            final_cilia_df["log_ratio"].mean(),
            final_cilia_df["log_ratio"].std(),
        ]
    })

    with pd.ExcelWriter(excel_path) as writer:
        final_df.to_excel(writer, sheet_name="all_data", index=False)
        qc_sample.to_excel(writer, sheet_name="qc_per_sample", index=False)
        qc_global.to_excel(writer, sheet_name="qc_global", index=False)

    print(f"Saved QC Excel: {excel_path}")


    # Scatter plot (the moment of truth)
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        x='dt_neurite',
        y='dt_nuclei',
        hue='file_short',
        data=final_cilia_df,
        palette='tab10'
    )

    plt.xlabel("Assigned Neurite Voxel thickness (um)")
    plt.ylabel("Assigned Neurite Voxel Distance to nuclei (um)")
    plt.title("DT Comparison per Sample")
    plt.legend(bbox_to_anchor=(0.5, 1), loc='upper left')

    fig_dir = os.path.join(output_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.savefig(os.path.join(fig_dir, "dt_scatterplot.png"), dpi=300)
    plt.close()

    # Scatter plot (the moment of truth)
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        x='log_dt_neurite',
        y='log_dt_nuclei',
        hue='file_short',
        data=final_cilia_df,
        palette='tab10'
    )

    plt.xlabel("Log of Assigned Neurite Voxel thickness (um)")
    plt.ylabel("Log of Assigned Neurite Voxel Distance to nuclei (um)")
    plt.title("Log DT Comparison per Sample")
    plt.legend(bbox_to_anchor=(0.5, 1), loc='upper left')

    fig_dir = os.path.join(output_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.savefig(os.path.join(fig_dir, "log_dt_scatterplot.png"), dpi=300)
    plt.close()


    print(f"Figures saved in: {fig_dir}")
    print("Pipeline complete.")