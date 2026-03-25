# qc.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_qc_excel(final_df: pd.DataFrame, csv_dir: str, excel_name: str = "all_cilia_features.xlsx"):
    """
    Save QC data to an Excel file with three sheets:
        - all_data
        - qc_per_sample
        - qc_global
    """
    excel_path = os.path.join(csv_dir, excel_name)

    # QC per sample
    qc_sample = final_df.groupby("file_short").agg(
        n_cilia=("cilia_id", "count"),
        mean_ratio=("ratio", "mean"),
        std_ratio=("ratio", "std"),
        mean_log_ratio=("log_ratio", "mean"),
        std_log_ratio=("log_ratio", "std"),
        mean_distance=("distance_to_neurite_um", "mean"),
        max_distance=("distance_to_neurite_um", "max"),
    ).reset_index()

    # Global QC
    qc_global = pd.DataFrame({
        "metric": [
            "n_total_cilia",
            "mean_ratio",
            "std_ratio",
            "mean_log_ratio",
            "std_log_ratio"
        ],
        "value": [
            len(final_df),
            final_df["ratio"].mean(),
            final_df["ratio"].std(),
            final_df["log_ratio"].mean(),
            final_df["log_ratio"].std(),
        ]
    })

    # Write to Excel
    with pd.ExcelWriter(excel_path) as writer:
        final_df.to_excel(writer, sheet_name="all_data", index=False)
        qc_sample.to_excel(writer, sheet_name="qc_per_sample", index=False)
        qc_global.to_excel(writer, sheet_name="qc_global", index=False)

    print(f"Saved QC Excel: {excel_path}")
    return excel_path, qc_sample, qc_global

def scatter_plot(
    final_df: pd.DataFrame,
    output_path: str,
    x: str,
    y: str,
    title: str,
    filename: str,
    hue: str = "file_short",
    palette: str = "tab10",
    figsize=(10, 6),
):
    """
    Create and save a scatterplot.
    """
    fig_dir = os.path.join(output_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=figsize)
    sns.scatterplot(x=x, y=y, hue=hue, data=final_df, palette=palette)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.title(title)
    plt.legend(bbox_to_anchor=(0.5, 1), loc='upper left')

    save_path = os.path.join(fig_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure: {save_path}")
    return save_path, fig_dir

def run_qc(final_df: pd.DataFrame, csv_dir: str, output_path: str):
    """
    Run full QC: Excel + scatter plots.
    """
    # Excel
    excel_path, qc_sample, qc_global = save_qc_excel(final_df, csv_dir)

    # Scatter plots
    scatter_plot(final_df, output_path,
                 x='dt_neurite',
                 y='dt_nuclei',
                 title='DT Comparison per Sample',
                 filename='dt_scatterplot.png')

    scatter_plot(final_df, output_path,
                 x='log_dt_neurite',
                 y='log_dt_nuclei',
                 title='Log DT Comparison per Sample',
                 filename='log_dt_scatterplot.png')

    print("Pipeline complete.")
    return excel_path, qc_sample, qc_global

def run_qc(final_df: pd.DataFrame, csv_dir: str, output_path: str):
    """
    Run full QC: Excel + scatter plots.
    """
    # Excel
    excel_path, qc_sample, qc_global = save_qc_excel(final_df, csv_dir)

    # Scatter plots
    scatter_plot(final_df, output_path,
                 x='dt_neurite',
                 y='dt_nuclei',
                 title='DT Comparison per Sample',
                 filename='dt_scatterplot.png')

    scatter_plot(final_df, output_path,
                 x='log_dt_neurite',
                 y='log_dt_nuclei',
                 title='Log DT Comparison per Sample',
                 filename='log_dt_scatterplot.png')

    print("Pipeline complete.")
    return excel_path, qc_sample, qc_global