# figures.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def overview_panels(final_df: pd.DataFrame, output_path: str, filename: str = "overview_panels.png"):
    """
    Create a 2x2 figure:
        - Raw ratio distribution
        - Log ratio distribution
        - Log ratio per sample (boxplot)
        - Class distribution per sample (countplot)
    """
    fig_dir = os.path.join(output_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Raw ratio
    sns.histplot(final_df["ratio"], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Raw Ratio Distribution")

    # Log ratio
    sns.histplot(final_df["log_ratio"], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Log Ratio Distribution")

    # Log ratio per sample (boxplot)
    sns.boxplot(data=final_df, x="file_short", y="log_ratio", ax=axes[1, 0])
    axes[1, 0].set_title("Log Ratio per Sample")
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Class distribution per sample
    sns.countplot(data=final_df, x="file_short", hue="class", ax=axes[1, 1])
    axes[1, 1].set_title("Class Distribution per Sample")
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_path = os.path.join(fig_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved overview panels: {save_path}")
    return save_path, fig_dir


def logratio_kde_per_sample(final_df: pd.DataFrame, output_path: str, filename: str = "logratio_kde_per_sample.png"):
    """
    Plot KDE of log_ratio per sample.
    """
    fig_dir = os.path.join(output_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=final_df,
        x="log_ratio",
        hue="file_short",
        common_norm=False,
    )
    plt.title("Log Ratio Distribution per Sample")
    plt.xlabel("Log Ratio")

    save_path = os.path.join(fig_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved logratio KDE per sample: {save_path}")
    return save_path, fig_dir