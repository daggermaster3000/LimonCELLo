import os
os.environ["PYOPENCL_NO_CACHE"] = "1"
import streamlit as st
import os
import pandas as pd
import sys
import time
from limoncello.analysis.pipeline import run_pipeline3

st.set_page_config(layout="wide")

# ---- TITLE ----
st.title("Limoncello 🍋")
st.markdown(
"""
Interactive interface for running the cilia–neurite analysis pipeline.

Adjust parameters in the sidebar, run the pipeline, and explore results.
"""
)

# ---- SIDEBAR ----
st.sidebar.header("⚙️ Pipeline Settings")

input_path = st.sidebar.text_input(
    "Input folder",
    help="Folder containing .ims files to process"
)

output_path = st.sidebar.text_input(
    "Output folder",
    value="tutorial/output",
    help="Results (figures, CSVs, Excel) will be saved here"
)

classifier_path = st.sidebar.text_input(
    "Cilia classifier",
    value=r"segmenters\cilia-segmenter.cl",
    help="Path to trained cilia segmentation model"
)

st.sidebar.markdown("---")

st.sidebar.subheader("Distance thresholds")
max_cilia = st.sidebar.slider(
    "Max cilia distance (µm)",
    0.5, 10.0, 2.0, 0.1,
    help="Maximum allowed distance from neurite for cilia to be considered valid"
)

max_basal = st.sidebar.slider(
    "Max basal body distance (µm)",
    0.5, 10.0, 2.0, 0.1,
    help="Same as above, but for basal bodies"
)

st.sidebar.markdown("---")

st.sidebar.subheader("Segmentation parameters")

nuclei_sigma = st.sidebar.slider(
    "Nuclei spot sigma",
    1, 50, 15,
    help="Controls smoothing scale for nuclei detection (higher = larger structures)"
)

tophat_radius = st.sidebar.slider(
    "Tophat radius",
    1, 50, 12,
    help="Background subtraction radius for nuclei segmentation"
)

neurite_sigma = st.sidebar.slider(
    "Neurite spot sigma",
    1, 20, 5,
    help="Scale for neurite detection (affects skeletonization)"
)

# ---- RUN BUTTON ----
run_clicked = st.sidebar.button("🚀 Run pipeline")

# ---- MAIN LAYOUT ----
col1, col2 = st.columns([2, 1])

# LEFT: execution + logs
with col1:
    st.subheader("Execution")

    if run_clicked:
        if not os.path.exists(input_path):
            st.error("Input path does not exist.")
        else:
            with st.spinner("Running pipeline... go hydrate"):
                
                try:
                    run_pipeline3(
                        input_path=input_path,
                        output_path=output_path,
                        max_cilia_dist_cutoff_um=max_cilia,
                        max_basal_body_cutoff_um=max_basal,
                        nuclei_spot_sigma=nuclei_sigma,
                        tophat_radius=tophat_radius,
                        neurite_spot_sigma=neurite_sigma,
                        cilia_classifier_path=classifier_path,
                    )
                    st.success("Pipeline complete 🎉")
                except Exception as e:
                    st.error(f"Something broke: {e}")

# RIGHT: results explorer
with col2:
    import glob
from PIL import Image

st.subheader("📊 Results Explorer")

csv_dir = os.path.join(output_path, "csv")
fig_dir = os.path.join(output_path, "figures")
overlay_dir = os.path.join(fig_dir, "overlays")

tabs = st.tabs(["📊 Data", "📈 Figures", "🔬 Overlays"])

# -----------------------
# 📊 DATA TAB
# -----------------------
with tabs[0]:
    excel_path = os.path.join(csv_dir, "all_cilia_features.xlsx")

    if os.path.exists(excel_path):
        sheet = st.selectbox(
            "Select dataset",
            ["all_data", "qc_per_sample", "qc_global"]
        )

        df = pd.read_excel(excel_path, sheet_name=sheet)

        st.dataframe(df, width='stretch')

        st.download_button(
            "⬇️ Download Excel",
            data=open(excel_path, "rb"),
            file_name="results.xlsx"
        )

        # quick plotting
        # if "log_ratio" in df.columns:
        #    st.subheader("Quick plot")
        #    st.line_chart(df["log_ratio"])

    else:
        st.info("No data found yet.")

# -----------------------
# 📈 FIGURES TAB
# -----------------------
with tabs[1]:
    if os.path.exists(fig_dir):
        figures = glob.glob(os.path.join(fig_dir, "*.png"))

        if figures:
            selected_fig = st.selectbox(
                "Select figure",
                figures,
                format_func=lambda x: os.path.basename(x)
            )

            st.image(selected_fig, width='stretch')

        else:
            st.info("No figures found.")

# -----------------------
# 🔬 OVERLAYS TAB
# -----------------------
with tabs[2]:
    if os.path.exists(overlay_dir):
        overlays = glob.glob(os.path.join(overlay_dir, "*.png"))

        if overlays:
            selected_overlay = st.selectbox(
                "Select overlay",
                overlays,
                format_func=lambda x: os.path.basename(x)
            )

            st.image(selected_overlay, width='stretch')

        else:
            st.info("No overlays found.")

    else:
        st.info("Overlay folder not found.")