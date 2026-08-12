# Quickstart

[← Home](Home.md)

Goal: go from a folder of `.ims` files to a classified per-cilium table you can
explore — in about ten minutes.

---

## 1. Launch the pipeline app

```powershell
python napari_app.py
```

![Pipeline app on launch](images/pipeline-app-launch.png)
*The napari window with the LimonCELLo dock on the right.*

## 2. Point it at your data

1. **Input folder** → the folder of `.ims` files.
2. Click **🔍 Scan folder**. The first file loads into the file dropdown.
3. Pick your **GPU** in the dropdown (NVIDIA preferred).

## 3. Set the channels

Tell it which channel is which (0-indexed) under **Channels**:
`Cilia`, `Neurites`, `Basal Bodies`, `Nuclei`. Toggle **MIP** (2-D) or
**make isotropic** (3-D) depending on your acquisition.

> Not sure which channel is which? Run **Load** and look at the raw layers in
> napari — toggle each `LC: Raw …` layer.

## 4. Run the steps on one image

Use the step buttons top-to-bottom and check each result in the viewer:

```
Load → Segment cilia → Segment nuclei → Segment neurites
     → Segment basal bodies → Distances → Assign & classify
```

Or hit **▶ Run all steps (this image)**. Cilia are coloured by `log_ratio`
(blue = soma-like, red = neurite-like). Tune parameters until the segmentation
looks right — see [Pipeline app](03-Pipeline-App.md) for what each knob does.

![Classified cilia overlay](images/quickstart-classified.png)

## 5. Batch the whole folder

Set an **Output folder**, then click **⚡ Run BATCH (whole folder)**. It writes a
timestamped `lc-analysis-*/` folder with the table, ROIs and overlays.

Have many day/coverslip subfolders? Use **🗂️ Run SUPER BATCH (all subfolders)** —
see [Batch & Super-batch](04-Batch-and-Super-Batch.md).

## 6. Explore the results

```powershell
streamlit run data_app.py
```

In the sidebar, paste your **Output folder** path → **Add ALL found runs** →
explore distributions, scatter, QC and screening. See [Data app](06-Data-App.md).

![Data app overview](images/dataapp-overview.png)

---

**Next:** [Pipeline app in detail →](03-Pipeline-App.md)
