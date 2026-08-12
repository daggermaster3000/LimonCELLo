# Batch & Super-batch

[← Home](Home.md)

Once the parameters look right on a test image, process many images at once. All
batch controls are in the **Batch** section of the [Pipeline app](03-Pipeline-App.md).

![Batch section](images/batch-section.png)

---

## Batch (one folder)

Set an **Output folder**, then click **⚡ Run BATCH (whole folder)**. Every `.ims`
in the **Input folder** (or listed in a `.txt` manifest) is processed with the
current parameters.

**Output:** a timestamped `lc-analysis-<date>_<time>/` folder containing:

```
lc-analysis-2026-08-11_10-58-53/
├─ csv/
│  ├─ all_cilia_features.xlsx   ← per-object table (the main result)
│  ├─ nuclei_summary.csv        ← per-sample nuclei volume/count (ciliation rate)
│  ├─ run_parameters.json       ← every parameter, for reproducibility
│  └─ run_label.txt             ← default label (= input folder name)
└─ figures/
   ├─ overlays/                 ← per-sample overview panels
   ├─ cilia_rois/               ← per-cilium RGB thumbnails + raw .npz crops
   ├─ mips/                     ← per-channel XY MIP arrays (.npy)
   └─ napari_overlays/          ← 3-D screenshots (if live capture is on)
```

### Batch options

| Option | Effect |
|--------|--------|
| **Capture overlay screenshots during batch** | Show each file in napari and save a screenshot (slower). |
| **Save per-cilium ROI thumbnails + crops** | Export `cilia_rois/` for screening / AI. |
| **Correct/normalise ROI display** | Contrast-normalise ROI thumbnails (off = raw intensities). |
| **AI-validate cilia during batch** | Score each cilium's ROI with a CNN → `human_validation.csv`. See [AI validator](07-AI-Cilia-Validator.md). |
| **ROI-only fast batch** | ⚠️ Segment cilia + BB and export ROIs **only** — **skips** neurites/nuclei/distances/ratio/classification. Use this for building a labelling set, **not** for analysis. |

> **⚠️ Common gotcha — "where did my ratio go?"**
> A **ROI-only** run produces a minimal table with **no `ratio`, `log_ratio`,
> `class`, or distance columns**. If the data app shows blank ratios, the run was
> ROI-only. Re-run with **ROI-only fast batch UNCHECKED** for the full analysis.

---

## Super-batch (a tree of folders)

Data is usually organised as **day / coverslip**:

```
super-batch-1/
├─ d11/cv1/*.ims
├─ d11/cv2/*.ims
├─ d14/cv1/*.ims
└─ d38/40x FOVs/coverslip 1/*.ims
```

**🗂️ Run SUPER BATCH (all subfolders)** walks the **Input folder** recursively,
finds every subfolder that directly contains `.ims`, and processes each one as its
own run — **mirroring the input tree under the Output folder** so names are kept:

```
Input:   super-batch-1/d11/cv2/*.ims
Output:  <out>/d11/cv2/lc-analysis-<timestamp>/…
```

- Nested paths are preserved (`d38/40x FOVs/coverslip 1` → same under output).
- Reuses **every** batch option above (full vs ROI-only, capture, AI).
- Progress shows `[folder i/N] <subpath> — <file>`.
- Manifest (`.txt`) input is **not** supported for super-batch — use a real folder.

### Re-running super-batch

A re-run writes a **fresh** `lc-analysis-*` folder *inside* each mirror folder. If
an earlier ROI-only run left a flat `csv/` there too, the data app's discovery
**supersedes the stale one** and uses the newest nested run automatically — you
don't need to delete anything. (See [Data app](06-Data-App.md#run-discovery).)

---

## Loading super-batch results

Point the [Data app](06-Data-App.md) at the top of the output tree (e.g.
`…/super-batch-1`) and click **Add ALL found runs**. Runs are keyed by their path
(`d11/cv2/…`), and the app derives **day** and **coverslip** columns so you can
compare **between days** while still splitting/colouring by **coverslip**.

---

**Next:** [Annotator app →](05-Annotator-App.md)
