# Pipeline app (`napari_app.py`)

[← Home](Home.md)

The pipeline app is where you **tune parameters on one image**, curate the
segmentation by hand, and then run a **batch**. It's a napari window with a docked
control panel and a bottom "Cilia properties" table.

```powershell
python napari_app.py
```

![Pipeline app layout](images/pipeline-app-annotated.png)
*① Workflow · ② Step buttons · ③ Parameter sections · ④ Batch · ⑤ Cilia table.*

---

## Workflow section

| Control | Meaning |
|---------|---------|
| **Input folder** | Folder of `.ims`, **or** a `.txt` manifest (one `.ims` path per line). |
| **🔍 Scan folder** | Lists `.ims` files into the dropdown. |
| **File dropdown / ◀ ▶** | Choose / step through images. |
| **GPU** | OpenCL device for segmentation. NVIDIA auto-preferred. |

Parameters persist between sessions (saved to `.napari_app_state.json` on quit).

---

## The steps

Each button runs one pure-compute step and refreshes only the layers it produces,
so you can iterate on a single step without redoing the rest.

### 1. Load
Reads the `.ims` `(C, Z, Y, X)`, does per-channel percentile normalisation, and
optionally **MIP** (Z-projection → 2-D) or **make isotropic** (resample to the
coarsest voxel spacing). The isotropic result is cached on disk, so re-loading
after changing only a *segmentation* parameter is instant.

- **Channels** — which channel index is Cilia / Neurites / Basal Bodies / Nuclei.
- **MIP** vs **make isotropic** — pick one. MIP is fast and 2-D; isotropic keeps
  3-D but resamples anisotropic Z.
- **Normalisation** `p_low` / `p_high` — global percentile clip (default 2–98).
  Per-channel overrides are available.

### 2. Segment cilia
APOC `ObjectSegmenter` (`.cl` classifier) on the **raw** cilia channel, size-gated.

- **Classifier** — path to the cilia `.cl` (see [Training APOC](09-Training-APOC.md)).
- **min / max size** — voxel-count gate (max `0` = no upper limit).
- **log transform** — apply `log` before segmenting (helps high-dynamic-range).

**Manual curation:** **➕ Add cilium** (3-D brush), **🗑️ Delete cilium** (click a
label to remove), then **✅ Apply cilia edits** to fold changes back in. Re-run
*Assign & classify* to update.

### 3. Segment nuclei
**Voronoi–Otsu** (tophat + spot/outline σ) *or* an **APOC classifier**.

### 4. Segment neurites
Spot-sigma Voronoi labelling + skeletonisation. Optionally **merge nuclei** into
the neurite channel before skeletonising (useful when somata are dim).

### 5. Segment basal bodies
**Voronoi–Otsu** *or* an **APOC classifier**, size-gated (same UI as nuclei).

### 6. Distances
EDT (in µm, using the voxel size) to nuclei and inside neurites; the regularised
**ratio map** `(d_nuclei + ε)/(d_neurite + ε)` and its `log`; distance to the
nearest neurite; nearest-skeleton-voxel index. See [Theory](08-Theory.md).

### 7. Assign & classify
Per cilium / basal body, sample the distance features at the centroid and classify
from `log_ratio`:

- `neurite` if `log_ratio > neurite_thr`
- `soma` if `log_ratio < soma_thr`
- else `ambiguous`

Then **pair cilia ↔ basal bodies** and (optionally) inherit the paired BB's ratio
before re-classifying. Distance/pairing controls live in the **Distances /
classification** section:

| Control | Meaning |
|---------|---------|
| **max cilia dist (µm)** | feature-sampling cutoff for cilia |
| **max BB dist (µm)** | pairing cutoff; also the "validated" gate |
| **Require basal body** | keep only cilia paired to a BB within cutoff |
| **Axis-directed BB pairing** | search a cone along the cilium's long axis (see Theory) |
| **ratio from basal body** | take the cilium's ratio from its paired BB |
| **neurite / soma threshold** | the classification cut points on `log_ratio` |

---

## The cilia properties table

The bottom dock lists every kept cilium with its features. You can:

- **Edit the AI decision** — toggle the `ai_validated` checkbox per row.
- **📋 Copy for Excel** — whole table as TSV.
- **📦 Copy boxes** — per-cilium XY bounding boxes in the annotator `cilia_boxes`
  schema (reloads in the [Annotator](05-Annotator-App.md) / [Data app](06-Data-App.md)).
- **💾 Export .xlsx** — a `cilia_table` sheet + a `cilia_boxes` sheet.

Selecting a row centres the camera on that cilium (and vice-versa).

---

## Training APOC segmenters here

The app has a **🎓 training section** to paint labels and train `.cl` segmenters
directly on your images — see [Training APOC segmenters](09-Training-APOC.md).

---

**Next:** [Batch & Super-batch →](04-Batch-and-Super-Batch.md)
