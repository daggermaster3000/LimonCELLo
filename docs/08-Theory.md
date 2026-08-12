# Theory

[← Home](Home.md)

How LimonCELLo decides whether a cilium sits on a **neurite** or on the **soma**,
and how it links each cilium to its basal body.

All physical quantities use the anisotropic voxel size `(vz, vy, vx)` in µm read
from the `.ims` metadata.

---

## 1. Preprocessing

- **Percentile normalisation** — each channel is min–max scaled between its
  `p_low`/`p_high` percentiles (default 2–98), clipping hot/cold outliers.
- **MIP** — maximum-intensity Z-projection → a single 2-D plane per channel. Fast,
  loses depth.
- **Make isotropic** — anisotropic acquisitions (fine XY, coarse Z) are resampled
  so every axis has the same spacing (the coarsest, usually Z). This makes 3-D
  distances and shapes physically meaningful. The result is cached on disk.

![Isotropic vs MIP](images/theory-iso-mip.png)

---

## 2. Segmentation (APOC)

Cilia and (optionally) basal bodies / nuclei are segmented with **APOC**
`ObjectSegmenter`s — random-forest pixel classifiers trained on a few hand-painted
labels, saved as `.cl` files. They run on the GPU via pyclesperanto.

Nuclei / basal bodies can alternatively use **Voronoi–Otsu** labelling
(tophat background removal + spot/outline σ), a classical, training-free method.

Every label image is **size-gated** (min/max voxel count) to drop noise and merged
blobs. See [Training APOC segmenters](09-Training-APOC.md).

---

## 3. Distance & ratio maps

The classification signal is **geometric**: a cilium on a neurite is *far* from
nuclei and *inside/near* a neurite; a cilium on the soma is *close* to a nucleus.

From the segmentations we compute Euclidean distance transforms (EDT, in µm):

- `d_nuclei(x)` — distance from voxel `x` to the nearest nucleus.
- `d_neurite(x)` — distance **inside** the neurite mask (a thickness proxy).

and a **regularised ratio**, with a small `ε` to avoid divide-by-zero and damp
noise where both distances are tiny:

```
ratio(x)     = (d_nuclei(x) + ε) / (d_neurite(x) + ε)
log_ratio(x) = log(ratio(x))
```

- **High `log_ratio`** → far from nuclei, near/among neurites → **neurite-like**.
- **Low `log_ratio`** → close to a nucleus → **soma-like**.

We also keep the distance to the nearest neurite and the index of the nearest
skeleton voxel (for the overlay lines).

![Ratio map](images/theory-ratio-map.png)

---

## 4. Feature assignment & classification

For each cilium (and basal body) we take its **centroid**, sample the maps there,
and attach 3-D **shape descriptors** (volume, sphericity, length, elongation, …).

Classification is a two-threshold cut on `log_ratio`:

```
neurite    if log_ratio > neurite_threshold
soma       if log_ratio < soma_threshold
ambiguous  otherwise
```

The gap between the two thresholds is the **ambiguous band** — cilia the geometry
can't confidently place. Widen it to be conservative, narrow it to force a call.

---

## 5. Cilia ↔ basal-body pairing

A cilium grows from a basal body, so a valid cilium should have a basal body
nearby. Pairing both **validates** detections and lets a cilium inherit its BB's
(often cleaner) ratio.

### Nearest (default)
Every `(cilium, basal body)` distance is sorted ascending and assigned **greedily,
closest first, strict 1:1**. Pairs beyond `max BB dist (µm)` are rejected.

### Axis-directed
A basal body sits at the **base** of a cilium, roughly on its long axis. For each
cilium we take its **major axis** (PCA of the label voxels) and, seeding from each
extremity, search a **cone** (half-angle `cone_angle_deg`, length
`search_dist_um`) pointing outward along the axis. The nearest basal body inside
the cone is the directional candidate; candidates are assigned closest-first,
strict 1:1 (`pairing_method = "axis"`). A cilium with no axis hit **falls back** to
the nearest rule, so it never pairs fewer cilia than the default.

![Pairing](images/theory-pairing.png)

- **Require basal body** on → only paired cilia survive (strict, fewer false
  positives).
- **ratio from basal body** on → the cilium's ratio is taken at the BB position,
  then re-classified, so `class` reflects the base's location.

---

## 6. Outputs

Per object: centroid, `ratio`, `log_ratio`, `class`, distances, shape descriptors,
pairing info, and the XY bounding box. Aggregated into
`csv/all_cilia_features.xlsx` and explored in the [Data app](06-Data-App.md).

---

**Next:** [Training APOC segmenters →](09-Training-APOC.md)
