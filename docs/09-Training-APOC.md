# Training APOC segmenters

[← Home](Home.md)

APOC segmenters are random-forest classifiers saved as `.cl` files. LimonCELLo
ships several (see `segmenters/`), but your microscope, magnification and staining
may differ — training your own usually beats tuning someone else's.

> **APOC does not transfer across pixel sizes.** A classifier trained at
> 0.152 µm/px (40×) will produce garbage at 0.305 µm/px. Train per acquisition
> scale, or use the batch **Classifier XY µm** filter to skip mismatched images.

All of this lives in the **🎓 training section** of the
[Pipeline app](03-Pipeline-App.md).

![Training section](images/training-section.png)

---

## Quick path: train on the current image

1. **① Annotate** — click **new label layer**, then paint with the napari brush:
   paint over a few **objects** (the thing you want, e.g. cilia) and some
   **background**. You don't need to label everything — a representative handful.
2. Pick that Labels layer in **train annotation**, and set **train channel** to the
   raw channel you're segmenting.
3. **② Features** — choose the Gaussian **sigmas** (feature scales) and whether to
   include the original intensities. Bigger sigmas capture bigger structures.
4. **③ Model** — set which label value is **positive** (the object), tree
   **max depth** and **num trees**, and an **output `.cl`** path.
5. **Train (current image)** → the classifier trains and you can **Predict** to
   preview the result on the current image. Iterate: add labels where it's wrong,
   retrain.

**Continue training** keeps the existing model and refines it with the new labels
instead of starting over.

---

## Batch training across many images

To train on labels drawn across a whole set:

1. Save each image's painted labels with **save labels** — they're written as
   `<stem>_labels.tif` into the **Labels folder**.
2. Set the **Labels folder** and (optionally) tick **use all labeled images** or
   pick a subset in **train images**.
3. **🎓 Batch-train from saved label files** → one classifier learned from every
   `<stem>_labels.tif` / image pair.

This is the robust route for a production segmenter: it sees variation across
samples, not just one field of view.

---

## Feature importances

After training, the app can show which features (channel × sigma × operation) the
forest relied on — useful for pruning the feature set to something faster that
generalises better.

---

## Tips

- **Balance your labels** — a few clear background strokes matter as much as object
  strokes; all-object labels overfit.
- **Match sigmas to size** — thin cilia need small sigmas; big nuclei need larger.
- **Size-gate afterwards** — set min/max voxel size in the segment step to remove
  specks and merged blobs the classifier can't fix.
- **Keep scale constant** — one `.cl` per magnification.

---

**Next:** [Troubleshooting →](10-Troubleshooting.md)
