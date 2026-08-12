# Annotator app (`napari_annotator_app.py`)

[← Home](Home.md)

The annotator produces **ground truth**: hand-drawn bounding boxes around cilia on
the XY MIP, each tagged `neurite` / `soma` / `uncertain`. This is what you use to
measure the pipeline's **recall** (cilia missed) and **classification accuracy**
(cilia mislabelled) in the [Data app](06-Data-App.md).

```powershell
python napari_annotator_app.py
```

![Annotator app](images/annotator-app.png)

---

## Workflow

1. **Open a folder** of `.ims` and step through images.
2. The XY **MIP** of the cilia (and basal-body) channel is shown.
3. Draw a **rectangle** around each cilium (a napari Shapes layer).
4. **Tag** each box `neurite`, `soma`, or `uncertain`.
5. **Export** an Excel of boxes.

## The `cilia_boxes` schema

Boxes are stored one row per cilium, in the same schema the pipeline app's
**📦 Copy boxes** / **💾 Export .xlsx** produce, so files interchange freely:

| column | meaning |
|--------|---------|
| `filename`, `box_id`, `class` | image, box index, hand tag |
| `y_min, x_min, y_max, x_max` | XY box on the MIP grid (inclusive max) |
| `img_height, img_width` | MIP dimensions |
| `voxel_z, voxel_y, voxel_x` | µm/voxel from the `.ims` |
| `mip`, `make_isotropic` | how the grid was produced |

The sheet is named `cilia_boxes`. Paste a copied TSV into a sheet with that name to
reload it in the annotator or the data app.

> **Vocabulary note:** the pipeline's `ambiguous` class maps to the annotator's
> `uncertain`, and they are scored as equal in the Data app's confusion matrix.

---

**Next:** [Data app →](06-Data-App.md)
