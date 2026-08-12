# Troubleshooting

[← Home](Home.md)

---

## `DLL load failed` / `0xc06d007f` crashes

**Symptoms**
- `ImportError: DLL load failed while importing defs` (from `h5py`).
- A hard crash with exit code `0xc06d007f` inside `scipy.linalg` / `skimage`.

**Cause:** a **pip** wheel was installed over a **conda** build of a native package,
so the Python extension and its DLLs mismatch (HDF5 for `h5py`; the OpenBLAS/MKL
runtime for `numpy`/`scipy`).

**Fix — align each package to a single source:**

```powershell
# h5py: remove the pip copy, reinstall from conda so it matches the conda hdf5
python -m pip uninstall -y h5py
conda install -n <env> -y "h5py=3.16.0"

# scipy: repair the vendored OpenBLAS by reinstalling the wheel cleanly
python -m pip install --force-reinstall --no-deps "scipy==1.15.2"
```

Then verify:

```powershell
python -c "import h5py, numpy, scipy, skimage.io, napari; print('OK')"
```

> Prefer creating the environment from `env.yml` in the first place to avoid the
> pip/conda split. See [Installation](01-Installation.md).

---

## `python napari_app.py` opens the wrong Python

On Windows, `python` often resolves to the **Windows Store stub**, not your conda
env. Run with the env's interpreter explicitly:

```powershell
& "C:\Users\<you>\.conda\envs\<env>\python.exe" napari_app.py
```

or activate the env first (e.g. via Anaconda Prompt).

---

## Only "CPU" shows in the GPU dropdown

OpenCL/CUDA isn't visible to Python.

```powershell
python -c "import pyclesperanto_prototype as cle; print(cle.available_device_names())"
```

Update your GPU driver, install the vendor OpenCL runtime, and confirm the device
appears. CPU works but is much slower.

---

## Batch skips a large image / GPU out of memory

3-D segmentation is capped (~`4e8` voxels/channel) to avoid a VRAM allocation
failure that would abort the whole batch. Large stitched images are skipped with a
message. **Options:** enable **MIP** mode, downsample, or tune the cap for your GPU.

---

## "Where did my ratio values go?" (blank ratio/class)

The run was a **ROI-only fast batch** — it exports ROIs and skips
neurites/nuclei/distances/ratio/classification. Re-run with **ROI-only fast batch
UNCHECKED**. See [Batch & Super-batch](04-Batch-and-Super-Batch.md).

---

## Data app is slow to load a super-batch

First load reads every `.xlsx` (slow) and writes a `csv/all_data.parquet` sidecar;
later loads use the parquet and are ~15× faster. If it's slow every time, the
output folder is likely **read-only** (parquet can't be written) — copy it
somewhere writable, or accept the slower workbook path.

---

## Data app shows an old ROI-only run after a full re-run

Fixed in discovery: when a folder holds both a stale flat run and a fresh nested
`lc-analysis-*` run, the newest one is used. If you still see the old one, **clear
the run list and re-add** (the stale entry was cached in the session by its old
path), or restart the app.

---

## Same coverslip appears merged across days

Runs are keyed by their path (`d11/cv2`), and **day**/**coverslip** columns are
derived from it. If you renamed runs to a bare name like `cv2`, different days'
`cv2` will collide — keep the path-based labels, or make labels unique.

---

[← Home](Home.md)
