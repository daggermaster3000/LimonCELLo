# Installation

[← Home](Home.md)

LimonCELLo runs on **Windows / Linux** with Python **3.10**. A GPU (NVIDIA, CUDA)
is strongly recommended — segmentation uses [pyclesperanto] and the AI validator
uses PyTorch.

[pyclesperanto]: https://github.com/clEsperanto/pyclesperanto_prototype

---

## Recommended: conda environment

The repo ships an `env.yml`. This is the most reliable route on Windows because it
pins native libraries (HDF5, OpenCL, BLAS) that pip alone often mismatches.

```powershell
git clone https://github.com/daggermaster3000/LimonCELLo.git
cd LimonCELLo
conda env create -f env.yml -n napari-env
conda activate napari-env
```

> **Windows tip:** run apps with the environment's Python explicitly if `conda
> activate` is not on your PATH:
> `& "C:\Users\<you>\.conda\envs\napari-env\python.exe" napari_app.py`

---

## Alternative: pip

```powershell
pip install numpy scipy pandas matplotlib pyclesperanto_prototype scikit-image \
            apoc imaris_ims_file_reader seaborn tqdm skan stackview
# GUIs:
pip install napari[all] magicgui superqt
# Data app:
pip install streamlit plotly openpyxl pyarrow
# AI cilia validator (optional, GPU recommended):
pip install torch pillow
```

> ⚠️ **Do not mix pip and conda for `h5py` / `numpy` / `scipy`.** A pip wheel
> installed over a conda build causes `DLL load failed` (h5py) or a
> `0xc06d007f` crash in `scipy.linalg` (BLAS mismatch). See
> [Troubleshooting](10-Troubleshooting.md#dll-load-failed--0xc06d007f-crashes).

---

## Verify the install

```powershell
python -c "import h5py, numpy, scipy, skimage, napari, pyclesperanto_prototype; print('OK')"
```

Check the GPU is seen:

```powershell
python -c "import pyclesperanto_prototype as cle; print(cle.available_device_names())"
```

The apps also list detected GPUs in a dropdown; if only *CPU* appears, OpenCL/CUDA
is not visible to Python.

---

## What each dependency is for

| Package | Used by |
|---------|---------|
| `pyclesperanto_prototype` | GPU segmentation, distance transforms |
| `apoc` | trainable pixel/object classifiers (`.cl` files) |
| `imaris_ims_file_reader`, `h5py` | reading `.ims` |
| `scikit-image`, `scipy` | shape properties, EDT, skeletons |
| `napari`, `magicgui`, `superqt` | the desktop GUIs |
| `streamlit`, `plotly`, `openpyxl`, `pyarrow` | the data app |
| `torch`, `pillow` | the AI cilia validator |

---

**Next:** [Quickstart →](02-Quickstart.md)
