# LimonCELLo — DT neuron & cilia analysis

Short and practical guide to the repository, how to run the pipeline (CLI and GUI), and developer notes.

Overview
--------
LimonCELLo contains analysis code for processing 3D/4D Imaris (.ims) microscopy
datasets, focusing on associating cilia to neurites and extracting per-object
features. The repository is organized so notebooks hold experiments and the
library modules provide reusable building blocks.

Key directories
- `data/` — example or local `.ims` files (not included here).
- `preprocessing/` — preprocessing helpers (e.g. percentile normalization).
- `segmentation/` — segmentation code.
- `analysis/` — analysis helpers (e.g. cilia association logic).
- `utils/` — I/O and exporter helpers (`reader.py`).

Quick start (recommended)
-------------------------
1. Create a venv and activate (PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
```

2. Install runtime dependencies:

```powershell
pip install numpy scipy pandas matplotlib
# Optional (GPU path + advanced segmentation/filters):
pip install pyclesperanto_prototype scikit-image pyopencl
```

3. Run the CLI on a folder of `.ims` files:

```powershell
python -m cli.run_pipeline --input path\to\ims_folder --output outputs --config config.yaml --group-by filename
```

4. Or run the GUI:

```powershell
python -m gui.app
```


