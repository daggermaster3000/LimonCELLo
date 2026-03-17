Quick context

This repository contains research / analysis code for processing 3D/4D Imaris (.ims) microscopy data, with a focus on DT neuron / cilia analysis pipelines.

The project is transitioning from notebook-based prototypes to a production-ready CLI tool with optional GUI, enabling reproducible, batch-friendly analysis in a lab setting.

Top-level layout (relevant folders/files):

data/ - dataset files (raw .ims and supporting metadata .txt)

notebooks/ - prototype pipelines (*.ipynb) used as reference implementations

preprocessing/ - reusable preprocessing functions

segmention/ - segmentation code (yes, still misspelled)

features/ - feature extraction logic (distances, associations, stats)

visualization/ - plotting and figure generation

utils/reader.py - .ims loader + metadata parser

cli/ - NEW: command-line interface entrypoints

gui/ - NEW (optional): GUI wrapper for non-technical users

outputs/ - generated CSVs, figures, logs

Big-picture architecture & dataflow
Pipeline (canonical flow)
.ims → load → preprocess → segment → associate (DT neurons ↔ cilia)
    → filter → extract features → export CSV + figures
Input

.ims files (shape: (T, C, Z, Y, X))

metadata .txt files (auto-discovered)

Output

CSV files

per-object features (cilia, neurites)

association tables (which cilia belong to which neurite)

experiment-level summaries

Figures

distributions (distance, class assignment, etc.)

QC plots (segmentation quality, filtering)

Optional

Napari layers (for visual inspection)

Transition: notebooks → CLI tool

The two existing notebooks define the reference pipelines.

Rule:

Notebooks are NOT production code. They are specifications.

Required refactor:

Extract logic into reusable modules:

preprocessing/

segmention/

features/

visualization/

Keep notebooks as:

validation

experimentation

documentation

CLI design (core requirement)

Implement a CLI tool to run pipelines in a user-friendly, reproducible way.

Entry point
python -m cli.run_pipeline \
    --input data/ \
    --output outputs/ \
    --pipeline dt_neuron \
    --group-by filename \
    --config config.yaml
Required features
1. Batch processing

Accept:

single file OR folder

Automatically process all .ims files

2. Condition grouping (IMPORTANT)

Infer experimental condition from filename

Example:

ctrl_fish1.ims → group: ctrl
mut_fish3.ims  → group: mut

Implementation:

Regex or split-based parsing

Configurable via:

grouping:
  mode: filename
  pattern: "^(?P<group>[^_]+)_"
3. Config-driven pipelines

No hardcoding parameters in code.

Use YAML:

preprocessing:
  normalize: true
  p_low: 1
  p_high: 99

segmentation:
  method: threshold
  min_size: 50

association:
  max_distance_um: 10

filtering:
  max_distance_um: 15
4. Outputs

Each run must generate:

outputs/
  ├── csv/
  │   ├── features.csv
  │   ├── associations.csv
  │   └── summary.csv
  ├── figures/
  │   ├── distance_distribution.png
  │   └── class_distribution.png
  ├── logs/
  │   └── run.log
5. Logging

Use structured logging

Log:

parameters

processed files

warnings (e.g. “cilia too far removed”)

GUI mode (optional but required feature)

Because not everyone enjoys terminals.

Requirements

Launch via:

python -m gui.app

Features:

file/folder selection

parameter input (forms)

run pipeline button

progress display

basic visualization (optional Napari integration)

Suggested stack

napari (for visualization)

PyQt or tkinter (lightweight UI)

Project-specific conventions and gotchas
Data shape

Always assume: (T, C, Z, Y, X)

Do not silently reshape

Channel handling
get_channel(img, channel=0, timepoint=0) → (Z, Y, X)
Distance units

Always convert to µm using metadata

Never mix pixel distances and physical distances

Filtering (critical for DT analysis)

Remove:

cilia too far from neurites

noise objects

Filtering must be:

configurable

logged

Metadata fallback

Defaults to 1.0 µm

This is dangerous, log a warning when used

The “segmention” situation

Folder is misspelled

Options:

keep it (safe)

or rename + provide compatibility layer

Best practices (non-negotiable if you want this to survive lab use)
1. Reproducibility

Every run must be reproducible from:

input data

config file

2. No hidden parameters

If it matters, it goes in config

3. Deterministic outputs

Same input = same output

4. Modular design

Bad:

notebook_cell_everything()

Good:

segment()
associate()
filter()
export()
5. Testing (yes, even in academia)

Add tests/

At least test:

loading

preprocessing

distance computation

6. Documentation

Each pipeline step:

inputs

outputs

assumptions

Integration examples
Load + preprocess
from utils.reader import load_image
from preprocessing.preprocessing import percentile_minmax_normalize

img, meta = load_image("data/sample.ims")
norm = percentile_minmax_normalize(img)
CLI usage
python -m cli.run_pipeline \
    --input data/experiment1 \
    --output outputs/exp1 \
    --config config.yaml
What to look for when making changes

Does it break the TCZYX assumption?

Are units still in µm?

Is the parameter exposed in config?

Does it work on multiple files, not just your favorite fish?

Minimal guidance for AI agents

Treat notebooks as reference only

Implement reusable logic in:

preprocessing/

segmention/

features/

Build CLI in cli/

Ensure outputs:

CSV

figures

Never hardcode experimental assumptions

Always prefer config-driven design