"""Build the fixed 20-ROI study set for the Cilia Consensus PWA.

Scores every ROI thumbnail from the tutorial run with the ``cilialpha-3`` CNN,
then picks a *max-disagreement* set of 20 (same for every rater):

    5  clear cilia   (model P(cilia) > 0.90)
    5  clear NOT     (model P(cilia) < 0.10)
    5  borderline    (0.30 <= P <= 0.70, spread across the band)
    5  uncertain     (P closest to 0.50)

The chosen PNGs are copied into ``static/rois/`` under **opaque ids**
(``roi01.png`` … ``roi20.png``) so a rater can't infer the model's guess from
the filename, and the model's scores/labels are written to ``dataset.json``
(read only by the server for the dashboard — never sent to the labeling UI).

Run once, locally, before the talk::

    python cilia_study/build_dataset.py
"""
from __future__ import annotations

import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# The actual cilialpha-3 training ROIs. `cilia/` = human-labelled cilia (keep),
# `background/` = non-cilia junk (reject). We pool both so the "clear not" bucket
# is drawn from real junk while the doubtful cilia supply the borderline set.
_TRAIN = Path(r"C:\Users\qfavey\Documents\Thomaso\training_dataset\rois_training\labelling_set")
ROI_DIRS = [_TRAIN / "cilia", _TRAIN / "background"]
MODEL = REPO / "models" / "cilialpha-3.pt"
OUT_ROIS = HERE / "static" / "rois"
OUT_JSON = HERE / "dataset.json"
OUT_SEL = HERE / "selection.json"

POOL_SIZE = 200       # candidate ROIs the admin (Quillan) can choose from
DEFAULT_SELECTED = 15  # pre-selected most-ambiguous ROIs shown to the group
SEED = 20260713       # fixes the display order so it's identical for everyone


def _load_roi_validator():
    """Import ``limoncello.ml.roi_validator`` in isolation.

    The ``limoncello`` package ``__init__`` pulls in heavy optional deps
    (pyclesperanto, seaborn) that need not be installed just to score ROIs.
    ``roi_validator`` itself only needs numpy/torch/PIL and has no intra-package
    imports, so we load it straight from its file.
    """
    path = REPO / "limoncello" / "ml" / "roi_validator.py"
    spec = importlib.util.spec_from_file_location("_lc_roi_validator", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _model_label(p: float) -> str:
    if p >= 0.66:
        return "cilia"
    if p <= 0.33:
        return "not"
    return "uncertain"


def main() -> int:
    missing = [d for d in ROI_DIRS if not d.is_dir()]
    if missing:
        print(f"[!] ROI source(s) not found: {missing}")
        return 1
    if not MODEL.is_file():
        print(f"[!] Model bundle not found: {MODEL}")
        return 1

    rv = _load_roi_validator()
    print(f"[+] Loading model {MODEL.name} …")
    model, meta = rv.load_bundle(str(MODEL))
    size = int(meta.get("size", 64))
    normalize = meta.get("normalize")

    pngs = sorted(p for d in ROI_DIRS for p in d.glob("*.png"))
    print(f"[+] Scoring {len(pngs)} ROIs (size={size}, normalize={normalize}) …")
    paths = [str(p) for p in pngs]
    scores = rv.predict_proba(model, paths, size=size, normalize=normalize)

    # keep only the ones that scored
    items = [(p, float(s)) for p, s in zip(pngs, scores) if np.isfinite(s)]
    if len(items) < POOL_SIZE:
        print(f"[!] Only {len(items)} scored ROIs; need >= {POOL_SIZE}.")
        return 1
    items.sort(key=lambda t: t[1])                       # ascending P(cilia)
    ps = np.array([s for _, s in items])
    idx = np.arange(len(items))

    # POOL: sample evenly across the score range so the admin sees the full
    # spread (clear cilia, junk, and everything ambiguous in between) to pick from.
    sel = np.linspace(0, len(items) - 1, POOL_SIZE).round().astype(int)
    pool_idx = sorted(dict.fromkeys(int(i) for i in sel))
    while len(pool_idx) < POOL_SIZE:                      # backfill if dedup shrank it
        for i in idx.tolist():
            if i not in pool_idx:
                pool_idx.append(i)
                break
    pool_idx = sorted(pool_idx)

    # freeze a shuffled display order (same for everyone)
    rng = random.Random(SEED)
    order = pool_idx[:]
    rng.shuffle(order)

    # write PNGs + manifest
    if OUT_ROIS.exists():
        for old in OUT_ROIS.glob("*.png"):
            old.unlink()
    OUT_ROIS.mkdir(parents=True, exist_ok=True)

    dataset = []
    print("\n  id      P(cilia)  model_label   source")
    print("  " + "-" * 62)
    for n, i in enumerate(order, start=1):
        src, p = items[i]
        rid = f"roi{n:02d}"
        shutil.copyfile(src, OUT_ROIS / f"{rid}.png")     # original training PNG as-is
        dataset.append({
            "id": rid, "file": f"{rid}.png",
            "model_score": round(p, 4), "model_label": _model_label(p),
            "src": src.name,
        })
        print(f"  {rid}    {p:6.3f}    {_model_label(p):<10}  {src.name[:34]}")

    OUT_JSON.write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    # default selection = the most ambiguous DEFAULT_SELECTED (P closest to 0.5),
    # so the group has a working set before the admin curates it.
    amb = sorted(dataset, key=lambda d: abs(d["model_score"] - 0.5))[:DEFAULT_SELECTED]
    selected = sorted(d["id"] for d in amb)
    OUT_SEL.write_text(json.dumps({"selected": selected}, indent=2), encoding="utf-8")

    print(f"\n[+] Wrote pool of {len(dataset)} ROIs -> {OUT_ROIS}")
    print(f"[+] Wrote manifest -> {OUT_JSON}")
    print(f"[+] Default-selected {len(selected)} ambiguous ROIs -> {OUT_SEL}")
    print("[+] Quillan can re-pick the shown set in the app's 'Curate' screen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
