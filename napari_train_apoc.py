"""
LimonCELLo — APOC classifier trainer (paint-and-train on a random subset).

A focused companion to ``napari_app.py`` for *building* the APOC pixel/object
classifiers (the ``.cl`` files the pipeline uses to segment cilia / basal bodies).
It reuses that app's image loader and device helpers; you just:

  1. point it at a folder (or a .txt manifest) of .ims files,
  2. **randomly pick N** of them into a working set (share / spread annotation),
  3. step through them, painting sparse labels on the chosen channel
     (background = 1, object = 2),
  4. **Train / add** each painted image into one accumulating classifier
     (APOC ``continue_training``), and **Preview** the result live.

Train on the *raw* channel (no normalisation) and with the same MIP / isotropic
geometry the pipeline uses, so the classifier matches inference.

Run with:  python napari_train_apoc.py
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import napari
from magicgui.widgets import (
    Container, PushButton, ComboBox, FileEdit, Label, CheckBox, SpinBox,
    FloatSpinBox, LineEdit,
)
from qtpy.QtWidgets import (
    QScrollArea, QTableWidget, QTableWidgetItem, QAbstractItemView,
)
from qtpy.QtCore import Qt
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning

# Reuse the pipeline app's loader + device helpers (same behaviour as inference).
from napari_app import (
    step_load, _detect_gpus, _default_gpu, _select_device, _add_image_safe,
    _clamp_ch,
)
from limoncello.segmentation.train import (
    train_segmenter_single, predict_segmenter, DEFAULT_FEATURES,
    feature_spec_from_pairs, FEATURE_OPERATIONS,
)

_ANNOT = "LC-APOC: Annotation"
_PREVIEW = "LC-APOC: Preview"
_RAW = "LC-APOC: Raw channel"
_SEGMENTERS = str((Path(__file__).parent / "segmenters").resolve())


class ApocTrainerApp:
    """Dock widget: random-subset picker + paint-and-train one APOC classifier."""

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.state: dict = {}
        self.files: list[str] = []          # all scanned .ims (basename)
        self.path_by_name: dict[str, str] = {}
        self.subset: list[str] = []         # the randomly-picked working set
        self._busy = False
        self._trained_any = False           # → drives APOC continue_training
        self.widget = self._build()
        self._build_feature_grid()
        self.viewer.window.add_dock_widget(
            self.feat_table, area="bottom", name="APOC features (tick to use)")

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build(self) -> Container:
        devices = _detect_gpus()
        self.gpu_combo = ComboBox(label="Compute device",
                                  choices=[(l, v) for l, v in devices],
                                  value=_default_gpu(devices))
        self.folder = FileEdit(label="Image folder / .txt manifest", mode="d")
        self.scan_btn = PushButton(text="Scan")
        self.scan_btn.clicked.connect(self._scan)
        self.n_pick = SpinBox(label="Pick N at random", value=8, min=1, max=999)
        self.pick_btn = PushButton(text="🎲 Pick random subset")
        self.pick_btn.clicked.connect(self._pick_random)
        self.file_combo = ComboBox(label="Subset image", choices=())
        self.file_combo.changed.connect(lambda *_: None)
        self.prev_btn = PushButton(text="◀ Prev")
        self.next_btn = PushButton(text="Next ▶")
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(1))

        self.ch = SpinBox(label="Channel to segment", value=1, min=0, max=16)
        self.use_mip = CheckBox(label="MIP (2-D)", value=False)
        self.make_iso = CheckBox(label="Make isotropic", value=True)
        self.load_btn = PushButton(text="① Load + new annotation layer")
        self.load_btn.clicked.connect(self._load)

        # Feature selection via a tickbox grid (in its own bottom dock); these
        # control its σ columns and whether the raw channel is a feature.
        self.train_feat_sigmas = LineEdit(label="Feature σ scales",
                                          value="1, 2, 3, 5, 10")
        self.train_feat_original = CheckBox(label="Include original image", value=True)
        self.rebuild_feat_btn = PushButton(text="↻ Rebuild σ columns")
        self.rebuild_feat_btn.clicked.connect(self._rebuild_feature_columns)
        self.max_depth = SpinBox(label="Tree depth", value=2, min=1, max=10)
        self.num_trees = SpinBox(label="Num trees", value=100, min=10, max=500)
        self.pos_class = SpinBox(label="Object label", value=2, min=2, max=10)
        self.output = FileEdit(label="Classifier .cl",
                               value=os.path.join(_SEGMENTERS, "my-cilia.cl"),
                               mode="w")
        self.new_btn = PushButton(text="✦ Start new classifier")
        self.new_btn.clicked.connect(self._new_classifier)
        self.train_btn = PushButton(text="② Train / add this image")
        self.train_btn.clicked.connect(self._train)
        self.preview_btn = PushButton(text="③ Preview segmentation")
        self.preview_btn.clicked.connect(self._preview)
        self.status = Label(value="Scan a folder, pick a random subset, paint, train.")

        self._buttons = [self.scan_btn, self.pick_btn, self.prev_btn, self.next_btn,
                         self.load_btn, self.new_btn, self.train_btn, self.preview_btn]

        return Container(widgets=[
            Label(value="<b>APOC classifier trainer</b>"),
            self.gpu_combo,
            Label(value="<b>1 · Pick a random subset</b>"),
            self.folder, self.scan_btn, self.n_pick, self.pick_btn,
            self.file_combo, self.prev_btn, self.next_btn,
            Label(value="<b>2 · Load + paint</b>  (bg=1, object=2)"),
            self.ch, self.use_mip, self.make_iso, self.load_btn,
            Label(value="<b>3 · Features</b>  (tick cells in the bottom grid)"),
            self.train_feat_sigmas, self.train_feat_original, self.rebuild_feat_btn,
            Label(value="<b>4 · Train</b>"),
            self.max_depth, self.num_trees, self.pos_class,
            self.output, self.new_btn, self.train_btn, self.preview_btn,
            self.status,
        ], labels=True, scrollable=False)

    # ── feature-selection grid (operation × sigma checkboxes) ────────────────
    _OP_SHORT = {"gaussian_blur": "Gauss", "difference_of_gaussian": "DoG",
                 "laplace_box_of_gaussian_blur": "LoG",
                 "sobel_of_gaussian_blur": "Sobel"}

    def _parse_sigmas(self):
        out = []
        for tok in str(self.train_feat_sigmas.value).replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(float(tok))
            except ValueError:
                continue
        return sorted(dict.fromkeys(out)) or [1.0, 2.0, 3.0, 5.0, 10.0]

    def _build_feature_grid(self):
        """Checkbox grid: rows = filter operations, columns = σ scales. Ticked
        cells become ``operation=sigma`` APOC features (napari-apoc style)."""
        self._feat_sigmas = self._parse_sigmas()
        t = QTableWidget()
        t.setRowCount(len(FEATURE_OPERATIONS))
        t.setVerticalHeaderLabels([self._OP_SHORT.get(o, o) for o in FEATURE_OPERATIONS])
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionMode(QAbstractItemView.NoSelection)
        t.horizontalHeader().setStretchLastSection(True)
        self.feat_table = t
        self._populate_feature_columns(default_checked=True)
        t.cellDoubleClicked.connect(self._toggle_feature_cell)

    def _populate_feature_columns(self, default_checked=True, preserve=True):
        prev = {}
        if preserve and self.feat_table.columnCount():
            for r, op in enumerate(FEATURE_OPERATIONS):
                for c, s in enumerate(getattr(self, "_feat_sigmas_shown", [])):
                    it = self.feat_table.item(r, c)
                    if it is not None:
                        prev[(op, s)] = it.checkState() == Qt.Checked
        sig = self._feat_sigmas
        self.feat_table.setColumnCount(len(sig))
        self.feat_table.setHorizontalHeaderLabels([f"σ{s:g}" for s in sig])
        for r, op in enumerate(FEATURE_OPERATIONS):
            for c, s in enumerate(sig):
                it = QTableWidgetItem()
                it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                checked = prev.get((op, s), default_checked)
                it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                it.setToolTip(f"{op}={s:g}")
                self.feat_table.setItem(r, c, it)
        self._feat_sigmas_shown = list(sig)
        self.feat_table.resizeColumnsToContents()

    def _rebuild_feature_columns(self):
        self._feat_sigmas = self._parse_sigmas()
        self._populate_feature_columns(default_checked=True, preserve=True)

    def _toggle_feature_cell(self, r, c):
        it = self.feat_table.item(r, c)
        if it is not None:
            it.setCheckState(Qt.Unchecked if it.checkState() == Qt.Checked else Qt.Checked)

    def _feature_spec_from_grid(self) -> str:
        """APOC feature_specification string from the ticked grid cells."""
        pairs = []
        for r, op in enumerate(FEATURE_OPERATIONS):
            for c, s in enumerate(self._feat_sigmas_shown):
                it = self.feat_table.item(r, c)
                if it is not None and it.checkState() == Qt.Checked:
                    pairs.append((op, s))
        return feature_spec_from_pairs(pairs, bool(self.train_feat_original.value))

    # ── helpers ────────────────────────────────────────────────────────────────
    def _set_busy(self, b: bool):
        self._busy = b
        for w in self._buttons:
            w.enabled = not b

    def _say(self, msg: str):
        self.status.value = msg
        show_info(msg)

    def _scan(self):
        src = Path(str(self.folder.value))
        if str(src).lower().endswith(".txt") and src.is_file():
            paths = [ln.strip() for ln in src.read_text("utf-8").splitlines()
                     if ln.strip() and not ln.lstrip().startswith("#")]
        elif src.is_dir():
            paths = [str(src / f) for f in sorted(os.listdir(src))
                     if f.lower().endswith(".ims")]
        else:
            show_warning("Pick a folder of .ims or a .txt manifest.")
            return
        if not paths:
            show_warning("No .ims files found.")
            return
        # Keep basenames unique for the dropdown; map back to full paths.
        self.path_by_name = {}
        for p in paths:
            name = os.path.basename(p)
            while name in self.path_by_name:        # disambiguate rare dup names
                name = f"{name}~{len(self.path_by_name)}"
            self.path_by_name[name] = p
        self.files = list(self.path_by_name)
        self._say(f"Found {len(self.files)} .ims file(s). Pick a random subset.")

    def _pick_random(self):
        if not self.files:
            show_warning("Scan a folder first.")
            return
        n = min(int(self.n_pick.value), len(self.files))
        self.subset = random.sample(self.files, n)
        self.file_combo.choices = self.subset
        self.file_combo.value = self.subset[0]
        self._say(f"🎲 Picked {n} random image(s) to annotate.")

    def _step(self, d: int):
        if not self.subset:
            return
        cur = self.file_combo.value
        i = self.subset.index(cur) if cur in self.subset else 0
        self.file_combo.value = self.subset[(i + d) % len(self.subset)]

    def _params(self) -> dict:
        name = self.file_combo.value
        return dict(
            ims_path=self.path_by_name.get(name, ""),
            gpu_device=self.gpu_combo.value,
            p_low=2.0, p_high=98.0, ch_norm={},
            use_mip=self.use_mip.value, make_isotropic=self.make_iso.value,
        )

    # ── load + annotation layer ────────────────────────────────────────────────
    def _load(self):
        if self._busy:
            return
        p = self._params()
        if not p["ims_path"]:
            show_warning("Pick a random subset and select an image first.")
            return
        self._set_busy(True)
        self._say("Loading image …")

        @thread_worker
        def _work():
            _select_device(p["gpu_device"])
            return step_load({}, p)

        def _done(st):
            self.state = st
            ch = _clamp_ch(st["raw"].shape[0], self.ch.value, "channel")
            vs = st["voxel_size"]
            for nm in (_RAW, _PREVIEW, _ANNOT):
                if nm in self.viewer.layers:
                    self.viewer.layers.remove(nm)
            _add_image_safe(self.viewer, st["raw"][ch], name=_RAW, scale=vs,
                            colormap="green", blending="additive")
            # Empty label layer to paint on (matches the raw channel geometry).
            annot = self.viewer.add_labels(
                np.zeros(st["raw"][ch].shape, dtype=np.uint8),
                name=_ANNOT, scale=vs)
            annot.brush_size = 6
            try:
                annot.mode = "paint"
                annot.selected_label = int(self.pos_class.value)
            except Exception:                                 # noqa: BLE001
                pass
            self.viewer.layers.selection.active = annot
            self._set_busy(False)
            self._say("Paint bg=1, object=2 on the Annotation layer, then Train.")

        def _err(e):
            self._set_busy(False); show_warning(f"Load failed: {e}")

        w = _work(); w.returned.connect(_done); w.errored.connect(_err); w.start()

    # ── train / preview ────────────────────────────────────────────────────────
    def _new_classifier(self):
        self._trained_any = False
        out = str(self.output.value)
        if out and os.path.exists(out):
            try:
                os.remove(out)
            except OSError:
                pass
        self._say("Started a new classifier — next Train writes it fresh.")

    def _train(self):
        if self._busy:
            return
        if _ANNOT not in self.viewer.layers or "raw" not in self.state:
            show_warning("Load an image and paint the Annotation layer first.")
            return
        ch = _clamp_ch(self.state["raw"].shape[0], self.ch.value, "channel")
        image = np.asarray(self.state["raw"][ch])
        gt = np.asarray(self.viewer.layers[_ANNOT].data)
        pos = int(self.pos_class.value)
        if int(gt.max()) < pos:
            show_warning(f"Paint some object pixels (label {pos}) before training.")
            return
        out = str(self.output.value)
        feats = self._feature_spec_from_grid() or DEFAULT_FEATURES
        cont = self._trained_any
        p = self._params()
        self._set_busy(True)
        self._say("Training APOC classifier …")

        @thread_worker
        def _work():
            return train_segmenter_single(
                image, gt, out, features=feats, continue_training=cont,
                positive_class=pos, max_depth=int(self.max_depth.value),
                num_trees=int(self.num_trees.value), gpu_device=p["gpu_device"])

        def _done(path):
            self._trained_any = True
            self._set_busy(False)
            self._say(f"✓ Trained into {os.path.basename(path)} "
                      f"({'added' if cont else 'fresh'}). Preview or go to next image.")

        def _err(e):
            self._set_busy(False); show_warning(f"Training failed: {e}")

        w = _work(); w.returned.connect(_done); w.errored.connect(_err); w.start()

    def _preview(self):
        if self._busy:
            return
        out = str(self.output.value)
        if not os.path.exists(out):
            show_warning("Train a classifier first.")
            return
        if "raw" not in self.state:
            show_warning("Load an image first.")
            return
        ch = _clamp_ch(self.state["raw"].shape[0], self.ch.value, "channel")
        image = np.asarray(self.state["raw"][ch])
        p = self._params()
        self._set_busy(True)
        self._say("Predicting …")

        @thread_worker
        def _work():
            return predict_segmenter(image, out, gpu_device=p["gpu_device"])

        def _done(labels):
            if _PREVIEW in self.viewer.layers:
                self.viewer.layers.remove(_PREVIEW)
            self.viewer.add_labels(np.asarray(labels).astype("int32"),
                                   name=_PREVIEW, scale=self.state["voxel_size"])
            n = int(len(np.unique(labels)) - 1)
            self._set_busy(False)
            self._say(f"Preview: {n} object(s). Paint corrections + Train again to refine.")

        def _err(e):
            self._set_busy(False); show_warning(f"Preview failed: {e}")

        w = _work(); w.returned.connect(_done); w.errored.connect(_err); w.start()


def main():
    viewer = napari.Viewer(title="LimonCELLo — APOC trainer")
    app = ApocTrainerApp(viewer)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(app.widget.native)
    viewer.window.add_dock_widget(scroll, area="right", name="APOC trainer")
    napari.run()


if __name__ == "__main__":
    main()
