"""
LimonCELLo — manual cilia annotator.

Draw bounding boxes around cilia on the XY maximum-intensity projection and tag
each one as ``neurite`` / ``soma`` / ``uncertain``. Boxes + tags are written to
an Excel file (one row per box, pixel-space y/x bounds).

The point of the export is *per-cilium manual validation* in ``data_app.py``:
a detected cilium (its ``coords`` centroid, projected onto the XY MIP) is scored
against these hand-drawn boxes — if its (y, x) falls inside a box we know its
true class, so we can measure which cilia the pipeline **misses** or
**misclassifies**.

Workflow:
  1. Open a folder of .ims files and step through them one at a time.
  2. Draw rectangles over cilia in the viewer (Shapes layer "Cilia boxes").
  3. Set each box's class in the live table (neurite / soma / uncertain).
  4. Save all boxes across all reviewed images to an Excel file.

Run with:  python napari_annotator_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import napari
from magicgui.widgets import (
    Container, PushButton, ComboBox, FileEdit, Label, CheckBox, SpinBox,
)
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QLabel, QFrame,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QComboBox, QPushButton,
)
from napari.utils.notifications import show_info, show_warning

from limoncello.utils.reader import load_image
from limoncello.preprocessing.preprocessing import (
    percentile_minmax_normalize, make_isotropic,
)

_CH_NAMES     = ["Cilia", "Neurites", "Basal Bodies", "Nuclei"]
_CH_COLORMAPS = ["green", "cyan", "magenta", "blue"]
_CLASSES      = ["neurite", "soma", "uncertain"]
_CLASS_EDGE   = {"neurite": "#2ecc71", "soma": "#e74c3c", "uncertain": "#f39c12"}

_BOX_LAYER = "Cilia boxes"

# Excel columns (one row per hand-drawn box).
_XLSX_COLS = [
    "filename", "box_id", "class",
    "y_min", "x_min", "y_max", "x_max",
    "img_height", "img_width",
    "voxel_z", "voxel_y", "voxel_x",
    "mip", "make_isotropic",
]
_XLSX_SHEET = "cilia_boxes"


class LimoncelloAnnotator:
    """Builds the dock widget and owns the per-image annotation state."""

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.files: list[str] = []
        # Per-file boxes kept in memory so switching images doesn't lose work:
        #   {filename: {"boxes": [np.ndarray(4,2) …], "classes": [str …],
        #               "meta": {img_height, …}}}
        self._store: dict[str, dict] = {}
        self._cur_file: str | None = None
        self._cur_meta: dict = {}
        self.shapes = None
        self._syncing = False          # guards table ↔ shapes two-way sync
        self.widget = self._build()
        self._build_table()

    # ── image loading / display ─────────────────────────────────────────────────
    def _load_channels(self, ims_path: str):
        """Return (mip_stack (C,Y,X), voxel_size, meta) for the current settings."""
        img, meta = load_image(ims_path)
        voxel_size = meta["voxel_size"] or (1.0, 1.0, 1.0)
        n_ch = img.shape[1]
        raw = np.stack([np.asarray(img[0, c]) for c in range(n_ch)])   # (C,Z,Y,X)
        norm = np.stack([
            percentile_minmax_normalize(raw[c], self.p_low.value, self.p_high.value)
            for c in range(n_ch)
        ])
        if self.make_iso.value and not self.use_mip.value:
            iso = max(voxel_size)
            if not all(abs(v - iso) < 1e-6 for v in voxel_size):
                norm = np.stack([make_isotropic(norm[c], voxel_size, iso)[0]
                                 for c in range(n_ch)])
                voxel_size = (iso, iso, iso)
        # Always project to a 2-D XY MIP for drawing boxes (matches data_app,
        # which projects detected cilia coords onto the XY MIP).
        mip = np.max(norm, axis=1)                                     # (C,Y,X)
        return mip, voxel_size, n_ch

    def _show_image(self, ims_path: str):
        try:
            mip, voxel_size, n_ch = self._load_channels(ims_path)
        except Exception as exc:                                       # noqa: BLE001
            show_warning(f"Could not load {Path(ims_path).name}: {exc}")
            return
        vy, vx = float(voxel_size[1]), float(voxel_size[2])
        # Refresh raw channel layers (2-D).
        for name in _CH_NAMES:
            lname = f"LC: {name}"
            if lname in self.viewer.layers:
                self.viewer.layers.remove(lname)
        ch_map = {"Cilia": self.ch_cilia, "Neurites": self.ch_neurites,
                  "Basal Bodies": self.ch_bb, "Nuclei": self.ch_nuclei}
        for name, cmap in zip(_CH_NAMES, _CH_COLORMAPS):
            ch = int(ch_map[name].value)
            if ch >= n_ch:
                continue
            self.viewer.add_image(
                mip[ch], name=f"LC: {name}", scale=(vy, vx), colormap=cmap,
                blending="additive", visible=(name == "Cilia"),
            )
        self._cur_meta = dict(
            img_height=int(mip.shape[1]), img_width=int(mip.shape[2]),
            voxel_z=float(voxel_size[0]), voxel_y=vy, voxel_x=vx,
            mip=True, make_isotropic=bool(self.make_iso.value),
        )
        self.viewer.dims.ndisplay = 2
        self._ensure_box_layer((vy, vx))
        self.viewer.reset_view()

    def _ensure_box_layer(self, scale):
        """(Re)create the Shapes layer used to draw cilia boxes."""
        if _BOX_LAYER in self.viewer.layers:
            self.viewer.layers.remove(_BOX_LAYER)
        self.shapes = self.viewer.add_shapes(
            name=_BOX_LAYER, scale=scale, ndim=2,
            edge_color="yellow", face_color="transparent", edge_width=2,
        )
        self.shapes.mode = "add_rectangle"
        # Any structural change (add/move/delete a box) refreshes the table.
        self.shapes.events.data.connect(self._on_shapes_changed)

    # ── navigation ──────────────────────────────────────────────────────────────
    def _scan_folder(self):
        folder = Path(str(self.folder.value))
        if not folder.is_dir():
            show_warning("Select a valid input folder first.")
            return
        self.files = sorted(f for f in os.listdir(folder)
                            if f.lower().endswith(".ims"))
        if not self.files:
            show_warning(f"No .ims files found in {folder}")
            self.file_combo.choices = ()
            return
        self.file_combo.choices = self.files
        self.file_combo.value = self.files[0]
        show_info(f"Found {len(self.files)} .ims file(s).")

    def _step_file(self, delta: int):
        if not self.files:
            return
        cur = self.file_combo.value
        idx = self.files.index(cur) if cur in self.files else 0
        self.file_combo.value = self.files[(idx + delta) % len(self.files)]

    def _on_file_changed(self, *_):
        # Persist the boxes drawn on the file we're leaving, then load the new one.
        self._snapshot_current()
        new = self.file_combo.value
        self._cur_file = new
        if not new or not self.folder.value:
            return
        path = str(Path(str(self.folder.value)) / new)
        self._show_image(path)
        self._restore_boxes(new)
        self._rebuild_table()

    # ── box persistence across image switches ───────────────────────────────────
    def _snapshot_current(self):
        if self._cur_file is None or self.shapes is None:
            return
        boxes = [np.asarray(d) for d in self.shapes.data]
        self._store[self._cur_file] = dict(
            boxes=boxes, classes=list(self._classes()), meta=dict(self._cur_meta),
        )

    def _restore_boxes(self, filename: str):
        rec = self._store.get(filename)
        if not rec or self.shapes is None or not rec["boxes"]:
            return
        self._syncing = True
        try:
            self.shapes.add(rec["boxes"], shape_type="rectangle")
            self.shapes.features = pd.DataFrame({"class": rec["classes"]})
        finally:
            self._syncing = False

    # ── class bookkeeping (authoritative = shapes.features["class"]) ─────────────
    def _classes(self) -> list[str]:
        n = 0 if self.shapes is None else len(self.shapes.data)
        feats = getattr(self.shapes, "features", None)
        if feats is not None and "class" in feats and len(feats) == n:
            return [str(c) for c in feats["class"]]
        return ["uncertain"] * n

    def _set_class(self, row: int, value: str):
        classes = self._classes()
        if 0 <= row < len(classes):
            classes[row] = value
            self.shapes.features = pd.DataFrame({"class": classes})
            self._recolor_boxes(classes)

    def _recolor_boxes(self, classes: list[str]):
        if self.shapes is None or not classes:
            return
        self.shapes.edge_color = [_CLASS_EDGE.get(c, "#f39c12") for c in classes]

    # ── live table ──────────────────────────────────────────────────────────────
    def _build_table(self):
        self.table = QTableWidget()
        cols = ["box_id", "class", "y_min", "x_min", "y_max", "x_max", ""]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        if self.viewer is not None:
            self.viewer.window.add_dock_widget(
                self.table, area="bottom", name="Cilia boxes")

    def _on_shapes_changed(self, *_):
        if self._syncing:
            return
        # A newly drawn box has no class row yet → default it to "uncertain".
        n = len(self.shapes.data)
        classes = self._classes()
        if len(classes) != n:
            classes = (classes + ["uncertain"] * n)[:n]
            self.shapes.features = pd.DataFrame({"class": classes})
        self._recolor_boxes(classes)
        self._rebuild_table()

    def _rebuild_table(self):
        if self.shapes is None:
            self.table.setRowCount(0)
            return
        data = self.shapes.data
        classes = self._classes()
        self.table.setRowCount(len(data))
        for r, box in enumerate(data):
            arr = np.asarray(box)
            y0, x0 = arr[:, 0].min(), arr[:, 1].min()
            y1, x1 = arr[:, 0].max(), arr[:, 1].max()
            self.table.setItem(r, 0, QTableWidgetItem(str(r + 1)))
            combo = QComboBox()
            combo.addItems(_CLASSES)
            combo.setCurrentText(classes[r] if r < len(classes) else "uncertain")
            combo.currentTextChanged.connect(
                lambda val, row=r: self._set_class(row, val))
            self.table.setCellWidget(r, 1, combo)
            for c, v in ((2, y0), (3, x0), (4, y1), (5, x1)):
                self.table.setItem(r, c, QTableWidgetItem(f"{v:.0f}"))
            del_btn = QPushButton("🗑")
            del_btn.clicked.connect(lambda _=False, row=r: self._delete_row(row))
            self.table.setCellWidget(r, 6, del_btn)
        self.table.resizeColumnsToContents()

    def _on_row_selected(self):
        """Table row click → select the matching box and centre it in the viewer."""
        if self._syncing or self.shapes is None:
            return
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        if not rows:
            return
        row = min(rows)
        if row >= len(self.shapes.data):
            return
        self._syncing = True
        try:
            self.shapes.mode = "select"
            self.shapes.selected_data = {row}
            self.shapes.refresh()
            arr = np.asarray(self.shapes.data[row])
            cy, cx = float(arr[:, 0].mean()), float(arr[:, 1].mean())
            sy, sx = self.shapes.scale[-2], self.shapes.scale[-1]
            self.viewer.camera.center = (cy * sy, cx * sx)   # world (scaled) coords
        finally:
            self._syncing = False

    def _delete_row(self, row: int):
        if self.shapes is None or row >= len(self.shapes.data):
            return
        classes = self._classes()
        boxes = [np.asarray(d) for i, d in enumerate(self.shapes.data) if i != row]
        classes = [c for i, c in enumerate(classes) if i != row]
        self._syncing = True
        try:
            self.shapes.data = []
            if boxes:
                self.shapes.add(boxes, shape_type="rectangle")
            self.shapes.features = pd.DataFrame({"class": classes})
        finally:
            self._syncing = False
        self._recolor_boxes(classes)
        self._rebuild_table()

    # ── Excel export / import ────────────────────────────────────────────────────
    def _rows_for_file(self, filename: str, boxes, classes, meta) -> list[dict]:
        rows = []
        for i, box in enumerate(boxes):
            arr = np.asarray(box)
            rows.append({
                "filename": filename, "box_id": i + 1,
                "class": classes[i] if i < len(classes) else "uncertain",
                "y_min": float(arr[:, 0].min()), "x_min": float(arr[:, 1].min()),
                "y_max": float(arr[:, 0].max()), "x_max": float(arr[:, 1].max()),
                "img_height": meta.get("img_height"), "img_width": meta.get("img_width"),
                "voxel_z": meta.get("voxel_z"), "voxel_y": meta.get("voxel_y"),
                "voxel_x": meta.get("voxel_x"),
                "mip": meta.get("mip", True),
                "make_isotropic": meta.get("make_isotropic", False),
            })
        return rows

    def _save_excel(self):
        self._snapshot_current()
        rows: list[dict] = []
        for fn, rec in self._store.items():
            rows.extend(self._rows_for_file(fn, rec["boxes"], rec["classes"],
                                            rec["meta"]))
        if not rows:
            show_warning("No boxes drawn yet — nothing to save.")
            return
        out = Path(str(self.out_file.value))
        if out.suffix.lower() not in (".xlsx", ".xls"):
            out = out.with_suffix(".xlsx")
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows, columns=_XLSX_COLS)
        try:
            df.to_excel(out, sheet_name=_XLSX_SHEET, index=False)
        except Exception as exc:                                       # noqa: BLE001
            show_warning(f"Could not write Excel: {exc}")
            return
        show_info(f"Saved {len(rows)} box(es) across {df['filename'].nunique()} "
                  f"image(s) → {out}")

    def _load_excel(self):
        """Load an existing annotation Excel back into memory to resume work."""
        src = Path(str(self.out_file.value))
        if not src.exists():
            show_warning(f"No file at {src}.")
            return
        try:
            df = pd.read_excel(src, sheet_name=_XLSX_SHEET)
        except Exception as exc:                                       # noqa: BLE001
            show_warning(f"Could not read {src}: {exc}")
            return
        for fn, g in df.groupby("filename"):
            boxes, classes = [], []
            for _, r in g.iterrows():
                y0, x0, y1, x1 = r["y_min"], r["x_min"], r["y_max"], r["x_max"]
                boxes.append(np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]],
                                      dtype=float))
                classes.append(str(r.get("class", "uncertain")))
            meta = {k: g.iloc[0].get(k) for k in
                    ("img_height", "img_width", "voxel_z", "voxel_y", "voxel_x",
                     "mip", "make_isotropic")}
            self._store[str(fn)] = dict(boxes=boxes, classes=classes, meta=meta)
        show_info(f"Loaded {len(df)} box(es) from {src.name}. "
                  "Select an image to see its boxes.")
        if self._cur_file:
            self._restore_boxes(self._cur_file)
            self._rebuild_table()

    # ── UI ──────────────────────────────────────────────────────────────────────
    def _build(self) -> QScrollArea:
        self.folder = FileEdit(label="Input folder", mode="d")
        scan_btn = PushButton(text="🔍 Scan folder")
        scan_btn.clicked.connect(self._scan_folder)
        self.file_combo = ComboBox(label="Image", choices=())
        self.file_combo.changed.connect(self._on_file_changed)
        prev_btn = PushButton(text="◀ Prev")
        next_btn = PushButton(text="Next ▶")
        prev_btn.clicked.connect(lambda: self._step_file(-1))
        next_btn.clicked.connect(lambda: self._step_file(+1))
        nav = Container(widgets=[prev_btn, next_btn], layout="horizontal", label="")

        self.ch_cilia    = SpinBox(label="Ch: Cilia",        value=1, min=0, max=9)
        self.ch_neurites = SpinBox(label="Ch: Neurites",     value=0, min=0, max=9)
        self.ch_bb       = SpinBox(label="Ch: Basal Bodies", value=2, min=0, max=9)
        self.ch_nuclei   = SpinBox(label="Ch: Nuclei",       value=3, min=0, max=9)
        self.use_mip  = CheckBox(label="Force MIP (always on for drawing)", value=True)
        self.use_mip.enabled = False
        self.make_iso = CheckBox(label="Make isotropic (match pipeline)", value=True)
        self.p_low  = SpinBox(label="p_low (%)",  value=0,  min=0, max=49)
        self.p_high = SpinBox(label="p_high (%)", value=100, min=51, max=100)

        self.out_file = FileEdit(
            label="Annotations Excel", mode="w",
            filter="Excel (*.xlsx)",
            value=str(Path("tutorial/output/cilia_annotations.xlsx").resolve()))
        save_btn = PushButton(text="💾 Save annotations (Excel)")
        save_btn.clicked.connect(self._save_excel)
        load_btn = PushButton(text="📂 Load existing annotations")
        load_btn.clicked.connect(self._load_excel)

        input_box = Container(
            widgets=[self.folder, scan_btn, self.file_combo, nav], labels=True)
        chan_box = Container(
            widgets=[self.ch_cilia, self.ch_neurites, self.ch_bb, self.ch_nuclei,
                     self.use_mip, self.make_iso, self.p_low, self.p_high],
            labels=True)
        save_box = Container(
            widgets=[self.out_file, save_btn, load_btn], labels=True)

        def _header(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight:600; margin-top:6px; color:#F5A623;")
            return lbl

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(4)
        lay.addWidget(_header("① Input"))
        lay.addWidget(input_box.native)
        lay.addWidget(_header("② Channels & display"))
        lay.addWidget(chan_box.native)
        _sep = QFrame(); _sep.setFrameShape(QFrame.HLine)
        _sep.setStyleSheet("color:#444;")
        lay.addWidget(_sep)
        lay.addWidget(_header("③ Annotate"))
        lay.addWidget(QLabel(
            "Draw rectangles over cilia in the viewer, then set each box's class "
            "in the table below.\nneurite / soma / uncertain."))
        lay.addWidget(_header("④ Save"))
        lay.addWidget(save_box.native)
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setMinimumWidth(340)
        return scroll


def main():
    viewer = napari.Viewer(title="LimonCELLo — Cilia Annotator 🍋")
    app = LimoncelloAnnotator(viewer)
    viewer.window.add_dock_widget(app.widget, area="right", name="Annotator")
    napari.run()


if __name__ == "__main__":
    main()
