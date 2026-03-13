"""Main application window for Quality_tool GUI.

Provides the central layout (toolbar → map viewer → signal inspector →
status bar) and wires user actions to backend calls.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from quality_tool.core.models import MetricMapResult, SignalSet, ThresholdResult
from quality_tool.evaluation.evaluator import evaluate_metric_map
from quality_tool.evaluation.thresholding import apply_threshold
from quality_tool.gui.dialogs.info_dialog import InfoDialog
from quality_tool.gui.widgets.map_viewer import MapViewer
from quality_tool.gui.widgets.signal_inspector import SignalInspector
from quality_tool.gui.windows.compare_window import CompareWindow
from quality_tool.metrics.baseline.fringe_visibility import FringeVisibility
from quality_tool.metrics.baseline.power_band_ratio import PowerBandRatio
from quality_tool.metrics.baseline.snr import SNR
from quality_tool.metrics.registry import MetricRegistry


def _build_default_registry() -> MetricRegistry:
    """Create a registry populated with the baseline metrics.

    Note: this duplicates the list of baseline metrics because the
    backend ``default_registry`` (in ``metrics.registry``) is currently
    empty — nothing populates it at import time.  Once the backend
    provides a pre-populated default registry, the GUI should reuse it
    instead of maintaining its own copy.
    """
    registry = MetricRegistry()
    registry.register(FringeVisibility())
    registry.register(SNR())
    registry.register(PowerBandRatio())
    return registry


class MainWindow(QMainWindow):
    """Central application window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Quality_tool")
        self.resize(900, 750)

        # ----- state -------------------------------------------------
        self._registry = _build_default_registry()
        self._signal_set: SignalSet | None = None
        self._metric_map: MetricMapResult | None = None
        self._threshold_result: ThresholdResult | None = None
        self._compare_windows: list[CompareWindow] = []

        # ----- widgets -----------------------------------------------
        self._map_viewer = MapViewer()
        self._signal_inspector = SignalInspector()

        # ----- layout ------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._map_viewer)
        splitter.addWidget(self._signal_inspector)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # ----- toolbar -----------------------------------------------
        self._build_toolbar()

        # ----- status bar --------------------------------------------
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — load a dataset to begin")

        # ----- connections -------------------------------------------
        self._map_viewer.pixel_selected.connect(self._on_pixel_selected)

    # ==================================================================
    # Toolbar
    # ==================================================================

    def _build_toolbar(self) -> None:
        tb = QToolBar("Actions")
        tb.setMovable(False)
        self.addToolBar(tb)

        # Load
        btn_load = QPushButton("Load")
        btn_load.clicked.connect(self._on_load)
        tb.addWidget(btn_load)

        tb.addSeparator()

        # Metric selector
        tb.addWidget(QLabel(" Metric: "))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems(self._registry.list_metrics())
        tb.addWidget(self._metric_combo)

        # Compute
        btn_compute = QPushButton("Compute")
        btn_compute.clicked.connect(self._on_compute)
        tb.addWidget(btn_compute)

        tb.addSeparator()

        # Map type selector
        tb.addWidget(QLabel(" Map: "))
        self._map_combo = QComboBox()
        self._map_combo.addItems(["score_map"])
        self._map_combo.currentTextChanged.connect(self._on_map_switch)
        tb.addWidget(self._map_combo)

        tb.addSeparator()

        # Compare
        btn_compare = QPushButton("Compare")
        btn_compare.clicked.connect(self._on_compare)
        tb.addWidget(btn_compare)

        # Info
        btn_info = QPushButton("Info")
        btn_info.clicked.connect(self._on_info)
        tb.addWidget(btn_info)

        # Export
        btn_export = QPushButton("Export")
        btn_export.clicked.connect(self._on_export)
        tb.addWidget(btn_export)

    # ==================================================================
    # Action handlers
    # ==================================================================

    def _on_load(self) -> None:
        """Open a dataset via a small dialog that chooses loader type."""
        dlg = _LoadDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            signal_set = dlg.load()
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            self._status.showMessage("Load failed")
            return

        self._signal_set = signal_set
        self._metric_map = None
        self._threshold_result = None
        self._map_viewer.clear()
        self._signal_inspector.clear()
        self._refresh_map_combo()

        h, w, m = signal_set.signals.shape
        self._status.showMessage(
            f"Loaded: {signal_set.source_type}  "
            f"({h} x {w} x {m})  "
            f"z_axis={'file' if signal_set.z_axis_path else 'index'}"
        )

    def _on_compute(self) -> None:
        """Run the selected metric on the loaded dataset."""
        if self._signal_set is None:
            self._status.showMessage("No dataset loaded")
            return

        metric_name = self._metric_combo.currentText()
        metric = self._registry.get(metric_name)

        self._status.showMessage(f"Computing {metric_name}…")
        # Force status bar repaint before blocking compute.
        self._status.repaint()

        try:
            self._metric_map = evaluate_metric_map(self._signal_set, metric)
        except Exception as exc:
            QMessageBox.critical(self, "Compute error", str(exc))
            self._status.showMessage("Compute failed")
            return

        # Auto-threshold at median of valid scores for convenience.
        valid = self._metric_map.valid_map
        scores = self._metric_map.score_map[valid]
        if scores.size > 0:
            median_val = float(np.median(scores))
            self._threshold_result = apply_threshold(
                self._metric_map, median_val, keep_rule="above",
            )
        else:
            self._threshold_result = None

        self._refresh_map_combo()
        self._show_current_map()
        self._status.showMessage(f"Computed: {metric_name}")

    def _on_map_switch(self, _text: str) -> None:
        self._show_current_map()

    def _on_pixel_selected(self, row: int, col: int) -> None:
        if self._signal_set is None:
            return

        signal = self._signal_set.signals[row, col, :]
        z_axis = self._signal_set.z_axis
        self._signal_inspector.update_signal(
            signal, z_axis,
            title=f"Pixel ({row}, {col})",
        )

        value = self._map_viewer.value_at(row, col)
        map_name = self._map_combo.currentText()
        val_str = f"{value:.4g}" if value is not None else "—"
        self._status.showMessage(
            f"Pixel ({row}, {col})  {map_name}={val_str}"
        )

    def _on_compare(self) -> None:
        data, title = self._map_viewer.get_snapshot()
        if data is None:
            self._status.showMessage("No map to compare")
            return

        win = CompareWindow(data, title=title)
        win.show()
        self._compare_windows.append(win)

    def _on_info(self) -> None:
        info: dict[str, str] = {}
        if self._signal_set is not None:
            ss = self._signal_set
            h, w, m = ss.signals.shape
            info["Source type"] = ss.source_type
            info["Source path"] = ss.source_path or "—"
            info["Dimensions"] = f"{h} x {w} x {m}  (H x W x M)"
            info["Z-axis"] = "from file" if ss.z_axis_path else "index-based"
            info["Metadata found"] = "yes" if ss.metadata else "no"
            if ss.metadata:
                for k, v in ss.metadata.items():
                    info[f"  {k}"] = str(v)
        else:
            info["Dataset"] = "none loaded"

        if self._metric_map is not None:
            info["Metric"] = self._metric_map.metric_name
        if self._threshold_result is not None:
            info["Threshold"] = (
                f"{self._threshold_result.threshold:.4g}  "
                f"({self._threshold_result.keep_rule})"
            )
            if self._threshold_result.stats:
                info["Kept pixels"] = str(
                    self._threshold_result.stats.get("kept_pixels", "—")
                )

        dlg = InfoDialog(info, parent=self)
        dlg.exec()

    def _on_export(self) -> None:
        if self._metric_map is None:
            self._status.showMessage("Nothing to export")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export current map", "", "Text files (*.txt)",
        )
        if not path:
            return

        map_type = self._map_combo.currentText()
        if map_type == "threshold_mask" and self._threshold_result is not None:
            data = self._threshold_result.mask.astype(int)
        else:
            data = self._metric_map.score_map

        np.savetxt(path, data, fmt="%.6g")
        self._status.showMessage(f"Exported {map_type} → {path}")

    # ==================================================================
    # Helpers
    # ==================================================================

    def _refresh_map_combo(self) -> None:
        """Update the map type combo box based on available results."""
        self._map_combo.blockSignals(True)
        self._map_combo.clear()

        if self._metric_map is not None:
            self._map_combo.addItem("score_map")
            if self._threshold_result is not None:
                self._map_combo.addItem("threshold_mask")

        self._map_combo.blockSignals(False)

    def _show_current_map(self) -> None:
        """Render the map currently selected in the combo box."""
        map_type = self._map_combo.currentText()

        if map_type == "score_map" and self._metric_map is not None:
            self._map_viewer.set_map(
                self._metric_map.score_map,
                title=f"{self._metric_map.metric_name} — score map",
            )
        elif map_type == "threshold_mask" and self._threshold_result is not None:
            self._map_viewer.set_binary_mask(
                self._threshold_result.mask,
                title=(
                    f"{self._metric_map.metric_name} — mask  "
                    f"({self._threshold_result.keep_rule})"
                ),
            )


# ======================================================================
# Load dialog
# ======================================================================

class _LoadDialog(QDialog):
    """Small dialog to choose the loading method and parameters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load dataset")
        self.resize(460, 200)

        form = QFormLayout()

        self._type_combo = QComboBox()
        self._type_combo.addItems(["image_stack", "txt_matrix"])
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        form.addRow("Source type:", self._type_combo)

        # Path selection
        path_row = QHBoxLayout()
        self._path_label = QLabel("(none)")
        self._path_label.setMinimumWidth(250)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._on_browse)
        path_row.addWidget(self._path_label)
        path_row.addWidget(btn_browse)
        form.addRow("Path:", path_row)

        # Width / height (only for txt_matrix)
        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 100_000)
        self._width_spin.setValue(1)
        self._height_spin = QSpinBox()
        self._height_spin.setRange(1, 100_000)
        self._height_spin.setValue(1)
        form.addRow("Width:", self._width_spin)
        form.addRow("Height:", self._height_spin)

        self._width_spin.setEnabled(False)
        self._height_spin.setEnabled(False)

        # OK / Cancel
        btn_row = QHBoxLayout()
        btn_ok = QPushButton("Load")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)

        outer = QVBoxLayout(self)
        outer.addLayout(form)
        outer.addLayout(btn_row)

        self._selected_path: str | None = None

    def _on_type_changed(self, text: str) -> None:
        is_txt = text == "txt_matrix"
        self._width_spin.setEnabled(is_txt)
        self._height_spin.setEnabled(is_txt)

    def _on_browse(self) -> None:
        src = self._type_combo.currentText()
        if src == "image_stack":
            path = QFileDialog.getExistingDirectory(
                self, "Select image stack directory",
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select TXT data file", "", "Text files (*.txt)",
            )
        if path:
            self._selected_path = path
            self._path_label.setText(path)

    def load(self) -> SignalSet:
        """Execute the appropriate backend loader and return a SignalSet.

        Must be called after the dialog has been accepted.
        """
        if not self._selected_path:
            raise ValueError("No path selected")

        src = self._type_combo.currentText()

        if src == "image_stack":
            from quality_tool.io.image_stack_loader import load_image_stack

            return load_image_stack(self._selected_path)
        else:
            from quality_tool.io.txt_matrix_loader import load_txt_matrix

            return load_txt_matrix(
                self._selected_path,
                width=self._width_spin.value(),
                height=self._height_spin.value(),
            )
