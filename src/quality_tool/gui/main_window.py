"""Main application window for Quality_tool GUI.

Provides the central layout (toolbar → map section → signal section →
status bar) and wires user actions to backend calls.

Layout:
    +--------------------------------------------------------------+
    | Toolbar (Load | Metrics Compute | Map View | Compare Info Export) |
    +--------------------------------------------------------------+
    | Map section:  [  MapViewer  |  MapToolsPanel  ]              |
    +--------------------------------------------------------------+
    | Signal section:  [  SignalInspector  |  SignalToolsPanel  ]   |
    +--------------------------------------------------------------+
    | Status bar                                                    |
    +--------------------------------------------------------------+
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
from quality_tool.gui.dialogs.metrics_dialog import MetricsDialog
from quality_tool.gui.widgets.map_viewer import MapViewer
from quality_tool.gui.widgets.signal_inspector import SignalInspector
from quality_tool.gui.widgets.tool_panels import MapToolsPanel, SignalToolsPanel
from quality_tool.gui.windows.compare_window import CompareWindow
from quality_tool.metrics.baseline.fringe_visibility import FringeVisibility
from quality_tool.metrics.baseline.power_band_ratio import PowerBandRatio
from quality_tool.metrics.baseline.snr import SNR
from quality_tool.metrics.registry import MetricRegistry

# Number of discrete steps the threshold slider is divided into.
_SLIDER_STEPS = 1000


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

        # ----- backend ---------------------------------------------------
        self._registry = _build_default_registry()

        # ----- session state ---------------------------------------------
        self._signal_set: SignalSet | None = None
        self._selected_metrics: list[str] = []
        self._computed_results: dict[str, MetricMapResult] = {}
        self._threshold_states: dict[str, ThresholdResult | None] = {}
        self._current_map_name: str | None = None
        self._display_mode: str = "score"  # "score" | "masked" | "mask_only"
        self._compare_windows: list[CompareWindow] = []

        # ----- widgets ---------------------------------------------------
        self._map_viewer = MapViewer()
        self._signal_inspector = SignalInspector()
        self._map_tools = MapToolsPanel(slider_steps=_SLIDER_STEPS)
        self._signal_tools = SignalToolsPanel()

        # Convenience aliases so threshold handlers and tests can access
        # the slider / spinbox directly on MainWindow.
        self._thresh_slider = self._map_tools.slider
        self._thresh_spin = self._map_tools.spinbox

        # ----- layout ----------------------------------------------------
        # Map section: viewer + map tools panel side by side
        map_section = QWidget()
        map_layout = QHBoxLayout(map_section)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.addWidget(self._map_viewer, stretch=1)
        map_layout.addWidget(self._map_tools, stretch=0)

        # Signal section: inspector + signal tools panel side by side
        signal_section = QWidget()
        signal_layout = QHBoxLayout(signal_section)
        signal_layout.setContentsMargins(0, 0, 0, 0)
        signal_layout.addWidget(self._signal_inspector, stretch=1)
        signal_layout.addWidget(self._signal_tools, stretch=0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(map_section)
        splitter.addWidget(signal_section)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # ----- toolbar ---------------------------------------------------
        self._build_toolbar()

        # ----- status bar ------------------------------------------------
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — load a dataset to begin")

        # ----- connections -----------------------------------------------
        self._map_viewer.pixel_selected.connect(self._on_pixel_selected)
        self._map_tools.slider_moved.connect(self._on_slider_moved)
        self._map_tools.spin_changed.connect(self._on_spin_changed)
        self._map_tools.apply_clicked.connect(self._on_threshold_apply)
        self._map_tools.reset_clicked.connect(self._on_threshold_reset)

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

        # Metrics… dialog button
        btn_metrics = QPushButton("Metrics…")
        btn_metrics.clicked.connect(self._on_metrics_dialog)
        tb.addWidget(btn_metrics)

        # Compute
        btn_compute = QPushButton("Compute")
        btn_compute.clicked.connect(self._on_compute)
        tb.addWidget(btn_compute)

        tb.addSeparator()

        # Map selector — populated from computed results
        tb.addWidget(QLabel(" Map: "))
        self._map_combo = QComboBox()
        self._map_combo.currentTextChanged.connect(self._on_map_switch)
        tb.addWidget(self._map_combo)

        # Display mode selector
        tb.addWidget(QLabel(" View: "))
        self._display_combo = QComboBox()
        self._display_combo.addItems(["score", "masked", "mask_only"])
        self._display_combo.currentTextChanged.connect(
            self._on_display_mode_changed,
        )
        tb.addWidget(self._display_combo)

        tb.addSeparator()

        # Compare / Info / Export
        btn_compare = QPushButton("Compare")
        btn_compare.clicked.connect(self._on_compare)
        tb.addWidget(btn_compare)

        btn_info = QPushButton("Info")
        btn_info.clicked.connect(self._on_info)
        tb.addWidget(btn_info)

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
        self._clear_session()
        self._map_viewer.clear()
        self._signal_inspector.clear()
        self._refresh_map_combo()

        h, w, m = signal_set.signals.shape
        self._status.showMessage(
            f"Loaded: {signal_set.source_type}  "
            f"({h} x {w} x {m})  "
            f"z_axis={'file' if signal_set.z_axis_path else 'index'}"
        )

    def _on_metrics_dialog(self) -> None:
        """Open the multi-metric selection dialog."""
        dlg = MetricsDialog(
            self._registry, self._selected_metrics, parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._selected_metrics = dlg.selected_metrics()

    def _on_compute(self) -> None:
        """Run all selected metrics on the loaded dataset."""
        if self._signal_set is None:
            self._status.showMessage("No dataset loaded")
            return
        if not self._selected_metrics:
            self._status.showMessage("No metrics selected — use Metrics…")
            return

        newly_computed: list[str] = []
        for name in self._selected_metrics:
            # Skip metrics already computed for this dataset/session.
            if name in self._computed_results:
                continue

            metric = self._registry.get(name)
            self._status.showMessage(f"Computing {name}…")
            self._status.repaint()

            try:
                result = evaluate_metric_map(self._signal_set, metric)
            except Exception as exc:
                QMessageBox.critical(self, "Compute error", str(exc))
                self._status.showMessage(f"Compute failed: {name}")
                return

            self._computed_results[name] = result
            self._threshold_states[name] = None
            newly_computed.append(name)

        self._refresh_map_combo()

        # Show first selected metric by default.
        if self._computed_results:
            first = self._selected_metrics[0]
            idx = self._map_combo.findText(first)
            if idx >= 0:
                self._map_combo.setCurrentIndex(idx)

        self._show_current_map()

        total = len(self._computed_results)
        new = len(newly_computed)
        reused = total - new
        self._status.showMessage(
            f"{total} metric(s) available  ({new} new, {reused} reused)"
        )

    def _on_map_switch(self, text: str) -> None:
        """Switch the displayed map to a different computed metric."""
        if not text:
            return
        self._current_map_name = text
        self._sync_slider_range()
        self._show_current_map()

    def _on_display_mode_changed(self, text: str) -> None:
        """Switch between score / masked / mask_only display."""
        self._display_mode = text
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
        name = self._current_map_name or "—"
        val_str = f"{value:.4g}" if value is not None else "—"
        self._status.showMessage(
            f"Pixel ({row}, {col})  {name}={val_str}"
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

        # List computed metrics.
        if self._computed_results:
            info["Computed metrics"] = ", ".join(self._computed_results.keys())

        # Current map and threshold info.
        if self._current_map_name:
            info["Displayed metric"] = self._current_map_name
            tr = self._threshold_states.get(self._current_map_name)
            if tr is not None:
                info["Threshold"] = (
                    f"{tr.threshold:.4g}  ({tr.keep_rule})"
                )
                if tr.stats:
                    info["Kept pixels"] = str(tr.stats.get("kept_pixels", "—"))

        dlg = InfoDialog(info, parent=self)
        dlg.exec()

    def _on_export(self) -> None:
        name = self._current_map_name
        if name is None or name not in self._computed_results:
            self._status.showMessage("Nothing to export")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export current map", "", "Text files (*.txt)",
        )
        if not path:
            return

        mode = self._display_mode
        result = self._computed_results[name]
        tr = self._threshold_states.get(name)

        if mode == "mask_only" and tr is not None:
            data = tr.mask.astype(int)
        else:
            data = result.score_map

        np.savetxt(path, data, fmt="%.6g")
        self._status.showMessage(f"Exported {name} ({mode}) → {path}")

    # ==================================================================
    # Threshold handlers
    # ==================================================================

    def _on_slider_moved(self, tick: int) -> None:
        """Map integer slider position to float value and sync spinbox."""
        value = self._tick_to_value(tick)
        self._thresh_spin.blockSignals(True)
        self._thresh_spin.setValue(value)
        self._thresh_spin.blockSignals(False)

    def _on_spin_changed(self, value: float) -> None:
        """Sync slider position when the spinbox is edited directly."""
        tick = self._value_to_tick(value)
        self._thresh_slider.blockSignals(True)
        self._thresh_slider.setValue(tick)
        self._thresh_slider.blockSignals(False)

    def _on_threshold_apply(self) -> None:
        """Apply threshold to the currently displayed metric map."""
        name = self._current_map_name
        if name is None or name not in self._computed_results:
            self._status.showMessage("No metric map to threshold")
            return

        threshold_value = self._thresh_spin.value()
        result = self._computed_results[name]

        self._threshold_states[name] = apply_threshold(
            result, threshold_value, keep_rule="above",
        )

        # Switch to masked view automatically.
        self._display_combo.blockSignals(True)
        self._display_combo.setCurrentText("masked")
        self._display_combo.blockSignals(False)
        self._display_mode = "masked"

        self._show_current_map()

        tr = self._threshold_states[name]
        kept = tr.stats.get("kept_pixels", "?") if tr.stats else "?"
        self._status.showMessage(
            f"Threshold {threshold_value:.4g} applied to {name}  "
            f"({kept} kept)"
        )

    def _on_threshold_reset(self) -> None:
        """Clear threshold for the currently displayed metric map."""
        name = self._current_map_name
        if name is not None:
            self._threshold_states[name] = None

        # Switch back to raw score view.
        self._display_combo.blockSignals(True)
        self._display_combo.setCurrentText("score")
        self._display_combo.blockSignals(False)
        self._display_mode = "score"

        self._show_current_map()
        self._status.showMessage("Threshold reset")

    # ==================================================================
    # Helpers
    # ==================================================================

    def _clear_session(self) -> None:
        """Reset session state when a new dataset is loaded."""
        self._computed_results.clear()
        self._threshold_states.clear()
        self._current_map_name = None
        self._display_mode = "score"
        self._display_combo.blockSignals(True)
        self._display_combo.setCurrentText("score")
        self._display_combo.blockSignals(False)

    def _refresh_map_combo(self) -> None:
        """Populate map selector with actually computed metric names."""
        self._map_combo.blockSignals(True)
        self._map_combo.clear()
        for name in self._computed_results:
            self._map_combo.addItem(name)
        self._map_combo.blockSignals(False)

        if self._computed_results:
            first = list(self._computed_results.keys())[0]
            self._current_map_name = first

    def _show_current_map(self) -> None:
        """Render the map currently selected in the combo box."""
        name = self._current_map_name
        if name is None or name not in self._computed_results:
            return

        result = self._computed_results[name]
        tr = self._threshold_states.get(name)

        # Compute stable color range from the full original score map.
        valid = result.valid_map
        valid_scores = result.score_map[valid]
        if valid_scores.size > 0:
            vmin = float(np.nanmin(valid_scores))
            vmax = float(np.nanmax(valid_scores))
        else:
            vmin, vmax = 0.0, 1.0

        if self._display_mode == "masked" and tr is not None:
            self._map_viewer.set_masked_map(
                result.score_map,
                tr.mask,
                title=f"{name} — masked ({tr.keep_rule})",
                vmin=vmin,
                vmax=vmax,
            )
        elif self._display_mode == "mask_only" and tr is not None:
            self._map_viewer.set_binary_mask(
                tr.mask,
                title=f"{name} — mask ({tr.keep_rule})",
            )
        else:
            self._map_viewer.set_map(
                result.score_map,
                title=f"{name} — score map",
            )

    def _sync_slider_range(self) -> None:
        """Update threshold slider/spinbox range to match current map."""
        name = self._current_map_name
        if name is None or name not in self._computed_results:
            return

        result = self._computed_results[name]
        valid = result.valid_map
        valid_scores = result.score_map[valid]

        if valid_scores.size > 0:
            lo = float(np.nanmin(valid_scores))
            hi = float(np.nanmax(valid_scores))
        else:
            lo, hi = 0.0, 1.0

        # Prevent degenerate range.
        if hi <= lo:
            hi = lo + 1.0

        self._slider_min = lo
        self._slider_max = hi

        self._thresh_spin.blockSignals(True)
        self._thresh_spin.setRange(lo, hi)
        self._thresh_spin.setSingleStep((hi - lo) / 100.0)
        self._thresh_spin.setValue(lo)
        self._thresh_spin.blockSignals(False)

        self._thresh_slider.blockSignals(True)
        self._thresh_slider.setValue(0)
        self._thresh_slider.blockSignals(False)

    def _tick_to_value(self, tick: int) -> float:
        """Convert integer slider tick to float threshold value."""
        lo = getattr(self, "_slider_min", 0.0)
        hi = getattr(self, "_slider_max", 1.0)
        return lo + (hi - lo) * tick / _SLIDER_STEPS

    def _value_to_tick(self, value: float) -> int:
        """Convert float threshold value to integer slider tick."""
        lo = getattr(self, "_slider_min", 0.0)
        hi = getattr(self, "_slider_max", 1.0)
        if hi <= lo:
            return 0
        frac = (value - lo) / (hi - lo)
        return max(0, min(_SLIDER_STEPS, int(round(frac * _SLIDER_STEPS))))


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
