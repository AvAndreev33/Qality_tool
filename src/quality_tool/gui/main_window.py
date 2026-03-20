"""Main application window for Quality_tool GUI.

Provides the central layout (toolbar -> map section -> signal section ->
status bar) and wires user actions to backend calls.

Layout:
    +--------------------------------------------------------------+
    | Toolbar (Load | Metrics Settings Compute | Map View |        |
    |          Compare Info Export)                                 |
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
from quality_tool.envelope.analytic import AnalyticEnvelopeMethod
from quality_tool.envelope.registry import EnvelopeRegistry
from quality_tool.evaluation.evaluator import evaluate_metric_maps
from quality_tool.evaluation.recipe import recipe_from_processing
from quality_tool.evaluation.thresholding import apply_threshold
from quality_tool.gui.dialogs.info_dialog import InfoDialog
from quality_tool.gui.dialogs.metrics_dialog import MetricsDialog
from quality_tool.gui.dialogs.processing_dialog import ProcessingDialog
from quality_tool.gui.widgets.map_viewer import MapViewer
from quality_tool.gui.widgets.signal_inspector import SignalInspector
from quality_tool.gui.widgets.tool_panels import MapToolsPanel, SignalToolsPanel
from quality_tool.gui.windows.compare_window import CompareWindow
from quality_tool.gui.windows.histogram_window import HistogramWindow
from quality_tool.metrics.baseline.fringe_visibility import FringeVisibility
from quality_tool.metrics.baseline.power_band_ratio import PowerBandRatio
from quality_tool.metrics.baseline.snr import SNR
from quality_tool.metrics.registry import MetricRegistry
from quality_tool.preprocessing.basic import (
    normalize_amplitude,
    smooth,
    subtract_baseline,
)
from quality_tool.preprocessing.roi import extract_roi
from quality_tool.spectral.fft import compute_spectrum

# Number of discrete steps the threshold slider is divided into.
_SLIDER_STEPS = 1000


def _build_default_registry() -> MetricRegistry:
    """Create a registry populated with the baseline metrics."""
    registry = MetricRegistry()
    registry.register(FringeVisibility())
    registry.register(SNR())
    registry.register(PowerBandRatio())
    return registry


def _build_envelope_registry() -> EnvelopeRegistry:
    """Create a registry populated with the available envelope methods."""
    registry = EnvelopeRegistry()
    registry.register(AnalyticEnvelopeMethod())
    return registry


# Default processing settings returned by ProcessingDialog when no
# prior settings exist.
_DEFAULT_PROCESSING: dict = {
    "baseline": False,
    "normalize": False,
    "smooth": False,
    "roi_enabled": False,
    "segment_size": 128,
    "envelope_enabled": False,
    "envelope_method": "analytic",
}


class MainWindow(QMainWindow):
    """Central application window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Quality_tool")
        self.resize(900, 750)

        # ----- backend ---------------------------------------------------
        self._registry = _build_default_registry()
        self._envelope_registry = _build_envelope_registry()

        # ----- session state ---------------------------------------------
        self._signal_set: SignalSet | None = None
        self._selected_metrics: list[str] = []
        self._computed_results: dict[str, MetricMapResult] = {}
        self._current_map_name: str | None = None
        self._display_mode: str = "score"  # "score" | "masked" | "mask_only"
        self._compare_windows: list[CompareWindow] = []
        self._histogram_windows: list[HistogramWindow] = []

        # Mask-source metric: which metric's score map drives the threshold.
        self._mask_source_metric: str | None = None
        # Current threshold result (built from the mask-source metric).
        self._current_threshold: ThresholdResult | None = None

        # Processing / envelope / signal-display state
        self._processing: dict = dict(_DEFAULT_PROCESSING)
        self._signal_display_mode: str = "Raw"
        self._envelope_overlay: bool = False
        # Track the last-selected pixel for signal-display-mode switching
        self._selected_pixel: tuple[int, int] | None = None

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
        self._map_tools.mask_source_changed.connect(
            self._on_mask_source_changed,
        )
        self._signal_tools.display_mode_changed.connect(
            self._on_signal_display_mode_changed,
        )
        self._signal_tools.envelope_toggled.connect(
            self._on_envelope_toggled,
        )

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

        # Settings… dialog button
        btn_settings = QPushButton("Settings…")
        btn_settings.clicked.connect(self._on_settings_dialog)
        tb.addWidget(btn_settings)

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

        btn_histogram = QPushButton("Histogram")
        btn_histogram.clicked.connect(self._on_histogram)
        tb.addWidget(btn_histogram)

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

    def _on_settings_dialog(self) -> None:
        """Open the processing settings dialog."""
        dlg = ProcessingDialog(
            envelope_methods=self._envelope_registry.list_methods(),
            current=self._processing,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        new_settings = dlg.settings()
        # If processing settings changed, invalidate cached results for
        # metrics with active recipe binding.  Fixed-recipe metrics are
        # unaffected by session pipeline changes.
        if new_settings != self._processing:
            names_to_drop = [
                name for name in self._computed_results
                if self._computed_results[name].metadata.get(
                    "recipe_binding", "active"
                ) != "fixed"
            ]
            for name in names_to_drop:
                del self._computed_results[name]

            # Always reset threshold — the mask-source metric's scores
            # may have been invalidated.
            self._current_threshold = None

            # Keep mask-source and current-map if they still exist in
            # the surviving results; otherwise fall back.
            if self._mask_source_metric not in self._computed_results:
                self._mask_source_metric = None
            if self._current_map_name not in self._computed_results:
                self._current_map_name = None

            self._refresh_map_combo()

            if self._computed_results:
                # Surviving results exist — show the first one.
                self._show_current_map()
            else:
                self._map_viewer.clear()

            self._status.showMessage(
                "Settings changed — press Compute to re-evaluate"
            )
        self._processing = new_settings

    def _on_compute(self) -> None:
        """Run all selected metrics on the loaded dataset."""
        if self._signal_set is None:
            self._status.showMessage("No dataset loaded")
            return
        if not self._selected_metrics:
            self._status.showMessage("No metrics selected — use Metrics…")
            return

        # Collect metrics that still need computation.
        metrics_to_compute = []
        for name in self._selected_metrics:
            if name not in self._computed_results:
                metrics_to_compute.append(self._registry.get(name))

        if not metrics_to_compute:
            # All selected metrics are already cached.
            self._refresh_map_combo()
            self._show_current_map()
            total = len(self._computed_results)
            self._status.showMessage(
                f"{total} metric(s) available  (0 new, {total} reused)"
            )
            return

        # Build active recipe from current processing settings.
        active_recipe = recipe_from_processing(self._processing)
        envelope_method = self._get_envelope_method()

        names_to_compute = [m.name for m in metrics_to_compute]
        self._status.showMessage(
            f"Computing {', '.join(names_to_compute)}…"
        )
        self._status.repaint()

        try:
            new_results = evaluate_metric_maps(
                self._signal_set,
                metrics_to_compute,
                active_recipe=active_recipe,
                envelope_method=envelope_method,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Compute error", str(exc))
            self._status.showMessage("Compute failed")
            return

        self._computed_results.update(new_results)

        self._refresh_map_combo()

        # Show first selected metric by default.
        if self._computed_results:
            first = self._selected_metrics[0]
            idx = self._map_combo.findText(first)
            if idx >= 0:
                self._map_combo.setCurrentIndex(idx)

        self._show_current_map()

        total = len(self._computed_results)
        new = len(new_results)
        reused = total - new
        self._status.showMessage(
            f"{total} metric(s) available  ({new} new, {reused} reused)"
        )

    def _on_map_switch(self, text: str) -> None:
        """Switch the displayed map to a different computed metric."""
        if not text:
            return
        self._current_map_name = text
        self._show_current_map()

    def _on_display_mode_changed(self, text: str) -> None:
        """Switch between score / masked / mask_only display."""
        self._display_mode = text
        self._show_current_map()

    def _on_pixel_selected(self, row: int, col: int) -> None:
        if self._signal_set is None:
            return

        self._selected_pixel = (row, col)
        self._update_signal_display(row, col)

        value = self._map_viewer.value_at(row, col)
        name = self._current_map_name or "—"
        val_str = f"{value:.4g}" if value is not None else "—"
        self._status.showMessage(
            f"Pixel ({row}, {col})  {name}={val_str}"
        )

    def _on_signal_display_mode_changed(self, text: str) -> None:
        """Switch signal inspector display mode and refresh."""
        self._signal_display_mode = text
        if self._selected_pixel is not None:
            self._update_signal_display(*self._selected_pixel)

    def _on_envelope_toggled(self, checked: bool) -> None:
        """Toggle envelope overlay on the signal inspector."""
        self._envelope_overlay = checked
        if self._selected_pixel is not None:
            self._update_signal_display(*self._selected_pixel)

    def _on_mask_source_changed(self, text: str) -> None:
        """Update mask-source metric and sync slider range."""
        if not text:
            return
        self._mask_source_metric = text
        self._sync_slider_range()

    def _on_compare(self) -> None:
        data, title = self._map_viewer.get_snapshot()
        if data is None:
            self._status.showMessage("No map to compare")
            return

        win = CompareWindow(data, title=title)
        win.show()
        self._compare_windows.append(win)

    def _on_histogram(self) -> None:
        """Open a histogram window for the currently displayed map."""
        name = self._current_map_name
        if name is None or name not in self._computed_results:
            self._status.showMessage("No map to show histogram for")
            return

        result = self._computed_results[name]
        valid_values = result.score_map[result.valid_map]

        # Determine threshold applicability: threshold must be active AND
        # built from the same metric that is currently displayed.
        tr = self._current_threshold
        if tr is not None and self._mask_source_metric == name:
            threshold_value = tr.threshold
            keep_rule = tr.keep_rule
            threshold_stats = dict(tr.stats) if tr.stats else None
        else:
            threshold_value = None
            keep_rule = None
            threshold_stats = None

        win = HistogramWindow(
            values=valid_values,
            metric_name=name,
            threshold_value=threshold_value,
            keep_rule=keep_rule,
            threshold_stats=threshold_stats,
        )
        win.show()
        self._histogram_windows.append(win)

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

        # Processing settings
        p = self._processing
        pp_items = []
        if p.get("baseline"):
            pp_items.append("baseline")
        if p.get("normalize"):
            pp_items.append("normalize")
        if p.get("smooth"):
            pp_items.append("smooth")
        info["Preprocessing"] = ", ".join(pp_items) if pp_items else "none"

        if p.get("roi_enabled"):
            info["ROI"] = f"segment_size={p.get('segment_size', '?')}, centering=raw_max"
        else:
            info["ROI"] = "disabled"

        if p.get("envelope_enabled"):
            info["Envelope"] = p.get("envelope_method", "?")
        else:
            info["Envelope"] = "disabled"

        # List computed metrics.
        if self._computed_results:
            info["Computed metrics"] = ", ".join(self._computed_results.keys())

        # Current map and threshold info.
        if self._current_map_name:
            info["Displayed metric"] = self._current_map_name

        if self._mask_source_metric:
            info["Mask source"] = self._mask_source_metric

        tr = self._current_threshold
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
        tr = self._current_threshold

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
        """Apply threshold using the mask-source metric's score map."""
        source = self._mask_source_metric
        if source is None or source not in self._computed_results:
            self._status.showMessage("No mask-source metric selected")
            return

        threshold_value = self._thresh_spin.value()
        result = self._computed_results[source]

        self._current_threshold = apply_threshold(
            result, threshold_value, keep_rule="above",
        )

        # Switch to masked view automatically.
        self._display_combo.blockSignals(True)
        self._display_combo.setCurrentText("masked")
        self._display_combo.blockSignals(False)
        self._display_mode = "masked"

        self._show_current_map()

        tr = self._current_threshold
        kept = tr.stats.get("kept_pixels", "?") if tr.stats else "?"
        self._status.showMessage(
            f"Threshold {threshold_value:.4g} on {source}  "
            f"({kept} kept)"
        )

    def _on_threshold_reset(self) -> None:
        """Clear the current threshold."""
        self._current_threshold = None

        # Switch back to raw score view.
        self._display_combo.blockSignals(True)
        self._display_combo.setCurrentText("score")
        self._display_combo.blockSignals(False)
        self._display_mode = "score"

        self._show_current_map()
        self._status.showMessage("Threshold reset")

    # ==================================================================
    # Signal display logic
    # ==================================================================

    def _update_signal_display(self, row: int, col: int) -> None:
        """Render the signal inspector for the given pixel using the
        current signal display mode.

        All processing is done via backend functions — the inspector
        widget only receives pre-computed data.
        """
        if self._signal_set is None:
            return

        signal = self._signal_set.signals[row, col, :].copy()
        z_axis = self._signal_set.z_axis
        title = f"Pixel ({row}, {col})"
        mode = self._signal_display_mode

        if mode == "Raw":
            envelope = self._compute_envelope_for_display(signal) if self._envelope_overlay else None
            self._signal_inspector.update_signal(
                signal, z_axis, label="raw", title=title, envelope=envelope,
            )

        elif mode == "Processed":
            processed, proc_z = self._apply_processing_pipeline(signal, z_axis)
            envelope = self._compute_envelope_for_display(processed) if self._envelope_overlay else None
            self._signal_inspector.update_signal(
                processed, proc_z, label="processed", title=title,
                envelope=envelope,
            )

        elif mode == "Spectrum":
            try:
                spectral = compute_spectrum(signal, z_axis)
            except Exception:
                self._signal_inspector.update_signal(
                    signal, z_axis, title=f"{title} (spectrum error)",
                )
                return
            self._signal_inspector.update_spectrum(
                spectral.frequencies, spectral.amplitude, title=title,
            )

        else:
            # Fallback — raw signal
            self._signal_inspector.update_signal(
                signal, z_axis, title=title,
            )

    def _apply_processing_pipeline(
        self, signal: np.ndarray, z_axis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the enabled preprocessing steps and ROI to a signal.

        Returns the (possibly shortened) signal and its z-axis.
        """
        processed = signal.copy()

        # Step 1: preprocessing
        for fn in self._build_preprocess_list():
            processed = fn(processed)

        # Step 2: ROI extraction
        seg_size = self._get_segment_size()
        if seg_size is not None:
            try:
                processed = extract_roi(processed, seg_size)
                # ROI shortens the signal — build a matching z-axis
                z_axis = np.arange(len(processed), dtype=float)
            except Exception:
                pass  # keep the full processed signal on ROI error

        return processed, z_axis

    def _compute_envelope_for_display(
        self, signal: np.ndarray,
    ) -> np.ndarray | None:
        """Compute envelope for display, baseline-subtracting first.

        Returns None if envelope computation is not possible.
        """
        env_method = self._get_envelope_method()
        if env_method is None:
            return None
        try:
            centered = subtract_baseline(signal)
            return env_method.compute(centered)
        except Exception:
            return None

    # ==================================================================
    # Processing helpers
    # ==================================================================

    def _build_preprocess_list(self) -> list:
        """Build an ordered list of preprocessing callables from settings."""
        fns = []
        if self._processing.get("baseline"):
            fns.append(subtract_baseline)
        if self._processing.get("normalize"):
            fns.append(normalize_amplitude)
        if self._processing.get("smooth"):
            fns.append(smooth)
        return fns

    def _get_segment_size(self) -> int | None:
        """Return segment_size if ROI is enabled, else None."""
        if self._processing.get("roi_enabled"):
            return self._processing.get("segment_size", 128)
        return None

    def _get_envelope_method(self):
        """Return the selected envelope method instance, or None."""
        if not self._processing.get("envelope_enabled"):
            return None
        name = self._processing.get("envelope_method", "")
        if not name:
            return None
        try:
            return self._envelope_registry.get(name)
        except KeyError:
            return None

    # ==================================================================
    # Helpers
    # ==================================================================

    def _clear_session(self) -> None:
        """Reset session state when a new dataset is loaded."""
        self._computed_results.clear()
        self._current_threshold = None
        self._mask_source_metric = None
        self._current_map_name = None
        self._display_mode = "score"
        self._selected_pixel = None
        self._envelope_overlay = False
        self._display_combo.blockSignals(True)
        self._display_combo.setCurrentText("score")
        self._display_combo.blockSignals(False)

    def _refresh_map_combo(self) -> None:
        """Populate map selector and mask-source combo with computed names.

        Syncs ``_current_map_name`` and ``_mask_source_metric`` so that
        they always reflect what the combo widgets display.  If the
        previous selection still exists in the results it is preserved;
        otherwise the first available result is used.
        """
        names = list(self._computed_results.keys())

        # --- map combo ---
        self._map_combo.blockSignals(True)
        self._map_combo.clear()
        for name in names:
            self._map_combo.addItem(name)
        self._map_combo.blockSignals(False)

        # --- mask-source combo ---
        self._map_tools.mask_source_combo.blockSignals(True)
        self._map_tools.mask_source_combo.clear()
        for name in names:
            self._map_tools.mask_source_combo.addItem(name)
        self._map_tools.mask_source_combo.blockSignals(False)

        # --- sync internal state with combo contents ---
        if names:
            # Preserve previous selection when it still exists.
            if self._current_map_name not in self._computed_results:
                self._current_map_name = names[0]
            # Point the map combo widget at the active name.
            idx = self._map_combo.findText(self._current_map_name)
            if idx >= 0:
                self._map_combo.blockSignals(True)
                self._map_combo.setCurrentIndex(idx)
                self._map_combo.blockSignals(False)

            if self._mask_source_metric not in self._computed_results:
                self._mask_source_metric = names[0]
            idx = self._map_tools.mask_source_combo.findText(
                self._mask_source_metric,
            )
            if idx >= 0:
                self._map_tools.mask_source_combo.blockSignals(True)
                self._map_tools.mask_source_combo.setCurrentIndex(idx)
                self._map_tools.mask_source_combo.blockSignals(False)
            self._sync_slider_range()
        else:
            self._current_map_name = None
            self._mask_source_metric = None

    def _show_current_map(self) -> None:
        """Render the map currently selected in the combo box."""
        name = self._current_map_name
        if name is None or name not in self._computed_results:
            return

        result = self._computed_results[name]
        tr = self._current_threshold

        # Compute stable color range from the full original score map.
        valid = result.valid_map
        valid_scores = result.score_map[valid]
        if valid_scores.size > 0:
            vmin = float(np.nanmin(valid_scores))
            vmax = float(np.nanmax(valid_scores))
        else:
            vmin, vmax = 0.0, 1.0

        if self._display_mode == "masked" and tr is not None:
            source = self._mask_source_metric or name
            self._map_viewer.set_masked_map(
                result.score_map,
                tr.mask,
                title=f"{name} — masked (source: {source})",
                vmin=vmin,
                vmax=vmax,
            )
        elif self._display_mode == "mask_only" and tr is not None:
            source = self._mask_source_metric or name
            self._map_viewer.set_binary_mask(
                tr.mask,
                title=f"{name} — mask (source: {source})",
            )
        else:
            self._map_viewer.set_map(
                result.score_map,
                title=f"{name} — score map",
            )

    def _sync_slider_range(self) -> None:
        """Update threshold slider/spinbox range to match mask-source metric."""
        source = self._mask_source_metric
        if source is None or source not in self._computed_results:
            return

        result = self._computed_results[source]
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
