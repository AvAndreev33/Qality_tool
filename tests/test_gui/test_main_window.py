"""Tests for the main window, session state, and helper dialogs."""

from __future__ import annotations

import numpy as np

from quality_tool.core.models import MetricMapResult, SignalSet
from quality_tool.evaluation.evaluator import evaluate_metric_map
from quality_tool.evaluation.thresholding import apply_threshold
from quality_tool.gui.dialogs.info_dialog import InfoDialog
from quality_tool.gui.main_window import MainWindow
from quality_tool.gui.windows.compare_window import CompareWindow


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_metric_map(name: str, h: int = 4, w: int = 5) -> MetricMapResult:
    """Create a small metric map result for testing."""
    return MetricMapResult(
        metric_name=name,
        score_map=np.random.rand(h, w),
        valid_map=np.ones((h, w), dtype=bool),
        feature_maps={},
        metadata={"metric_name": name},
    )


# ------------------------------------------------------------------
# MainWindow
# ------------------------------------------------------------------

class TestMainWindow:
    def test_creation(self):
        window = MainWindow()
        assert window is not None

    def test_registry_populated(self):
        window = MainWindow()
        names = window._registry.list_metrics()
        assert "fringe_visibility" in names
        assert "snr" in names
        assert "power_band_ratio" in names

    def test_initial_session_state(self):
        window = MainWindow()
        assert window._signal_set is None
        assert window._selected_metrics == []
        assert window._computed_results == {}
        assert window._current_map_name is None
        assert window._display_mode == "score"
        assert window._selected_pixel is None
        assert window._signal_display_mode == "Raw"
        assert window._mask_source_metric is None
        assert window._current_threshold is None
        assert window._envelope_overlay is False

    def test_initial_processing_state(self):
        window = MainWindow()
        p = window._processing
        assert p["baseline"] is False
        assert p["normalize"] is False
        assert p["smooth"] is False
        assert p["roi_enabled"] is False
        assert p["segment_size"] == 128
        assert p["envelope_enabled"] is False
        assert p["envelope_method"] == "analytic"

    def test_envelope_registry_populated(self):
        window = MainWindow()
        methods = window._envelope_registry.list_methods()
        assert "analytic" in methods

    def test_map_combo_empty_initially(self):
        window = MainWindow()
        assert window._map_combo.count() == 0

    def test_layout_has_tool_panels(self):
        """Map and signal tool panels must be present in the layout."""
        window = MainWindow()
        assert window._map_tools is not None
        assert window._signal_tools is not None

    def test_thresh_controls_in_map_tools_panel(self):
        """Threshold slider/spinbox must be children of the map tools panel."""
        window = MainWindow()
        assert window._thresh_slider is window._map_tools.slider
        assert window._thresh_spin is window._map_tools.spinbox

    def test_refresh_map_combo_from_results(self):
        window = MainWindow()
        window._computed_results["snr"] = _make_metric_map("snr")
        window._computed_results["fringe_visibility"] = _make_metric_map(
            "fringe_visibility"
        )
        window._refresh_map_combo()
        items = [
            window._map_combo.itemText(i)
            for i in range(window._map_combo.count())
        ]
        assert "snr" in items
        assert "fringe_visibility" in items

    def test_refresh_map_combo_populates_mask_source(self):
        """Mask-source combo must be populated alongside map combo."""
        window = MainWindow()
        window._computed_results["snr"] = _make_metric_map("snr")
        window._computed_results["fringe_visibility"] = _make_metric_map(
            "fringe_visibility"
        )
        window._refresh_map_combo()
        combo = window._map_tools.mask_source_combo
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "snr" in items
        assert "fringe_visibility" in items

    def test_clear_session_resets_state(self):
        window = MainWindow()
        window._computed_results["snr"] = _make_metric_map("snr")
        window._current_threshold = apply_threshold(
            window._computed_results["snr"], 0.5,
        )
        window._mask_source_metric = "snr"
        window._current_map_name = "snr"
        window._display_mode = "masked"
        window._envelope_overlay = True

        window._clear_session()

        assert window._computed_results == {}
        assert window._current_threshold is None
        assert window._mask_source_metric is None
        assert window._current_map_name is None
        assert window._display_mode == "score"
        assert window._envelope_overlay is False

    def test_threshold_apply_stores_result(self):
        window = MainWindow()
        mm = _make_metric_map("snr")
        window._computed_results["snr"] = mm
        window._mask_source_metric = "snr"
        window._current_map_name = "snr"
        window._slider_min = 0.0
        window._slider_max = 1.0
        window._thresh_spin.setValue(0.5)

        window._on_threshold_apply()

        tr = window._current_threshold
        assert tr is not None
        assert tr.threshold == 0.5
        assert tr.mask.shape == mm.score_map.shape
        assert window._display_mode == "masked"

    def test_threshold_apply_uses_mask_source(self):
        """Threshold should be built from mask-source metric, not displayed."""
        window = MainWindow()
        mm_snr = _make_metric_map("snr")
        mm_fv = _make_metric_map("fringe_visibility")
        window._computed_results["snr"] = mm_snr
        window._computed_results["fringe_visibility"] = mm_fv
        window._current_map_name = "fringe_visibility"
        window._mask_source_metric = "snr"
        window._slider_min = 0.0
        window._slider_max = 1.0
        window._thresh_spin.setValue(0.5)

        window._on_threshold_apply()

        tr = window._current_threshold
        assert tr is not None
        # Mask shape matches the source metric's score map
        assert tr.mask.shape == mm_snr.score_map.shape

    def test_threshold_reset_clears(self):
        window = MainWindow()
        mm = _make_metric_map("snr")
        window._computed_results["snr"] = mm
        window._current_map_name = "snr"
        window._mask_source_metric = "snr"
        window._current_threshold = apply_threshold(mm, 0.5)

        window._on_threshold_reset()

        assert window._current_threshold is None
        assert window._display_mode == "score"

    def test_show_current_map_score(self):
        """Score mode should call set_map without error."""
        window = MainWindow()
        window._computed_results["snr"] = _make_metric_map("snr")
        window._current_map_name = "snr"
        window._display_mode = "score"
        window._show_current_map()
        assert window._map_viewer._data is not None

    def test_show_current_map_masked(self):
        """Masked mode should call set_masked_map without error."""
        window = MainWindow()
        mm = _make_metric_map("snr")
        window._computed_results["snr"] = mm
        window._current_threshold = apply_threshold(mm, 0.5)
        window._current_map_name = "snr"
        window._display_mode = "masked"
        window._show_current_map()
        assert window._map_viewer._data is not None

    def test_show_current_map_mask_only(self):
        """Mask-only mode should call set_binary_mask without error."""
        window = MainWindow()
        mm = _make_metric_map("snr")
        window._computed_results["snr"] = mm
        window._current_threshold = apply_threshold(mm, 0.5)
        window._current_map_name = "snr"
        window._display_mode = "mask_only"
        window._show_current_map()
        assert window._map_viewer._data is not None

    def test_slider_tick_round_trip(self):
        """tick_to_value and value_to_tick should be inverses."""
        window = MainWindow()
        window._slider_min = 0.0
        window._slider_max = 10.0
        for tick in [0, 250, 500, 750, 1000]:
            val = window._tick_to_value(tick)
            assert window._value_to_tick(val) == tick

    def test_compute_incremental_skips_existing(self, monkeypatch):
        """Already-computed metrics must not be recomputed."""
        window = MainWindow()
        # Fake a loaded signal set (never actually used because the
        # metric that would trigger compute is already cached).
        window._signal_set = object()  # truthy sentinel

        existing = _make_metric_map("snr")
        window._computed_results["snr"] = existing

        # Select snr (cached) + fringe_visibility (new).
        window._selected_metrics = ["snr", "fringe_visibility"]

        # Track calls to evaluate_metric_map.
        calls: list[str] = []
        original_evaluate = evaluate_metric_map

        def tracking_evaluate(signal_set, metric, **kw):
            calls.append(metric.name)
            return _make_metric_map(metric.name)

        from quality_tool.evaluation import evaluator
        monkeypatch.setattr(evaluator, "evaluate_metric_map", tracking_evaluate)
        # Also patch the name used inside main_window module.
        import quality_tool.gui.main_window as mw_mod
        monkeypatch.setattr(mw_mod, "evaluate_metric_map", tracking_evaluate)

        window._on_compute()

        # Only fringe_visibility should have been computed.
        assert calls == ["fringe_visibility"]
        # Both should be in results.
        assert "snr" in window._computed_results
        assert "fringe_visibility" in window._computed_results
        # The pre-existing snr result must be the same object (not replaced).
        assert window._computed_results["snr"] is existing

    def test_compute_incremental_preserves_threshold(self, monkeypatch):
        """Global threshold must survive a second Compute when cached."""
        window = MainWindow()
        window._signal_set = object()

        mm = _make_metric_map("snr")
        window._computed_results["snr"] = mm
        tr = apply_threshold(mm, 0.5)
        window._current_threshold = tr

        window._selected_metrics = ["snr"]

        import quality_tool.gui.main_window as mw_mod
        monkeypatch.setattr(
            mw_mod, "evaluate_metric_map",
            lambda *a, **k: _make_metric_map("snr"),
        )

        window._on_compute()

        # Threshold must still be the same object — not reset.
        assert window._current_threshold is tr

    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------

    def test_build_preprocess_list_empty(self):
        window = MainWindow()
        assert window._build_preprocess_list() == []

    def test_build_preprocess_list_all(self):
        window = MainWindow()
        window._processing["baseline"] = True
        window._processing["normalize"] = True
        window._processing["smooth"] = True
        fns = window._build_preprocess_list()
        assert len(fns) == 3
        names = [fn.__name__ for fn in fns]
        assert "subtract_baseline" in names
        assert "normalize_amplitude" in names
        assert "smooth" in names

    def test_get_segment_size_disabled(self):
        window = MainWindow()
        assert window._get_segment_size() is None

    def test_get_segment_size_enabled(self):
        window = MainWindow()
        window._processing["roi_enabled"] = True
        window._processing["segment_size"] = 64
        assert window._get_segment_size() == 64

    def test_get_envelope_method_disabled(self):
        window = MainWindow()
        assert window._get_envelope_method() is None

    def test_get_envelope_method_enabled(self):
        window = MainWindow()
        window._processing["envelope_enabled"] = True
        window._processing["envelope_method"] = "analytic"
        method = window._get_envelope_method()
        assert method is not None
        assert method.name == "analytic"

    # ------------------------------------------------------------------
    # Signal display mode
    # ------------------------------------------------------------------

    def test_signal_display_mode_switching(self):
        """Switching signal display mode must update internal state."""
        window = MainWindow()
        window._on_signal_display_mode_changed("Spectrum")
        assert window._signal_display_mode == "Spectrum"

    def test_signal_display_raw(self):
        """Raw mode must render without error on a loaded dataset."""
        window = MainWindow()
        ss = SignalSet(
            signals=np.random.rand(4, 5, 100),
            width=5, height=4,
            z_axis=np.arange(100, dtype=float),
        )
        window._signal_set = ss
        window._signal_display_mode = "Raw"
        window._update_signal_display(0, 0)

    def test_signal_display_spectrum(self):
        """Spectrum mode must render without error."""
        window = MainWindow()
        ss = SignalSet(
            signals=np.random.rand(4, 5, 100),
            width=5, height=4,
            z_axis=np.arange(100, dtype=float),
        )
        window._signal_set = ss
        window._signal_display_mode = "Spectrum"
        window._update_signal_display(0, 0)

    def test_signal_display_processed(self):
        """Processed mode must render without error."""
        window = MainWindow()
        ss = SignalSet(
            signals=np.random.rand(4, 5, 100),
            width=5, height=4,
            z_axis=np.arange(100, dtype=float),
        )
        window._signal_set = ss
        window._processing["baseline"] = True
        window._signal_display_mode = "Processed"
        window._update_signal_display(0, 0)

    def test_signal_display_processed_with_roi(self):
        """Processed mode with ROI must render without error."""
        window = MainWindow()
        ss = SignalSet(
            signals=np.random.rand(4, 5, 100),
            width=5, height=4,
            z_axis=np.arange(100, dtype=float),
        )
        window._signal_set = ss
        window._processing["roi_enabled"] = True
        window._processing["segment_size"] = 32
        window._signal_display_mode = "Processed"
        window._update_signal_display(0, 0)

    def test_signal_display_envelope_overlay(self):
        """Raw + Envelope overlay must render when envelope is enabled."""
        window = MainWindow()
        ss = SignalSet(
            signals=np.random.rand(4, 5, 100),
            width=5, height=4,
            z_axis=np.arange(100, dtype=float),
        )
        window._signal_set = ss
        window._processing["envelope_enabled"] = True
        window._processing["envelope_method"] = "analytic"
        window._envelope_overlay = True
        window._signal_display_mode = "Raw"
        window._update_signal_display(0, 0)

    def test_signal_display_envelope_overlay_disabled(self):
        """Envelope overlay off should show raw without envelope."""
        window = MainWindow()
        ss = SignalSet(
            signals=np.random.rand(4, 5, 100),
            width=5, height=4,
            z_axis=np.arange(100, dtype=float),
        )
        window._signal_set = ss
        window._processing["envelope_enabled"] = False
        window._envelope_overlay = True
        window._signal_display_mode = "Raw"
        # Should not crash — envelope method returns None
        window._update_signal_display(0, 0)

    def test_envelope_toggled(self):
        """Toggling envelope overlay must update internal state."""
        window = MainWindow()
        window._on_envelope_toggled(True)
        assert window._envelope_overlay is True
        window._on_envelope_toggled(False)
        assert window._envelope_overlay is False

    def test_mask_source_changed(self):
        """Changing mask source must update internal state."""
        window = MainWindow()
        window._computed_results["snr"] = _make_metric_map("snr")
        window._on_mask_source_changed("snr")
        assert window._mask_source_metric == "snr"

    def test_clear_session_resets_selected_pixel(self):
        window = MainWindow()
        window._selected_pixel = (2, 3)
        window._clear_session()
        assert window._selected_pixel is None


# ------------------------------------------------------------------
# CompareWindow
# ------------------------------------------------------------------

class TestCompareWindow:
    def test_creation_continuous(self):
        data = np.random.rand(8, 10)
        win = CompareWindow(data, title="test")
        assert win is not None

    def test_creation_bool(self):
        data = np.ones((8, 10), dtype=bool)
        win = CompareWindow(data, title="mask")
        assert win is not None


# ------------------------------------------------------------------
# InfoDialog
# ------------------------------------------------------------------

class TestInfoDialog:
    def test_creation(self):
        info = {"Key1": "value1", "Key2": "value2"}
        dlg = InfoDialog(info)
        assert dlg is not None
