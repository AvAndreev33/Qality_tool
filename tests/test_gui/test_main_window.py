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
        assert window._threshold_states == {}
        assert window._current_map_name is None
        assert window._display_mode == "score"

    def test_map_combo_empty_initially(self):
        window = MainWindow()
        assert window._map_combo.count() == 0

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

    def test_clear_session_resets_state(self):
        window = MainWindow()
        window._computed_results["snr"] = _make_metric_map("snr")
        window._threshold_states["snr"] = None
        window._current_map_name = "snr"
        window._display_mode = "masked"

        window._clear_session()

        assert window._computed_results == {}
        assert window._threshold_states == {}
        assert window._current_map_name is None
        assert window._display_mode == "score"

    def test_threshold_apply_stores_result(self):
        window = MainWindow()
        mm = _make_metric_map("snr")
        window._computed_results["snr"] = mm
        window._threshold_states["snr"] = None
        window._current_map_name = "snr"
        window._slider_min = 0.0
        window._slider_max = 1.0
        window._thresh_spin.setValue(0.5)

        window._on_threshold_apply()

        tr = window._threshold_states["snr"]
        assert tr is not None
        assert tr.threshold == 0.5
        assert tr.mask.shape == mm.score_map.shape
        assert window._display_mode == "masked"

    def test_threshold_reset_clears(self):
        window = MainWindow()
        mm = _make_metric_map("snr")
        window._computed_results["snr"] = mm
        window._current_map_name = "snr"
        window._threshold_states["snr"] = apply_threshold(mm, 0.5)

        window._on_threshold_reset()

        assert window._threshold_states["snr"] is None
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
        window._threshold_states["snr"] = apply_threshold(mm, 0.5)
        window._current_map_name = "snr"
        window._display_mode = "masked"
        window._show_current_map()
        assert window._map_viewer._data is not None

    def test_show_current_map_mask_only(self):
        """Mask-only mode should call set_binary_mask without error."""
        window = MainWindow()
        mm = _make_metric_map("snr")
        window._computed_results["snr"] = mm
        window._threshold_states["snr"] = apply_threshold(mm, 0.5)
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
        window._threshold_states["snr"] = None

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
        """Threshold on a cached metric must survive a second Compute."""
        window = MainWindow()
        window._signal_set = object()

        mm = _make_metric_map("snr")
        window._computed_results["snr"] = mm
        tr = apply_threshold(mm, 0.5)
        window._threshold_states["snr"] = tr

        window._selected_metrics = ["snr"]

        import quality_tool.gui.main_window as mw_mod
        monkeypatch.setattr(
            mw_mod, "evaluate_metric_map",
            lambda *a, **k: _make_metric_map("snr"),
        )

        window._on_compute()

        # Threshold must still be the same object — not reset to None.
        assert window._threshold_states["snr"] is tr


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
