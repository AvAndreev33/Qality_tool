"""Tests for the histogram window and its integration with MainWindow."""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.core.models import MetricMapResult
from quality_tool.evaluation.thresholding import apply_threshold
from quality_tool.gui.main_window import MainWindow
from quality_tool.gui.windows.histogram_window import (
    HistogramWindow,
    compute_map_statistics,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_metric_map(name: str, h: int = 4, w: int = 5) -> MetricMapResult:
    rng = np.random.RandomState(42)
    return MetricMapResult(
        metric_name=name,
        score_map=rng.rand(h, w),
        valid_map=np.ones((h, w), dtype=bool),
        feature_maps={},
        metadata={"metric_name": name},
    )


def _make_metric_map_with_invalid(name: str) -> MetricMapResult:
    rng = np.random.RandomState(42)
    score_map = rng.rand(4, 5)
    valid_map = np.ones((4, 5), dtype=bool)
    valid_map[0, 0] = False
    valid_map[3, 4] = False
    return MetricMapResult(
        metric_name=name,
        score_map=score_map,
        valid_map=valid_map,
        feature_maps={},
        metadata={"metric_name": name},
    )


# ------------------------------------------------------------------
# compute_map_statistics
# ------------------------------------------------------------------

class TestComputeMapStatistics:
    def test_basic_stats(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = compute_map_statistics(values)
        assert s["min"] == 1.0
        assert s["max"] == 5.0
        assert s["mean"] == 3.0
        assert s["median"] == 3.0
        assert abs(s["std"] - np.std(values)) < 1e-10

    def test_empty_values(self):
        s = compute_map_statistics(np.array([]))
        assert np.isnan(s["min"])
        assert np.isnan(s["max"])
        assert np.isnan(s["mean"])

    def test_single_value(self):
        s = compute_map_statistics(np.array([7.0]))
        assert s["min"] == 7.0
        assert s["max"] == 7.0
        assert s["mean"] == 7.0
        assert s["std"] == 0.0


# ------------------------------------------------------------------
# HistogramWindow
# ------------------------------------------------------------------

class TestHistogramWindow:
    def test_creation_without_threshold(self):
        values = np.random.rand(100)
        win = HistogramWindow(values, metric_name="snr")
        assert win is not None
        assert win._threshold_value is None
        assert win._threshold_stats is None

    def test_creation_with_threshold(self):
        values = np.random.rand(100)
        stats = {
            "total_pixels": 100,
            "valid_pixels": 100,
            "kept_pixels": 60,
            "rejected_pixels": 40,
            "kept_fraction": 0.6,
        }
        win = HistogramWindow(
            values,
            metric_name="snr",
            threshold_value=0.5,
            keep_rule="score >= 0.5",
            threshold_stats=stats,
        )
        assert win._threshold_value == 0.5
        assert win._threshold_stats["kept_pixels"] == 60

    def test_snapshot_values_are_copied(self):
        """Modifying the source array must not affect the window."""
        values = np.array([1.0, 2.0, 3.0])
        win = HistogramWindow(values, metric_name="test")
        values[0] = 999.0
        assert win._values[0] == 1.0

    def test_map_stats_computed(self):
        values = np.array([2.0, 4.0, 6.0, 8.0])
        win = HistogramWindow(values, metric_name="test")
        assert win._map_stats["min"] == 2.0
        assert win._map_stats["max"] == 8.0
        assert win._map_stats["mean"] == 5.0

    def test_window_title(self):
        win = HistogramWindow(np.array([1.0]), metric_name="fringe_visibility")
        assert "fringe_visibility" in win.windowTitle()

    def test_empty_values(self):
        """Window must not crash on empty valid values."""
        win = HistogramWindow(np.array([]), metric_name="empty")
        assert win is not None
        assert np.isnan(win._map_stats["min"])


# ------------------------------------------------------------------
# MainWindow histogram integration
# ------------------------------------------------------------------

class TestMainWindowHistogram:
    def test_histogram_windows_list_initially_empty(self):
        window = MainWindow()
        assert window._histogram_windows == []

    def test_on_histogram_no_results(self):
        """Histogram action with no results should not crash."""
        window = MainWindow()
        window._on_histogram()
        assert window._histogram_windows == []

    def test_on_histogram_creates_window(self):
        window = MainWindow()
        mm = _make_metric_map("snr")
        window._computed_results["snr"] = mm
        window._current_map_name = "snr"

        window._on_histogram()

        assert len(window._histogram_windows) == 1
        win = window._histogram_windows[0]
        assert win._metric_name == "snr"
        assert win._threshold_value is None

    def test_on_histogram_with_matching_threshold(self):
        window = MainWindow()
        mm = _make_metric_map("snr")
        window._computed_results["snr"] = mm
        window._current_map_name = "snr"
        window._mask_source_metric = "snr"
        window._current_threshold = apply_threshold(mm, 0.5)

        window._on_histogram()

        win = window._histogram_windows[0]
        assert win._threshold_value == 0.5
        assert win._threshold_stats is not None
        assert "kept_pixels" in win._threshold_stats

    def test_on_histogram_threshold_different_source(self):
        """Threshold from a different source metric should NOT show on histogram."""
        window = MainWindow()
        mm_snr = _make_metric_map("snr")
        mm_fv = _make_metric_map("fringe_visibility")
        window._computed_results["snr"] = mm_snr
        window._computed_results["fringe_visibility"] = mm_fv
        window._current_map_name = "fringe_visibility"
        window._mask_source_metric = "snr"
        window._current_threshold = apply_threshold(mm_snr, 0.5)

        window._on_histogram()

        win = window._histogram_windows[0]
        assert win._metric_name == "fringe_visibility"
        assert win._threshold_value is None
        assert win._threshold_stats is None

    def test_multiple_histogram_windows(self):
        """Opening multiple histograms should create independent windows."""
        window = MainWindow()
        mm_snr = _make_metric_map("snr")
        mm_fv = _make_metric_map("fringe_visibility")
        window._computed_results["snr"] = mm_snr
        window._computed_results["fringe_visibility"] = mm_fv

        window._current_map_name = "snr"
        window._on_histogram()

        window._current_map_name = "fringe_visibility"
        window._on_histogram()

        assert len(window._histogram_windows) == 2
        assert window._histogram_windows[0]._metric_name == "snr"
        assert window._histogram_windows[1]._metric_name == "fringe_visibility"

    def test_histogram_uses_valid_pixels_only(self):
        """Histogram should contain only valid-pixel values."""
        window = MainWindow()
        mm = _make_metric_map_with_invalid("snr")
        window._computed_results["snr"] = mm
        window._current_map_name = "snr"

        window._on_histogram()

        win = window._histogram_windows[0]
        expected_count = int(mm.valid_map.sum())
        assert len(win._values) == expected_count

    def test_histogram_is_snapshot(self):
        """Changing main window state after opening must not affect histogram."""
        window = MainWindow()
        mm = _make_metric_map("snr")
        window._computed_results["snr"] = mm
        window._current_map_name = "snr"

        window._on_histogram()
        win = window._histogram_windows[0]
        original_values = win._values.copy()

        # Mutate the score map in session — histogram must be unaffected.
        mm.score_map[:] = 999.0
        np.testing.assert_array_equal(win._values, original_values)
