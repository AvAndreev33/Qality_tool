"""Tests for signal recipe and recipe binding — the metric input contract.

Covers:
- baseline metrics declare correct signal_recipe and recipe_binding
- evaluator respects fixed recipe (ignores active_recipe)
- evaluator respects active recipe (uses active_recipe)
- fixed-recipe metrics receive original signal regardless of active settings
- active-recipe metrics receive preprocessed/ROI signal
- mixed-binding multi-metric runs produce correct results
- result metadata records effective recipe and binding
"""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.core.models import MetricResult, SignalSet
from quality_tool.evaluation.evaluator import evaluate_metric_map, evaluate_metric_maps
from quality_tool.evaluation.recipe import RAW, SignalRecipe
from quality_tool.metrics.baseline.fringe_visibility import FringeVisibility
from quality_tool.metrics.baseline.power_band_ratio import PowerBandRatio
from quality_tool.metrics.baseline.snr import SNR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal_set(h: int = 2, w: int = 2, m: int = 64) -> SignalSet:
    """Create a small non-negative signal set suitable for fringe_visibility."""
    rng = np.random.default_rng(99)
    signals = 10.0 + rng.standard_normal((h, w, m)).clip(-5, 5)
    z_axis = np.arange(m, dtype=float)
    return SignalSet(
        signals=signals,
        width=w,
        height=h,
        z_axis=z_axis,
        source_type="test",
    )


class _RecordingMetric:
    """Metric that records the signals it receives."""

    def __init__(self, *, binding: str = "active") -> None:
        self.name = f"recording_{binding}"
        self.signal_recipe = RAW
        self.recipe_binding = binding
        self.received_signals: list[np.ndarray] = []

    def evaluate(self, signal, z_axis=None, envelope=None, context=None):
        self.received_signals.append(signal.copy())
        return MetricResult(score=float(np.sum(signal)), features={})


# Active recipe with baseline subtraction.
_BASELINE_RECIPE = SignalRecipe(baseline=True)


# ---------------------------------------------------------------------------
# Tests — declared recipe declarations on baseline metrics
# ---------------------------------------------------------------------------

class TestRecipeDeclaration:
    def test_fringe_visibility_is_fixed_raw(self):
        fv = FringeVisibility()
        assert fv.signal_recipe == RAW
        assert fv.recipe_binding == "fixed"

    def test_snr_is_active(self):
        snr = SNR()
        assert snr.recipe_binding == "active"

    def test_power_band_ratio_is_active(self):
        pbr = PowerBandRatio()
        assert pbr.recipe_binding == "active"


# ---------------------------------------------------------------------------
# Tests — evaluator respects recipe binding
# ---------------------------------------------------------------------------

class TestEvaluatorRecipeBinding:
    def test_fixed_metric_ignores_active_recipe(self):
        """A fixed-recipe metric should receive the same signal regardless
        of the active recipe."""
        ss = _make_signal_set(h=1, w=1, m=32)

        metric_no_pp = _RecordingMetric(binding="fixed")
        evaluate_metric_map(ss, metric_no_pp)

        metric_with_pp = _RecordingMetric(binding="fixed")
        evaluate_metric_map(
            ss, metric_with_pp, active_recipe=_BASELINE_RECIPE,
        )

        np.testing.assert_array_equal(
            metric_no_pp.received_signals[0],
            metric_with_pp.received_signals[0],
        )

    def test_active_metric_receives_preprocessed_signal(self):
        """An active-binding metric should receive a different signal when
        an active recipe with preprocessing is provided."""
        ss = _make_signal_set(h=1, w=1, m=32)

        metric_no_pp = _RecordingMetric(binding="active")
        evaluate_metric_map(ss, metric_no_pp)

        metric_with_pp = _RecordingMetric(binding="active")
        evaluate_metric_map(
            ss, metric_with_pp, active_recipe=_BASELINE_RECIPE,
        )

        assert not np.array_equal(
            metric_no_pp.received_signals[0],
            metric_with_pp.received_signals[0],
        )

    def test_fixed_metric_ignores_roi(self):
        """A fixed-recipe metric should receive the full-length signal even
        when the active recipe includes ROI."""
        ss = _make_signal_set(h=1, w=1, m=64)

        metric = _RecordingMetric(binding="fixed")
        evaluate_metric_map(
            ss, metric,
            active_recipe=SignalRecipe(roi_enabled=True, segment_size=16),
        )

        assert len(metric.received_signals[0]) == 64

    def test_active_metric_receives_roi(self):
        """An active-binding metric should receive a shorter signal when
        the active recipe includes ROI."""
        ss = _make_signal_set(h=1, w=1, m=64)

        metric = _RecordingMetric(binding="active")
        evaluate_metric_map(
            ss, metric,
            active_recipe=SignalRecipe(roi_enabled=True, segment_size=16),
        )

        assert len(metric.received_signals[0]) == 16

    def test_fixed_metric_receives_original_values(self):
        """The signal passed to a fixed-recipe metric must be numerically
        identical to the original dataset signal."""
        ss = _make_signal_set(h=1, w=1, m=32)
        expected = ss.signals[0, 0, :].copy()

        metric = _RecordingMetric(binding="fixed")
        evaluate_metric_map(
            ss, metric,
            active_recipe=SignalRecipe(
                baseline=True, roi_enabled=True, segment_size=16,
            ),
        )

        np.testing.assert_array_equal(metric.received_signals[0], expected)


# ---------------------------------------------------------------------------
# Tests — metadata records recipe information
# ---------------------------------------------------------------------------

class TestMetadataRecipe:
    def test_fixed_metric_metadata(self):
        ss = _make_signal_set(h=1, w=1, m=32)
        result = evaluate_metric_map(
            ss, FringeVisibility(),
            active_recipe=SignalRecipe(
                baseline=True, roi_enabled=True, segment_size=16,
            ),
        )
        md = result.metadata
        assert md["recipe_binding"] == "fixed"
        assert md["effective_recipe"] == RAW

    def test_active_metric_metadata(self):
        ss = _make_signal_set(h=1, w=1, m=32)
        active = SignalRecipe(baseline=True, roi_enabled=True, segment_size=16)
        result = evaluate_metric_map(
            ss, SNR(), active_recipe=active,
        )
        md = result.metadata
        assert md["recipe_binding"] == "active"
        assert md["effective_recipe"] == active


# ---------------------------------------------------------------------------
# Tests — mixed-binding multi-metric run
# ---------------------------------------------------------------------------

class TestMixedBindingRun:
    def test_mixed_metrics_produce_correct_results(self):
        """When computing fringe_visibility (fixed/raw) and snr (active) on
        the same dataset with an active recipe, they should use different
        effective signals."""
        ss = _make_signal_set(h=2, w=2, m=64)

        results = evaluate_metric_maps(
            ss, [FringeVisibility(), SNR()],
            active_recipe=_BASELINE_RECIPE,
        )

        assert results["fringe_visibility"].metadata["recipe_binding"] == "fixed"
        assert results["snr"].metadata["recipe_binding"] == "active"
        assert results["fringe_visibility"].metadata["effective_recipe"] == RAW
        assert results["snr"].metadata["effective_recipe"] == _BASELINE_RECIPE
        assert results["fringe_visibility"].score_map.shape == (2, 2)
        assert results["snr"].score_map.shape == (2, 2)

    def test_fringe_visibility_stable_across_active_recipe(self):
        """Fringe visibility should produce identical results regardless
        of the active recipe."""
        ss = _make_signal_set(h=2, w=2, m=64)

        result_plain = evaluate_metric_map(ss, FringeVisibility())
        result_with_pp = evaluate_metric_map(
            ss, FringeVisibility(),
            active_recipe=SignalRecipe(
                baseline=True, roi_enabled=True, segment_size=32,
            ),
        )

        np.testing.assert_array_equal(
            result_plain.score_map,
            result_with_pp.score_map,
        )
        np.testing.assert_array_equal(
            result_plain.valid_map,
            result_with_pp.valid_map,
        )

    def test_multi_metric_evaluate_maps(self):
        """evaluate_metric_maps should return results for all metrics."""
        ss = _make_signal_set(h=2, w=2, m=64)

        results = evaluate_metric_maps(
            ss, [FringeVisibility(), SNR(), PowerBandRatio()],
            active_recipe=_BASELINE_RECIPE,
        )

        assert set(results.keys()) == {"fringe_visibility", "snr", "power_band_ratio"}
        for name, result in results.items():
            assert result.score_map.shape == (2, 2)
