"""Tests for regularity metric implementations.

Covers: metadata, scalar evaluation, batch evaluation, scalar/batch
consistency, invalid-case handling, analysis-context parameter usage,
and helper functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.core.analysis_context import AnalysisContext
from quality_tool.core.models import MetricResult
from quality_tool.metrics.regularity.autocorrelation_peak_strength import (
    AutocorrelationPeakStrength,
)
from quality_tool.metrics.regularity.local_oscillation_regularity import (
    LocalOscillationRegularity,
)
from quality_tool.metrics.regularity.jitter_of_extrema import JitterOfExtrema
from quality_tool.metrics.regularity.zero_crossing_stability import (
    ZeroCrossingStability,
)
from quality_tool.metrics.regularity._regularity_helpers import (
    find_local_maxima,
    find_upward_zero_crossings,
    resample_normalize_cycle,
)


# ---- test signal helpers ----

_PERIOD = 20  # samples per cycle
_N_CYCLES = 8
_M = _PERIOD * _N_CYCLES  # 160 samples


def _make_context(ctx: AnalysisContext | None = None) -> dict:
    return {"analysis_context": ctx or AnalysisContext(
        expected_period_samples=_PERIOD,
    )}


def _clean_sinusoid(n: int = _M, period: int = _PERIOD) -> np.ndarray:
    """Return a clean zero-mean sinusoid with the given period."""
    t = np.arange(n, dtype=float)
    return np.sin(2 * np.pi * t / period)


def _noisy_sinusoid(
    n: int = _M, period: int = _PERIOD, noise_std: float = 0.3,
) -> np.ndarray:
    rng = np.random.default_rng(42)
    return _clean_sinusoid(n, period) + rng.normal(0, noise_std, n)


def _constant_signal(n: int = _M) -> np.ndarray:
    return np.zeros(n)


# ---- metadata tests ----

_ALL_METRICS = [
    AutocorrelationPeakStrength(),
    LocalOscillationRegularity(),
    JitterOfExtrema(),
    ZeroCrossingStability(),
]


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_category_is_regularity(metric):
    assert metric.category == "regularity"


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_display_name_is_readable(metric):
    assert metric.display_name != metric.name
    assert len(metric.display_name) > 5


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_recipe_binding_is_fixed(metric):
    assert metric.recipe_binding == "fixed"


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_signal_recipe_has_no_smoothing(metric):
    assert not metric.signal_recipe.smooth


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_signal_recipe_has_baseline_and_detrend(metric):
    assert metric.signal_recipe.baseline
    assert metric.signal_recipe.detrend
    assert metric.signal_recipe.roi_enabled


# ---- helper tests ----

class TestFindLocalMaxima:
    def test_simple_peaks(self):
        sig = np.sin(2 * np.pi * np.arange(100) / 20)
        peaks = find_local_maxima(sig, min_distance=10)
        # Expect peaks near indices 5, 25, 45, 65, 85.
        assert len(peaks) >= 4
        # All peaks should be at positive values.
        assert np.all(sig[peaks] > 0.9)

    def test_constant_signal_no_peaks(self):
        peaks = find_local_maxima(np.ones(50), min_distance=5)
        assert len(peaks) == 0

    def test_short_signal(self):
        peaks = find_local_maxima(np.array([1.0, 2.0]), min_distance=1)
        assert len(peaks) == 0

    def test_min_distance_enforced(self):
        sig = np.sin(2 * np.pi * np.arange(100) / 10)
        peaks = find_local_maxima(sig, min_distance=15)
        if len(peaks) >= 2:
            dists = np.diff(peaks)
            assert np.all(dists >= 15)


class TestResampleNormalizeCycle:
    def test_output_shape(self):
        sig = np.sin(2 * np.pi * np.arange(40) / 20)
        c = resample_normalize_cycle(sig, 0, 20, length=32)
        assert c is not None
        assert c.shape == (32,)

    def test_zero_mean(self):
        sig = np.sin(2 * np.pi * np.arange(40) / 20)
        c = resample_normalize_cycle(sig, 0, 20, length=32)
        assert c is not None
        np.testing.assert_allclose(c.mean(), 0.0, atol=1e-10)

    def test_unit_norm(self):
        sig = np.sin(2 * np.pi * np.arange(40) / 20)
        c = resample_normalize_cycle(sig, 0, 20, length=32)
        assert c is not None
        np.testing.assert_allclose(np.linalg.norm(c), 1.0, atol=1e-10)

    def test_degenerate_segment_returns_none(self):
        assert resample_normalize_cycle(np.array([1.0]), 0, 1, 32) is None

    def test_constant_segment_returns_none(self):
        assert resample_normalize_cycle(np.ones(20), 0, 20, 32) is None


class TestFindUpwardZeroCrossings:
    def test_sinusoid_crossings(self):
        sig = np.sin(2 * np.pi * np.arange(100) / 20)
        crossings = find_upward_zero_crossings(sig)
        # Expect ~5 upward crossings (at 0, 20, 40, 60, 80).
        assert len(crossings) >= 4

    def test_interpolation_accuracy(self):
        # Linear ramp: crosses zero at exactly index 5.
        sig = np.arange(-5, 6, dtype=float)
        crossings = find_upward_zero_crossings(sig)
        assert len(crossings) == 1
        np.testing.assert_allclose(crossings[0], 5.0, atol=1e-10)

    def test_constant_no_crossings(self):
        crossings = find_upward_zero_crossings(np.ones(50))
        assert len(crossings) == 0


# ---- AutocorrelationPeakStrength ----

class TestAutocorrelationPeakStrength:
    metric = AutocorrelationPeakStrength()

    def test_clean_signal_high_score(self):
        sig = _clean_sinusoid()
        r = self.metric.evaluate(sig, context=_make_context())
        assert r.valid
        assert r.score > 0.8  # strong periodicity

    def test_noisy_signal_lower_score(self):
        ctx = _make_context()
        clean_r = self.metric.evaluate(_clean_sinusoid(), context=ctx)
        noisy_r = self.metric.evaluate(_noisy_sinusoid(noise_std=0.8), context=ctx)
        assert noisy_r.score < clean_r.score

    def test_constant_signal_invalid(self):
        r = self.metric.evaluate(_constant_signal(), context=_make_context())
        assert not r.valid

    def test_short_signal_invalid(self):
        r = self.metric.evaluate(np.array([1.0, 2.0]), context=_make_context())
        assert not r.valid

    def test_search_window_outside_range(self):
        # Very long expected period relative to signal.
        ctx = _make_context(AnalysisContext(expected_period_samples=500))
        r = self.metric.evaluate(_clean_sinusoid(n=100), context=ctx)
        assert not r.valid

    def test_batch_scalar_consistency(self):
        signals = np.array([_clean_sinusoid(), _noisy_sinusoid()])
        ctx = _make_context()
        batch = self.metric.evaluate_batch(signals, context=ctx)
        for i in range(2):
            scalar = self.metric.evaluate(signals[i], context=ctx)
            if scalar.valid:
                np.testing.assert_allclose(
                    batch.scores[i], scalar.score, rtol=1e-10,
                )
            else:
                assert not batch.valid[i]

    def test_uses_expected_period(self):
        sig = _clean_sinusoid()
        # Correct period -> high score.
        ctx_right = _make_context(AnalysisContext(expected_period_samples=_PERIOD))
        r_right = self.metric.evaluate(sig, context=ctx_right)
        # Wrong period -> lower or invalid score.
        ctx_wrong = _make_context(AnalysisContext(
            expected_period_samples=_PERIOD * 3,
            period_search_tolerance_fraction=0.1,
        ))
        r_wrong = self.metric.evaluate(sig, context=ctx_wrong)
        if r_wrong.valid:
            assert r_right.score > r_wrong.score
        else:
            assert r_right.valid


# ---- LocalOscillationRegularity ----

class TestLocalOscillationRegularity:
    metric = LocalOscillationRegularity()

    def test_clean_signal_high_score(self):
        sig = _clean_sinusoid()
        r = self.metric.evaluate(sig, context=_make_context())
        assert r.valid
        assert r.score > 0.9  # identical cycles

    def test_noisy_signal_lower_score(self):
        ctx = _make_context()
        clean_r = self.metric.evaluate(_clean_sinusoid(), context=ctx)
        noisy_r = self.metric.evaluate(_noisy_sinusoid(noise_std=0.8), context=ctx)
        if noisy_r.valid:
            assert noisy_r.score < clean_r.score

    def test_too_few_peaks_invalid(self):
        # Very short signal with not enough peaks.
        sig = np.sin(2 * np.pi * np.arange(30) / _PERIOD)
        r = self.metric.evaluate(sig, context=_make_context())
        assert not r.valid

    def test_batch_scalar_consistency(self):
        signals = np.array([_clean_sinusoid(), _noisy_sinusoid()])
        ctx = _make_context()
        batch = self.metric.evaluate_batch(signals, context=ctx)
        for i in range(2):
            scalar = self.metric.evaluate(signals[i], context=ctx)
            if scalar.valid:
                np.testing.assert_allclose(
                    batch.scores[i], scalar.score, rtol=1e-10,
                )
            else:
                assert not batch.valid[i]


# ---- JitterOfExtrema ----

class TestJitterOfExtrema:
    metric = JitterOfExtrema()

    def test_clean_signal_low_jitter(self):
        sig = _clean_sinusoid()
        r = self.metric.evaluate(sig, context=_make_context())
        assert r.valid
        assert r.score < 0.05  # very regular spacing

    def test_noisy_signal_higher_jitter(self):
        ctx = _make_context()
        clean_r = self.metric.evaluate(_clean_sinusoid(), context=ctx)
        noisy_r = self.metric.evaluate(_noisy_sinusoid(noise_std=0.5), context=ctx)
        if noisy_r.valid:
            assert noisy_r.score >= clean_r.score

    def test_too_few_peaks_invalid(self):
        sig = np.sin(2 * np.pi * np.arange(25) / _PERIOD)
        r = self.metric.evaluate(sig, context=_make_context())
        assert not r.valid

    def test_batch_scalar_consistency(self):
        signals = np.array([_clean_sinusoid(), _noisy_sinusoid()])
        ctx = _make_context()
        batch = self.metric.evaluate_batch(signals, context=ctx)
        for i in range(2):
            scalar = self.metric.evaluate(signals[i], context=ctx)
            if scalar.valid:
                np.testing.assert_allclose(
                    batch.scores[i], scalar.score, rtol=1e-10,
                )
            else:
                assert not batch.valid[i]


# ---- ZeroCrossingStability ----

class TestZeroCrossingStability:
    metric = ZeroCrossingStability()

    def test_clean_signal_low_jitter(self):
        sig = _clean_sinusoid()
        r = self.metric.evaluate(sig, context=_make_context())
        assert r.valid
        assert r.score < 0.05  # very regular crossings

    def test_noisy_signal_higher_jitter(self):
        ctx = _make_context()
        clean_r = self.metric.evaluate(_clean_sinusoid(), context=ctx)
        noisy_r = self.metric.evaluate(
            _noisy_sinusoid(noise_std=0.5), context=ctx,
        )
        if noisy_r.valid:
            assert noisy_r.score >= clean_r.score

    def test_constant_signal_invalid(self):
        r = self.metric.evaluate(_constant_signal(), context=_make_context())
        assert not r.valid

    def test_short_signal_invalid(self):
        r = self.metric.evaluate(np.array([1.0, 2.0]), context=_make_context())
        assert not r.valid

    def test_batch_scalar_consistency(self):
        signals = np.array([_clean_sinusoid(), _noisy_sinusoid()])
        ctx = _make_context()
        batch = self.metric.evaluate_batch(signals, context=ctx)
        for i in range(2):
            scalar = self.metric.evaluate(signals[i], context=ctx)
            if scalar.valid:
                np.testing.assert_allclose(
                    batch.scores[i], scalar.score, rtol=1e-10,
                )
            else:
                assert not batch.valid[i]


# ---- Registry and GUI grouping ----

class TestRegularityRegistration:
    def test_regularity_metrics_in_default_registry(self):
        from quality_tool.metrics.registry import default_registry
        import quality_tool.metrics.regularity  # noqa: F401 — triggers registration

        names = default_registry.list_metrics()
        assert "autocorrelation_peak_strength" in names
        assert "local_oscillation_regularity" in names
        assert "jitter_of_extrema" in names
        assert "zero_crossing_stability" in names

    def test_grouped_includes_regularity(self):
        from quality_tool.metrics.registry import MetricRegistry
        reg = MetricRegistry()
        reg.register(AutocorrelationPeakStrength())
        reg.register(JitterOfExtrema())

        groups = reg.list_grouped()
        cats = [g[0] for g in groups]
        assert "regularity" in cats

    def test_gui_category_label_exists(self):
        from quality_tool.gui.dialogs.metrics_dialog import _CATEGORY_LABELS
        assert "regularity" in _CATEGORY_LABELS
        assert _CATEGORY_LABELS["regularity"] == "Regularity metrics"
