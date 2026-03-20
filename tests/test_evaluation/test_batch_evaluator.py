"""Tests for batch/chunked evaluation in quality_tool.evaluation.evaluator.

Covers:
- correct output shapes from batch evaluation
- equivalence of batch vs old per-pixel results
- raw metrics without FFT
- spectral metrics with FFT
- envelope-required metrics
- processed metrics with preprocessing
- chunked evaluation consistency (small chunk sizes)
- conditional FFT: metrics without needs_spectral get no spectral context
- direct array assembly (no MetricResult list)
"""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.core.models import MetricMapResult, MetricResult, SignalSet
from quality_tool.envelope.analytic import AnalyticEnvelopeMethod
from quality_tool.evaluation.evaluator import evaluate_metric_map
from quality_tool.evaluation.recipe import RAW, SignalRecipe
from quality_tool.metrics.baseline.fringe_visibility import FringeVisibility
from quality_tool.metrics.baseline.power_band_ratio import PowerBandRatio
from quality_tool.metrics.baseline.snr import SNR
from quality_tool.metrics.batch_result import BatchMetricArrays
from quality_tool.preprocessing.basic import subtract_baseline
from quality_tool.spectral.fft import SpectralResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_signal_set(h: int = 4, w: int = 5, m: int = 64) -> SignalSet:
    """Create a small non-negative signal set for testing."""
    rng = np.random.default_rng(42)
    signals = 10.0 + rng.standard_normal((h, w, m)).clip(-5, 5)
    z_axis = np.arange(m, dtype=float)
    return SignalSet(
        signals=signals, width=w, height=h, z_axis=z_axis, source_type="test",
    )


# Active recipe that applies baseline subtraction.
_BASELINE_RECIPE = SignalRecipe(baseline=True)


# ---------------------------------------------------------------------------
# Tests — batch output shape and correctness
# ---------------------------------------------------------------------------

class TestBatchOutputShape:
    def test_score_map_shape(self):
        ss = _make_signal_set(h=4, w=5, m=64)
        result = evaluate_metric_map(ss, FringeVisibility())
        assert result.score_map.shape == (4, 5)
        assert result.valid_map.shape == (4, 5)

    def test_all_valid_fringe_visibility(self):
        ss = _make_signal_set()
        result = evaluate_metric_map(ss, FringeVisibility())
        assert np.all(result.valid_map)
        assert not np.any(np.isnan(result.score_map))

    def test_feature_maps_present(self):
        ss = _make_signal_set()
        result = evaluate_metric_map(ss, FringeVisibility())
        assert "i_max" in result.feature_maps
        assert "i_min" in result.feature_maps
        assert result.feature_maps["i_max"].shape == (4, 5)


class TestBatchCorrectness:
    """Verify that batch evaluation produces numerically identical results
    to what the old per-pixel loop would give."""

    def test_fringe_visibility_values(self):
        ss = _make_signal_set(h=3, w=4, m=64)
        result = evaluate_metric_map(ss, FringeVisibility())
        for r in range(3):
            for c in range(4):
                sig = ss.signals[r, c, :]
                i_max = np.max(sig)
                i_min = np.min(sig)
                expected = (i_max - i_min) / (i_max + i_min)
                assert result.score_map[r, c] == pytest.approx(expected, rel=1e-12)

    def test_snr_values(self):
        ss = _make_signal_set(h=2, w=3, m=64)
        result = evaluate_metric_map(
            ss, SNR(), active_recipe=_BASELINE_RECIPE,
        )
        for r in range(2):
            for c in range(3):
                sig = ss.signals[r, c, :] - np.mean(ss.signals[r, c, :])
                n = len(sig)
                q = max(n // 4, 1)
                noise = np.concatenate([sig[:q], sig[-q:]])
                noise_std = np.std(noise, ddof=0)
                p2p = np.max(sig) - np.min(sig)
                expected = p2p / noise_std if noise_std >= 1e-12 else 0.0
                assert result.score_map[r, c] == pytest.approx(expected, rel=1e-10)

    def test_power_band_ratio_values(self):
        ss = _make_signal_set(h=2, w=2, m=64)
        metric = PowerBandRatio()
        result = evaluate_metric_map(
            ss, metric, active_recipe=_BASELINE_RECIPE,
        )
        for r in range(2):
            for c in range(2):
                sig = ss.signals[r, c, :] - np.mean(ss.signals[r, c, :])
                from quality_tool.spectral.fft import compute_spectrum
                spectral = compute_spectrum(sig, None)
                power = spectral.amplitude ** 2
                total_power = np.sum(power[1:])
                freq_mask = (
                    (spectral.frequencies >= 0.05)
                    & (spectral.frequencies <= 0.45)
                )
                signal_power = np.sum(power[freq_mask])
                expected = signal_power / total_power
                assert result.score_map[r, c] == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Tests — raw metric without FFT
# ---------------------------------------------------------------------------

class TestRawMetricNoFFT:
    def test_raw_metric_no_fft_overhead(self):
        """FringeVisibility (raw, needs_spectral=False) should work correctly
        without any FFT computation."""
        ss = _make_signal_set()
        assert FringeVisibility().needs_spectral is False
        result = evaluate_metric_map(ss, FringeVisibility())
        assert np.all(result.valid_map)

    def test_raw_metric_ignores_preprocessing(self):
        """Raw metric produces identical results regardless of active recipe."""
        ss = _make_signal_set()
        result_plain = evaluate_metric_map(ss, FringeVisibility())
        result_pp = evaluate_metric_map(
            ss, FringeVisibility(),
            active_recipe=SignalRecipe(baseline=True, roi_enabled=True, segment_size=32),
        )
        np.testing.assert_array_equal(result_plain.score_map, result_pp.score_map)


# ---------------------------------------------------------------------------
# Tests — spectral metric with FFT
# ---------------------------------------------------------------------------

class TestSpectralMetric:
    def test_pbr_needs_spectral(self):
        assert PowerBandRatio().needs_spectral is True

    def test_pbr_gets_valid_results(self):
        ss = _make_signal_set()
        result = evaluate_metric_map(
            ss, PowerBandRatio(), active_recipe=_BASELINE_RECIPE,
        )
        assert result.score_map.shape == (4, 5)
        valid_scores = result.score_map[result.valid_map]
        assert np.all(valid_scores >= 0.0)
        assert np.all(valid_scores <= 1.0)


# ---------------------------------------------------------------------------
# Tests — envelope
# ---------------------------------------------------------------------------

class TestBatchEnvelope:
    def test_envelope_with_snr(self):
        ss = _make_signal_set(h=2, w=2, m=64)
        env = AnalyticEnvelopeMethod()
        result = evaluate_metric_map(
            ss, SNR(),
            active_recipe=_BASELINE_RECIPE,
            envelope_method=env,
        )
        assert result.score_map.shape == (2, 2)
        assert result.metadata["envelope_method"] == "analytic"


# ---------------------------------------------------------------------------
# Tests — chunked evaluation consistency
# ---------------------------------------------------------------------------

class TestChunkedConsistency:
    def test_small_chunks_same_as_large(self):
        """Tiny chunk size should produce identical results to a single chunk."""
        ss = _make_signal_set(h=4, w=5, m=64)

        result_big = evaluate_metric_map(
            ss, SNR(),
            active_recipe=_BASELINE_RECIPE,
            chunk_size=1_000_000,
        )
        result_small = evaluate_metric_map(
            ss, SNR(),
            active_recipe=_BASELINE_RECIPE,
            chunk_size=3,
        )

        np.testing.assert_allclose(
            result_big.score_map, result_small.score_map, rtol=1e-12,
        )
        np.testing.assert_array_equal(result_big.valid_map, result_small.valid_map)

    def test_chunk_size_1(self):
        """Degenerate case: chunk_size=1 should still work correctly."""
        ss = _make_signal_set(h=2, w=2, m=32)
        result = evaluate_metric_map(
            ss, FringeVisibility(), chunk_size=1,
        )
        assert result.score_map.shape == (2, 2)
        assert np.all(result.valid_map)

    def test_pbr_chunked_consistency(self):
        ss = _make_signal_set(h=3, w=3, m=64)
        result_big = evaluate_metric_map(
            ss, PowerBandRatio(),
            active_recipe=_BASELINE_RECIPE,
            chunk_size=100_000,
        )
        result_small = evaluate_metric_map(
            ss, PowerBandRatio(),
            active_recipe=_BASELINE_RECIPE,
            chunk_size=4,
        )
        np.testing.assert_allclose(
            result_big.score_map, result_small.score_map, rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# Tests — conditional FFT (no spectral for non-spectral metrics)
# ---------------------------------------------------------------------------

class _NoSpectralRecordingMetric:
    """Metric that records whether spectral context was provided."""
    name = "no_spectral_recording"
    signal_recipe = RAW
    recipe_binding = "active"
    needs_spectral = False

    def __init__(self):
        self.contexts: list[dict | None] = []

    def evaluate(self, signal, z_axis=None, envelope=None, context=None):
        self.contexts.append(context)
        return MetricResult(score=float(np.sum(signal)), features={})


class _SpectralRecordingMetric:
    """Metric that records spectral context, needs_spectral=True."""
    name = "spectral_recording"
    signal_recipe = RAW
    recipe_binding = "active"
    needs_spectral = True

    def __init__(self):
        self.contexts: list[dict | None] = []

    def evaluate(self, signal, z_axis=None, envelope=None, context=None):
        self.contexts.append(context)
        return MetricResult(score=float(np.sum(signal)), features={})


class TestConditionalFFT:
    def test_non_spectral_metric_gets_no_spectral_context(self):
        ss = _make_signal_set(h=1, w=2, m=32)
        metric = _NoSpectralRecordingMetric()
        evaluate_metric_map(ss, metric)
        for ctx in metric.contexts:
            assert isinstance(ctx, dict)
            assert "spectral_result" not in ctx

    def test_spectral_metric_gets_spectral_context(self):
        ss = _make_signal_set(h=1, w=2, m=32)
        metric = _SpectralRecordingMetric()
        evaluate_metric_map(ss, metric)
        for ctx in metric.contexts:
            assert isinstance(ctx, dict)
            assert "spectral_result" in ctx
            assert isinstance(ctx["spectral_result"], SpectralResult)


# ---------------------------------------------------------------------------
# Tests — ROI with batch evaluation
# ---------------------------------------------------------------------------

class TestBatchROI:
    def test_roi_reduces_signal_length(self):
        ss = _make_signal_set(h=2, w=2, m=64)
        result = evaluate_metric_map(
            ss, SNR(),
            active_recipe=SignalRecipe(baseline=True, roi_enabled=True, segment_size=16),
        )
        assert result.score_map.shape == (2, 2)
        assert result.metadata["effective_recipe"].segment_size == 16

    def test_roi_batch_consistent_with_per_signal(self):
        """Batch ROI extraction should match per-signal extraction."""
        from quality_tool.preprocessing.batch import extract_roi_batch
        from quality_tool.preprocessing.roi import extract_roi

        rng = np.random.default_rng(123)
        signals = rng.standard_normal((10, 64))
        seg = 16

        batch_result = extract_roi_batch(signals, seg)
        for i in range(10):
            single_result = extract_roi(signals[i], seg)
            np.testing.assert_array_equal(batch_result[i], single_result)


# ---------------------------------------------------------------------------
# Tests — metadata
# ---------------------------------------------------------------------------

class TestBatchMetadata:
    def test_metadata_keys(self):
        ss = _make_signal_set()
        result = evaluate_metric_map(
            ss, SNR(),
            active_recipe=SignalRecipe(baseline=True, roi_enabled=True, segment_size=32),
            envelope_method=AnalyticEnvelopeMethod(),
        )
        md = result.metadata
        assert md["metric_name"] == "snr"
        assert md["recipe_binding"] == "active"
        assert md["effective_recipe"] == SignalRecipe(
            baseline=True, roi_enabled=True, segment_size=32,
        )
        assert md["envelope_method"] == "analytic"
        assert md["image_shape"] == (4, 5)

    def test_raw_metric_metadata_ignores_settings(self):
        ss = _make_signal_set()
        result = evaluate_metric_map(
            ss, FringeVisibility(),
            active_recipe=SignalRecipe(baseline=True, roi_enabled=True, segment_size=32),
        )
        md = result.metadata
        assert md["recipe_binding"] == "fixed"
        assert md["effective_recipe"] == RAW
