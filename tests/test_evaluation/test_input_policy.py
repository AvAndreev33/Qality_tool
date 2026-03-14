"""Tests for metric input policy — raw vs processed signal contract.

Covers:
- fringe_visibility declares input_policy="raw"
- snr and power_band_ratio declare input_policy="processed"
- evaluator skips preprocessing/ROI for raw-only metrics
- evaluator applies preprocessing/ROI for processed metrics
- mixed-policy multi-metric runs produce correct results
- result metadata records the effective input_policy
"""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.core.models import MetricResult, SignalSet
from quality_tool.evaluation.evaluator import evaluate_metric_map
from quality_tool.metrics.baseline.fringe_visibility import FringeVisibility
from quality_tool.metrics.baseline.power_band_ratio import PowerBandRatio
from quality_tool.metrics.baseline.snr import SNR
from quality_tool.preprocessing.basic import subtract_baseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal_set(h: int = 2, w: int = 2, m: int = 64) -> SignalSet:
    """Create a small non-negative signal set suitable for fringe_visibility."""
    rng = np.random.default_rng(99)
    # Non-negative signals so fringe_visibility is valid.
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

    def __init__(self, *, policy: str = "processed") -> None:
        self.name = f"recording_{policy}"
        self.input_policy = policy
        self.received_signals: list[np.ndarray] = []

    def evaluate(self, signal, z_axis=None, envelope=None, context=None):
        self.received_signals.append(signal.copy())
        return MetricResult(score=float(np.sum(signal)), features={})


# ---------------------------------------------------------------------------
# Tests — declared input_policy on baseline metrics
# ---------------------------------------------------------------------------

class TestInputPolicyDeclaration:
    def test_fringe_visibility_is_raw(self):
        assert FringeVisibility().input_policy == "raw"

    def test_snr_is_processed(self):
        assert SNR().input_policy == "processed"

    def test_power_band_ratio_is_processed(self):
        assert PowerBandRatio().input_policy == "processed"


# ---------------------------------------------------------------------------
# Tests — evaluator respects input_policy
# ---------------------------------------------------------------------------

class TestEvaluatorInputPolicy:
    def test_raw_metric_ignores_preprocessing(self):
        """A raw-only metric should receive the same signal regardless of
        whether preprocessing is configured."""
        ss = _make_signal_set(h=1, w=1, m=32)

        metric_no_pp = _RecordingMetric(policy="raw")
        evaluate_metric_map(ss, metric_no_pp)

        metric_with_pp = _RecordingMetric(policy="raw")
        evaluate_metric_map(
            ss, metric_with_pp, preprocess=[subtract_baseline],
        )

        np.testing.assert_array_equal(
            metric_no_pp.received_signals[0],
            metric_with_pp.received_signals[0],
        )

    def test_processed_metric_receives_preprocessed_signal(self):
        """A processed metric should receive a different signal when
        preprocessing is enabled."""
        ss = _make_signal_set(h=1, w=1, m=32)

        metric_no_pp = _RecordingMetric(policy="processed")
        evaluate_metric_map(ss, metric_no_pp)

        metric_with_pp = _RecordingMetric(policy="processed")
        evaluate_metric_map(
            ss, metric_with_pp, preprocess=[subtract_baseline],
        )

        # After baseline subtraction the signal should differ.
        assert not np.array_equal(
            metric_no_pp.received_signals[0],
            metric_with_pp.received_signals[0],
        )

    def test_raw_metric_ignores_roi(self):
        """A raw-only metric should receive the full-length signal even
        when segment_size is configured."""
        ss = _make_signal_set(h=1, w=1, m=64)

        metric = _RecordingMetric(policy="raw")
        evaluate_metric_map(ss, metric, segment_size=16)

        # Should receive full signal length, not ROI.
        assert len(metric.received_signals[0]) == 64

    def test_processed_metric_receives_roi(self):
        """A processed metric should receive a shorter signal when
        segment_size is configured."""
        ss = _make_signal_set(h=1, w=1, m=64)

        metric = _RecordingMetric(policy="processed")
        evaluate_metric_map(ss, metric, segment_size=16)

        assert len(metric.received_signals[0]) == 16

    def test_raw_metric_receives_original_values(self):
        """The raw signal passed to a raw-only metric must be numerically
        identical to the original dataset signal."""
        ss = _make_signal_set(h=1, w=1, m=32)
        expected = ss.signals[0, 0, :].copy()

        metric = _RecordingMetric(policy="raw")
        evaluate_metric_map(
            ss, metric,
            preprocess=[subtract_baseline],
            segment_size=16,
        )

        np.testing.assert_array_equal(metric.received_signals[0], expected)


# ---------------------------------------------------------------------------
# Tests — metadata records input_policy
# ---------------------------------------------------------------------------

class TestMetadataInputPolicy:
    def test_raw_metric_metadata(self):
        ss = _make_signal_set(h=1, w=1, m=32)
        result = evaluate_metric_map(
            ss, FringeVisibility(),
            preprocess=[subtract_baseline],
            segment_size=16,
        )
        md = result.metadata
        assert md["input_policy"] == "raw"
        # Raw metrics should show empty preprocess / None segment_size
        # in metadata, even if caller passed those arguments.
        assert md["preprocess"] == []
        assert md["segment_size"] is None

    def test_processed_metric_metadata(self):
        ss = _make_signal_set(h=1, w=1, m=32)
        result = evaluate_metric_map(
            ss, SNR(),
            preprocess=[subtract_baseline],
            segment_size=16,
        )
        md = result.metadata
        assert md["input_policy"] == "processed"
        assert "subtract_baseline" in md["preprocess"]
        assert md["segment_size"] == 16


# ---------------------------------------------------------------------------
# Tests — mixed-policy multi-metric run
# ---------------------------------------------------------------------------

class TestMixedPolicyRun:
    def test_mixed_metrics_produce_different_results(self):
        """When computing fringe_visibility (raw) and snr (processed) on
        the same dataset with preprocessing enabled, they should use
        different effective signals and produce independent results."""
        ss = _make_signal_set(h=2, w=2, m=64)

        result_fv = evaluate_metric_map(
            ss, FringeVisibility(),
            preprocess=[subtract_baseline],
        )
        result_snr = evaluate_metric_map(
            ss, SNR(),
            preprocess=[subtract_baseline],
        )

        assert result_fv.metadata["input_policy"] == "raw"
        assert result_snr.metadata["input_policy"] == "processed"

        # fringe_visibility should be valid (non-negative raw signals)
        # while snr on baseline-subtracted signals may differ.
        assert result_fv.score_map.shape == (2, 2)
        assert result_snr.score_map.shape == (2, 2)

    def test_fringe_visibility_stable_across_settings(self):
        """Fringe visibility should produce identical results regardless
        of preprocessing settings."""
        ss = _make_signal_set(h=2, w=2, m=64)

        result_plain = evaluate_metric_map(ss, FringeVisibility())
        result_with_pp = evaluate_metric_map(
            ss, FringeVisibility(),
            preprocess=[subtract_baseline],
            segment_size=32,
        )

        np.testing.assert_array_equal(
            result_plain.score_map,
            result_with_pp.score_map,
        )
        np.testing.assert_array_equal(
            result_plain.valid_map,
            result_with_pp.valid_map,
        )
