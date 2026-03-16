"""Tests for quality_tool.evaluation.evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.core.models import MetricMapResult, MetricResult, SignalSet
from quality_tool.evaluation.evaluator import evaluate_metric_map
from quality_tool.preprocessing.basic import subtract_baseline
from quality_tool.preprocessing.roi import extract_roi
from quality_tool.spectral.fft import SpectralResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal_set(h: int = 2, w: int = 3, m: int = 32) -> SignalSet:
    """Create a small deterministic SignalSet for testing."""
    rng = np.random.default_rng(42)
    signals = rng.standard_normal((h, w, m))
    z_axis = np.arange(m, dtype=float)
    return SignalSet(
        signals=signals,
        width=w,
        height=h,
        z_axis=z_axis,
        source_type="test",
    )


class _DummyMetric:
    """Minimal metric that returns sum(signal) as score."""
    name = "dummy"

    def evaluate(self, signal, z_axis=None, envelope=None, context=None):
        return MetricResult(
            score=float(np.sum(signal)),
            features={"length": float(len(signal))},
        )


class _InvalidPixelMetric:
    """Metric that marks a specific pixel invalid.

    Pixels at ``(row=0, col=0)`` return ``valid=False``.
    All others return a normal score.
    """
    name = "invalid_pixel"

    def __init__(self) -> None:
        self._call_count = 0

    def evaluate(self, signal, z_axis=None, envelope=None, context=None):
        idx = self._call_count
        self._call_count += 1
        if idx == 0:  # first pixel (0, 0)
            return MetricResult(score=0.0, features={}, valid=False,
                                notes="test invalid")
        return MetricResult(
            score=float(np.sum(signal)),
            features={"length": float(len(signal))},
        )


class _VariableFeatureMetric:
    """Metric that returns different feature keys per pixel."""
    name = "variable_features"

    def __init__(self) -> None:
        self._call_count = 0

    def evaluate(self, signal, z_axis=None, envelope=None, context=None):
        idx = self._call_count
        self._call_count += 1
        features: dict = {"common": 1.0}
        if idx % 2 == 0:
            features["even_only"] = 2.0
        else:
            features["odd_only"] = 3.0
        return MetricResult(score=float(idx), features=features)


class _ContextCheckMetric:
    """Metric that declares needs_spectral and asserts context contains spectral_result."""
    name = "context_check"
    needs_spectral = True

    def evaluate(self, signal, z_axis=None, envelope=None, context=None):
        assert context is not None, "context must not be None"
        sr = context.get("spectral_result")
        assert isinstance(sr, SpectralResult), "spectral_result missing"
        return MetricResult(score=1.0, features={})


class _EnvelopeRecordingMetric:
    """Metric that records whether an envelope was received."""
    name = "envelope_recording"

    def __init__(self) -> None:
        self.received_envelopes: list[np.ndarray | None] = []

    def evaluate(self, signal, z_axis=None, envelope=None, context=None):
        self.received_envelopes.append(envelope)
        return MetricResult(score=1.0, features={})


class _DummyEnvelope:
    """Envelope method that returns abs(signal)."""
    name = "dummy_abs"

    def compute(self, signal, z_axis=None, context=None):
        return np.abs(signal)


# ---------------------------------------------------------------------------
# Tests — basic evaluation
# ---------------------------------------------------------------------------

class TestBasicEvaluation:
    def test_output_shape(self):
        ss = _make_signal_set(h=2, w=3, m=32)
        result = evaluate_metric_map(ss, _DummyMetric())

        assert isinstance(result, MetricMapResult)
        assert result.score_map.shape == (2, 3)
        assert result.valid_map.shape == (2, 3)
        assert result.metric_name == "dummy"

    def test_all_valid(self):
        ss = _make_signal_set()
        result = evaluate_metric_map(ss, _DummyMetric())
        assert np.all(result.valid_map)
        assert not np.any(np.isnan(result.score_map))

    def test_score_values(self):
        ss = _make_signal_set(h=2, w=3, m=32)
        result = evaluate_metric_map(ss, _DummyMetric())
        for r in range(2):
            for c in range(3):
                expected = float(np.sum(ss.signals[r, c, :]))
                assert result.score_map[r, c] == pytest.approx(expected)

    def test_metadata_keys(self):
        ss = _make_signal_set()
        result = evaluate_metric_map(ss, _DummyMetric())
        md = result.metadata
        assert md["metric_name"] == "dummy"
        assert md["preprocess"] == []
        assert md["segment_size"] is None
        assert md["envelope_method"] is None
        assert md["image_shape"] == (2, 3)


# ---------------------------------------------------------------------------
# Tests — preprocessing integration
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_preprocessing_applied(self):
        ss = _make_signal_set()
        result_raw = evaluate_metric_map(ss, _DummyMetric())
        result_pre = evaluate_metric_map(
            ss, _DummyMetric(), preprocess=[subtract_baseline]
        )
        # After baseline subtraction the sum should be ~0.0 for each pixel.
        assert np.allclose(result_pre.score_map, 0.0, atol=1e-10)
        # Raw scores should generally not be zero.
        assert not np.allclose(result_raw.score_map, 0.0, atol=1e-10)

    def test_preprocessing_in_metadata(self):
        ss = _make_signal_set()
        result = evaluate_metric_map(
            ss, _DummyMetric(), preprocess=[subtract_baseline]
        )
        assert "subtract_baseline" in result.metadata["preprocess"]


# ---------------------------------------------------------------------------
# Tests — ROI integration
# ---------------------------------------------------------------------------

class TestROI:
    def test_roi_reduces_signal_length(self):
        """Feature 'length' should reflect the ROI size, not the full signal."""
        ss = _make_signal_set(h=1, w=1, m=32)
        seg = 16
        result = evaluate_metric_map(
            ss, _DummyMetric(), segment_size=seg
        )
        assert result.feature_maps["length"][0, 0] == pytest.approx(seg)

    def test_roi_in_metadata(self):
        ss = _make_signal_set()
        result = evaluate_metric_map(ss, _DummyMetric(), segment_size=16)
        assert result.metadata["segment_size"] == 16


# ---------------------------------------------------------------------------
# Tests — envelope integration
# ---------------------------------------------------------------------------

class TestEnvelope:
    def test_envelope_passed_to_metric(self):
        ss = _make_signal_set(h=1, w=2, m=32)
        metric = _EnvelopeRecordingMetric()
        evaluate_metric_map(ss, metric, envelope_method=_DummyEnvelope())
        assert len(metric.received_envelopes) == 2
        for env in metric.received_envelopes:
            assert env is not None
            assert isinstance(env, np.ndarray)

    def test_no_envelope_by_default(self):
        ss = _make_signal_set(h=1, w=1, m=32)
        metric = _EnvelopeRecordingMetric()
        evaluate_metric_map(ss, metric)
        assert metric.received_envelopes == [None]

    def test_envelope_in_metadata(self):
        ss = _make_signal_set()
        result = evaluate_metric_map(
            ss, _DummyMetric(), envelope_method=_DummyEnvelope()
        )
        assert result.metadata["envelope_method"] == "dummy_abs"


# ---------------------------------------------------------------------------
# Tests — spectral context
# ---------------------------------------------------------------------------

class TestSpectralContext:
    def test_spectral_result_present(self):
        ss = _make_signal_set(h=2, w=2, m=32)
        # _ContextCheckMetric asserts inside evaluate(); test passes
        # only if context["spectral_result"] is always present.
        result = evaluate_metric_map(ss, _ContextCheckMetric())
        assert np.all(result.valid_map)


# ---------------------------------------------------------------------------
# Tests — invalid pixels
# ---------------------------------------------------------------------------

class TestInvalidPixels:
    def test_invalid_pixel_nan_and_flag(self):
        ss = _make_signal_set(h=2, w=3, m=32)
        result = evaluate_metric_map(ss, _InvalidPixelMetric())
        # Pixel (0, 0) should be invalid.
        assert result.valid_map[0, 0] is np.bool_(False)
        assert np.isnan(result.score_map[0, 0])
        # Other pixels should be valid.
        assert np.all(result.valid_map.ravel()[1:])
        assert not np.any(np.isnan(result.score_map.ravel()[1:]))


# ---------------------------------------------------------------------------
# Tests — feature maps
# ---------------------------------------------------------------------------

class TestFeatureMaps:
    def test_feature_union(self):
        ss = _make_signal_set(h=2, w=3, m=32)
        result = evaluate_metric_map(ss, _VariableFeatureMetric())
        fm = result.feature_maps
        assert "common" in fm
        assert "even_only" in fm
        assert "odd_only" in fm
        for key in fm:
            assert fm[key].shape == (2, 3)

    def test_missing_features_are_nan(self):
        ss = _make_signal_set(h=1, w=2, m=32)
        result = evaluate_metric_map(ss, _VariableFeatureMetric())
        fm = result.feature_maps
        # Pixel 0 (even) has "even_only" but not "odd_only"
        assert not np.isnan(fm["even_only"][0, 0])
        assert np.isnan(fm["odd_only"][0, 0])
        # Pixel 1 (odd) has "odd_only" but not "even_only"
        assert np.isnan(fm["even_only"][0, 1])
        assert not np.isnan(fm["odd_only"][0, 1])
