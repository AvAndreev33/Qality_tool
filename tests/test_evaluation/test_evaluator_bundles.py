"""Tests for evaluator bundle-based evaluation.

Covers:
- envelope reuse per recipe group
- spectral reuse per recipe group
- representations from different recipes remain distinct
- analysis context propagation
- mixed-needs recipe groups
"""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.core.analysis_context import AnalysisContext, default_analysis_context
from quality_tool.core.models import MetricMapResult, MetricResult, SignalSet
from quality_tool.evaluation.evaluator import evaluate_metric_map, evaluate_metric_maps
from quality_tool.evaluation.recipe import RAW, SignalRecipe
from quality_tool.metrics.base import RepresentationNeeds
from quality_tool.metrics.batch_result import BatchMetricArrays
from quality_tool.spectral.fft import SpectralResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal_set(h=2, w=3, m=32, metadata=None):
    rng = np.random.default_rng(42)
    signals = 10.0 + rng.standard_normal((h, w, m)).clip(-5, 5)
    z_axis = np.arange(m, dtype=float)
    return SignalSet(
        signals=signals, width=w, height=h, z_axis=z_axis, source_type="test",
        metadata=metadata,
    )


class _ContextRecordingMetric:
    """Records the context dicts it receives."""
    def __init__(self, name="ctx_recorder", needs=None, needs_spectral=False):
        self.name = name
        self.signal_recipe = RAW
        self.recipe_binding = "active"
        self.needs_spectral = needs_spectral
        if needs is not None:
            self.representation_needs = needs
        self.recorded_contexts: list[dict] = []

    def evaluate(self, signal, z_axis=None, envelope=None, context=None):
        self.recorded_contexts.append(dict(context) if context else {})
        return MetricResult(score=float(np.sum(signal)), features={})


class _BatchContextRecordingMetric:
    """Batch metric that records its context."""
    def __init__(self, name="batch_ctx", needs=None):
        self.name = name
        self.signal_recipe = RAW
        self.recipe_binding = "active"
        if needs is not None:
            self.representation_needs = needs
        self.recorded_contexts: list[dict] = []

    def evaluate(self, signal, z_axis=None, envelope=None, context=None):
        return MetricResult(score=0.0)

    def evaluate_batch(self, signals, z_axis=None, envelopes=None, context=None):
        self.recorded_contexts.append(dict(context) if context else {})
        n = signals.shape[0]
        return BatchMetricArrays(
            scores=np.sum(signals, axis=1),
            valid=np.ones(n, dtype=bool),
            features={},
        )


class _DummyEnvelope:
    name = "dummy_abs"
    def compute(self, signal, z_axis=None, context=None):
        return np.abs(signal)


_BASELINE = SignalRecipe(baseline=True)


# ---------------------------------------------------------------------------
# Tests — analysis context propagation
# ---------------------------------------------------------------------------

class TestAnalysisContextPropagation:
    def test_default_context_propagated(self):
        ss = _make_signal_set(h=1, w=1, m=16)
        metric = _ContextRecordingMetric()
        evaluate_metric_map(ss, metric)
        assert len(metric.recorded_contexts) == 1
        ctx = metric.recorded_contexts[0]
        assert "analysis_context" in ctx
        assert isinstance(ctx["analysis_context"], AnalysisContext)

    def test_custom_context_propagated(self):
        ss = _make_signal_set(h=1, w=1, m=16)
        custom = AnalysisContext(epsilon=1e-6)
        metric = _ContextRecordingMetric()
        evaluate_metric_map(ss, metric, analysis_context=custom)
        ctx = metric.recorded_contexts[0]
        assert ctx["analysis_context"].epsilon == 1e-6

    def test_batch_metric_gets_context(self):
        ss = _make_signal_set(h=1, w=2, m=16)
        metric = _BatchContextRecordingMetric(
            needs=RepresentationNeeds(amplitude=True),
        )
        evaluate_metric_map(ss, metric)
        assert len(metric.recorded_contexts) >= 1
        ctx = metric.recorded_contexts[0]
        assert "analysis_context" in ctx
        assert "batch_frequencies" in ctx
        assert "batch_amplitude" in ctx


# ---------------------------------------------------------------------------
# Tests — spectral reuse per recipe
# ---------------------------------------------------------------------------

class TestSpectralReuse:
    def test_two_spectral_metrics_same_group_share_spectral(self):
        """Two metrics with spectral needs in the same recipe group should
        both receive spectral data computed once."""
        ss = _make_signal_set(h=1, w=2, m=16)
        m1 = _BatchContextRecordingMetric(
            name="spec_a", needs=RepresentationNeeds(amplitude=True),
        )
        m2 = _BatchContextRecordingMetric(
            name="spec_b", needs=RepresentationNeeds(amplitude=True),
        )
        results = evaluate_metric_maps(ss, [m1, m2])
        assert "spec_a" in results
        assert "spec_b" in results
        # Both should have received spectral data.
        assert "batch_frequencies" in m1.recorded_contexts[0]
        assert "batch_frequencies" in m2.recorded_contexts[0]
        # The frequency arrays should be the same object (reuse).
        assert m1.recorded_contexts[0]["batch_frequencies"] is \
               m2.recorded_contexts[0]["batch_frequencies"]

    def test_non_spectral_metric_gets_no_spectral(self):
        ss = _make_signal_set(h=1, w=1, m=16)
        metric = _ContextRecordingMetric()
        evaluate_metric_map(ss, metric)
        ctx = metric.recorded_contexts[0]
        assert "spectral_result" not in ctx
        assert "batch_frequencies" not in ctx


# ---------------------------------------------------------------------------
# Tests — representations from different recipes are distinct
# ---------------------------------------------------------------------------

class TestRecipeDistinctness:
    def test_different_recipes_produce_different_spectral(self):
        """Spectral data from raw vs preprocessed recipe must differ."""
        ss = _make_signal_set(h=1, w=1, m=32)

        m_raw = _BatchContextRecordingMetric(
            name="raw_spec",
            needs=RepresentationNeeds(amplitude=True),
        )
        m_raw.signal_recipe = RAW
        m_raw.recipe_binding = "fixed"

        m_pre = _BatchContextRecordingMetric(
            name="pre_spec",
            needs=RepresentationNeeds(amplitude=True),
        )
        m_pre.recipe_binding = "active"

        results = evaluate_metric_maps(
            ss, [m_raw, m_pre], active_recipe=_BASELINE,
        )

        raw_amp = m_raw.recorded_contexts[0]["batch_amplitude"]
        pre_amp = m_pre.recorded_contexts[0]["batch_amplitude"]

        # After baseline subtraction the spectrum changes → amplitudes differ.
        assert not np.allclose(raw_amp, pre_amp)


# ---------------------------------------------------------------------------
# Tests — envelope reuse
# ---------------------------------------------------------------------------

class TestEnvelopeReuse:
    def test_envelope_passed_via_bundle(self):
        ss = _make_signal_set(h=1, w=2, m=16)

        class _EnvRecorder:
            name = "env_rec"
            signal_recipe = RAW
            recipe_binding = "active"
            received: list = []
            def evaluate(self, signal, z_axis=None, envelope=None, context=None):
                _EnvRecorder.received.append(envelope)
                return MetricResult(score=1.0)

        _EnvRecorder.received = []
        evaluate_metric_map(ss, _EnvRecorder(), envelope_method=_DummyEnvelope())
        assert len(_EnvRecorder.received) == 2
        for env in _EnvRecorder.received:
            assert env is not None
            assert isinstance(env, np.ndarray)


# ---------------------------------------------------------------------------
# Tests — per-signal fallback with spectral context
# ---------------------------------------------------------------------------

class TestPerSignalFallbackSpectral:
    def test_per_signal_gets_spectral_result(self):
        ss = _make_signal_set(h=1, w=2, m=16)
        metric = _ContextRecordingMetric(
            needs=RepresentationNeeds(amplitude=True),
        )
        evaluate_metric_map(ss, metric)
        for ctx in metric.recorded_contexts:
            assert "spectral_result" in ctx
            sr = ctx["spectral_result"]
            assert isinstance(sr, SpectralResult)
            assert sr.amplitude is not None

    def test_per_signal_power_when_requested(self):
        ss = _make_signal_set(h=1, w=1, m=16)
        metric = _ContextRecordingMetric(
            needs=RepresentationNeeds(amplitude=True, power=True),
        )
        evaluate_metric_map(ss, metric)
        sr = metric.recorded_contexts[0]["spectral_result"]
        assert sr.power is not None
        np.testing.assert_allclose(sr.power, sr.amplitude ** 2)


# ---------------------------------------------------------------------------
# Tests — backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_existing_pbr_still_works(self):
        from quality_tool.metrics.baseline.power_band_ratio import PowerBandRatio
        ss = _make_signal_set(h=2, w=2, m=64)
        result = evaluate_metric_map(
            ss, PowerBandRatio(), active_recipe=_BASELINE,
        )
        assert result.score_map.shape == (2, 2)
        valid_scores = result.score_map[result.valid_map]
        assert np.all(valid_scores >= 0.0)
        assert np.all(valid_scores <= 1.0)

    def test_existing_fringe_visibility_still_works(self):
        from quality_tool.metrics.baseline.fringe_visibility import FringeVisibility
        ss = _make_signal_set(h=2, w=2, m=32)
        result = evaluate_metric_map(ss, FringeVisibility())
        assert np.all(result.valid_map)

    def test_existing_snr_still_works(self):
        from quality_tool.metrics.baseline.snr import SNR
        ss = _make_signal_set(h=2, w=2, m=64)
        result = evaluate_metric_map(ss, SNR(), active_recipe=_BASELINE)
        assert result.score_map.shape == (2, 2)


# ---------------------------------------------------------------------------
# Tests — metadata-aware context resolution in evaluator
# ---------------------------------------------------------------------------

class TestMetadataAwareContext:
    def test_oversampling_resolved_automatically(self):
        """Evaluator builds context with oversampling scaling when no
        explicit analysis_context is provided."""
        ss = _make_signal_set(h=1, w=1, m=16, metadata={
            "oversampling_factor": 2,
        })
        metric = _ContextRecordingMetric()
        evaluate_metric_map(ss, metric)
        ctx = metric.recorded_contexts[0]["analysis_context"]
        assert ctx.band_half_width_bins == 10
        assert ctx.default_segment_size == 256
        assert ctx.expected_period_samples == 8

    def test_no_metadata_gives_base_defaults(self):
        """Without metadata the evaluator uses base defaults."""
        ss = _make_signal_set(h=1, w=1, m=16, metadata=None)
        metric = _ContextRecordingMetric()
        evaluate_metric_map(ss, metric)
        ctx = metric.recorded_contexts[0]["analysis_context"]
        assert ctx.band_half_width_bins == 5
        assert ctx.default_segment_size == 128
        assert ctx.expected_period_samples == 4

    def test_explicit_context_overrides_builder(self):
        """An explicitly passed analysis_context takes precedence."""
        ss = _make_signal_set(h=1, w=1, m=16, metadata={
            "oversampling_factor": 2,
        })
        explicit = AnalysisContext(band_half_width_bins=99)
        metric = _ContextRecordingMetric()
        evaluate_metric_map(ss, metric, analysis_context=explicit)
        ctx = metric.recorded_contexts[0]["analysis_context"]
        assert ctx.band_half_width_bins == 99

    def test_wavelength_propagated_to_metrics(self):
        ss = _make_signal_set(h=1, w=1, m=16, metadata={
            "wavelength_nm": 550.0,
        })
        metric = _ContextRecordingMetric()
        evaluate_metric_map(ss, metric)
        ctx = metric.recorded_contexts[0]["analysis_context"]
        assert ctx.wavelength_nm == 550.0

    def test_two_metrics_same_recipe_share_resolved_context(self):
        """Both metrics in the same group receive the same resolved
        context object."""
        ss = _make_signal_set(h=1, w=2, m=16, metadata={
            "oversampling_factor": 3,
        })
        m1 = _ContextRecordingMetric(name="a")
        m2 = _ContextRecordingMetric(name="b")
        evaluate_metric_maps(ss, [m1, m2])
        ctx1 = m1.recorded_contexts[0]["analysis_context"]
        ctx2 = m2.recorded_contexts[0]["analysis_context"]
        assert ctx1 is ctx2
        assert ctx1.expected_period_samples == 12
