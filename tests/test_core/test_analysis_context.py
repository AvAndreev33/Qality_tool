"""Tests for quality_tool.core.analysis_context."""

import math

import numpy as np

from quality_tool.core.analysis_context import (
    AnalysisContext,
    build_analysis_context,
    default_analysis_context,
    _effective_oversampling,
)
from quality_tool.core.models import SignalSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal_set(metadata=None):
    """Minimal SignalSet for context-builder tests."""
    signals = np.zeros((2, 3, 16))
    z_axis = np.arange(16, dtype=float)
    return SignalSet(
        signals=signals, width=3, height=2, z_axis=z_axis,
        source_type="test", metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Original tests (preserved)
# ---------------------------------------------------------------------------

class TestAnalysisContext:
    def test_defaults(self):
        ctx = default_analysis_context()
        assert ctx.epsilon == 1e-12
        assert ctx.dc_exclude is True
        assert ctx.default_low_freq == 0.05
        assert ctx.default_high_freq == 0.45
        assert ctx.noise_quarter_fraction == 0.25

    def test_frozen(self):
        ctx = AnalysisContext()
        try:
            ctx.epsilon = 1e-6  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_custom_values(self):
        ctx = AnalysisContext(epsilon=1e-8, dc_exclude=False)
        assert ctx.epsilon == 1e-8
        assert ctx.dc_exclude is False

    def test_equality(self):
        a = AnalysisContext()
        b = AnalysisContext()
        assert a == b

    def test_inequality(self):
        a = AnalysisContext()
        b = AnalysisContext(epsilon=1e-6)
        assert a != b

    def test_default_metadata_fields_are_none(self):
        ctx = AnalysisContext()
        assert ctx.wavelength_nm is None
        assert ctx.coherence_length_nm is None

    def test_metadata_fields_settable(self):
        ctx = AnalysisContext(wavelength_nm=550.0, coherence_length_nm=5000.0)
        assert ctx.wavelength_nm == 550.0
        assert ctx.coherence_length_nm == 5000.0


# ---------------------------------------------------------------------------
# Oversampling scaling rule
# ---------------------------------------------------------------------------

class TestEffectiveOversampling:
    def test_none_metadata(self):
        assert _effective_oversampling(None) == 1

    def test_missing_key(self):
        assert _effective_oversampling({}) == 1

    def test_none_value(self):
        assert _effective_oversampling({"oversampling_factor": None}) == 1

    def test_nan_value(self):
        assert _effective_oversampling({"oversampling_factor": float("nan")}) == 1

    def test_one(self):
        assert _effective_oversampling({"oversampling_factor": 1}) == 1

    def test_one_float(self):
        assert _effective_oversampling({"oversampling_factor": 1.0}) == 1

    def test_two(self):
        assert _effective_oversampling({"oversampling_factor": 2}) == 2

    def test_float_two(self):
        assert _effective_oversampling({"oversampling_factor": 2.0}) == 2

    def test_three(self):
        assert _effective_oversampling({"oversampling_factor": 3}) == 3

    def test_invalid_string(self):
        assert _effective_oversampling({"oversampling_factor": "bad"}) == 1

    def test_less_than_one(self):
        assert _effective_oversampling({"oversampling_factor": 0.5}) == 1


# ---------------------------------------------------------------------------
# build_analysis_context
# ---------------------------------------------------------------------------

class TestBuildAnalysisContext:
    def test_no_metadata(self):
        ss = _make_signal_set(metadata=None)
        ctx = build_analysis_context(ss)
        assert ctx.band_half_width_bins == 5
        assert ctx.default_segment_size == 128
        assert ctx.expected_period_samples == 4
        assert ctx.wavelength_nm is None
        assert ctx.coherence_length_nm is None

    def test_empty_metadata(self):
        ss = _make_signal_set(metadata={})
        ctx = build_analysis_context(ss)
        assert ctx.band_half_width_bins == 5
        assert ctx.default_segment_size == 128
        assert ctx.expected_period_samples == 4

    def test_oversampling_factor_1(self):
        ss = _make_signal_set(metadata={"oversampling_factor": 1})
        ctx = build_analysis_context(ss)
        assert ctx.band_half_width_bins == 5
        assert ctx.default_segment_size == 128
        assert ctx.expected_period_samples == 4

    def test_oversampling_factor_2(self):
        ss = _make_signal_set(metadata={"oversampling_factor": 2})
        ctx = build_analysis_context(ss)
        assert ctx.band_half_width_bins == 10
        assert ctx.default_segment_size == 256
        assert ctx.expected_period_samples == 8

    def test_oversampling_factor_3(self):
        ss = _make_signal_set(metadata={"oversampling_factor": 3})
        ctx = build_analysis_context(ss)
        assert ctx.band_half_width_bins == 15
        assert ctx.default_segment_size == 384
        assert ctx.expected_period_samples == 12

    def test_oversampling_nan_falls_back(self):
        ss = _make_signal_set(metadata={"oversampling_factor": float("nan")})
        ctx = build_analysis_context(ss)
        assert ctx.band_half_width_bins == 5
        assert ctx.default_segment_size == 128
        assert ctx.expected_period_samples == 4

    def test_wavelength_propagated(self):
        ss = _make_signal_set(metadata={"wavelength_nm": 550.0})
        ctx = build_analysis_context(ss)
        assert ctx.wavelength_nm == 550.0
        assert ctx.coherence_length_nm is None

    def test_coherence_length_propagated(self):
        ss = _make_signal_set(metadata={"coherence_length_nm": 5000.0})
        ctx = build_analysis_context(ss)
        assert ctx.coherence_length_nm == 5000.0
        assert ctx.wavelength_nm is None

    def test_both_optical_params(self):
        ss = _make_signal_set(metadata={
            "wavelength_nm": 632.8,
            "coherence_length_nm": 8000.0,
        })
        ctx = build_analysis_context(ss)
        assert ctx.wavelength_nm == 632.8
        assert ctx.coherence_length_nm == 8000.0

    def test_nan_wavelength_ignored(self):
        ss = _make_signal_set(metadata={"wavelength_nm": float("nan")})
        ctx = build_analysis_context(ss)
        assert ctx.wavelength_nm is None

    def test_nan_coherence_length_ignored(self):
        ss = _make_signal_set(metadata={"coherence_length_nm": float("nan")})
        ctx = build_analysis_context(ss)
        assert ctx.coherence_length_nm is None

    def test_combined_oversampling_and_optical(self):
        ss = _make_signal_set(metadata={
            "oversampling_factor": 2,
            "wavelength_nm": 550.0,
            "coherence_length_nm": 5000.0,
        })
        ctx = build_analysis_context(ss)
        assert ctx.band_half_width_bins == 10
        assert ctx.default_segment_size == 256
        assert ctx.expected_period_samples == 8
        assert ctx.wavelength_nm == 550.0
        assert ctx.coherence_length_nm == 5000.0

    def test_non_oversampled_fields_unchanged(self):
        """Non-oversampling fields preserve their defaults."""
        ss = _make_signal_set(metadata={"oversampling_factor": 2})
        ctx = build_analysis_context(ss)
        assert ctx.epsilon == 1e-12
        assert ctx.dc_exclude is True
        assert ctx.drift_window == 31
        assert ctx.peak_min_distance_fraction == 0.6
        assert ctx.alpha_main_support == 0.1

    def test_result_is_frozen(self):
        ss = _make_signal_set(metadata={"oversampling_factor": 2})
        ctx = build_analysis_context(ss)
        try:
            ctx.band_half_width_bins = 99  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_integer_types(self):
        """Oversampling-scaled values should be int."""
        ss = _make_signal_set(metadata={"oversampling_factor": 2.0})
        ctx = build_analysis_context(ss)
        assert isinstance(ctx.band_half_width_bins, int)
        assert isinstance(ctx.default_segment_size, int)
        assert isinstance(ctx.expected_period_samples, int)
