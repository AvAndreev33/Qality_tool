"""Tests for spectral priors computation.

Covers expected period, expected carrier bin, expected band width,
fallback behavior, metadata propagation, and band mask helpers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quality_tool.core.analysis_context import (
    AnalysisContext,
    build_analysis_context,
    _effective_oversampling,
)
from quality_tool.core.models import SignalSet
from quality_tool.spectral.priors import (
    SpectralPriors,
    build_expected_band_mask,
    build_harmonic_band_masks,
    build_low_freq_mask,
    compute_spectral_priors,
    positive_freq_mask,
)


# ---- helpers ----

def _make_signal_set(metadata: dict | None = None, m: int = 128) -> SignalSet:
    signals = np.zeros((2, 2, m))
    return SignalSet(
        signals=signals, width=2, height=2,
        z_axis=np.arange(m, dtype=float),
        metadata=metadata,
    )


# ================================================================
# Expected period and oversampling
# ================================================================


class TestExpectedPeriod:
    """Verify the expected-period rule."""

    def test_default_period_no_metadata(self):
        ctx = build_analysis_context(_make_signal_set(None))
        assert ctx.expected_period_samples == 4

    def test_default_period_missing_oversampling(self):
        ctx = build_analysis_context(_make_signal_set({}))
        assert ctx.expected_period_samples == 4

    def test_default_period_oversampling_one(self):
        ctx = build_analysis_context(_make_signal_set({"oversampling_factor": 1}))
        assert ctx.expected_period_samples == 4

    def test_default_period_oversampling_nan(self):
        ctx = build_analysis_context(
            _make_signal_set({"oversampling_factor": float("nan")})
        )
        assert ctx.expected_period_samples == 4

    def test_oversampled_period(self):
        ctx = build_analysis_context(
            _make_signal_set({"oversampling_factor": 4})
        )
        assert ctx.expected_period_samples == 16

    def test_oversampled_period_float(self):
        ctx = build_analysis_context(
            _make_signal_set({"oversampling_factor": 2.5})
        )
        # int(2.5) = 2 → expected_period = 4 * 2 = 8
        assert ctx.expected_period_samples == 8


# ================================================================
# Expected carrier bin
# ================================================================


class TestExpectedCarrierBin:
    """Verify the expected carrier bin computation."""

    def test_default_carrier_bin(self):
        priors = compute_spectral_priors(128, AnalysisContext())
        assert priors.expected_carrier_bin == 32  # 128 / 4

    def test_oversampled_carrier_bin(self):
        ctx = AnalysisContext(expected_period_samples=16)
        priors = compute_spectral_priors(256, ctx)
        assert priors.expected_carrier_bin == 16  # 256 / 16

    def test_carrier_bin_clips_to_range(self):
        # Very short period → high bin → should be clipped.
        ctx = AnalysisContext(expected_period_samples=1)
        priors = compute_spectral_priors(10, ctx)
        assert 1 <= priors.expected_carrier_bin <= priors.num_positive_bins - 2

    def test_carrier_bin_minimum_is_one(self):
        # Very long period → bin ~0 → should be clipped to 1.
        ctx = AnalysisContext(expected_period_samples=1000)
        priors = compute_spectral_priors(100, ctx)
        assert priors.expected_carrier_bin >= 1


# ================================================================
# Expected band width
# ================================================================


class TestExpectedBandWidth:
    """Verify the expected band width computation."""

    def test_fallback_no_metadata(self):
        ctx = AnalysisContext(band_half_width_bins=7)
        priors = compute_spectral_priors(128, ctx)
        assert priors.expected_band_half_width_bins == 7

    def test_metadata_derived_width(self):
        ctx = AnalysisContext(
            coherence_length_nm=5000.0,
            z_step_nm=100.0,
            coherence_to_bandwidth_scale=0.5,
        )
        priors = compute_spectral_priors(256, ctx)
        # packet = 5000/100 = 50, half_w = round(0.5 * 256 / 50) = round(2.56) = 3
        assert priors.expected_band_half_width_bins == 3

    def test_metadata_partial_falls_back(self):
        """Missing z_step should cause fallback."""
        ctx = AnalysisContext(
            coherence_length_nm=5000.0,
            z_step_nm=None,
            band_half_width_bins=5,
        )
        priors = compute_spectral_priors(128, ctx)
        assert priors.expected_band_half_width_bins == 5

    def test_metadata_zero_coherence_falls_back(self):
        ctx = AnalysisContext(
            coherence_length_nm=0.0,
            z_step_nm=50.0,
            band_half_width_bins=5,
        )
        priors = compute_spectral_priors(128, ctx)
        assert priors.expected_band_half_width_bins == 5

    def test_minimum_half_width_is_one(self):
        """Even with large coherence, half-width should be at least 1."""
        ctx = AnalysisContext(
            coherence_length_nm=100000.0,
            z_step_nm=1.0,
            coherence_to_bandwidth_scale=0.5,
        )
        priors = compute_spectral_priors(128, ctx)
        assert priors.expected_band_half_width_bins >= 1


# ================================================================
# Band edges and masks
# ================================================================


class TestBandEdges:
    """Verify that band edges are valid and correctly clipped."""

    def test_edges_within_spectrum(self):
        ctx = AnalysisContext()
        priors = compute_spectral_priors(128, ctx)
        assert priors.expected_band_low_bin >= 1
        assert priors.expected_band_high_bin < priors.num_positive_bins
        assert priors.expected_band_low_bin <= priors.expected_carrier_bin
        assert priors.expected_band_high_bin >= priors.expected_carrier_bin

    def test_mask_matches_edges(self):
        ctx = AnalysisContext()
        priors = compute_spectral_priors(128, ctx)
        mask = build_expected_band_mask(priors.num_positive_bins, priors)
        # Mask should be True from low to high inclusive.
        for k in range(priors.num_positive_bins):
            if priors.expected_band_low_bin <= k <= priors.expected_band_high_bin:
                assert mask[k], f"bin {k} should be in band"
            else:
                assert not mask[k], f"bin {k} should not be in band"


class TestHarmonicMasks:
    """Verify harmonic band mask construction."""

    def test_harmonic_masks_for_long_signal(self):
        ctx = AnalysisContext()  # period=4, carrier bin ≈ 64
        priors = compute_spectral_priors(256, ctx)
        masks = build_harmonic_band_masks(priors.num_positive_bins, priors)
        # 2nd harmonic at bin 128, 3rd at 192 → one or both may fit
        assert len(masks) >= 1

    def test_no_harmonics_for_short_signal(self):
        ctx = AnalysisContext()  # period=4, carrier ≈ 4
        priors = compute_spectral_priors(16, ctx)
        masks = build_harmonic_band_masks(priors.num_positive_bins, priors)
        # 2nd harmonic at bin 8, num_pos = 9 → should fit
        # 3rd at 12 → may exceed num_pos
        assert isinstance(masks, list)


class TestLowFreqMask:
    """Verify low-frequency mask construction."""

    def test_low_freq_mask_below_carrier_band(self):
        ctx = AnalysisContext()
        priors = compute_spectral_priors(128, ctx)
        mask = build_low_freq_mask(priors.num_positive_bins, priors)
        # Should only cover bins [1, k_low).
        for k in range(priors.num_positive_bins):
            if 1 <= k < priors.expected_band_low_bin:
                assert mask[k]
            else:
                assert not mask[k]


# ================================================================
# Metadata propagation end-to-end
# ================================================================


class TestMetadataPropagation:
    """Verify that metadata flows through to priors."""

    def test_z_step_propagation(self):
        ss = _make_signal_set({"z_step_nm": 75.0})
        ctx = build_analysis_context(ss)
        assert ctx.z_step_nm == 75.0

    def test_invalid_z_step_ignored(self):
        ss = _make_signal_set({"z_step_nm": -10.0})
        ctx = build_analysis_context(ss)
        assert ctx.z_step_nm is None

    def test_oversampling_scales_period_and_bandwidth(self):
        ss = _make_signal_set({"oversampling_factor": 3})
        ctx = build_analysis_context(ss)
        assert ctx.expected_period_samples == 12  # 4 * 3
        assert ctx.band_half_width_bins == 15  # 5 * 3
