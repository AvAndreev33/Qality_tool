"""Tests for refined spectral layer (batch API, power, complex, band indices)."""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.spectral.fft import (
    BatchSpectralResult,
    SpectralResult,
    compute_spectrum,
    compute_spectrum_batch,
    frequency_band_indices,
)


# ---------------------------------------------------------------------------
# Tests — compute_spectrum refinements
# ---------------------------------------------------------------------------

class TestComputeSpectrumRefined:
    def test_power_included_when_requested(self):
        signal = np.sin(np.linspace(0, 4 * np.pi, 64))
        result = compute_spectrum(signal, include_power=True)
        assert result.power is not None
        np.testing.assert_allclose(result.power, result.amplitude ** 2)

    def test_power_none_by_default(self):
        signal = np.ones(16)
        result = compute_spectrum(signal)
        assert result.power is None

    def test_complex_included_when_requested(self):
        signal = np.sin(np.linspace(0, 2 * np.pi, 32))
        result = compute_spectrum(signal, include_complex=True)
        assert result.complex_fft is not None
        assert result.complex_fft.dtype == np.complex128 or np.iscomplexobj(result.complex_fft)
        # Amplitude should be |complex_fft|.
        np.testing.assert_allclose(result.amplitude, np.abs(result.complex_fft))

    def test_complex_none_by_default(self):
        result = compute_spectrum(np.ones(8))
        assert result.complex_fft is None

    def test_dc_index_is_zero(self):
        result = compute_spectrum(np.ones(8))
        assert result.dc_index == 0

    def test_backward_compatible_basic(self):
        """Old usage (no extra kwargs) still works identically."""
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_spectrum(signal)
        assert isinstance(result, SpectralResult)
        assert result.frequencies.shape == (3,)
        assert result.amplitude.shape == (3,)


# ---------------------------------------------------------------------------
# Tests — compute_spectrum_batch
# ---------------------------------------------------------------------------

class TestComputeSpectrumBatch:
    def test_basic_shapes(self):
        signals = np.random.default_rng(0).standard_normal((10, 64))
        result = compute_spectrum_batch(signals)
        assert isinstance(result, BatchSpectralResult)
        expected_f = 64 // 2 + 1
        assert result.frequencies.shape == (expected_f,)
        assert result.amplitude is not None
        assert result.amplitude.shape == (10, expected_f)

    def test_amplitude_matches_single(self):
        rng = np.random.default_rng(42)
        signals = rng.standard_normal((5, 32))
        batch = compute_spectrum_batch(signals)
        for i in range(5):
            single = compute_spectrum(signals[i])
            np.testing.assert_allclose(
                batch.amplitude[i], single.amplitude, rtol=1e-12,
            )

    def test_power_included(self):
        signals = np.random.default_rng(0).standard_normal((3, 16))
        result = compute_spectrum_batch(signals, include_power=True)
        assert result.power is not None
        np.testing.assert_allclose(result.power, result.amplitude ** 2)

    def test_complex_included(self):
        signals = np.random.default_rng(0).standard_normal((3, 16))
        result = compute_spectrum_batch(signals, include_complex=True)
        assert result.complex_fft is not None
        np.testing.assert_allclose(
            result.amplitude, np.abs(result.complex_fft),
        )

    def test_no_amplitude_when_disabled(self):
        signals = np.random.default_rng(0).standard_normal((3, 16))
        result = compute_spectrum_batch(
            signals, include_amplitude=False, include_power=True,
        )
        assert result.amplitude is None
        assert result.power is not None

    def test_physical_z_axis(self):
        n = 32
        spacing = 0.5
        z_axis = np.arange(n) * spacing
        signals = np.random.default_rng(0).standard_normal((4, n))
        result = compute_spectrum_batch(signals, z_axis=z_axis)
        expected_f1 = 1.0 / (n * spacing)
        assert result.frequencies[1] == pytest.approx(expected_f1, rel=1e-10)

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_spectrum_batch(np.ones(10))

    def test_dc_index(self):
        signals = np.ones((2, 8))
        result = compute_spectrum_batch(signals)
        assert result.dc_index == 0


# ---------------------------------------------------------------------------
# Tests — frequency_band_indices
# ---------------------------------------------------------------------------

class TestFrequencyBandIndices:
    def test_basic(self):
        freqs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        mask = frequency_band_indices(freqs, 0.1, 0.3)
        expected = np.array([False, True, True, True, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_inclusive_bounds(self):
        freqs = np.array([0.0, 0.1, 0.2, 0.3])
        mask = frequency_band_indices(freqs, 0.1, 0.3)
        assert mask[1] is np.bool_(True)
        assert mask[3] is np.bool_(True)

    def test_empty_band(self):
        freqs = np.array([0.0, 0.1, 0.2])
        mask = frequency_band_indices(freqs, 0.5, 0.9)
        assert not np.any(mask)

    def test_full_band(self):
        freqs = np.array([0.1, 0.2, 0.3])
        mask = frequency_band_indices(freqs, 0.0, 1.0)
        assert np.all(mask)
