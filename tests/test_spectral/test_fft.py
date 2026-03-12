"""Tests for quality_tool.spectral.fft."""

import numpy as np
import pytest

from quality_tool.spectral.fft import SpectralResult, compute_spectrum


class TestComputeSpectrum:
    """Tests for the shared FFT helper."""

    def test_returns_spectral_result(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_spectrum(signal)
        assert isinstance(result, SpectralResult)

    def test_output_shapes(self):
        signal = np.ones(8)
        result = compute_spectrum(signal)
        expected_len = 8 // 2 + 1  # rfft length
        assert result.frequencies.shape == (expected_len,)
        assert result.amplitude.shape == (expected_len,)

    def test_dc_signal(self):
        """A constant signal should have all energy at frequency 0."""
        signal = np.full(16, 5.0)
        result = compute_spectrum(signal)
        assert result.amplitude[0] == pytest.approx(5.0 * 16, rel=1e-10)
        # All other bins should be zero.
        assert np.allclose(result.amplitude[1:], 0.0, atol=1e-10)

    def test_known_sine_peak(self):
        """A pure sine should peak at the correct frequency bin."""
        n = 64
        freq_idx = 4  # desired frequency bin index
        t = np.arange(n, dtype=float)
        signal = np.sin(2 * np.pi * freq_idx / n * t)

        result = compute_spectrum(signal)

        peak_bin = int(np.argmax(result.amplitude))
        assert peak_bin == freq_idx

    def test_physical_z_axis_frequency_scale(self):
        """With a physical z-axis, frequencies should reflect the spacing."""
        n = 32
        spacing = 0.5  # microns
        z_axis = np.arange(n) * spacing
        signal = np.sin(2 * np.pi * 1.0 / (n * spacing) * z_axis * n / n)

        result = compute_spectrum(signal, z_axis=z_axis)
        # First non-zero frequency should be 1/(n*spacing)
        expected_f1 = 1.0 / (n * spacing)
        assert result.frequencies[1] == pytest.approx(expected_f1, rel=1e-10)

    def test_index_based_default_spacing(self):
        """Without z_axis, spacing defaults to 1.0."""
        n = 16
        result = compute_spectrum(np.ones(n))
        expected_f1 = 1.0 / n  # rfftfreq with d=1.0
        assert result.frequencies[1] == pytest.approx(expected_f1, rel=1e-10)

    def test_single_sample(self):
        """Length-1 signal should still work."""
        result = compute_spectrum(np.array([42.0]))
        assert result.frequencies.shape == (1,)
        assert result.amplitude[0] == pytest.approx(42.0)

    def test_rejects_2d_signal(self):
        with pytest.raises(ValueError, match="1-D"):
            compute_spectrum(np.ones((2, 3)))

    def test_rejects_empty_signal(self):
        with pytest.raises(ValueError, match="at least 1"):
            compute_spectrum(np.array([]))

    def test_amplitude_is_nonnegative(self):
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(64)
        result = compute_spectrum(signal)
        assert np.all(result.amplitude >= 0)
