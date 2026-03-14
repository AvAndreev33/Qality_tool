"""Tests for baseline quality metrics."""

import numpy as np
import pytest

from quality_tool.core.models import MetricResult
from quality_tool.metrics.baseline.fringe_visibility import FringeVisibility
from quality_tool.metrics.baseline.snr import SNR
from quality_tool.metrics.baseline.power_band_ratio import PowerBandRatio
from quality_tool.spectral.fft import compute_spectrum


# ---------------------------------------------------------------------------
# Fringe visibility
# ---------------------------------------------------------------------------

class TestFringeVisibility:

    def setup_method(self):
        self.metric = FringeVisibility()

    def test_name(self):
        assert self.metric.name == "fringe_visibility"

    def test_returns_metric_result(self):
        signal = np.array([1.0, 3.0, 1.0, 3.0])
        result = self.metric.evaluate(signal)
        assert isinstance(result, MetricResult)

    def test_constant_signal_zero_visibility(self):
        signal = np.full(16, 5.0)
        result = self.metric.evaluate(signal)
        assert result.valid is True
        assert result.score == pytest.approx(0.0)

    def test_known_visibility(self):
        """Signal oscillating between 1 and 3: V = (3-1)/(3+1) = 0.5."""
        signal = np.array([1.0, 3.0, 1.0, 3.0])
        result = self.metric.evaluate(signal)
        assert result.valid is True
        assert result.score == pytest.approx(0.5)
        assert result.features["i_max"] == pytest.approx(3.0)
        assert result.features["i_min"] == pytest.approx(1.0)

    def test_full_contrast(self):
        """Signal between 0 and 1: V = 1.0."""
        signal = np.array([0.0, 1.0, 0.0, 1.0])
        result = self.metric.evaluate(signal)
        assert result.valid is True
        assert result.score == pytest.approx(1.0)

    def test_all_zero_invalid(self):
        signal = np.zeros(8)
        result = self.metric.evaluate(signal)
        assert result.valid is False

    def test_too_short_invalid(self):
        result = self.metric.evaluate(np.array([5.0]))
        assert result.valid is False

    def test_2d_signal_invalid(self):
        result = self.metric.evaluate(np.ones((2, 3)))
        assert result.valid is False

    def test_negative_signal_invalid(self):
        """Signals with negative values must be rejected (e.g. after baseline subtraction)."""
        signal = np.array([-1.0, 2.0, -0.5, 3.0])
        result = self.metric.evaluate(signal)
        assert result.valid is False
        assert "negative" in result.notes


# ---------------------------------------------------------------------------
# SNR
# ---------------------------------------------------------------------------

class TestSNR:

    def setup_method(self):
        self.metric = SNR()

    def test_name(self):
        assert self.metric.name == "snr"

    def test_returns_metric_result(self):
        signal = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = self.metric.evaluate(signal)
        assert isinstance(result, MetricResult)

    def test_constant_signal_invalid(self):
        signal = np.full(16, 3.0)
        result = self.metric.evaluate(signal)
        assert result.valid is False
        assert result.notes != ""

    def test_known_snr(self):
        """Build a signal with controlled peak-to-peak and known noise."""
        rng = np.random.default_rng(42)
        n = 100
        noise_level = 0.5
        signal = np.zeros(n)
        # Outer quarters get noise only
        quarter = n // 4
        signal[:quarter] = rng.normal(0, noise_level, quarter)
        signal[-quarter:] = rng.normal(0, noise_level, quarter)
        # Central peak
        signal[n // 2] = 20.0

        result = self.metric.evaluate(signal)
        assert result.valid is True
        assert result.score > 1.0  # should be well above 1
        assert "peak_to_peak" in result.features
        assert "noise_std" in result.features

    def test_too_short_invalid(self):
        result = self.metric.evaluate(np.array([1.0, 2.0, 3.0]))
        assert result.valid is False

    def test_features_populated(self):
        signal = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        result = self.metric.evaluate(signal)
        assert "peak_to_peak" in result.features
        assert "noise_std" in result.features


# ---------------------------------------------------------------------------
# Power band ratio
# ---------------------------------------------------------------------------

class TestPowerBandRatio:

    def setup_method(self):
        # Use wide band so a simple sine at bin 4/64 falls inside.
        self.metric = PowerBandRatio(low_freq=0.01, high_freq=0.45)

    def test_name(self):
        assert self.metric.name == "power_band_ratio"

    def test_returns_metric_result(self):
        signal = np.sin(np.linspace(0, 2 * np.pi, 32))
        result = self.metric.evaluate(signal)
        assert isinstance(result, MetricResult)

    def test_pure_sine_high_pbr(self):
        """A pure sine within the band should capture most of the power."""
        n = 64
        freq_idx = 4
        t = np.arange(n, dtype=float)
        signal = np.sin(2 * np.pi * freq_idx / n * t)

        result = self.metric.evaluate(signal)
        assert result.valid is True
        assert result.score > 0.9

    def test_dc_signal_invalid(self):
        """Constant signal: all power at DC which is excluded → zero total."""
        signal = np.full(32, 5.0)
        result = self.metric.evaluate(signal)
        assert result.valid is False

    def test_precomputed_context_reuse(self):
        """Metric should accept precomputed SpectralResult from context."""
        n = 64
        freq_idx = 4
        t = np.arange(n, dtype=float)
        signal = np.sin(2 * np.pi * freq_idx / n * t)

        spectral = compute_spectrum(signal)
        context = {"spectral_result": spectral}

        result = self.metric.evaluate(signal, context=context)
        assert result.valid is True
        assert result.score > 0.9

    def test_features_populated(self):
        signal = np.sin(np.linspace(0, 4 * np.pi, 64))
        result = self.metric.evaluate(signal)
        assert "signal_power" in result.features
        assert "total_power" in result.features

    def test_too_short_invalid(self):
        result = self.metric.evaluate(np.array([1.0]))
        assert result.valid is False

    def test_configurable_band_limits(self):
        """Narrow band that excludes the sine should give low ratio."""
        n = 64
        freq_idx = 4
        t = np.arange(n, dtype=float)
        signal = np.sin(2 * np.pi * freq_idx / n * t)

        # Band that is far from freq_idx/n ≈ 0.0625
        metric = PowerBandRatio(low_freq=0.3, high_freq=0.45)
        result = metric.evaluate(signal)
        assert result.valid is True
        assert result.score < 0.1
