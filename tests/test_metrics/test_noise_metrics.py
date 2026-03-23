"""Tests for noise metric implementations.

Covers: scalar evaluation, batch evaluation, scalar/batch consistency,
invalid-case handling, and analysis-context parameter usage.
"""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.core.analysis_context import AnalysisContext
from quality_tool.core.models import MetricResult
from quality_tool.metrics.noise.spectral_snr import SpectralSNR
from quality_tool.metrics.noise.local_snr import LocalSNR
from quality_tool.metrics.noise.envelope_peak_to_background_ratio import (
    EnvelopePeakToBackgroundRatio,
)
from quality_tool.metrics.noise.residual_noise_energy import ResidualNoiseEnergy
from quality_tool.metrics.noise.high_frequency_noise_level import (
    HighFrequencyNoiseLevel,
)
from quality_tool.metrics.noise.low_frequency_drift_level import (
    LowFrequencyDriftLevel,
)


# ---- helpers ----

def _make_context(ctx: AnalysisContext | None = None) -> dict:
    """Build a context dict wrapping an AnalysisContext."""
    return {"analysis_context": ctx or AnalysisContext()}


def _clean_sinusoid(n: int = 128, freq: float = 0.15) -> np.ndarray:
    """Return a clean sinusoid of *n* samples at normalised *freq*."""
    t = np.arange(n, dtype=float)
    return np.sin(2 * np.pi * freq * t)


def _noisy_sinusoid(
    n: int = 128, freq: float = 0.15, noise_std: float = 0.5,
) -> np.ndarray:
    rng = np.random.default_rng(42)
    return _clean_sinusoid(n, freq) + rng.normal(0, noise_std, n)


def _envelope_for(signal: np.ndarray) -> np.ndarray:
    """Simple analytic envelope via Hilbert transform."""
    from scipy.signal import hilbert
    return np.abs(hilbert(signal))


# ---- metadata ----

_ALL_METRICS = [
    SpectralSNR(),
    LocalSNR(),
    EnvelopePeakToBackgroundRatio(),
    ResidualNoiseEnergy(),
    HighFrequencyNoiseLevel(),
    LowFrequencyDriftLevel(),
]


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_category_is_noise(metric):
    assert metric.category == "noise"


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_display_name_is_readable(metric):
    assert metric.display_name != metric.name
    assert len(metric.display_name) > 3


@pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.name)
def test_recipe_binding_is_fixed(metric):
    assert metric.recipe_binding == "fixed"


# ---- SpectralSNR ----

class TestSpectralSNR:
    metric = SpectralSNR()

    def test_clean_signal_high_snr(self):
        sig = _clean_sinusoid()
        r = self.metric.evaluate(sig, context=_make_context())
        assert r.valid
        assert r.score > 10  # clean signal should have very high SNR

    def test_noisy_signal_lower_snr(self):
        sig = _noisy_sinusoid(noise_std=1.0)
        r = self.metric.evaluate(sig, context=_make_context())
        assert r.valid
        # Noisy signal should have lower SNR than clean.
        clean_r = self.metric.evaluate(_clean_sinusoid(), context=_make_context())
        assert r.score < clean_r.score

    def test_short_signal_invalid(self):
        r = self.metric.evaluate(np.array([1.0, 2.0]), context=_make_context())
        assert not r.valid

    def test_batch_scalar_consistency(self):
        signals = np.array([_clean_sinusoid(), _noisy_sinusoid()])
        ctx = _make_context()
        batch = self.metric.evaluate_batch(signals, context=ctx)
        for i in range(2):
            scalar = self.metric.evaluate(signals[i], context=ctx)
            np.testing.assert_allclose(batch.scores[i], scalar.score, rtol=1e-10)

    def test_uses_band_half_width(self):
        sig = _clean_sinusoid()
        ctx_narrow = _make_context(AnalysisContext(band_half_width_bins=1))
        ctx_wide = _make_context(AnalysisContext(band_half_width_bins=15))
        r_narrow = self.metric.evaluate(sig, context=ctx_narrow)
        r_wide = self.metric.evaluate(sig, context=ctx_wide)
        # Wider band captures more signal power -> higher SNR.
        assert r_wide.score > r_narrow.score


# ---- LocalSNR ----

class TestLocalSNR:
    metric = LocalSNR()

    def test_valid_with_envelope(self):
        sig = _clean_sinusoid()
        env = _envelope_for(sig)
        r = self.metric.evaluate(sig, envelope=env, context=_make_context())
        assert r.valid
        assert r.score > 0

    def test_invalid_without_envelope(self):
        r = self.metric.evaluate(_clean_sinusoid(), context=_make_context())
        assert not r.valid

    def test_batch_scalar_consistency(self):
        sigs = np.array([_clean_sinusoid(), _noisy_sinusoid()])
        envs = np.array([_envelope_for(s) for s in sigs])
        ctx = _make_context()
        batch = self.metric.evaluate_batch(sigs, envelopes=envs, context=ctx)
        for i in range(2):
            scalar = self.metric.evaluate(sigs[i], envelope=envs[i], context=ctx)
            np.testing.assert_allclose(batch.scores[i], scalar.score, rtol=1e-10)


# ---- EnvelopePeakToBackgroundRatio ----

def _windowed_sinusoid(n: int = 128, freq: float = 0.15) -> np.ndarray:
    """Sinusoid with a Gaussian envelope — ensures clear peak/background."""
    t = np.arange(n, dtype=float)
    carrier = np.sin(2 * np.pi * freq * t)
    gauss = np.exp(-((t - n / 2) ** 2) / (2 * (n / 8) ** 2))
    return carrier * gauss


class TestEnvelopePBR:
    metric = EnvelopePeakToBackgroundRatio()

    def test_valid_with_envelope(self):
        sig = _windowed_sinusoid()
        env = _envelope_for(sig)
        r = self.metric.evaluate(sig, envelope=env, context=_make_context())
        assert r.valid
        assert r.score > 1.0  # peak should exceed background

    def test_invalid_without_envelope(self):
        r = self.metric.evaluate(_clean_sinusoid(), context=_make_context())
        assert not r.valid

    def test_batch_scalar_consistency(self):
        sigs = np.array([_windowed_sinusoid(), _noisy_sinusoid()])
        envs = np.array([_envelope_for(s) for s in sigs])
        ctx = _make_context()
        batch = self.metric.evaluate_batch(sigs, envelopes=envs, context=ctx)
        for i in range(2):
            scalar = self.metric.evaluate(sigs[i], envelope=envs[i], context=ctx)
            if scalar.valid:
                np.testing.assert_allclose(batch.scores[i], scalar.score, rtol=1e-10)
            else:
                assert not batch.valid[i]


# ---- ResidualNoiseEnergy ----

class TestResidualNoiseEnergy:
    metric = ResidualNoiseEnergy()

    def test_clean_signal_low_residual(self):
        sig = _clean_sinusoid()
        r = self.metric.evaluate(sig, context=_make_context())
        assert r.valid
        assert r.score < 0.3  # most energy should be in the carrier band

    def test_noisy_signal_higher_residual(self):
        clean_r = self.metric.evaluate(_clean_sinusoid(), context=_make_context())
        noisy_r = self.metric.evaluate(_noisy_sinusoid(noise_std=1.0),
                                       context=_make_context())
        assert noisy_r.score > clean_r.score

    def test_batch_scalar_consistency(self):
        sigs = np.array([_clean_sinusoid(), _noisy_sinusoid()])
        ctx = _make_context()
        batch = self.metric.evaluate_batch(sigs, context=ctx)
        for i in range(2):
            scalar = self.metric.evaluate(sigs[i], context=ctx)
            np.testing.assert_allclose(batch.scores[i], scalar.score, rtol=1e-10)


# ---- HighFrequencyNoiseLevel ----

class TestHighFrequencyNoiseLevel:
    metric = HighFrequencyNoiseLevel()

    def test_clean_signal_low_hf(self):
        sig = _clean_sinusoid()
        r = self.metric.evaluate(sig, context=_make_context())
        assert r.valid
        assert r.score < 0.5

    def test_high_freq_noise_increases_score(self):
        rng = np.random.default_rng(42)
        sig = _clean_sinusoid()
        # Add high-frequency noise at Nyquist.
        hf = 0.5 * np.cos(2 * np.pi * 0.48 * np.arange(128))
        r_clean = self.metric.evaluate(sig, context=_make_context())
        r_hf = self.metric.evaluate(sig + hf, context=_make_context())
        assert r_hf.score > r_clean.score

    def test_batch_scalar_consistency(self):
        sigs = np.array([_clean_sinusoid(), _noisy_sinusoid()])
        ctx = _make_context()
        batch = self.metric.evaluate_batch(sigs, context=ctx)
        for i in range(2):
            scalar = self.metric.evaluate(sigs[i], context=ctx)
            np.testing.assert_allclose(batch.scores[i], scalar.score, rtol=1e-10)


# ---- LowFrequencyDriftLevel ----

class TestLowFrequencyDriftLevel:
    metric = LowFrequencyDriftLevel()

    def test_zero_drift_sinusoid(self):
        sig = _clean_sinusoid()
        r = self.metric.evaluate(sig, context=_make_context())
        assert r.valid
        # Pure sinusoid has some residual trend but not dominant.
        assert r.score < 1.0

    def test_constant_offset_high_drift(self):
        # A constant signal is all drift.
        sig = np.full(128, 5.0)
        r = self.metric.evaluate(sig, context=_make_context())
        assert r.valid
        assert r.score > 0.85

    def test_short_signal_invalid(self):
        ctx = _make_context(AnalysisContext(drift_window=200))
        sig = np.ones(50)
        r = self.metric.evaluate(sig, context=ctx)
        assert not r.valid

    def test_batch_scalar_consistency(self):
        sigs = np.array([_clean_sinusoid(), np.full(128, 3.0)])
        ctx = _make_context()
        batch = self.metric.evaluate_batch(sigs, context=ctx)
        for i in range(2):
            scalar = self.metric.evaluate(sigs[i], context=ctx)
            np.testing.assert_allclose(batch.scores[i], scalar.score, rtol=1e-10)


# ---- detrend_linear_batch ----

class TestDetrendLinearBatch:
    def test_removes_linear_trend(self):
        from quality_tool.preprocessing.batch import detrend_linear_batch
        n, m = 3, 100
        x = np.arange(m, dtype=float)
        signals = np.stack([2.0 * x + 5.0, -x + 10.0, np.zeros(m)])
        result = detrend_linear_batch(signals)
        # After detrending, each row should be close to zero.
        for i in range(n):
            np.testing.assert_allclose(result[i], 0.0, atol=1e-10)


# ---- registry grouping ----

class TestRegistryGrouping:
    def test_list_grouped_returns_categories(self):
        from quality_tool.metrics.registry import MetricRegistry
        from quality_tool.metrics.baseline.snr import SNR
        from quality_tool.metrics.noise.spectral_snr import SpectralSNR

        reg = MetricRegistry()
        reg.register(SNR())
        reg.register(SpectralSNR())

        groups = reg.list_grouped()
        cats = [g[0] for g in groups]
        assert "baseline" in cats
        assert "noise" in cats

    def test_display_names_in_grouped(self):
        from quality_tool.metrics.registry import MetricRegistry
        from quality_tool.metrics.baseline.snr import SNR

        reg = MetricRegistry()
        reg.register(SNR())

        groups = reg.list_grouped()
        items = groups[0][1]
        assert items[0] == ("snr", "SNR")
