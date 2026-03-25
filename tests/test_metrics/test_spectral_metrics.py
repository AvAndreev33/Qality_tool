"""Tests for spectral metric implementations.

Covers: spectral priors computation, metric metadata, scalar evaluation,
batch evaluation, scalar/batch consistency, invalid-case handling,
analysis-context parameter usage, helper functions, and GUI grouping.
"""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.core.analysis_context import AnalysisContext, build_analysis_context
from quality_tool.core.models import MetricResult, SignalSet
from quality_tool.spectral.priors import (
    SpectralPriors,
    build_expected_band_mask,
    build_harmonic_band_masks,
    build_low_freq_mask,
    compute_spectral_priors,
    positive_freq_mask,
)

# Import all spectral metrics.
from quality_tool.metrics.spectral.presence_of_expected_carrier_frequency import (
    PresenceOfExpectedCarrierFrequency,
)
from quality_tool.metrics.spectral.dominant_spectral_peak_prominence import (
    DominantSpectralPeakProminence,
)
from quality_tool.metrics.spectral.carrier_to_background_spectral_ratio import (
    CarrierToBackgroundSpectralRatio,
)
from quality_tool.metrics.spectral.energy_concentration_in_working_band import (
    EnergyConcentrationInWorkingBand,
)
from quality_tool.metrics.spectral.spectral_centroid_offset import SpectralCentroidOffset
from quality_tool.metrics.spectral.spectral_spread import SpectralSpread
from quality_tool.metrics.spectral.spectral_entropy import SpectralEntropy
from quality_tool.metrics.spectral.spectral_kurtosis import SpectralKurtosis
from quality_tool.metrics.spectral.spectral_peak_sharpness import SpectralPeakSharpness
from quality_tool.metrics.spectral.envelope_spectrum_consistency import (
    EnvelopeSpectrumConsistency,
)


# ---- test signal helpers ----

_M = 128


def _make_context(
    ctx: AnalysisContext | None = None,
    signal_length: int = _M,
) -> dict:
    """Build a context dict with priors."""
    c = ctx or AnalysisContext()
    priors = compute_spectral_priors(signal_length, c)
    return {"analysis_context": c, "spectral_priors": priors}


def _pure_carrier(m: int = _M, period: float = 4.0, amp: float = 1.0) -> np.ndarray:
    """Create a pure sinusoidal signal at the expected carrier frequency."""
    t = np.arange(m, dtype=float)
    return amp * np.cos(2 * np.pi * t / period)


def _carrier_with_noise(
    m: int = _M, period: float = 4.0, snr_linear: float = 10.0,
) -> np.ndarray:
    rng = np.random.RandomState(42)
    carrier = np.cos(2 * np.pi * np.arange(m) / period)
    noise = rng.randn(m) / snr_linear
    return carrier + noise


def _flat_noise(m: int = _M) -> np.ndarray:
    rng = np.random.RandomState(123)
    return rng.randn(m)


def _make_power_context(signal: np.ndarray, ctx: AnalysisContext | None = None):
    """Build a full context with power spectrum for scalar evaluate."""
    from quality_tool.spectral.fft import compute_spectrum, SpectralResult
    c = ctx or AnalysisContext()
    m = len(signal)
    priors = compute_spectral_priors(m, c)
    spectral = compute_spectrum(signal, include_power=True)
    return {
        "analysis_context": c,
        "spectral_priors": priors,
        "spectral_result": spectral,
    }


def _make_batch_power_context(
    signals: np.ndarray, ctx: AnalysisContext | None = None,
):
    """Build a full context with batch power spectrum."""
    from quality_tool.spectral.fft import compute_spectrum_batch
    c = ctx or AnalysisContext()
    m = signals.shape[1]
    priors = compute_spectral_priors(m, c)
    batch = compute_spectrum_batch(signals, include_power=True)
    return {
        "analysis_context": c,
        "spectral_priors": priors,
        "batch_power": batch.power,
        "spectral": batch,
    }


# ================================================================
# Spectral priors tests
# ================================================================


class TestSpectralPriors:
    """Tests for spectral prior computation."""

    def test_default_expected_period(self):
        ctx = AnalysisContext()
        priors = compute_spectral_priors(128, ctx)
        assert priors.expected_period_samples == 4.0

    def test_oversampled_expected_period(self):
        ctx = AnalysisContext(expected_period_samples=16)
        priors = compute_spectral_priors(256, ctx)
        assert priors.expected_period_samples == 16.0

    def test_expected_carrier_bin_default(self):
        ctx = AnalysisContext()  # period = 4
        priors = compute_spectral_priors(128, ctx)
        # expected = round(128 / 4) = 32
        assert priors.expected_carrier_bin == 32

    def test_expected_carrier_bin_oversampled(self):
        ctx = AnalysisContext(expected_period_samples=16)
        priors = compute_spectral_priors(256, ctx)
        # expected = round(256 / 16) = 16
        assert priors.expected_carrier_bin == 16

    def test_carrier_bin_clipping(self):
        """Carrier bin should not exceed max usable bin."""
        ctx = AnalysisContext(expected_period_samples=2)
        priors = compute_spectral_priors(10, ctx)
        # round(10/2) = 5, num_pos = 6, max_bin = 4
        assert priors.expected_carrier_bin <= priors.num_positive_bins - 2

    def test_band_width_fallback(self):
        """Without metadata, half-width should fall back to default."""
        ctx = AnalysisContext(band_half_width_bins=5)
        priors = compute_spectral_priors(128, ctx)
        assert priors.expected_band_half_width_bins == 5

    def test_band_width_from_metadata(self):
        """With metadata, half-width should be derived from coherence."""
        ctx = AnalysisContext(
            coherence_length_nm=2000.0,
            z_step_nm=50.0,
            coherence_to_bandwidth_scale=0.5,
        )
        priors = compute_spectral_priors(128, ctx)
        # packet_width = 2000/50 = 40, half_w = round(0.5 * 128 / 40) = round(1.6) = 2
        assert priors.expected_band_half_width_bins == 2

    def test_band_edges_valid(self):
        ctx = AnalysisContext()
        priors = compute_spectral_priors(128, ctx)
        assert priors.expected_band_low_bin >= 1
        assert priors.expected_band_high_bin < priors.num_positive_bins
        assert priors.expected_band_low_bin <= priors.expected_band_high_bin

    def test_build_analysis_context_with_z_step(self):
        """build_analysis_context should propagate z_step_nm."""
        signals = np.zeros((2, 2, 128))
        ss = SignalSet(
            signals=signals, width=2, height=2,
            z_axis=np.arange(128, dtype=float),
            metadata={"z_step_nm": 50.0},
        )
        ctx = build_analysis_context(ss)
        assert ctx.z_step_nm == 50.0

    def test_build_analysis_context_missing_z_step(self):
        signals = np.zeros((2, 2, 128))
        ss = SignalSet(
            signals=signals, width=2, height=2,
            z_axis=np.arange(128, dtype=float),
            metadata={},
        )
        ctx = build_analysis_context(ss)
        assert ctx.z_step_nm is None


class TestBandMasks:
    """Tests for band mask construction helpers."""

    def test_expected_band_mask_shape(self):
        priors = compute_spectral_priors(128, AnalysisContext())
        mask = build_expected_band_mask(65, priors)
        assert mask.shape == (65,)
        assert mask.dtype == bool

    def test_expected_band_mask_coverage(self):
        priors = compute_spectral_priors(128, AnalysisContext())
        mask = build_expected_band_mask(65, priors)
        assert np.any(mask)
        # Band should include the expected carrier bin.
        assert mask[priors.expected_carrier_bin]

    def test_harmonic_masks(self):
        priors = compute_spectral_priors(128, AnalysisContext())
        masks = build_harmonic_band_masks(65, priors)
        assert len(masks) >= 1  # at least 2nd harmonic should fit

    def test_low_freq_mask_excludes_dc(self):
        priors = compute_spectral_priors(128, AnalysisContext())
        mask = build_low_freq_mask(65, priors)
        if mask.any():
            assert not mask[0]  # DC excluded

    def test_positive_freq_mask_dc_exclude(self):
        mask = positive_freq_mask(65, dc_exclude=True)
        assert not mask[0]
        assert mask[1]

    def test_positive_freq_mask_dc_include(self):
        mask = positive_freq_mask(65, dc_exclude=False)
        assert mask[0]


# ================================================================
# Metric metadata tests
# ================================================================

ALL_SPECTRAL_METRIC_CLASSES = [
    PresenceOfExpectedCarrierFrequency,
    DominantSpectralPeakProminence,
    CarrierToBackgroundSpectralRatio,
    EnergyConcentrationInWorkingBand,
    SpectralCentroidOffset,
    SpectralSpread,
    SpectralEntropy,
    SpectralKurtosis,
    SpectralPeakSharpness,
    EnvelopeSpectrumConsistency,
]


class TestSpectralMetricMetadata:
    """All spectral metrics must have correct metadata."""

    @pytest.mark.parametrize("cls", ALL_SPECTRAL_METRIC_CLASSES)
    def test_category_is_spectral(self, cls):
        m = cls()
        assert m.category == "spectral"

    @pytest.mark.parametrize("cls", ALL_SPECTRAL_METRIC_CLASSES)
    def test_has_display_name(self, cls):
        m = cls()
        assert isinstance(m.display_name, str)
        assert len(m.display_name) > 0

    @pytest.mark.parametrize("cls", ALL_SPECTRAL_METRIC_CLASSES)
    def test_recipe_binding_is_fixed(self, cls):
        m = cls()
        assert m.recipe_binding == "fixed"

    @pytest.mark.parametrize("cls", ALL_SPECTRAL_METRIC_CLASSES)
    def test_has_evaluate_and_evaluate_batch(self, cls):
        m = cls()
        assert hasattr(m, "evaluate")
        assert hasattr(m, "evaluate_batch")


# ================================================================
# Scalar evaluation tests on pure carrier
# ================================================================

class TestScalarEvaluationPureCarrier:
    """Scalar evaluate on a pure carrier signal should produce valid scores."""

    def test_presence_of_expected_carrier(self):
        signal = _pure_carrier()
        ctx = _make_power_context(signal)
        m = PresenceOfExpectedCarrierFrequency()
        result = m.evaluate(signal, context=ctx)
        assert result.valid
        assert result.score > 0.5  # carrier is where expected

    def test_dominant_peak_prominence(self):
        signal = _pure_carrier()
        ctx = _make_power_context(signal)
        m = DominantSpectralPeakProminence()
        result = m.evaluate(signal, context=ctx)
        assert result.valid
        assert result.score > 1.0

    def test_carrier_to_background(self):
        signal = _pure_carrier()
        ctx = _make_power_context(signal)
        m = CarrierToBackgroundSpectralRatio()
        result = m.evaluate(signal, context=ctx)
        assert result.valid
        assert result.score > 1.0

    def test_energy_concentration(self):
        signal = _pure_carrier()
        ctx = _make_power_context(signal)
        m = EnergyConcentrationInWorkingBand()
        result = m.evaluate(signal, context=ctx)
        assert result.valid
        assert result.score > 0.5

    def test_centroid_offset(self):
        signal = _pure_carrier()
        ctx = _make_power_context(signal)
        m = SpectralCentroidOffset()
        result = m.evaluate(signal, context=ctx)
        assert result.valid

    def test_spread(self):
        signal = _pure_carrier()
        ctx = _make_power_context(signal)
        m = SpectralSpread()
        result = m.evaluate(signal, context=ctx)
        assert result.valid

    def test_entropy(self):
        signal = _pure_carrier()
        ctx = _make_power_context(signal)
        m = SpectralEntropy()
        result = m.evaluate(signal, context=ctx)
        assert result.valid
        assert result.score < 0.5  # concentrated spectrum

    def test_kurtosis(self):
        # Pure carrier has near-zero variance → valid=False is correct.
        # Use carrier + noise so variance is stable.
        signal = _carrier_with_noise()
        ctx = _make_power_context(signal)
        m = SpectralKurtosis()
        result = m.evaluate(signal, context=ctx)
        assert result.valid
        assert result.score > 3.0  # peaked distribution

    def test_peak_sharpness(self):
        signal = _pure_carrier()
        ctx = _make_power_context(signal)
        m = SpectralPeakSharpness()
        result = m.evaluate(signal, context=ctx)
        assert result.valid
        assert result.score > 0.0


# ================================================================
# Scalar evaluation on noise
# ================================================================

class TestScalarEvaluationNoise:
    """Noise-only signals should produce contrasting scores."""

    def test_energy_concentration_low_on_noise(self):
        signal = _flat_noise()
        ctx = _make_power_context(signal)
        m = EnergyConcentrationInWorkingBand()
        result = m.evaluate(signal, context=ctx)
        assert result.valid
        # Flat noise → energy spread broadly → low concentration.
        assert result.score < 0.5

    def test_entropy_high_on_noise(self):
        signal = _flat_noise()
        ctx = _make_power_context(signal)
        m = SpectralEntropy()
        result = m.evaluate(signal, context=ctx)
        assert result.valid
        assert result.score > 0.7  # noise is high-entropy


# ================================================================
# Batch evaluation and scalar/batch consistency
# ================================================================

class TestBatchConsistency:
    """Batch and scalar results should be consistent."""

    @pytest.mark.parametrize("cls", [
        PresenceOfExpectedCarrierFrequency,
        DominantSpectralPeakProminence,
        CarrierToBackgroundSpectralRatio,
        EnergyConcentrationInWorkingBand,
        SpectralCentroidOffset,
        SpectralSpread,
        SpectralEntropy,
        SpectralKurtosis,
        SpectralPeakSharpness,
    ])
    def test_scalar_batch_consistency(self, cls):
        signal = _carrier_with_noise(m=128)
        signals = signal[np.newaxis, :]  # (1, M)
        metric = cls()

        # Scalar context.
        scalar_ctx = _make_power_context(signal)
        scalar_result = metric.evaluate(signal, context=scalar_ctx)

        # Batch context.
        batch_ctx = _make_batch_power_context(signals)
        batch_result = metric.evaluate_batch(signals, context=batch_ctx)

        if scalar_result.valid:
            assert batch_result.valid[0]
            np.testing.assert_allclose(
                batch_result.scores[0], scalar_result.score, rtol=1e-5,
            )


# ================================================================
# Invalid-case handling tests
# ================================================================

class TestInvalidCases:
    """Metrics should return valid=False for invalid inputs."""

    def test_missing_power_spectrum(self):
        signal = _pure_carrier()
        ctx = {"analysis_context": AnalysisContext(), "spectral_priors": None}
        m = PresenceOfExpectedCarrierFrequency()
        result = m.evaluate(signal, context=ctx)
        assert not result.valid

    def test_missing_priors(self):
        from quality_tool.spectral.fft import compute_spectrum
        signal = _pure_carrier()
        spectral = compute_spectrum(signal, include_power=True)
        ctx = {"analysis_context": AnalysisContext(), "spectral_result": spectral}
        m = EnergyConcentrationInWorkingBand()
        result = m.evaluate(signal, context=ctx)
        assert not result.valid

    def test_envelope_spectrum_no_envelope(self):
        signal = _pure_carrier()
        ctx = _make_power_context(signal)
        m = EnvelopeSpectrumConsistency()
        result = m.evaluate(signal, context=ctx)
        assert not result.valid

    def test_envelope_spectrum_no_metadata(self):
        signal = _pure_carrier()
        ctx = _make_power_context(signal)
        envelope = np.abs(signal)
        m = EnvelopeSpectrumConsistency()
        result = m.evaluate(signal, envelope=envelope, context=ctx)
        # No wavelength/coherence → valid=False
        assert not result.valid


# ================================================================
# Envelope–spectrum consistency with metadata
# ================================================================

class TestEnvelopeSpectrumConsistencyWithMetadata:
    """Test ESC when metadata is available."""

    def test_valid_with_metadata(self):
        m_len = 128
        ctx = AnalysisContext(
            wavelength_nm=600.0,
            coherence_length_nm=3000.0,
            z_step_nm=50.0,
        )
        signal = _pure_carrier(m=m_len)
        power_ctx = _make_power_context(signal, ctx)

        # Create a reasonable envelope.
        t = np.arange(m_len, dtype=float)
        envelope = np.exp(-0.5 * ((t - m_len / 2) / 20) ** 2)

        metric = EnvelopeSpectrumConsistency()
        result = metric.evaluate(signal, envelope=envelope, context=power_ctx)
        assert result.valid
        assert result.score >= 0.0


# ================================================================
# GUI metric grouping and registration
# ================================================================

class TestGUIGrouping:
    """Spectral metrics must appear in the registry grouped correctly."""

    def test_all_spectral_metrics_registered(self):
        from quality_tool.metrics.registry import default_registry
        names = default_registry.list_metrics()
        expected_names = [
            "presence_of_expected_carrier_frequency",
            "dominant_spectral_peak_prominence",
            "carrier_to_background_spectral_ratio",
            "energy_concentration_in_working_band",
            "spectral_centroid_offset",
            "spectral_spread",
            "spectral_entropy",
            "spectral_kurtosis",
            "spectral_peak_sharpness",
            "envelope_spectrum_consistency",
        ]
        for name in expected_names:
            assert name in names, f"{name} not registered"

    def test_removed_metrics_not_registered(self):
        """Verify that removed spectral metrics are no longer registered."""
        from quality_tool.metrics.registry import default_registry
        names = default_registry.list_metrics()
        removed = [
            "low_frequency_trend_energy_fraction",
            "harmonic_distortion_level",
            "spectral_correlation_score",
        ]
        for name in removed:
            assert name not in names, f"{name} should have been removed"

    def test_spectral_group_in_registry(self):
        from quality_tool.metrics.registry import default_registry
        groups = default_registry.list_grouped()
        group_names = [cat for cat, _ in groups]
        assert "spectral" in group_names

    def test_spectral_group_has_all_metrics(self):
        from quality_tool.metrics.registry import default_registry
        groups = dict(default_registry.list_grouped())
        spectral_items = groups.get("spectral", [])
        assert len(spectral_items) == 10

    def test_category_labels_include_spectral(self):
        from quality_tool.gui.dialogs.metrics_dialog import _CATEGORY_LABELS
        assert "spectral" in _CATEGORY_LABELS
        assert _CATEGORY_LABELS["spectral"] == "Spectral metrics"


# ================================================================
# Dual-band visualization helpers
# ================================================================

class TestDualBandVisualization:
    """Test the expected band info computation for GUI."""

    def test_compute_spectral_priors_returns_valid_bins(self):
        ctx = AnalysisContext()
        priors = compute_spectral_priors(128, ctx)
        assert priors.expected_carrier_bin > 0
        assert priors.expected_band_low_bin > 0
        assert priors.expected_band_high_bin > priors.expected_band_low_bin

    def test_signal_inspector_accepts_expected_band_info(self):
        """Verify the update_spectrum signature supports dual bands."""
        from quality_tool.gui.widgets.signal_inspector import SignalInspector
        import inspect
        sig = inspect.signature(SignalInspector.update_spectrum)
        assert "expected_band_info" in sig.parameters
