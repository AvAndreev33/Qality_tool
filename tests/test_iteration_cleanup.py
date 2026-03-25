"""Targeted tests for the metric cleanup + normalized map view +
canonical processed signal + autocorrelation inspection iteration.

Covers:
- removal of three spectral metrics from the registry
- normalized_score map-view mode
- preservation of native score maps when normalized view is selected
- Canonical processed signal-inspector mode
- Autocorrelation signal-inspector mode with expected-period guidance
- envelope overlay independence from metric-evaluation state
- no regression of existing signal modes and current GUI workflow
"""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.comparison.normalization import normalize_score_map
from quality_tool.core.analysis_context import AnalysisContext, build_analysis_context
from quality_tool.core.models import MetricMapResult, SignalSet
from quality_tool.metrics.base import resolve_score_direction, resolve_score_scale
from quality_tool.preprocessing.basic import subtract_baseline, detrend_linear
from quality_tool.preprocessing.roi import extract_roi
from quality_tool.spectral.autocorrelation import (
    compute_normalized_autocorrelation,
    find_autocorrelation_peak,
)


# ================================================================
# Helpers
# ================================================================

def _make_signal_set(h: int = 4, w: int = 5, m: int = 128) -> SignalSet:
    """Create a minimal SignalSet with a carrier signal."""
    t = np.arange(m, dtype=float)
    carrier = np.cos(2 * np.pi * t / 4.0)
    signals = np.tile(carrier, (h, w, 1))
    return SignalSet(
        signals=signals, width=w, height=h,
        z_axis=t, metadata={},
    )


def _make_carrier(m: int = 128, period: float = 4.0) -> np.ndarray:
    t = np.arange(m, dtype=float)
    return np.cos(2 * np.pi * t / period)


# ================================================================
# Metric removal
# ================================================================


class TestMetricRemoval:
    """Verify the three spectral metrics are no longer registered."""

    def test_removed_from_default_registry(self):
        from quality_tool.metrics.registry import default_registry
        names = default_registry.list_metrics()
        for removed in [
            "low_frequency_trend_energy_fraction",
            "harmonic_distortion_level",
            "spectral_correlation_score",
        ]:
            assert removed not in names

    def test_remaining_spectral_metrics_still_registered(self):
        from quality_tool.metrics.registry import default_registry
        names = default_registry.list_metrics()
        for expected in [
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
        ]:
            assert expected in names

    def test_spectral_group_count(self):
        from quality_tool.metrics.registry import default_registry
        groups = dict(default_registry.list_grouped())
        spectral_items = groups.get("spectral", [])
        assert len(spectral_items) == 10


# ================================================================
# Normalized score map view
# ================================================================


class TestNormalizedScoreMapView:
    """Verify normalized_score display logic."""

    def test_normalize_preserves_native_map(self):
        """Native score map must not be modified after normalization."""
        score_map = np.random.rand(4, 5).astype(float) * 10
        valid_map = np.ones((4, 5), dtype=bool)
        original = score_map.copy()

        _ = normalize_score_map(score_map, valid_map, "higher_better", "positive_unbounded")

        np.testing.assert_array_equal(score_map, original)

    def test_normalized_output_range(self):
        """Normalized scores should be in [0, 1]."""
        score_map = np.random.rand(4, 5).astype(float) * 100
        valid_map = np.ones((4, 5), dtype=bool)

        norm = normalize_score_map(score_map, valid_map, "higher_better", "positive_unbounded")

        valid_norm = norm[valid_map]
        assert np.all(valid_norm >= 0.0)
        assert np.all(valid_norm <= 1.0)

    def test_normalized_bounded_01_higher_better(self):
        """bounded_01 + higher_better: no transformation needed."""
        score_map = np.array([[0.2, 0.8], [0.5, 0.9]])
        valid_map = np.ones_like(score_map, dtype=bool)

        norm = normalize_score_map(score_map, valid_map, "higher_better", "bounded_01")

        np.testing.assert_allclose(norm, score_map)

    def test_normalized_bounded_01_lower_better(self):
        """bounded_01 + lower_better: should flip via 1 - x."""
        score_map = np.array([[0.2, 0.8], [0.5, 0.9]])
        valid_map = np.ones_like(score_map, dtype=bool)

        norm = normalize_score_map(score_map, valid_map, "lower_better", "bounded_01")

        np.testing.assert_allclose(norm, 1.0 - score_map)

    def test_invalid_pixels_remain_nan(self):
        """Invalid pixels should be NaN in normalized output."""
        score_map = np.ones((3, 3))
        valid_map = np.ones((3, 3), dtype=bool)
        valid_map[1, 1] = False

        norm = normalize_score_map(score_map, valid_map, "higher_better", "bounded_01")

        assert np.isnan(norm[1, 1])


# ================================================================
# Autocorrelation computation
# ================================================================


class TestAutocorrelation:
    """Tests for the autocorrelation utility."""

    def test_autocorrelation_at_lag_zero_is_one(self):
        signal = _make_carrier()
        lags, autocorr = compute_normalized_autocorrelation(signal)
        assert autocorr[0] == pytest.approx(1.0)

    def test_autocorrelation_shape_default_max_lag(self):
        signal = _make_carrier(m=64)
        lags, autocorr = compute_normalized_autocorrelation(signal)
        # Default max_lag = M // 2 = 32, so length is 33 (0..32).
        assert len(lags) == 33
        assert len(autocorr) == 33

    def test_autocorrelation_shape_explicit_max_lag(self):
        signal = _make_carrier(m=64)
        lags, autocorr = compute_normalized_autocorrelation(signal, max_lag=10)
        assert len(lags) == 11
        assert len(autocorr) == 11

    def test_autocorrelation_periodic_signal_peaks(self):
        """A periodic signal should have autocorrelation peaks at multiples of the period."""
        signal = _make_carrier(m=128, period=4.0)
        _, autocorr = compute_normalized_autocorrelation(signal)
        # Autocorrelation at lag=4 (one period) should be close to 1.
        assert autocorr[4] > 0.9

    def test_autocorrelation_zero_signal(self):
        signal = np.zeros(32)
        lags, autocorr = compute_normalized_autocorrelation(signal)
        assert len(autocorr) == 17  # max_lag = 32 // 2 = 16, len = 17
        np.testing.assert_array_equal(autocorr, 0.0)

    def test_find_peak_in_window(self):
        signal = _make_carrier(m=128, period=4.0)
        _, autocorr = compute_normalized_autocorrelation(signal)
        peak = find_autocorrelation_peak(autocorr, 3, 5)
        assert peak == 4

    def test_find_peak_invalid_window(self):
        autocorr = np.zeros(10)
        result = find_autocorrelation_peak(autocorr, 20, 30)
        assert result is None


# ================================================================
# Canonical processed signal
# ================================================================


class TestCanonicalProcessed:
    """Test the canonical processed pipeline."""

    def test_canonical_processed_output(self):
        """Canonical processed should apply baseline, detrend, ROI."""
        signal = _make_carrier(m=256) + 5.0  # add offset
        processed = subtract_baseline(signal.copy())
        processed = detrend_linear(processed)
        processed = extract_roi(processed, 128)

        # Mean should be close to zero after baseline subtraction.
        assert abs(np.mean(processed)) < 0.1
        assert len(processed) == 128


# ================================================================
# Signal inspector modes (widget-level smoke tests)
# ================================================================


class TestSignalInspectorModes:
    """Verify the signal inspector widget accepts all display modes."""

    def test_update_autocorrelation(self):
        from quality_tool.gui.widgets.signal_inspector import SignalInspector
        inspector = SignalInspector()
        lags = np.arange(64, dtype=float)
        autocorr = np.random.rand(64)
        autocorr[0] = 1.0
        inspector.update_autocorrelation(
            lags, autocorr,
            title="test",
            expected_period=4.0,
            search_window=(3.0, 5.0),
            detected_peak_lag=4.0,
        )

    def test_update_autocorrelation_no_guidance(self):
        from quality_tool.gui.widgets.signal_inspector import SignalInspector
        inspector = SignalInspector()
        lags = np.arange(64, dtype=float)
        autocorr = np.random.rand(64)
        inspector.update_autocorrelation(lags, autocorr, title="no guidance")

    def test_signal_mode_combo_has_new_modes(self):
        from quality_tool.gui.widgets.tool_panels import SignalToolsPanel
        panel = SignalToolsPanel()
        items = [panel.mode_combo.itemText(i) for i in range(panel.mode_combo.count())]
        assert "Canonical processed" in items
        assert "Autocorrelation" in items
        # Existing modes preserved.
        assert "Raw" in items
        assert "Processed" in items
        assert "Spectrum" in items
        assert "Processed spectrum" in items

    def test_display_combo_has_normalized_score(self):
        """The map view combo should include normalized_score."""
        from quality_tool.gui.main_window import MainWindow
        window = MainWindow()
        items = [
            window._display_combo.itemText(i)
            for i in range(window._display_combo.count())
        ]
        assert "normalized_score" in items
        assert "score" in items
        assert "masked" in items
        assert "mask_only" in items


# ================================================================
# Envelope overlay independence
# ================================================================


class TestEnvelopeOverlayCleanup:
    """Verify envelope overlay is viewer-side, not metric-execution dependent."""

    def test_envelope_computed_from_displayed_signal(self):
        """Envelope overlay should work even when envelope is disabled in settings."""
        from quality_tool.gui.main_window import MainWindow
        window = MainWindow()
        ss = _make_signal_set()
        window._signal_set = ss
        window._processing = {
            "baseline": False,
            "normalize": False,
            "smooth": False,
            "roi_enabled": False,
            "segment_size": 128,
            "envelope_enabled": False,  # disabled!
            "envelope_method": "analytic",
        }
        signal = ss.signals[0, 0, :].copy()

        # Should still compute envelope because it's viewer-side.
        envelope = window._compute_envelope_for_display(signal)
        assert envelope is not None
        assert len(envelope) == len(signal)

    def test_envelope_modes_classification(self):
        """Envelope overlay should only be meaningful for signal modes."""
        from quality_tool.gui.main_window import MainWindow
        valid = MainWindow._ENVELOPE_MODES
        assert "Raw" in valid
        assert "Processed" in valid
        assert "Canonical processed" in valid
        assert "Spectrum" not in valid
        assert "Autocorrelation" not in valid
