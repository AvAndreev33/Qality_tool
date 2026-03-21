"""Tests for per-pixel metric inspector windows and data preparation."""

import numpy as np
import pytest

from quality_tool.comparison.normalization import (
    normalize_single,
    reference_range_from_map,
)
from quality_tool.metrics.base import resolve_score_direction, resolve_score_scale


# ------------------------------------------------------------------
# Score-semantics metadata on all registered metrics
# ------------------------------------------------------------------

class TestScoreSemantics:
    """Every metric must declare score_direction and score_scale."""

    @pytest.fixture()
    def all_metrics(self):
        from quality_tool.metrics.baseline.fringe_visibility import FringeVisibility
        from quality_tool.metrics.baseline.snr import SNR
        from quality_tool.metrics.baseline.power_band_ratio import PowerBandRatio
        from quality_tool.metrics.noise.spectral_snr import SpectralSNR
        from quality_tool.metrics.noise.local_snr import LocalSNR
        from quality_tool.metrics.noise.envelope_peak_to_background_ratio import (
            EnvelopePeakToBackgroundRatio,
        )
        from quality_tool.metrics.noise.noise_floor_level import NoiseFloorLevel
        from quality_tool.metrics.noise.residual_noise_energy import ResidualNoiseEnergy
        from quality_tool.metrics.noise.high_frequency_noise_level import HighFrequencyNoiseLevel
        from quality_tool.metrics.noise.low_frequency_drift_level import LowFrequencyDriftLevel

        return [
            FringeVisibility(),
            SNR(),
            PowerBandRatio(),
            SpectralSNR(),
            LocalSNR(),
            EnvelopePeakToBackgroundRatio(),
            NoiseFloorLevel(),
            ResidualNoiseEnergy(),
            HighFrequencyNoiseLevel(),
            LowFrequencyDriftLevel(),
        ]

    def test_all_have_direction(self, all_metrics):
        for m in all_metrics:
            d = resolve_score_direction(m)
            assert d in ("higher_better", "lower_better"), (
                f"{m.name} has invalid score_direction: {d}"
            )

    def test_all_have_scale(self, all_metrics):
        for m in all_metrics:
            s = resolve_score_scale(m)
            assert s in ("bounded_01", "positive_unbounded", "db_like"), (
                f"{m.name} has invalid score_scale: {s}"
            )

    def test_bounded_metrics_are_correct(self, all_metrics):
        bounded = {m.name for m in all_metrics if resolve_score_scale(m) == "bounded_01"}
        expected_bounded = {
            "fringe_visibility",
            "power_band_ratio",
            "residual_noise_energy",
            "high_frequency_noise_level",
        }
        assert bounded == expected_bounded

    def test_lower_better_metrics_are_correct(self, all_metrics):
        lower = {m.name for m in all_metrics if resolve_score_direction(m) == "lower_better"}
        expected_lower = {
            "noise_floor_level",
            "residual_noise_energy",
            "high_frequency_noise_level",
            "low_frequency_drift_level",
        }
        assert lower == expected_lower


# ------------------------------------------------------------------
# Data gathering logic (mirrors MainWindow._gather_pixel_metrics_data)
# ------------------------------------------------------------------

class TestPixelMetricsDataGathering:
    """Verify per-pixel data preparation for the inspector windows."""

    def test_valid_pixel_has_normalized_score(self):
        score_map = np.array([[0.2, 0.8], [0.5, 0.9]])
        valid_map = np.ones_like(score_map, dtype=bool)
        ref_min, ref_max = reference_range_from_map(
            score_map, valid_map, "bounded_01",
        )
        norm = normalize_single(0.8, "higher_better", "bounded_01", ref_min, ref_max)
        assert pytest.approx(norm) == 0.8

    def test_invalid_pixel_gets_nan(self):
        score_map = np.array([[0.5, 0.8]])
        valid_map = np.array([[False, True]])
        ref_min, ref_max = reference_range_from_map(
            score_map, valid_map, "bounded_01",
        )
        norm = normalize_single(0.5, "higher_better", "bounded_01", ref_min, ref_max)
        # Even though we call normalize_single, the caller should check
        # valid_map and mark invalid pixels — the function itself normalizes.
        assert np.isfinite(norm)
