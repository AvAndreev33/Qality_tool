"""Spectral correlation score metric.

Measures stability of local spectral structure along the ROI
by computing correlations between adjacent windowed spectra::

    SCS = median(rho_j)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quality_tool.core.models import MetricResult
from quality_tool.evaluation.recipe import (
    ROI_MEAN_SUBTRACTED_LINEAR_DETRENDED,
    RecipeBinding,
    SignalRecipe,
)
from quality_tool.metrics.base import RepresentationNeeds
from quality_tool.spectral.priors import SpectralPriors, build_expected_band_mask

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


def _local_spectral_correlations(
    signal: np.ndarray,
    priors: SpectralPriors,
    window_len: int,
    hop_len: int,
) -> float | None:
    """Compute median correlation between adjacent local spectra.

    Returns None if fewer than 2 windows are available.
    """
    m = len(signal)
    if window_len < 4 or hop_len < 1 or window_len > m:
        return None

    # Extract windows.
    starts = list(range(0, m - window_len + 1, hop_len))
    if len(starts) < 2:
        return None

    window = np.hanning(window_len)
    num_pos = window_len // 2 + 1

    # Build expected band mask for the local window length.
    local_priors = SpectralPriors(
        expected_period_samples=priors.expected_period_samples,
        expected_carrier_bin=max(1, round(window_len / priors.expected_period_samples)),
        expected_band_half_width_bins=priors.expected_band_half_width_bins,
        expected_band_low_bin=max(
            1,
            round(window_len / priors.expected_period_samples)
            - priors.expected_band_half_width_bins,
        ),
        expected_band_high_bin=min(
            num_pos - 1,
            round(window_len / priors.expected_period_samples)
            + priors.expected_band_half_width_bins,
        ),
        signal_length=window_len,
        num_positive_bins=num_pos,
    )

    band_mask = build_expected_band_mask(num_pos, local_priors)
    if not np.any(band_mask):
        return None

    # Compute normalised band-power vectors for each window.
    spectra = []
    for s in starts:
        seg = signal[s:s + window_len] * window
        p = np.abs(np.fft.rfft(seg)) ** 2
        q = p[band_mask]
        total = q.sum()
        if total > 0:
            q = q / total
        spectra.append(q)

    if len(spectra) < 2:
        return None

    # Correlations between adjacent windows.
    corrs: list[float] = []
    for j in range(len(spectra) - 1):
        a = spectra[j]
        b = spectra[j + 1]
        a_m = a - a.mean()
        b_m = b - b.mean()
        denom = np.sqrt((a_m ** 2).sum() * (b_m ** 2).sum())
        if denom > 0:
            corrs.append(float((a_m * b_m).sum() / denom))

    if not corrs:
        return None

    return float(np.median(corrs))


class SpectralCorrelationScore:
    """Spectral correlation score (local spectral stability).

    Score meaning: higher is better.
    """

    name: str = "spectral_correlation_score"
    category: str = "spectral"
    display_name: str = "Spectral Correlation Score"
    score_direction: str = "higher_better"
    score_scale: str = "bounded_01"
    signal_recipe: SignalRecipe = ROI_MEAN_SUBTRACTED_LINEAR_DETRENDED
    recipe_binding: RecipeBinding = "fixed"
    representation_needs: RepresentationNeeds = RepresentationNeeds(power=True)

    def evaluate(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelope: np.ndarray | None = None,
        context: dict | None = None,
    ) -> MetricResult:
        ctx = (context or {}).get("analysis_context")
        priors: SpectralPriors | None = (context or {}).get("spectral_priors")

        if priors is None:
            return MetricResult(score=0.0, valid=False, notes="no priors")

        c_win = getattr(ctx, "local_spectrum_window_cycles", 4.0)
        c_hop = getattr(ctx, "local_spectrum_hop_cycles", 2.0)
        t_exp = priors.expected_period_samples

        win_len = max(4, round(c_win * t_exp))
        hop_len = max(1, round(c_hop * t_exp))

        result = _local_spectral_correlations(signal, priors, win_len, hop_len)
        if result is None:
            return MetricResult(score=0.0, valid=False, notes="too few windows")

        return MetricResult(score=result)

    def evaluate_batch(
        self,
        signals: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelopes: np.ndarray | None = None,
        context: dict | None = None,
    ) -> BatchMetricArrays:
        """Per-signal — local windowed spectra are inherently per-signal."""
        from quality_tool.metrics.batch_result import BatchMetricArrays

        n = signals.shape[0]
        scores = np.full(n, np.nan)
        valid_arr = np.zeros(n, dtype=bool)

        ctx = (context or {}).get("analysis_context")
        priors: SpectralPriors | None = (context or {}).get("spectral_priors")

        if priors is None:
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        c_win = getattr(ctx, "local_spectrum_window_cycles", 4.0)
        c_hop = getattr(ctx, "local_spectrum_hop_cycles", 2.0)
        t_exp = priors.expected_period_samples

        win_len = max(4, round(c_win * t_exp))
        hop_len = max(1, round(c_hop * t_exp))

        for i in range(n):
            result = _local_spectral_correlations(
                signals[i], priors, win_len, hop_len,
            )
            if result is not None:
                scores[i] = result
                valid_arr[i] = True

        return BatchMetricArrays(scores=scores, valid=valid_arr)
