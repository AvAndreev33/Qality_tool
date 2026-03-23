"""Spectral peak sharpness metric.

Inverse of the half-maximum width of the dominant peak in B_exp::

    SPS = 1 / (W_half + eps)
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


def _half_max_width(power: np.ndarray, k_c: int) -> float | None:
    """Compute interpolated half-maximum width around bin k_c.

    Returns None when crossings cannot be found.
    """
    f = len(power)
    p_c = power[k_c]
    if p_c <= 0:
        return None
    h = 0.5 * p_c

    # Search left.
    k_l: float | None = None
    for i in range(k_c - 1, -1, -1):
        if power[i] <= h:
            # Linear interpolation.
            if power[i + 1] > power[i]:
                frac = (h - power[i]) / (power[i + 1] - power[i])
                k_l = i + frac
            else:
                k_l = float(i)
            break
    if k_l is None:
        k_l = 0.0

    # Search right.
    k_r: float | None = None
    for i in range(k_c + 1, f):
        if power[i] <= h:
            if power[i - 1] > power[i]:
                frac = (h - power[i]) / (power[i - 1] - power[i])
                k_r = i - frac
            else:
                k_r = float(i)
            break
    if k_r is None:
        k_r = float(f - 1)

    w = k_r - k_l
    return w if w > 0 else None


class SpectralPeakSharpness:
    """Spectral peak sharpness (inverse half-max width).

    Score meaning: higher is better.
    """

    name: str = "spectral_peak_sharpness"
    category: str = "spectral"
    display_name: str = "Spectral Peak Sharpness"
    score_direction: str = "higher_better"
    score_scale: str = "positive_unbounded"
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
        eps = getattr(ctx, "epsilon", 1e-12)

        spectral = (context or {}).get("spectral_result")
        if spectral is None or spectral.power is None or priors is None:
            return MetricResult(score=0.0, valid=False, notes="missing data")

        power = spectral.power
        f = len(power)
        band = build_expected_band_mask(f, priors)
        if not np.any(band):
            return MetricResult(score=0.0, valid=False, notes="empty band")

        band_power = np.where(band, power, -np.inf)
        k_c = int(np.argmax(band_power))

        w = _half_max_width(power, k_c)
        if w is None:
            return MetricResult(score=0.0, valid=False, notes="no crossings")

        score = 1.0 / (w + eps)
        return MetricResult(score=float(score), features={"half_width": w})

    def evaluate_batch(
        self,
        signals: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelopes: np.ndarray | None = None,
        context: dict | None = None,
    ) -> BatchMetricArrays:
        """Per-signal fallback — half-max crossing requires interpolation."""
        from quality_tool.metrics.batch_result import BatchMetricArrays

        n = signals.shape[0]
        scores = np.full(n, np.nan)
        valid_arr = np.zeros(n, dtype=bool)

        ctx = (context or {}).get("analysis_context")
        priors: SpectralPriors | None = (context or {}).get("spectral_priors")
        eps = getattr(ctx, "epsilon", 1e-12)
        power = (context or {}).get("batch_power")

        if power is None or priors is None:
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        f = power.shape[1]
        band = build_expected_band_mask(f, priors)
        if not np.any(band):
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        band_power = np.where(band[np.newaxis, :], power, -np.inf)
        k_c_arr = np.argmax(band_power, axis=1)

        for i in range(n):
            w = _half_max_width(power[i], int(k_c_arr[i]))
            if w is not None:
                scores[i] = 1.0 / (w + eps)
                valid_arr[i] = True

        return BatchMetricArrays(scores=scores, valid=valid_arr)
