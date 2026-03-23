"""Dominant spectral peak prominence metric.

Computes local spectral contrast around the dominant peak::

    DSPP = P[k_0] / (median(N_loc) + eps)
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

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class DominantSpectralPeakProminence:
    """Dominant spectral peak prominence.

    Score meaning: higher is better.
    """

    name: str = "dominant_spectral_peak_prominence"
    category: str = "spectral"
    display_name: str = "Dominant Peak Prominence"
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
        eps = getattr(ctx, "epsilon", 1e-12)
        w_prom = getattr(ctx, "prominence_window_bins", 20)
        dk_prom = getattr(ctx, "prominence_exclusion_half_width_bins", 3)

        spectral = (context or {}).get("spectral_result")
        if spectral is None or spectral.power is None:
            return MetricResult(score=0.0, valid=False, notes="no power spectrum")

        power = spectral.power
        f = len(power)
        search = power.copy()
        if f > 1:
            search[0] = -np.inf
        k_0 = int(np.argmax(search))

        # Build neighbourhood mask.
        bins = np.arange(f)
        dist = np.abs(bins - k_0)
        n_loc = (dist <= w_prom) & (dist > dk_prom)
        if not np.any(n_loc):
            return MetricResult(score=0.0, valid=False, notes="empty neighbourhood")

        med = float(np.median(power[n_loc]))
        score = float(power[k_0] / (med + eps))
        return MetricResult(score=score, features={"k_0": float(k_0), "local_median": med})

    def evaluate_batch(
        self,
        signals: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelopes: np.ndarray | None = None,
        context: dict | None = None,
    ) -> BatchMetricArrays:
        from quality_tool.metrics.batch_result import BatchMetricArrays

        n = signals.shape[0]
        scores = np.full(n, np.nan)
        valid_arr = np.zeros(n, dtype=bool)

        ctx = (context or {}).get("analysis_context")
        eps = getattr(ctx, "epsilon", 1e-12)
        w_prom = getattr(ctx, "prominence_window_bins", 20)
        dk_prom = getattr(ctx, "prominence_exclusion_half_width_bins", 3)

        power = (context or {}).get("batch_power")
        if power is None:
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        f = power.shape[1]
        search = power.copy()
        if f > 1:
            search[:, 0] = -np.inf
        k_0 = np.argmax(search, axis=1)  # (N,)

        p_peak = power[np.arange(n), k_0]  # (N,)

        # Build per-signal neighbourhood masks.
        bins = np.arange(f)[np.newaxis, :]  # (1, F)
        dist = np.abs(bins - k_0[:, np.newaxis])  # (N, F)
        n_loc = (dist <= w_prom) & (dist > dk_prom)  # (N, F)

        has_loc = n_loc.any(axis=1)

        # Compute median for valid signals.
        # Use masked approach: set non-neighbourhood bins to NaN, then nanmedian.
        masked_power = np.where(n_loc, power, np.nan)
        med = np.nanmedian(masked_power, axis=1)  # (N,)

        scores[has_loc] = p_peak[has_loc] / (med[has_loc] + eps)
        valid_arr = has_loc

        return BatchMetricArrays(
            scores=scores, valid=valid_arr,
            features={"k_0": k_0.astype(float)},
        )
