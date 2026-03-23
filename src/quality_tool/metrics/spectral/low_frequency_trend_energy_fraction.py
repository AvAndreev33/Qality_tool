"""Low-frequency trend energy fraction metric.

Measures residual low-frequency content below the expected carrier band::

    LFTEF = sum(P[L]) / (sum(P) + eps)

where L = {k : 1 <= k < k_low}.
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
from quality_tool.spectral.priors import SpectralPriors, build_low_freq_mask

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class LowFrequencyTrendEnergyFraction:
    """Low-frequency trend energy fraction.

    Score meaning: lower is better.
    """

    name: str = "low_frequency_trend_energy_fraction"
    category: str = "spectral"
    display_name: str = "Low-Freq Trend Energy"
    score_direction: str = "lower_better"
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
        eps = getattr(ctx, "epsilon", 1e-12)

        spectral = (context or {}).get("spectral_result")
        if spectral is None or spectral.power is None or priors is None:
            return MetricResult(score=0.0, valid=False, notes="missing data")

        power = spectral.power
        f = len(power)
        lf_mask = build_low_freq_mask(f, priors)
        if not np.any(lf_mask):
            return MetricResult(score=0.0, features={"low_freq_energy": 0.0})

        score = float(np.sum(power[lf_mask]) / (np.sum(power) + eps))
        return MetricResult(score=score)

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
        priors: SpectralPriors | None = (context or {}).get("spectral_priors")
        eps = getattr(ctx, "epsilon", 1e-12)
        power = (context or {}).get("batch_power")

        if power is None or priors is None:
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        f = power.shape[1]
        lf_mask = build_low_freq_mask(f, priors)

        e_total = power.sum(axis=1) + eps
        if np.any(lf_mask):
            e_lf = power[:, lf_mask].sum(axis=1)
        else:
            e_lf = np.zeros(n)

        scores = e_lf / e_total
        valid_arr = np.ones(n, dtype=bool)

        return BatchMetricArrays(scores=scores, valid=valid_arr)
