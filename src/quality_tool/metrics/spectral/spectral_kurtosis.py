"""Spectral kurtosis metric.

Measures how peaked the spectral distribution is::

    SK = mu_4 / (sigma^4 + eps)
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
from quality_tool.metrics.spectral._spectral_batch_helpers import (
    normalized_spectral_weights,
    spectral_centroid_batch,
    spectral_variance_batch,
)

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class SpectralKurtosis:
    """Spectral kurtosis.

    Score meaning: higher is better (sharper peaks).
    """

    name: str = "spectral_kurtosis"
    category: str = "spectral"
    display_name: str = "Spectral Kurtosis"
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

        spectral = (context or {}).get("spectral_result")
        if spectral is None or spectral.power is None:
            return MetricResult(score=0.0, valid=False, notes="missing data")

        power = spectral.power[np.newaxis, :]
        bins = np.arange(power.shape[1], dtype=float)

        p = normalized_spectral_weights(power, eps)
        mu = spectral_centroid_batch(p, bins)
        var = spectral_variance_batch(p, bins, mu)

        diff = bins[np.newaxis, :] - mu[:, np.newaxis]
        mu4 = (p * diff ** 4).sum(axis=1)

        sigma4 = var ** 2
        if sigma4[0] < eps:
            return MetricResult(score=0.0, valid=False, notes="unstable variance")

        score = float(mu4[0] / (sigma4[0] + eps))
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
        eps = getattr(ctx, "epsilon", 1e-12)
        power = (context or {}).get("batch_power")

        if power is None:
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        bins = np.arange(power.shape[1], dtype=float)
        p = normalized_spectral_weights(power, eps)
        mu = spectral_centroid_batch(p, bins)
        var = spectral_variance_batch(p, bins, mu)

        diff = bins[np.newaxis, :] - mu[:, np.newaxis]
        mu4 = (p * diff ** 4).sum(axis=1)
        sigma4 = var ** 2

        stable = sigma4 >= eps
        scores[stable] = mu4[stable] / (sigma4[stable] + eps)
        valid_arr = stable

        return BatchMetricArrays(scores=scores, valid=valid_arr)
