"""Spectral centroid offset metric.

Measures how far the spectral centre of mass is from the expected carrier::

    SCO = |mu - k_exp| / (k_exp + eps)
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
)
from quality_tool.spectral.priors import SpectralPriors

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class SpectralCentroidOffset:
    """Spectral centroid offset.

    Score meaning: lower is better.
    """

    name: str = "spectral_centroid_offset"
    category: str = "spectral"
    display_name: str = "Spectral Centroid Offset"
    score_direction: str = "lower_better"
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

        power = spectral.power[np.newaxis, :]  # (1, F)
        k_exp = float(priors.expected_carrier_bin)
        bins = np.arange(power.shape[1], dtype=float)

        p = normalized_spectral_weights(power, eps)
        mu = float(spectral_centroid_batch(p, bins)[0])
        score = abs(mu - k_exp) / (k_exp + eps)
        return MetricResult(score=float(score), features={"centroid": mu})

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

        k_exp = float(priors.expected_carrier_bin)
        bins = np.arange(power.shape[1], dtype=float)

        p = normalized_spectral_weights(power, eps)
        mu = spectral_centroid_batch(p, bins)

        scores = np.abs(mu - k_exp) / (k_exp + eps)
        valid_arr = np.ones(n, dtype=bool)

        return BatchMetricArrays(
            scores=scores, valid=valid_arr,
            features={"centroid": mu},
        )
