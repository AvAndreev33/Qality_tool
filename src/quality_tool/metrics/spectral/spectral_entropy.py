"""Spectral entropy metric.

Normalised entropy of the spectral power distribution::

    SE = -sum(p[k] * log(p[k] + eps)) / log(K)
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
)
from quality_tool.spectral.priors import SpectralPriors, positive_freq_mask

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class SpectralEntropy:
    """Normalised spectral entropy.

    Score meaning: lower is better (lower = more concentrated).
    """

    name: str = "spectral_entropy"
    category: str = "spectral"
    display_name: str = "Spectral Entropy"
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
        dc_exclude = getattr(ctx, "dc_exclude", True)

        spectral = (context or {}).get("spectral_result")
        if spectral is None or spectral.power is None:
            return MetricResult(score=0.0, valid=False, notes="missing data")

        power = spectral.power
        f = len(power)
        pos = positive_freq_mask(f, dc_exclude)
        k = int(np.sum(pos))
        if k <= 1:
            return MetricResult(score=0.0, valid=False, notes="too few bins")

        p_pos = power[pos]
        total = p_pos.sum() + eps
        p_norm = p_pos / total

        entropy = -np.sum(p_norm * np.log(p_norm + eps)) / np.log(k)
        return MetricResult(score=float(entropy))

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
        dc_exclude = getattr(ctx, "dc_exclude", True)
        power = (context or {}).get("batch_power")

        if power is None:
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        f = power.shape[1]
        pos = positive_freq_mask(f, dc_exclude)
        k = int(np.sum(pos))
        if k <= 1:
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        p_pos = power[:, pos]  # (N, K)
        total = p_pos.sum(axis=1, keepdims=True) + eps
        p_norm = p_pos / total

        entropy = -np.sum(p_norm * np.log(p_norm + eps), axis=1) / np.log(k)
        scores = entropy
        valid_arr = np.ones(n, dtype=bool)

        return BatchMetricArrays(scores=scores, valid=valid_arr)
