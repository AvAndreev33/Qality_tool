"""Envelope height metric for Quality_tool.

Computes the peak value of the envelope::

    EH = max_n e[n]
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


class EnvelopeHeight:
    """Envelope height metric.

    Score meaning: higher is better.
    """

    name: str = "envelope_height"
    category: str = "envelope"
    display_name: str = "Envelope Height"
    score_direction: str = "higher_better"
    score_scale: str = "positive_unbounded"
    signal_recipe: SignalRecipe = ROI_MEAN_SUBTRACTED_LINEAR_DETRENDED
    recipe_binding: RecipeBinding = "fixed"
    representation_needs: RepresentationNeeds = RepresentationNeeds(envelope=True)

    def evaluate(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelope: np.ndarray | None = None,
        context: dict | None = None,
    ) -> MetricResult:
        if envelope is None or envelope.size == 0:
            return MetricResult(score=0.0, valid=False,
                                notes="envelope not available")
        if not np.all(np.isfinite(envelope)):
            return MetricResult(score=0.0, valid=False,
                                notes="envelope contains non-finite values")

        e_peak = float(np.max(envelope))
        return MetricResult(score=e_peak, features={"e_peak": e_peak})

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
        valid = np.zeros(n, dtype=bool)

        if envelopes is None:
            return BatchMetricArrays(scores=scores, valid=valid)

        finite_mask = np.all(np.isfinite(envelopes), axis=1)
        e_peak = np.max(envelopes, axis=1)

        valid = finite_mask & (envelopes.shape[1] > 0)
        scores[valid] = e_peak[valid]

        return BatchMetricArrays(
            scores=scores, valid=valid,
            features={"e_peak": e_peak},
        )
