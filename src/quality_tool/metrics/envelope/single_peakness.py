"""Single-peakness metric for Quality_tool.

Measures the fraction of total envelope mass concentrated in the
main-peak support region::

    SP = sum_{W_main} e[n] / (sum_all e[n] + epsilon)

where W_main = {n : e[n] >= alpha * e_peak}.
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
from quality_tool.metrics.envelope._envelope_helpers import main_support_mask_batch

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class SinglePeakness:
    """Single-peakness metric.

    Score meaning: higher is better (near 1 = unimodal).
    """

    name: str = "single_peakness"
    category: str = "envelope"
    display_name: str = "Single-Peakness"
    score_direction: str = "higher_better"
    score_scale: str = "bounded_01"
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

        ctx = (context or {}).get("analysis_context")
        alpha = getattr(ctx, "alpha_main_support", 0.1)
        eps = getattr(ctx, "epsilon", 1e-12)

        e_peak = float(np.max(envelope))
        total = float(np.sum(envelope))
        if total < eps or e_peak <= 0:
            return MetricResult(score=0.0, valid=False,
                                notes="near-zero envelope mass")

        w_main = envelope >= alpha * e_peak
        if not np.any(w_main):
            return MetricResult(score=0.0, valid=False,
                                notes="empty main support")

        main_mass = float(np.sum(envelope[w_main]))
        sp = main_mass / (total + eps)
        return MetricResult(score=float(sp), features={"main_mass": main_mass,
                                                        "total_mass": total})

    def evaluate_batch(
        self,
        signals: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelopes: np.ndarray | None = None,
        context: dict | None = None,
    ) -> BatchMetricArrays:
        from quality_tool.metrics.batch_result import BatchMetricArrays

        n = signals.shape[0]
        ctx = (context or {}).get("analysis_context")
        alpha = getattr(ctx, "alpha_main_support", 0.1)
        eps = getattr(ctx, "epsilon", 1e-12)

        scores = np.full(n, np.nan)
        valid_arr = np.zeros(n, dtype=bool)

        if envelopes is None:
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        finite_mask = np.all(np.isfinite(envelopes), axis=1)
        e_peak = np.max(envelopes, axis=1)  # (N,)
        total = np.sum(envelopes, axis=1)   # (N,)

        w_main = main_support_mask_batch(envelopes, e_peak, alpha)  # (N, M)
        main_mass = np.sum(envelopes * w_main, axis=1)  # (N,)

        ok = finite_mask & (e_peak > 0) & (total > eps) & np.any(w_main, axis=1)
        scores[ok] = main_mass[ok] / (total[ok] + eps)
        valid_arr = ok

        return BatchMetricArrays(
            scores=scores, valid=valid_arr,
            features={"main_mass": main_mass, "total_mass": total},
        )
