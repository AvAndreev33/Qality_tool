"""Main-peak to sidelobe ratio metric for Quality_tool.

Computes the ratio of the main envelope peak to the strongest
secondary peak outside the main-peak support::

    MPSR = e_peak / (e_side + epsilon)
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
from quality_tool.metrics.envelope._envelope_helpers import (
    detect_secondary_peaks,
    main_support_mask_batch,
)

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class MainPeakToSidelobeRatio:
    """Main-peak to sidelobe ratio metric.

    Score meaning: higher is better.
    """

    name: str = "main_peak_to_sidelobe_ratio"
    category: str = "envelope"
    display_name: str = "Main Peak / Sidelobe"
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

        ctx = (context or {}).get("analysis_context")
        alpha = getattr(ctx, "alpha_main_support", 0.1)
        eps = getattr(ctx, "epsilon", 1e-12)
        min_dist = getattr(ctx, "secondary_peak_min_distance", 3)
        min_prom = getattr(ctx, "secondary_peak_min_prominence", 0.0)

        e_peak = float(np.max(envelope))
        if e_peak <= 0:
            return MetricResult(score=0.0, valid=False,
                                notes="envelope peak is zero")

        w_main = envelope >= alpha * e_peak
        sec = detect_secondary_peaks(envelope, w_main, min_dist, min_prom)
        e_side = float(np.max(sec)) if len(sec) > 0 else 0.0

        mpsr = e_peak / (e_side + eps)
        return MetricResult(
            score=float(mpsr),
            features={"e_peak": e_peak, "e_side": e_side},
        )

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
        min_dist = getattr(ctx, "secondary_peak_min_distance", 3)
        min_prom = getattr(ctx, "secondary_peak_min_prominence", 0.0)

        scores = np.full(n, np.nan)
        valid_arr = np.zeros(n, dtype=bool)
        feat_peak = np.zeros(n)
        feat_side = np.zeros(n)

        if envelopes is None:
            return BatchMetricArrays(
                scores=scores, valid=valid_arr,
                features={"e_peak": feat_peak, "e_side": feat_side},
            )

        finite_mask = np.all(np.isfinite(envelopes), axis=1)
        e_peak = np.max(envelopes, axis=1)
        w_main = main_support_mask_batch(envelopes, e_peak, alpha)

        e_side = np.zeros(n)
        for i in range(n):
            if not finite_mask[i] or e_peak[i] <= 0:
                continue
            sec = detect_secondary_peaks(
                envelopes[i], w_main[i], min_dist, min_prom,
            )
            if len(sec) > 0:
                e_side[i] = float(np.max(sec))

        ok = finite_mask & (e_peak > 0)
        scores[ok] = e_peak[ok] / (e_side[ok] + eps)
        valid_arr = ok
        feat_peak[:] = e_peak
        feat_side[:] = e_side

        return BatchMetricArrays(
            scores=scores, valid=valid_arr,
            features={"e_peak": feat_peak, "e_side": feat_side},
        )
