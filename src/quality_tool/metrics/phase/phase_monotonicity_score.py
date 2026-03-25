"""Phase monotonicity score metric.

Measures the weighted fraction of local phase slopes that are
monotone inliers::

    PMS = sum_{i in M} w[i] / (sum_i w[i] + ε)
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
from quality_tool.metrics.phase._helpers import (
    compute_analytic_batch,
    compute_local_coordinate,
    compute_local_slopes,
    compute_phase_support,
    validate_phase_support,
)

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class PhaseMonotonicityScore:
    """Phase monotonicity score metric.

    Score meaning: higher is better.
    """

    name: str = "phase_monotonicity_score"
    category: str = "phase"
    display_name: str = "Phase Monotonicity"
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
        ctx = (context or {}).get("analysis_context")
        if ctx is None:
            return MetricResult(score=0.0, valid=False, notes="no context")
        result = self.evaluate_batch(
            signal[np.newaxis, :], z_axis, None, context,
        )
        if not result.valid[0]:
            return MetricResult(score=0.0, valid=False, notes="invalid support")
        return MetricResult(score=float(result.scores[0]))

    def evaluate_batch(
        self,
        signals: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelopes: np.ndarray | None = None,
        context: dict | None = None,
    ) -> BatchMetricArrays:
        from quality_tool.metrics.batch_result import BatchMetricArrays

        ctx = (context or {}).get("analysis_context")
        n, m = signals.shape
        scores = np.full(n, np.nan)
        valid = np.zeros(n, dtype=bool)
        if ctx is None:
            return BatchMetricArrays(scores=scores, valid=valid)

        eps = ctx.epsilon
        tau_mon = ctx.phase_monotonicity_tolerance_fraction
        env, phase = compute_analytic_batch(signals)
        support, n0, e_peak = compute_phase_support(env, ctx)
        u = compute_local_coordinate(m, z_axis, ctx)
        slopes_list, pair_idx_list = compute_local_slopes(
            phase, support, u, eps,
        )
        sup_valid = validate_phase_support(support, slopes_list, u, ctx)

        for i in range(n):
            if not sup_valid[i]:
                continue
            d = slopes_list[i]
            pidx = pair_idx_list[i]
            d_med = np.median(d)
            s_ref = np.sign(d_med)

            # Monotone inlier: same sign as reference and within tolerance.
            inlier = (s_ref * d > 0) & (np.abs(d - d_med) <= tau_mon * abs(d_med))

            # Pair weights: min(e[i], e[i+1]) / (e_peak + ε).
            e_i = env[i]
            ep = e_peak[i]
            w = np.minimum(e_i[pidx], e_i[pidx + 1]) / (ep + eps)

            total_w = np.sum(w) + eps
            score = np.sum(w[inlier]) / total_w
            scores[i] = score
            valid[i] = True

        return BatchMetricArrays(scores=scores, valid=valid)
