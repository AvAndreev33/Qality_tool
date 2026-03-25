"""Phase curvature index metric.

Measures non-linearity of the phase across the support::

    PCI = |γ2| * L / (|γ1| + ε)

where ``γ1, γ2`` are from a weighted quadratic fit and ``L`` is the
support span.
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


class PhaseCurvatureIndex:
    """Phase curvature index metric.

    Score meaning: lower is better.
    """

    name: str = "phase_curvature_index"
    category: str = "phase"
    display_name: str = "Phase Curvature Index"
    score_direction: str = "lower_better"
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
        p = ctx.phase_weight_power
        env, phase = compute_analytic_batch(signals)
        support, n0, e_peak = compute_phase_support(env, ctx)
        u = compute_local_coordinate(m, z_axis, ctx)
        slopes_list, _ = compute_local_slopes(phase, support, u, eps)
        sup_valid = validate_phase_support(support, slopes_list, u, ctx)

        for i in range(n):
            if not sup_valid[i]:
                continue
            idx = np.nonzero(support[i])[0]
            phi = phase[i, idx]
            ui = u[idx]
            w = (env[i, idx] / (e_peak[i] + eps)) ** p

            # Weighted quadratic fit via normal equations.
            # Design matrix columns: 1, u, u^2
            sw = w.sum() + eps
            A = np.column_stack([np.ones_like(ui), ui, ui ** 2])
            Aw = A * w[:, None]
            AtWA = Aw.T @ A
            AtWp = Aw.T @ phi
            try:
                coeffs = np.linalg.solve(AtWA, AtWp)
            except np.linalg.LinAlgError:
                continue
            gamma1, gamma2 = coeffs[1], coeffs[2]

            span = ui[-1] - ui[0]
            score = abs(gamma2) * span / (abs(gamma1) + eps)
            scores[i] = score
            valid[i] = True

        return BatchMetricArrays(scores=scores, valid=valid)
