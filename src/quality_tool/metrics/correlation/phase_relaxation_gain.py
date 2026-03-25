"""Phase relaxation gain metric.

Diagnostic metric measuring how much freeing the carrier phase
improves model agreement::

    PRG = BPRC - CRC
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
from quality_tool.metrics.correlation._helpers import (
    build_reference_model,
    build_reference_support,
    normalize_on_support,
    orthonormalize_basis,
    resolve_reference_scales,
)

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class PhaseRelaxationGain:
    """Phase relaxation gain metric.

    Score meaning: lower is better.
    """

    name: str = "phase_relaxation_gain"
    category: str = "correlation"
    display_name: str = "Phase Relaxation Gain"
    score_direction: str = "lower_better"
    score_scale: str = "positive_unbounded"
    signal_recipe: SignalRecipe = ROI_MEAN_SUBTRACTED_LINEAR_DETRENDED
    recipe_binding: RecipeBinding = "fixed"

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
            return MetricResult(score=0.0, valid=False, notes="invalid reference")
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
        T_ref, L_ref, u = resolve_reference_scales(ctx, z_axis, m)
        if T_ref is None or L_ref is None or T_ref <= 0 or L_ref <= 0:
            return BatchMetricArrays(scores=scores, valid=valid)

        g_ref, r_c, r_s = build_reference_model(u, T_ref, L_ref)
        support = build_reference_support(g_ref, ctx.reference_support_threshold_fraction)

        if np.sum(support) < ctx.minimum_reference_support_samples:
            return BatchMetricArrays(scores=scores, valid=valid)

        c_norm = normalize_on_support(r_c, support, eps)
        s_norm = normalize_on_support(r_s, support, eps)
        q1, q2, q2_norm = orthonormalize_basis(c_norm, s_norm, support, eps)

        if q2_norm < eps:
            return BatchMetricArrays(scores=scores, valid=valid)

        x_norm = normalize_on_support(signals, support, eps)
        x_energy = np.sum(x_norm[:, support] ** 2, axis=1)
        sig_valid = x_energy > eps

        # CRC = <x̃, c̃>
        crc = np.sum(x_norm[:, support] * c_norm[support], axis=1)

        # BPRC = sqrt(a1^2 + a2^2)
        a1 = np.sum(x_norm[:, support] * q1[support], axis=1)
        a2 = np.sum(x_norm[:, support] * q2[support], axis=1)
        bprc = np.sqrt(a1 ** 2 + a2 ** 2)

        prg = bprc - crc
        scores[sig_valid] = prg[sig_valid]
        valid[:] = sig_valid

        return BatchMetricArrays(
            scores=scores, valid=valid,
            features={"crc": crc, "bprc": bprc},
        )
