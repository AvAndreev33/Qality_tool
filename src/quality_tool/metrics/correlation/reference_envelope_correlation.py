"""Reference envelope correlation metric.

Compares the observed envelope shape with the expected Gaussian
reference envelope on the reference support::

    REC = <ẽ, ĝ>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import hilbert

from quality_tool.core.models import MetricResult
from quality_tool.evaluation.recipe import (
    ROI_MEAN_SUBTRACTED_LINEAR_DETRENDED,
    RecipeBinding,
    SignalRecipe,
)
from quality_tool.metrics.base import RepresentationNeeds
from quality_tool.metrics.correlation._helpers import (
    build_reference_model,
    build_reference_support,
    normalize_on_support,
    resolve_reference_scales,
)

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class ReferenceEnvelopeCorrelation:
    """Reference envelope correlation metric.

    Score meaning: higher is better.
    """

    name: str = "reference_envelope_correlation"
    category: str = "correlation"
    display_name: str = "Ref. Envelope Correlation"
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
            signal[np.newaxis, :],
            z_axis,
            envelope[np.newaxis, :] if envelope is not None else None,
            context,
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

        g_ref, _, _ = build_reference_model(u, T_ref, L_ref)
        support = build_reference_support(g_ref, ctx.reference_support_threshold_fraction)

        if np.sum(support) < ctx.minimum_reference_support_samples:
            return BatchMetricArrays(scores=scores, valid=valid)

        # Use precomputed envelopes from the bundle when available,
        # otherwise compute from signals.
        if envelopes is not None:
            e_obs = envelopes
        else:
            e_obs = np.abs(hilbert(signals, axis=1))

        # Normalize reference envelope (shared).
        g_norm = normalize_on_support(g_ref, support, eps)

        # Normalize observed envelopes (batch).
        e_norm = normalize_on_support(e_obs, support, eps)

        e_energy = np.sum(e_norm[:, support] ** 2, axis=1)
        sig_valid = e_energy > eps

        dot = np.sum(e_norm[:, support] * g_norm[support], axis=1)

        scores[sig_valid] = dot[sig_valid]
        valid[:] = sig_valid

        return BatchMetricArrays(scores=scores, valid=valid)
