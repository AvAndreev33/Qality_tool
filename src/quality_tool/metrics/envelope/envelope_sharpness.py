"""Envelope sharpness metric for Quality_tool.

Computes the ratio of envelope height to envelope width::

    ES = EH / (EW + epsilon)
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
    half_max_crossings_batch,
)
from quality_tool.metrics.envelope.envelope_width import _scalar_half_max_width

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class EnvelopeSharpness:
    """Envelope sharpness (peak-over-width) metric.

    Score meaning: higher is better.
    """

    name: str = "envelope_sharpness"
    category: str = "envelope"
    display_name: str = "Envelope Sharpness"
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
        eps = getattr(ctx, "epsilon", 1e-12)

        e_peak = float(np.max(envelope))
        width, ok = _scalar_half_max_width(envelope)
        if not ok:
            return MetricResult(score=0.0, valid=False,
                                notes="envelope width invalid")

        sharpness = e_peak / (width + eps)
        return MetricResult(
            score=float(sharpness),
            features={"e_peak": e_peak, "fwhm": width},
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
        eps = getattr(ctx, "epsilon", 1e-12)

        scores = np.full(n, np.nan)
        valid_arr = np.zeros(n, dtype=bool)

        if envelopes is None:
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        finite_mask = np.all(np.isfinite(envelopes), axis=1)
        n0 = np.argmax(envelopes, axis=1)
        e_peak = np.max(envelopes, axis=1)

        z_l, z_r, crossing_valid = half_max_crossings_batch(envelopes, n0, e_peak)
        fwhm = z_r - z_l

        ok = finite_mask & crossing_valid & (e_peak > 0)
        scores[ok] = e_peak[ok] / (fwhm[ok] + eps)
        valid_arr = ok

        return BatchMetricArrays(
            scores=scores, valid=valid_arr,
            features={"e_peak": e_peak, "fwhm": np.where(ok, fwhm, 0.0)},
        )
