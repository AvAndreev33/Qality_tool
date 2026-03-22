"""Envelope symmetry metric for Quality_tool.

Measures left-right symmetry of the envelope around the main peak::

    ESYM = 1 - D / (S + epsilon)

where D = sum |e[n0-m] - e[n0+m]| and S = sum (e[n0-m] + e[n0+m])
for m = 1..H, and H = min(n0, M-1-n0).
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


def _scalar_symmetry(envelope: np.ndarray, eps: float) -> tuple[float, bool]:
    """Return (symmetry_score, valid) for a single envelope."""
    m = len(envelope)
    n0 = int(np.argmax(envelope))
    h = min(n0, m - 1 - n0)
    if h < 1:
        return 0.0, False

    left = envelope[n0 - h:n0][::-1]   # m=1..H reversed
    right = envelope[n0 + 1:n0 + h + 1]
    d = float(np.sum(np.abs(left - right)))
    s = float(np.sum(left + right))
    return 1.0 - d / (s + eps), True


class EnvelopeSymmetry:
    """Envelope symmetry metric.

    Score meaning: higher is better (near 1 = symmetric).
    """

    name: str = "envelope_symmetry"
    category: str = "envelope"
    display_name: str = "Envelope Symmetry"
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
        eps = getattr(ctx, "epsilon", 1e-12)

        score, ok = _scalar_symmetry(envelope, eps)
        if not ok:
            return MetricResult(score=0.0, valid=False,
                                notes="no mirrored comparison range")
        return MetricResult(score=score)

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
        m = envelopes.shape[1]
        n0 = np.argmax(envelopes, axis=1)  # (N,)

        # Vectorised symmetry for signals with sufficient mirror range.
        h_arr = np.minimum(n0, m - 1 - n0)  # (N,)
        h_max = int(np.max(h_arr)) if n > 0 else 0

        if h_max >= 1:
            # Build index offsets 1..h_max.
            offsets = np.arange(1, h_max + 1)  # (h_max,)
            left_idx = n0[:, np.newaxis] - offsets[np.newaxis, :]   # (N, h_max)
            right_idx = n0[:, np.newaxis] + offsets[np.newaxis, :]  # (N, h_max)

            # Mask valid offsets per signal.
            offset_valid = offsets[np.newaxis, :] <= h_arr[:, np.newaxis]  # (N, h_max)

            # Clip to valid range for indexing (masked out later).
            left_idx_safe = np.clip(left_idx, 0, m - 1)
            right_idx_safe = np.clip(right_idx, 0, m - 1)

            # Gather values.
            rows = np.arange(n)[:, np.newaxis]
            left_vals = envelopes[rows, left_idx_safe]    # (N, h_max)
            right_vals = envelopes[rows, right_idx_safe]  # (N, h_max)

            diff = np.abs(left_vals - right_vals) * offset_valid
            summ = (left_vals + right_vals) * offset_valid

            d = diff.sum(axis=1)  # (N,)
            s = summ.sum(axis=1)  # (N,)

            ok = finite_mask & (h_arr >= 1)
            scores[ok] = 1.0 - d[ok] / (s[ok] + eps)
            valid_arr = ok

        return BatchMetricArrays(scores=scores, valid=valid_arr)
