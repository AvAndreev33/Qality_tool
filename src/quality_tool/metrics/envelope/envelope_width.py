"""Envelope width metric for Quality_tool.

Computes the full-width at half-maximum (FWHM) of the envelope::

    EW = z_R - z_L

where z_L and z_R are the linearly interpolated half-maximum crossing
positions around the main peak.
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

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


def _scalar_half_max_width(envelope: np.ndarray) -> tuple[float, bool]:
    """Return (FWHM, valid) for a single envelope."""
    n0 = int(np.argmax(envelope))
    e_peak = envelope[n0]
    if e_peak <= 0:
        return 0.0, False

    h = 0.5 * e_peak
    m = len(envelope)

    # Left crossing — scan backwards from peak.
    z_l = None
    for j in range(n0, 0, -1):
        if envelope[j - 1] <= h <= envelope[j]:
            denom = envelope[j] - envelope[j - 1]
            z_l = (j - 1) + (h - envelope[j - 1]) / denom if denom > 0 else float(j - 1)
            break

    # Right crossing — scan forwards from peak.
    z_r = None
    for j in range(n0, m - 1):
        if envelope[j + 1] <= h <= envelope[j]:
            denom = envelope[j] - envelope[j + 1]
            z_r = j + (envelope[j] - h) / denom if denom > 0 else float(j + 1)
            break

    if z_l is None or z_r is None:
        return 0.0, False
    return z_r - z_l, True


class EnvelopeWidth:
    """Envelope width (FWHM) metric.

    Score meaning: lower is better.
    """

    name: str = "envelope_width"
    category: str = "envelope"
    display_name: str = "Envelope Width"
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
        if envelope is None or envelope.size == 0:
            return MetricResult(score=0.0, valid=False,
                                notes="envelope not available")
        if not np.all(np.isfinite(envelope)):
            return MetricResult(score=0.0, valid=False,
                                notes="envelope contains non-finite values")

        width, ok = _scalar_half_max_width(envelope)
        if not ok:
            return MetricResult(score=0.0, valid=False,
                                notes="half-maximum crossing not found")
        return MetricResult(score=width, features={"fwhm": width})

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

        if envelopes is None:
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        finite_mask = np.all(np.isfinite(envelopes), axis=1)
        n0 = np.argmax(envelopes, axis=1)
        e_peak = np.max(envelopes, axis=1)

        z_l, z_r, crossing_valid = half_max_crossings_batch(envelopes, n0, e_peak)

        ok = finite_mask & crossing_valid & (e_peak > 0)
        scores[ok] = z_r[ok] - z_l[ok]
        valid_arr = ok

        return BatchMetricArrays(
            scores=scores, valid=valid_arr,
            features={"fwhm": np.where(ok, z_r - z_l, 0.0)},
        )
