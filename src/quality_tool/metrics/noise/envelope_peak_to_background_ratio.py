"""Envelope peak-to-background ratio metric for Quality_tool.

Computes the ratio of the envelope peak to the robust background
level (median of envelope outside the main peak support)::

    PBR = (e_peak + ε) / (e_bg + ε)
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

# Minimum samples outside the main peak for a reliable background estimate.
_MIN_BG_SAMPLES = 3


class EnvelopePeakToBackgroundRatio:
    """Envelope peak-to-background ratio metric.

    Score meaning: higher is better.
    """

    name: str = "envelope_peak_to_background_ratio"
    category: str = "noise"
    display_name: str = "Envelope Peak/Background"
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
        if envelope is None or envelope.size < 4:
            return MetricResult(score=0.0, valid=False,
                                notes="envelope not available or too short")

        ctx = (context or {}).get("analysis_context")
        eps = getattr(ctx, "epsilon", 1e-12)

        e_peak = float(np.max(envelope))
        if e_peak <= 0:
            return MetricResult(score=0.0, valid=False,
                                notes="envelope peak is zero")

        w_main = envelope >= 0.5 * e_peak
        bg_mask = ~w_main

        if np.sum(bg_mask) < _MIN_BG_SAMPLES:
            return MetricResult(score=0.0, valid=False,
                                notes="too few background samples")

        e_bg = float(np.median(envelope[bg_mask]))
        pbr = (e_peak + eps) / (e_bg + eps)

        return MetricResult(
            score=float(pbr),
            features={"e_peak": e_peak, "e_bg": e_bg},
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
        valid = np.zeros(n, dtype=bool)
        feat_peak = np.zeros(n)
        feat_bg = np.zeros(n)

        if envelopes is None:
            return BatchMetricArrays(scores=scores, valid=valid,
                                     features={"e_peak": feat_peak,
                                                "e_bg": feat_bg})

        e_peak = np.max(envelopes, axis=1)               # (N,)
        w_main = envelopes >= 0.5 * e_peak[:, np.newaxis]  # (N, M)
        bg_mask = ~w_main
        bg_count = bg_mask.sum(axis=1)

        # Per-signal median of background — vectorise with loop (median
        # does not broadcast over masked arrays efficiently).
        e_bg = np.zeros(n)
        for i in range(n):
            if bg_count[i] >= _MIN_BG_SAMPLES:
                e_bg[i] = np.median(envelopes[i, bg_mask[i]])

        valid = (e_peak > 0) & (bg_count >= _MIN_BG_SAMPLES)
        scores[valid] = (e_peak[valid] + eps) / (e_bg[valid] + eps)

        feat_peak[:] = e_peak
        feat_bg[:] = e_bg
        return BatchMetricArrays(
            scores=scores,
            valid=valid,
            features={"e_peak": feat_peak, "e_bg": feat_bg},
        )
