"""Low-frequency drift level metric for Quality_tool.

Estimates the slow trend energy relative to total signal energy::

    LFD = sum(t[n]^2) / (sum(x[n]^2) + ε)

where ``t[n]`` is a moving-average trend of the ROI signal.

This metric deliberately uses the ROI signal *without* mean subtraction
or detrending, so that drift is preserved and measurable.  Lower is better.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quality_tool.core.models import MetricResult
from quality_tool.evaluation.recipe import (
    ROI_ONLY,
    RecipeBinding,
    SignalRecipe,
)
from quality_tool.metrics.base import RepresentationNeeds

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


def _moving_average_batch(signals: np.ndarray, window: int) -> np.ndarray:
    """Compute moving average along axis=1 with 'same' output length."""
    n, m = signals.shape
    if window >= m:
        # Degenerate: return the mean replicated.
        return np.broadcast_to(
            np.mean(signals, axis=1, keepdims=True), (n, m),
        ).copy()

    kernel = np.ones(window, dtype=signals.dtype) / window
    out = np.empty_like(signals)
    for i in range(n):
        out[i] = np.convolve(signals[i], kernel, mode="same")
    return out


class LowFrequencyDriftLevel:
    """Low-frequency drift level metric.

    Score meaning: lower is better.
    """

    name: str = "low_frequency_drift_level"
    category: str = "noise"
    display_name: str = "Low-Freq Drift Level"
    score_direction: str = "lower_better"
    score_scale: str = "positive_unbounded"
    signal_recipe: SignalRecipe = ROI_ONLY
    recipe_binding: RecipeBinding = "fixed"
    representation_needs: RepresentationNeeds = RepresentationNeeds()

    def evaluate(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelope: np.ndarray | None = None,
        context: dict | None = None,
    ) -> MetricResult:
        if signal.ndim != 1 or signal.size < 4:
            return MetricResult(score=0.0, valid=False,
                                notes="signal too short")

        ctx = (context or {}).get("analysis_context")
        eps = getattr(ctx, "epsilon", 1e-12)
        drift_w = getattr(ctx, "drift_window", 31)

        m = signal.size
        if drift_w > m:
            return MetricResult(score=0.0, valid=False,
                                notes="ROI too short for drift window")

        kernel = np.ones(drift_w, dtype=signal.dtype) / drift_w
        trend = np.convolve(signal, kernel, mode="same")

        total_energy = float(np.sum(signal ** 2))
        trend_energy = float(np.sum(trend ** 2))

        lfd = trend_energy / (total_energy + eps)

        return MetricResult(
            score=float(lfd),
            features={"trend_energy": trend_energy,
                       "total_energy": total_energy},
        )

    def evaluate_batch(
        self,
        signals: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelopes: np.ndarray | None = None,
        context: dict | None = None,
    ) -> BatchMetricArrays:
        from quality_tool.metrics.batch_result import BatchMetricArrays

        n, m = signals.shape
        ctx = (context or {}).get("analysis_context")
        eps = getattr(ctx, "epsilon", 1e-12)
        drift_w = getattr(ctx, "drift_window", 31)

        scores = np.full(n, np.nan)
        valid = np.zeros(n, dtype=bool)
        trend_energy = np.zeros(n)
        total_energy = np.sum(signals ** 2, axis=1)

        if drift_w > m:
            return BatchMetricArrays(
                scores=scores,
                valid=valid,
                features={"trend_energy": trend_energy,
                           "total_energy": total_energy},
            )

        trend = _moving_average_batch(signals, drift_w)
        trend_energy = np.sum(trend ** 2, axis=1)

        valid = total_energy > eps
        scores[valid] = trend_energy[valid] / (total_energy[valid] + eps)

        return BatchMetricArrays(
            scores=scores,
            valid=valid,
            features={"trend_energy": trend_energy,
                       "total_energy": total_energy},
        )
