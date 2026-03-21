"""Noise floor level metric for Quality_tool.

Estimates the spectral noise floor relative to the carrier band::

    NFL = median(P[out-of-band]) / (median(P[in-band]) + ε)

Uses Hann windowing before FFT.  Lower is better.
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
from quality_tool.metrics.noise._spectral_helpers import (
    find_carrier_and_band,
    hann_windowed_rfft_power,
)

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class NoiseFloorLevel:
    """Spectral noise-floor level metric.

    Score meaning: lower is better.
    """

    name: str = "noise_floor_level"
    category: str = "noise"
    display_name: str = "Noise Floor Level"
    signal_recipe: SignalRecipe = ROI_MEAN_SUBTRACTED_LINEAR_DETRENDED
    recipe_binding: RecipeBinding = "fixed"
    representation_needs: RepresentationNeeds = RepresentationNeeds(power=True)

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
        bw = getattr(ctx, "band_half_width_bins", 5)

        power, dc = hann_windowed_rfft_power(signal[np.newaxis, :])
        k_c, in_band, out_band = find_carrier_and_band(power, bw, dc)

        ib = in_band[0]
        ob = out_band[0]
        if not np.any(ib) or not np.any(ob):
            return MetricResult(score=0.0, valid=False,
                                notes="empty in-band or out-of-band region")

        floor = float(np.median(power[0, ob]))
        ref = float(np.median(power[0, ib]))
        nfl = floor / (ref + eps)

        return MetricResult(
            score=float(nfl),
            features={"floor_median": floor, "ref_median": ref},
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
        bw = getattr(ctx, "band_half_width_bins", 5)

        power, dc = hann_windowed_rfft_power(signals)
        k_c, in_band, out_band = find_carrier_and_band(power, bw, dc)

        scores = np.full(n, np.nan)
        valid = np.zeros(n, dtype=bool)
        floor_med = np.zeros(n)
        ref_med = np.zeros(n)

        has_ib = in_band.any(axis=1)
        has_ob = out_band.any(axis=1)

        for i in range(n):
            if has_ib[i] and has_ob[i]:
                floor_med[i] = np.median(power[i, out_band[i]])
                ref_med[i] = np.median(power[i, in_band[i]])
                valid[i] = True
                scores[i] = floor_med[i] / (ref_med[i] + eps)

        return BatchMetricArrays(
            scores=scores,
            valid=valid,
            features={"floor_median": floor_med, "ref_median": ref_med},
        )
