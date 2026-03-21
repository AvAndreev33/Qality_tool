"""High-frequency noise level metric for Quality_tool.

Measures the fraction of total power that lies above the carrier band::

    HFN = sum(P[k > k_c + Δk]) / (sum(P) + ε)

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
    hann_windowed_rfft_power,
)

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class HighFrequencyNoiseLevel:
    """High-frequency noise level metric.

    Score meaning: lower is better.
    """

    name: str = "high_frequency_noise_level"
    category: str = "noise"
    display_name: str = "High-Freq Noise Level"
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
        p = power[0]
        f = p.shape[0]

        # Exclude DC for carrier search.
        search = p.copy()
        search[dc] = -np.inf
        k_c = int(np.argmax(search))

        # High-frequency region: bins strictly above the carrier band.
        hf_start = k_c + bw + 1
        if hf_start >= f:
            return MetricResult(score=0.0, valid=False,
                                notes="no high-frequency bins above carrier")

        # Exclude DC from total power.
        total_power = float(np.sum(p[1:]))
        if total_power < eps:
            return MetricResult(score=0.0, valid=False,
                                notes="total power is zero")

        hf_power = float(np.sum(p[hf_start:]))
        hfn = hf_power / (total_power + eps)

        return MetricResult(
            score=float(hfn),
            features={"hf_power": hf_power, "total_power": total_power},
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
        f = power.shape[1]

        # Find carrier per signal (exclude DC).
        search = power.copy()
        search[:, dc] = -np.inf
        k_c = np.argmax(search, axis=1)  # (N,)

        # Total power excluding DC.
        total_power = power[:, 1:].sum(axis=1)

        # High-frequency power: sum of bins above k_c + bw.
        bin_idx = np.arange(f)[np.newaxis, :]  # (1, F)
        hf_mask = bin_idx > (k_c[:, np.newaxis] + bw)  # (N, F)
        hf_power = np.where(hf_mask, power, 0.0).sum(axis=1)

        valid = (total_power >= eps) & hf_mask.any(axis=1)
        scores = np.full(n, np.nan)
        scores[valid] = hf_power[valid] / (total_power[valid] + eps)

        return BatchMetricArrays(
            scores=scores,
            valid=valid,
            features={"hf_power": hf_power, "total_power": total_power},
        )
