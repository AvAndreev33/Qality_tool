"""Spectral SNR noise metric for Quality_tool.

Computes the signal-to-noise ratio in the frequency domain by comparing
power inside a narrow carrier band to out-of-band power::

    SNR = 10 * log10((P_signal + ε) / (P_noise + ε))

Uses Hann windowing before FFT.
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


class SpectralSNR:
    """Spectral signal-to-noise ratio metric.

    Score meaning: higher is better.
    """

    name: str = "spectral_snr"
    category: str = "noise"
    display_name: str = "Spectral SNR"
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

        p_sig = float(np.sum(power[0, in_band[0]]))
        p_noise = float(np.sum(power[0, out_band[0]]))

        if not np.any(out_band[0]):
            return MetricResult(score=0.0, valid=False,
                                notes="no out-of-band bins")

        snr = 10.0 * np.log10((p_sig + eps) / (p_noise + eps))
        return MetricResult(
            score=float(snr),
            features={"p_signal": p_sig, "p_noise": p_noise,
                       "carrier_bin": int(k_c[0])},
        )

    def evaluate_batch(
        self,
        signals: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelopes: np.ndarray | None = None,
        context: dict | None = None,
    ) -> BatchMetricArrays:
        from quality_tool.metrics.batch_result import BatchMetricArrays

        ctx = (context or {}).get("analysis_context")
        eps = getattr(ctx, "epsilon", 1e-12)
        bw = getattr(ctx, "band_half_width_bins", 5)

        n = signals.shape[0]
        power, dc = hann_windowed_rfft_power(signals)
        k_c, in_band, out_band = find_carrier_and_band(power, bw, dc)

        # Masked sums.
        p_sig = np.where(in_band, power, 0.0).sum(axis=1)
        p_noise = np.where(out_band, power, 0.0).sum(axis=1)

        valid = out_band.any(axis=1)
        scores = np.full(n, np.nan)
        scores[valid] = 10.0 * np.log10(
            (p_sig[valid] + eps) / (p_noise[valid] + eps)
        )

        return BatchMetricArrays(
            scores=scores,
            valid=valid,
            features={"p_signal": p_sig, "p_noise": p_noise,
                       "carrier_bin": k_c.astype(float)},
        )
