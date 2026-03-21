"""Local SNR noise metric for Quality_tool.

Computes an envelope-based energy ratio: energy inside the main packet
support (where envelope >= 50% of peak) versus energy outside::

    LocalSNR = 10 * log10((E_signal + ε) / (E_noise + ε))
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


class LocalSNR:
    """Local signal-to-noise ratio based on envelope support.

    Score meaning: higher is better.
    """

    name: str = "local_snr"
    category: str = "noise"
    display_name: str = "Local SNR"
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
        if signal.ndim != 1 or signal.size < 4:
            return MetricResult(score=0.0, valid=False,
                                notes="signal too short")
        if envelope is None:
            return MetricResult(score=0.0, valid=False,
                                notes="envelope not provided")

        ctx = (context or {}).get("analysis_context")
        eps = getattr(ctx, "epsilon", 1e-12)

        e_max = float(np.max(envelope))
        if e_max <= 0:
            return MetricResult(score=0.0, valid=False,
                                notes="envelope peak is zero")

        w_sig = envelope >= 0.5 * e_max
        w_noise = ~w_sig

        if not np.any(w_sig) or not np.any(w_noise):
            return MetricResult(score=0.0, valid=False,
                                notes="empty signal or noise region")

        e_signal = float(np.sum(signal[w_sig] ** 2))
        e_noise = float(np.sum(signal[w_noise] ** 2))

        snr = 10.0 * np.log10((e_signal + eps) / (e_noise + eps))
        return MetricResult(
            score=float(snr),
            features={"e_signal": e_signal, "e_noise": e_noise},
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

        scores = np.full(n, np.nan)
        valid = np.zeros(n, dtype=bool)
        e_signal = np.zeros(n)
        e_noise = np.zeros(n)

        if envelopes is None:
            return BatchMetricArrays(scores=scores, valid=valid,
                                     features={"e_signal": e_signal,
                                                "e_noise": e_noise})

        e_max = np.max(envelopes, axis=1, keepdims=True)  # (N, 1)
        w_sig = envelopes >= 0.5 * e_max                   # (N, M)
        w_noise = ~w_sig

        sig_sq = signals ** 2
        e_signal = np.where(w_sig, sig_sq, 0.0).sum(axis=1)
        e_noise = np.where(w_noise, sig_sq, 0.0).sum(axis=1)

        has_sig = w_sig.any(axis=1)
        has_noise = w_noise.any(axis=1)
        has_peak = e_max.squeeze() > 0
        valid = has_sig & has_noise & has_peak

        scores[valid] = 10.0 * np.log10(
            (e_signal[valid] + eps) / (e_noise[valid] + eps)
        )

        return BatchMetricArrays(
            scores=scores,
            valid=valid,
            features={"e_signal": e_signal, "e_noise": e_noise},
        )
