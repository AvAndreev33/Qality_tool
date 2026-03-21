"""Residual noise energy metric for Quality_tool.

Reconstructs a band-limited signal from the carrier band using full
complex FFT, then measures the residual energy fraction::

    RNE = sum(r^2) / (sum(x_w^2) + ε)

Uses Hann windowing and full (symmetric) FFT for reconstruction.
Lower is better.
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


def _compute_rne_batch(
    signals: np.ndarray,
    band_half_width: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Core RNE computation for a batch of signals.

    Returns (scores, valid, rne_values, carrier_bins).
    """
    n, m = signals.shape
    window = np.hanning(m).astype(signals.dtype)
    x_w = signals * window[np.newaxis, :]

    # Full complex FFT.
    X = np.fft.fft(x_w, axis=1)  # (N, M)

    # Find carrier on positive frequencies (indices 1..M//2).
    power_pos = np.abs(X[:, 1:m // 2 + 1]) ** 2
    k_c_pos = np.argmax(power_pos, axis=1) + 1  # offset by 1 for DC skip

    # Build symmetric band mask.
    bin_idx = np.arange(m)[np.newaxis, :]        # (1, M)
    k_c_2d = k_c_pos[:, np.newaxis]              # (N, 1)

    # Positive-side band.
    pos_band = np.abs(bin_idx - k_c_2d) <= band_half_width

    # Mirror for negative frequencies: index (M - k) is the conjugate.
    neg_k_c = m - k_c_pos                        # (N,)
    neg_k_c_2d = neg_k_c[:, np.newaxis]
    neg_band = np.abs(bin_idx - neg_k_c_2d) <= band_half_width

    band_mask = pos_band | neg_band  # (N, M)

    # Band-limited reconstruction.
    X_b = np.where(band_mask, X, 0.0)
    s_hat = np.real(np.fft.ifft(X_b, axis=1))

    residual = x_w - s_hat
    res_energy = np.sum(residual ** 2, axis=1)
    total_energy = np.sum(x_w ** 2, axis=1)

    valid = total_energy > epsilon
    scores = np.full(n, np.nan)
    scores[valid] = res_energy[valid] / (total_energy[valid] + epsilon)

    return scores, valid, res_energy, k_c_pos.astype(float)


class ResidualNoiseEnergy:
    """Residual noise energy metric.

    Score meaning: lower is better.
    """

    name: str = "residual_noise_energy"
    category: str = "noise"
    display_name: str = "Residual Noise Energy"
    score_direction: str = "lower_better"
    score_scale: str = "bounded_01"
    signal_recipe: SignalRecipe = ROI_MEAN_SUBTRACTED_LINEAR_DETRENDED
    recipe_binding: RecipeBinding = "fixed"
    representation_needs: RepresentationNeeds = RepresentationNeeds(complex_fft=True)

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

        scores, valid, res_e, k_c = _compute_rne_batch(
            signal[np.newaxis, :], bw, eps,
        )

        if not valid[0]:
            return MetricResult(score=0.0, valid=False,
                                notes="total energy too low")

        return MetricResult(
            score=float(scores[0]),
            features={"residual_energy": float(res_e[0]),
                       "carrier_bin": float(k_c[0])},
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

        scores, valid, res_e, k_c = _compute_rne_batch(signals, bw, eps)

        return BatchMetricArrays(
            scores=scores,
            valid=valid,
            features={"residual_energy": res_e, "carrier_bin": k_c},
        )
