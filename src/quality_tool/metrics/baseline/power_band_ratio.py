"""Power band ratio metric for Quality_tool.

Computes the ratio of power in a configurable signal-frequency band to the
total spectral power (excluding DC)::

    PBR = sum(amplitude[signal_band]^2) / sum(amplitude[total_band]^2)

Uses the shared FFT helper from :mod:`quality_tool.spectral.fft`.
Can consume a precomputed :class:`SpectralResult` via
``context["spectral_result"]`` to avoid redundant FFT computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quality_tool.core.models import MetricResult
from quality_tool.evaluation.recipe import RAW, RecipeBinding, SignalRecipe
from quality_tool.metrics.base import RepresentationNeeds
from quality_tool.spectral.fft import SpectralResult, compute_spectrum

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class PowerBandRatio:
    """Spectral power-band-ratio metric.

    Parameters
    ----------
    low_freq : float
        Lower bound of the signal frequency band (inclusive).
    high_freq : float
        Upper bound of the signal frequency band (inclusive).

    Formula::

        PBR = sum(amplitude[signal_band]^2) / sum(amplitude[total_band]^2)

    ``total_band`` excludes DC (frequency index 0).

    Returns ``valid=False`` when total power is effectively zero or the
    signal is too short.
    """

    name: str = "power_band_ratio"
    category: str = "baseline"
    display_name: str = "Power Band Ratio"
    signal_recipe: SignalRecipe = RAW
    recipe_binding: RecipeBinding = "active"
    needs_spectral: bool = True

    representation_needs: RepresentationNeeds = RepresentationNeeds(amplitude=True)

    def __init__(
        self,
        low_freq: float = 0.05,
        high_freq: float = 0.45,
    ) -> None:
        self.low_freq = low_freq
        self.high_freq = high_freq

    def evaluate(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelope: np.ndarray | None = None,
        context: dict | None = None,
    ) -> MetricResult:
        if signal.ndim != 1 or signal.size < 2:
            return MetricResult(
                score=0.0,
                features={},
                valid=False,
                notes="signal must be 1-D with at least 2 samples",
            )

        # Use precomputed spectral result when available.
        spectral: SpectralResult | None = None
        if context is not None:
            spectral = context.get("spectral_result")

        if spectral is None:
            spectral = compute_spectrum(signal, z_axis)

        frequencies = spectral.frequencies
        amplitude = spectral.amplitude

        # Derive power locally from amplitude.
        power = amplitude ** 2

        # Total band: everything except DC (index 0).
        total_mask = np.ones(len(frequencies), dtype=bool)
        total_mask[0] = False
        total_power = float(np.sum(power[total_mask]))

        if total_power < 1e-20:
            return MetricResult(
                score=0.0,
                features={"signal_power": 0.0, "total_power": total_power},
                valid=False,
                notes="total spectral power is effectively zero",
            )

        # Signal band.
        signal_mask = (
            (frequencies >= self.low_freq) & (frequencies <= self.high_freq)
        )
        signal_power = float(np.sum(power[signal_mask]))

        pbr = signal_power / total_power

        return MetricResult(
            score=float(pbr),
            features={"signal_power": signal_power, "total_power": total_power},
        )

    def evaluate_batch(
        self,
        signals: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelopes: np.ndarray | None = None,
        context: dict | None = None,
    ) -> BatchMetricArrays:
        """Vectorised evaluation over a chunk of signals ``(N, M)``."""
        from quality_tool.metrics.batch_result import BatchMetricArrays

        n, m = signals.shape

        # Get batch spectral data from context if available
        if context is not None and "batch_amplitude" in context:
            amplitude = context["batch_amplitude"]      # (N, F)
            frequencies = context["batch_frequencies"]   # (F,)
        else:
            # Fallback: compute batch FFT
            if z_axis is not None and len(z_axis) >= 2:
                spacing = float(np.mean(np.diff(z_axis)))
                if spacing <= 0:
                    spacing = 1.0
            else:
                spacing = 1.0
            fft_coeffs = np.fft.rfft(signals, axis=1)
            frequencies = np.fft.rfftfreq(m, d=spacing)
            amplitude = np.abs(fft_coeffs)

        power = amplitude ** 2  # (N, F)

        # Total power: exclude DC (index 0)
        total_power = np.sum(power[:, 1:], axis=1)  # (N,)

        # Signal band
        signal_mask = (
            (frequencies >= self.low_freq) & (frequencies <= self.high_freq)
        )
        signal_power = np.sum(power[:, signal_mask], axis=1)  # (N,)

        scores = np.full(n, np.nan)
        valid = total_power >= 1e-20

        with np.errstate(divide="ignore", invalid="ignore"):
            scores[valid] = signal_power[valid] / total_power[valid]

        features = {"signal_power": signal_power, "total_power": total_power}
        return BatchMetricArrays(scores=scores, valid=valid, features=features)
