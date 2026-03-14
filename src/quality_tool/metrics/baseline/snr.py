"""Signal-to-noise ratio metric for Quality_tool.

Computes a simple heuristic SNR::

    SNR = (I_max - I_min) / sigma_noise

where ``sigma_noise`` is the standard deviation of the outer quarters of
the signal (samples likely away from the central fringe region).
"""

from __future__ import annotations

import numpy as np

from quality_tool.core.models import MetricResult


class SNR:
    """Heuristic signal-to-noise ratio metric.

    Formula::

        SNR = (I_max - I_min) / sigma_noise

    ``sigma_noise`` is estimated from the first and last quarter of the
    signal, which are assumed to lie outside the main fringe region.

    Returns ``valid=False`` when the noise estimate is effectively zero
    or the signal is too short.
    """

    name: str = "snr"
    input_policy: str = "processed"

    def evaluate(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelope: np.ndarray | None = None,
        context: dict | None = None,
    ) -> MetricResult:
        if signal.ndim != 1 or signal.size < 4:
            return MetricResult(
                score=0.0,
                features={},
                valid=False,
                notes="signal must be 1-D with at least 4 samples",
            )

        n = len(signal)
        quarter = max(n // 4, 1)
        noise_region = np.concatenate([signal[:quarter], signal[-quarter:]])
        noise_std = float(np.std(noise_region, ddof=0))

        peak_to_peak = float(np.max(signal) - np.min(signal))

        if noise_std < 1e-12:
            return MetricResult(
                score=0.0,
                features={"peak_to_peak": peak_to_peak, "noise_std": noise_std},
                valid=False,
                notes="noise standard deviation is effectively zero",
            )

        snr_value = peak_to_peak / noise_std

        return MetricResult(
            score=float(snr_value),
            features={"peak_to_peak": peak_to_peak, "noise_std": noise_std},
        )
