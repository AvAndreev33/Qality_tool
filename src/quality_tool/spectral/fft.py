"""Shared FFT / spectrum helper for Quality_tool.

Provides a lightweight spectral representation that multiple metrics can
reuse.  This module intentionally returns only *frequencies* and *amplitude
spectrum* — metrics that need power can derive it locally from the amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpectralResult:
    """Result of computing the spectrum of a 1-D signal.

    Attributes:
        frequencies: 1-D array of frequency bins (from ``np.fft.rfftfreq``).
        amplitude: 1-D array of amplitude values (``|FFT coefficients|``).
    """

    frequencies: np.ndarray
    amplitude: np.ndarray


def compute_spectrum(
    signal: np.ndarray,
    z_axis: np.ndarray | None = None,
) -> SpectralResult:
    """Compute the amplitude spectrum of a 1-D signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.
    z_axis : np.ndarray | None
        Optional physical z-axis.  If provided, the sample spacing is
        derived from the mean step size.  Otherwise a spacing of ``1.0``
        is assumed (index-based).

    Returns
    -------
    SpectralResult
        Frequencies and amplitude spectrum.

    Raises
    ------
    ValueError
        If *signal* is not 1-D or has fewer than 1 sample.
    """
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
    if signal.size < 1:
        raise ValueError("signal must have at least 1 sample")

    if z_axis is not None and len(z_axis) >= 2:
        spacing = float(np.mean(np.diff(z_axis)))
        if spacing <= 0:
            spacing = 1.0
    else:
        spacing = 1.0

    n = len(signal)
    fft_coeffs = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, d=spacing)
    amplitude = np.abs(fft_coeffs)

    return SpectralResult(frequencies=frequencies, amplitude=amplitude)
