"""Shared FFT / spectrum helper for Quality_tool.

Provides a lightweight spectral representation that multiple metrics can
reuse.  The module supports:

- one-sided positive-frequency view (via ``np.fft.rfft``)
- amplitude and power spectra
- optional raw complex FFT coefficients
- DC-bin index marker
- frequency-band index selection helper
- both single-signal and batch APIs

The spectral layer is intentionally minimal.  It provides the
representations needed by baseline and upcoming metric groups without
building a full DSP framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------

@dataclass
class SpectralResult:
    """Result of computing the spectrum of a 1-D signal.

    Attributes
    ----------
    frequencies : np.ndarray
        1-D array of one-sided frequency bins (from ``np.fft.rfftfreq``).
    amplitude : np.ndarray
        1-D array of amplitude values ``|FFT coefficients|``.
    power : np.ndarray | None
        1-D array of power values ``amplitude ** 2``.  ``None`` when
        power was not requested.
    complex_fft : np.ndarray | None
        1-D array of raw complex FFT coefficients.  ``None`` when
        complex output was not requested.
    dc_index : int
        Index of the DC bin (always ``0``).
    """

    frequencies: np.ndarray
    amplitude: np.ndarray
    power: np.ndarray | None = None
    complex_fft: np.ndarray | None = None
    dc_index: int = 0


@dataclass
class BatchSpectralResult:
    """Batch spectral result for ``(N, F)`` shaped arrays.

    Attributes
    ----------
    frequencies : np.ndarray
        1-D array of shape ``(F,)`` — shared frequency axis.
    amplitude : np.ndarray | None
        2-D array of shape ``(N, F)`` with per-signal amplitudes.
    power : np.ndarray | None
        2-D array of shape ``(N, F)`` with per-signal power values.
    complex_fft : np.ndarray | None
        2-D array of shape ``(N, F)`` with raw complex coefficients.
    dc_index : int
        Index of the DC bin (always ``0``).
    """

    frequencies: np.ndarray
    amplitude: np.ndarray | None = None
    power: np.ndarray | None = None
    complex_fft: np.ndarray | None = None
    dc_index: int = 0


# ------------------------------------------------------------------
# Spacing helper
# ------------------------------------------------------------------

def _resolve_spacing(z_axis: np.ndarray | None) -> float:
    """Determine sample spacing from *z_axis*, defaulting to 1.0."""
    if z_axis is not None and len(z_axis) >= 2:
        spacing = float(np.mean(np.diff(z_axis)))
        if spacing <= 0:
            spacing = 1.0
    else:
        spacing = 1.0
    return spacing


# ------------------------------------------------------------------
# Single-signal API
# ------------------------------------------------------------------

def compute_spectrum(
    signal: np.ndarray,
    z_axis: np.ndarray | None = None,
    *,
    include_power: bool = False,
    include_complex: bool = False,
) -> SpectralResult:
    """Compute the spectrum of a 1-D signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.
    z_axis : np.ndarray | None
        Optional physical z-axis.  If provided, the sample spacing is
        derived from the mean step size.  Otherwise a spacing of ``1.0``
        is assumed (index-based).
    include_power : bool
        If ``True``, also compute and return the power spectrum.
    include_complex : bool
        If ``True``, also return the raw complex FFT coefficients.

    Returns
    -------
    SpectralResult

    Raises
    ------
    ValueError
        If *signal* is not 1-D or has fewer than 1 sample.
    """
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
    if signal.size < 1:
        raise ValueError("signal must have at least 1 sample")

    spacing = _resolve_spacing(z_axis)
    n = len(signal)

    fft_coeffs = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, d=spacing)
    amplitude = np.abs(fft_coeffs)

    power = (amplitude ** 2) if include_power else None
    complex_out = fft_coeffs if include_complex else None

    return SpectralResult(
        frequencies=frequencies,
        amplitude=amplitude,
        power=power,
        complex_fft=complex_out,
    )


# ------------------------------------------------------------------
# Batch API
# ------------------------------------------------------------------

def compute_spectrum_batch(
    signals: np.ndarray,
    z_axis: np.ndarray | None = None,
    *,
    include_amplitude: bool = True,
    include_power: bool = False,
    include_complex: bool = False,
) -> BatchSpectralResult:
    """Compute spectra for a batch of signals.

    Parameters
    ----------
    signals : np.ndarray
        2-D array of shape ``(N, M)``.
    z_axis : np.ndarray | None
        Optional shared physical z-axis of length ``M``.
    include_amplitude : bool
        Compute amplitude spectrum (default ``True``).
    include_power : bool
        Compute power spectrum.
    include_complex : bool
        Return raw complex FFT coefficients.

    Returns
    -------
    BatchSpectralResult
    """
    if signals.ndim != 2:
        raise ValueError(f"signals must be 2-D, got shape {signals.shape}")

    spacing = _resolve_spacing(z_axis)
    m = signals.shape[1]

    fft_coeffs = np.fft.rfft(signals, axis=1)
    frequencies = np.fft.rfftfreq(m, d=spacing)

    amplitude = np.abs(fft_coeffs) if include_amplitude else None
    power = (np.abs(fft_coeffs) ** 2) if include_power else None
    complex_out = fft_coeffs if include_complex else None

    return BatchSpectralResult(
        frequencies=frequencies,
        amplitude=amplitude,
        power=power,
        complex_fft=complex_out,
    )


# ------------------------------------------------------------------
# Frequency-band helper
# ------------------------------------------------------------------

def frequency_band_indices(
    frequencies: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Return a boolean mask for frequencies in ``[low, high]``.

    Parameters
    ----------
    frequencies : np.ndarray
        1-D frequency array.
    low, high : float
        Inclusive frequency bounds.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(len(frequencies),)``.
    """
    return (frequencies >= low) & (frequencies <= high)
