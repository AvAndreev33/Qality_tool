"""Shared spectral helpers for noise metrics.

Provides Hann-windowed power spectrum computation and carrier-band
utilities used by multiple noise metrics.  These are lightweight
helpers, not a replacement for the main spectral layer.
"""

from __future__ import annotations

import numpy as np


def hann_windowed_rfft_power(signals: np.ndarray) -> tuple[np.ndarray, int]:
    """Compute Hann-windowed one-sided power spectrum for a batch.

    Parameters
    ----------
    signals : np.ndarray
        2-D array of shape ``(N, M)``.

    Returns
    -------
    power : np.ndarray
        2-D array of shape ``(N, F)`` where ``F = M // 2 + 1``.
        Power values ``|rFFT(windowed)|^2``.
    dc_index : int
        Always ``0``.
    """
    n, m = signals.shape
    window = np.hanning(m).astype(signals.dtype)
    windowed = signals * window[np.newaxis, :]
    fft_coeffs = np.fft.rfft(windowed, axis=1)
    power = np.abs(fft_coeffs) ** 2
    return power, 0


def find_carrier_and_band(
    power: np.ndarray,
    band_half_width: int,
    dc_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the dominant carrier bin and build band masks.

    Parameters
    ----------
    power : np.ndarray
        2-D power spectrum ``(N, F)``.
    band_half_width : int
        Half-width Δk around the carrier bin.
    dc_index : int
        DC bin index to exclude when searching for the carrier.

    Returns
    -------
    k_c : np.ndarray
        1-D array ``(N,)`` of carrier bin indices (int).
    in_band : np.ndarray
        2-D bool mask ``(N, F)`` for carrier-band bins.
    out_band : np.ndarray
        2-D bool mask ``(N, F)`` for out-of-band bins (excludes DC).
    """
    n, f = power.shape

    # Exclude DC for carrier search.
    search = power.copy()
    search[:, dc_index] = -np.inf
    k_c = np.argmax(search, axis=1)  # (N,)

    # Build per-signal band masks.
    bin_idx = np.arange(f)[np.newaxis, :]     # (1, F)
    k_c_2d = k_c[:, np.newaxis]               # (N, 1)
    in_band = np.abs(bin_idx - k_c_2d) <= band_half_width  # (N, F)

    # Out-of-band: not in band, and not DC.
    dc_mask = np.zeros(f, dtype=bool)
    dc_mask[dc_index] = True
    out_band = ~in_band & ~dc_mask[np.newaxis, :]

    return k_c, in_band, out_band
