"""Shared helpers for envelope quality metrics.

Provides batch-oriented helper functions that multiple envelope metrics
share, avoiding duplicated logic across individual metric modules.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def half_max_crossings_batch(
    envelopes: np.ndarray,
    n0: np.ndarray,
    e_peak: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find left and right half-maximum crossings with linear interpolation.

    Parameters
    ----------
    envelopes : np.ndarray
        Shape ``(N, M)`` — envelope values.
    n0 : np.ndarray
        Shape ``(N,)`` — index of the envelope peak per signal.
    e_peak : np.ndarray
        Shape ``(N,)`` — envelope peak value per signal.

    Returns
    -------
    z_L : np.ndarray
        Shape ``(N,)`` — interpolated left crossing position.
        ``NaN`` where no valid crossing was found.
    z_R : np.ndarray
        Shape ``(N,)`` — interpolated right crossing position.
        ``NaN`` where no valid crossing was found.
    valid : np.ndarray
        Shape ``(N,)`` — boolean, ``True`` where both crossings exist.
    """
    n_signals, m = envelopes.shape
    half = 0.5 * e_peak  # (N,)

    z_l = np.full(n_signals, np.nan)
    z_r = np.full(n_signals, np.nan)

    for i in range(n_signals):
        h = half[i]
        peak_idx = n0[i]
        env = envelopes[i]

        # --- left crossing: scan backwards from peak ---
        for j in range(peak_idx, 0, -1):
            if env[j - 1] <= h <= env[j]:
                # Linear interpolation between j-1 and j.
                denom = env[j] - env[j - 1]
                if denom > 0:
                    z_l[i] = (j - 1) + (h - env[j - 1]) / denom
                else:
                    z_l[i] = float(j - 1)
                break

        # --- right crossing: scan forwards from peak ---
        for j in range(peak_idx, m - 1):
            if env[j + 1] <= h <= env[j]:
                # Linear interpolation between j and j+1.
                denom = env[j] - env[j + 1]
                if denom > 0:
                    z_r[i] = j + (env[j] - h) / denom
                else:
                    z_r[i] = float(j + 1)
                break

    valid = np.isfinite(z_l) & np.isfinite(z_r)
    return z_l, z_r, valid


def main_support_mask_batch(
    envelopes: np.ndarray,
    e_peak: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Boolean mask of main-peak support region.

    Parameters
    ----------
    envelopes : np.ndarray
        Shape ``(N, M)``.
    e_peak : np.ndarray
        Shape ``(N,)``.
    alpha : float
        Fraction of peak defining the support threshold.

    Returns
    -------
    np.ndarray
        Shape ``(N, M)`` boolean mask where ``e[n] >= alpha * e_peak``.
    """
    return envelopes >= alpha * e_peak[:, np.newaxis]


def detect_secondary_peaks(
    envelope: np.ndarray,
    main_mask: np.ndarray,
    min_distance: int = 3,
    min_prominence: float = 0.0,
) -> np.ndarray:
    """Detect local maxima of a single envelope outside the main support.

    Parameters
    ----------
    envelope : np.ndarray
        1-D envelope values.
    main_mask : np.ndarray
        1-D boolean mask of the main-peak support region.
    min_distance : int
        Minimum distance between secondary peaks.
    min_prominence : float
        Minimum prominence for a secondary peak.

    Returns
    -------
    np.ndarray
        1-D array of peak heights for secondary peaks found outside
        *main_mask*.  Empty array if none found.
    """
    kwargs: dict = {"distance": max(1, min_distance)}
    if min_prominence > 0:
        kwargs["prominence"] = min_prominence

    # Find local maxima of the original envelope, then keep only those
    # outside the main support.  This avoids false peaks created by
    # zeroing out the main support region.
    peaks, _ = find_peaks(envelope, **kwargs)
    if len(peaks) == 0:
        return np.empty(0)

    outside_peaks = peaks[~main_mask[peaks]]
    if len(outside_peaks) == 0:
        return np.empty(0)
    return envelope[outside_peaks]
