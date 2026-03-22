"""Shared helpers for regularity metrics.

Provides extrema detection, cycle resampling, and zero-crossing
interpolation used by multiple regularity metrics.  These are
lightweight helpers, not a general signal-analysis framework.
"""

from __future__ import annotations

import numpy as np


def find_local_maxima(signal: np.ndarray, min_distance: int) -> np.ndarray:
    """Detect local maxima with a minimum inter-peak distance.

    Parameters
    ----------
    signal : np.ndarray
        1-D signal array.
    min_distance : int
        Minimum number of samples between consecutive peaks.
        Must be >= 1.

    Returns
    -------
    np.ndarray
        Sorted 1-D integer array of peak indices.
    """
    min_distance = max(1, int(min_distance))
    n = signal.size
    if n < 3:
        return np.array([], dtype=int)

    # Find all local maxima (strictly greater than both neighbours).
    candidates = np.where(
        (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
    )[0] + 1

    if candidates.size == 0:
        return np.array([], dtype=int)

    # Greedy selection: keep strongest peaks first, enforce min_distance.
    order = np.argsort(-signal[candidates])
    selected = np.zeros(n, dtype=bool)
    keep = []
    for idx in candidates[order]:
        lo = max(0, idx - min_distance)
        hi = min(n, idx + min_distance + 1)
        if not np.any(selected[lo:hi]):
            selected[idx] = True
            keep.append(idx)

    return np.sort(np.array(keep, dtype=int))


def resample_normalize_cycle(
    signal: np.ndarray,
    start: int,
    end: int,
    length: int,
    epsilon: float = 1e-12,
) -> np.ndarray | None:
    """Resample a signal segment to fixed length, mean-subtract, L2-normalise.

    Parameters
    ----------
    signal : np.ndarray
        1-D source signal.
    start, end : int
        Inclusive start, exclusive end of the cycle segment.
    length : int
        Target resampled length.
    epsilon : float
        Guard against zero-norm division.

    Returns
    -------
    np.ndarray or None
        Normalised 1-D array of shape ``(length,)``, or ``None`` if
        the segment is degenerate (fewer than 2 samples or zero norm).
    """
    seg = signal[start:end]
    if seg.size < 2:
        return None

    # Resample to fixed length via linear interpolation.
    x_old = np.linspace(0, 1, seg.size)
    x_new = np.linspace(0, 1, length)
    resampled = np.interp(x_new, x_old, seg)

    # Mean-subtract and L2-normalise.
    resampled -= resampled.mean()
    norm = np.linalg.norm(resampled)
    if norm < epsilon:
        return None
    resampled /= norm
    return resampled


def find_upward_zero_crossings(
    signal: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Detect upward zero crossings with linear interpolation.

    An upward crossing exists between samples *n* and *n+1* when
    ``signal[n] < 0`` and ``signal[n+1] >= 0``.

    Parameters
    ----------
    signal : np.ndarray
        1-D signal array (should be mean-subtracted / detrended).
    epsilon : float
        Guard for denominator in interpolation.

    Returns
    -------
    np.ndarray
        1-D float array of interpolated crossing positions.
    """
    neg = signal[:-1] < 0
    non_neg = signal[1:] >= 0
    mask = neg & non_neg
    indices = np.where(mask)[0]

    if indices.size == 0:
        return np.array([], dtype=float)

    x0 = signal[indices]
    x1 = signal[indices + 1]
    denom = x1 - x0 + epsilon
    positions = indices + (-x0) / denom
    return positions
