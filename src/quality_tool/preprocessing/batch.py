"""Batch preprocessing functions for 2-D signal arrays.

Each function operates on a ``(N, M)`` array and returns a new array of
the same shape.  These are the vectorised counterparts of the per-signal
functions in :mod:`quality_tool.preprocessing.basic`.
"""

from __future__ import annotations

import numpy as np


def subtract_baseline_batch(signals: np.ndarray) -> np.ndarray:
    """Subtract the per-signal mean from each row of *signals*.

    Parameters
    ----------
    signals : np.ndarray
        2-D array of shape ``(N, M)``.

    Returns
    -------
    np.ndarray
        Baseline-subtracted signals, same shape.
    """
    return signals - np.mean(signals, axis=1, keepdims=True)


def normalize_amplitude_batch(signals: np.ndarray) -> np.ndarray:
    """Scale each signal to [0, 1] independently.

    Flat signals (max == min) become all zeros.
    """
    lo = np.min(signals, axis=1, keepdims=True)
    hi = np.max(signals, axis=1, keepdims=True)
    span = hi - lo
    # Avoid division by zero for flat signals
    safe_span = np.where(span == 0.0, 1.0, span)
    result = (signals - lo) / safe_span
    # Zero out flat signals
    result[span.squeeze(axis=1) == 0.0] = 0.0
    return result


def smooth_batch(signals: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply uniform moving-average smoothing to each signal.

    Uses 1-D convolution along axis=1 via a strided view approach.

    Parameters
    ----------
    signals : np.ndarray
        2-D array ``(N, M)``.
    window_size : int
        Must be a positive odd integer, <= M.

    Returns
    -------
    np.ndarray
        Smoothed signals, same shape.
    """
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(
            f"window_size must be a positive odd integer, got {window_size}"
        )
    n, m = signals.shape
    if window_size > m:
        raise ValueError(
            f"window_size ({window_size}) exceeds signal length ({m})"
        )
    # np.apply_along_axis would be slow; use a direct vectorised approach
    # via cumsum-based moving average for 'same' mode.
    kernel = np.ones(window_size, dtype=signals.dtype) / window_size
    # Use numpy convolve per signal — still per-row but through C,
    # much faster than Python-level per-pixel loop.
    out = np.empty_like(signals)
    for i in range(n):
        out[i] = np.convolve(signals[i], kernel, mode="same")
    return out


def extract_roi_batch(
    signals: np.ndarray,
    segment_size: int,
    mode: str = "raw_max",
) -> np.ndarray:
    """Extract ROI segments from a batch of signals.

    Parameters
    ----------
    signals : np.ndarray
        2-D array ``(N, M)``.
    segment_size : int
        Length of the extracted segment.
    mode : str
        Centering strategy.  Only ``"raw_max"`` is supported.

    Returns
    -------
    np.ndarray
        2-D array ``(N, segment_size)``.
    """
    if mode != "raw_max":
        raise ValueError(f"unsupported centering mode {mode!r}")

    n, m = signals.shape
    if segment_size < 1 or segment_size > m:
        raise ValueError(
            f"segment_size must be in [1, {m}], got {segment_size}"
        )

    # Vectorised center finding
    centers = np.argmax(signals, axis=1)  # (N,)
    half = segment_size // 2
    starts = centers - half

    # Clamp to valid range
    starts = np.clip(starts, 0, m - segment_size)

    # Gather ROI segments — use advanced indexing
    # Build index array: (N, segment_size)
    offsets = np.arange(segment_size)  # (segment_size,)
    indices = starts[:, np.newaxis] + offsets[np.newaxis, :]  # (N, seg)
    return np.take_along_axis(signals, indices, axis=1)
