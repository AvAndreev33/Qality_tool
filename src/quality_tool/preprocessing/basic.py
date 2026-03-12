"""Basic preprocessing functions for 1-D WLI signals.

Each function takes a 1-D numpy array and returns a **new** 1-D array of
the same length.  Input arrays are never modified in-place.
"""

from __future__ import annotations

import numpy as np


def _validate_signal(signal: np.ndarray) -> None:
    """Validate that *signal* is a non-empty 1-D array."""
    if signal.ndim != 1:
        raise ValueError(
            f"signal must be 1-D, got shape {signal.shape}"
        )
    if signal.size == 0:
        raise ValueError("signal must not be empty")


def subtract_baseline(signal: np.ndarray) -> np.ndarray:
    """Subtract the mean value from *signal*.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.

    Returns
    -------
    np.ndarray
        Signal with zero mean.
    """
    _validate_signal(signal)
    return signal - np.mean(signal)


def normalize_amplitude(signal: np.ndarray) -> np.ndarray:
    """Scale *signal* to the [0, 1] range.

    If the signal is flat (max == min), returns an array of zeros.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.

    Returns
    -------
    np.ndarray
        Amplitude-normalized signal.
    """
    _validate_signal(signal)
    lo = np.min(signal)
    hi = np.max(signal)
    span = hi - lo
    if span == 0.0:
        return np.zeros_like(signal, dtype=float)
    return (signal - lo) / span


def smooth(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply a uniform moving-average smoothing filter.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.
    window_size : int
        Width of the averaging window.  Must be a positive odd integer.

    Returns
    -------
    np.ndarray
        Smoothed signal of the same length as the input.
    """
    _validate_signal(signal)
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")
    if window_size > len(signal):
        raise ValueError(
            f"window_size ({window_size}) must not exceed "
            f"signal length ({len(signal)})"
        )
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(signal, kernel, mode="same")
