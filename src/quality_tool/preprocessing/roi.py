"""ROI extraction for 1-D WLI signals.

Extracts a segment of a given length from a signal, centered around
a reference point determined by the chosen centering mode.
"""

from __future__ import annotations

import numpy as np


_SUPPORTED_MODES = ("raw_max",)


def extract_roi(
    signal: np.ndarray,
    segment_size: int,
    mode: str = "raw_max",
) -> np.ndarray:
    """Extract a region-of-interest segment from *signal*.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.
    segment_size : int
        Desired length of the output segment.
    mode : str
        Centering strategy.  Currently only ``"raw_max"`` is supported.

    Returns
    -------
    np.ndarray
        A **copy** of the extracted segment with length *segment_size*.

    Raises
    ------
    ValueError
        If *signal* is not 1-D, is empty, *segment_size* is invalid,
        or *mode* is unsupported.
    """
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
    if signal.size == 0:
        raise ValueError("signal must not be empty")
    if segment_size < 1:
        raise ValueError(f"segment_size must be >= 1, got {segment_size}")
    if segment_size > len(signal):
        raise ValueError(
            f"segment_size ({segment_size}) exceeds "
            f"signal length ({len(signal)})"
        )
    if mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"unsupported centering mode {mode!r}, "
            f"choose from {_SUPPORTED_MODES}"
        )

    center = _find_center(signal, mode)
    start, end = _clamp_window(center, segment_size, len(signal))
    return signal[start:end].copy()


def _find_center(signal: np.ndarray, mode: str) -> int:
    """Return the center index for the given centering *mode*."""
    if mode == "raw_max":
        return int(np.argmax(signal))
    # Guard against future modes added to _SUPPORTED_MODES without
    # a corresponding implementation here.
    raise NotImplementedError(f"centering mode {mode!r} not implemented")


def _clamp_window(center: int, size: int, length: int) -> tuple[int, int]:
    """Compute a [start, end) window of *size* around *center*, clamped to [0, length)."""
    half = size // 2
    start = center - half
    end = start + size

    if start < 0:
        start = 0
        end = size
    elif end > length:
        end = length
        start = end - size

    return start, end
