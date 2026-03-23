"""Shared batch helpers for spectral metrics.

Provides reusable vectorised building blocks used by multiple
spectral metrics.  These operate on 2-D power arrays of shape
``(N, F)`` and shared :class:`SpectralPriors`.
"""

from __future__ import annotations

import numpy as np

from quality_tool.spectral.priors import SpectralPriors


def hann_windowed_power_batch(signals: np.ndarray) -> np.ndarray:
    """Hann-window signals and return one-sided power spectrum.

    Parameters
    ----------
    signals : np.ndarray
        2-D array of shape ``(N, M)``.

    Returns
    -------
    np.ndarray
        Power spectrum of shape ``(N, F)`` where ``F = M // 2 + 1``.
    """
    m = signals.shape[1]
    window = np.hanning(m).astype(signals.dtype)
    windowed = signals * window[np.newaxis, :]
    fft_coeffs = np.fft.rfft(windowed, axis=1)
    return np.abs(fft_coeffs) ** 2


def normalized_spectral_weights(
    power: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Normalize power to probability weights (row-wise sum = 1).

    Parameters
    ----------
    power : np.ndarray
        2-D power spectrum ``(N, F)``.
    eps : float
        Numeric stability guard.

    Returns
    -------
    np.ndarray
        Weights ``(N, F)`` with each row summing to ~1.
    """
    total = power.sum(axis=1, keepdims=True) + eps
    return power / total


def spectral_centroid_batch(
    p: np.ndarray,
    bin_indices: np.ndarray,
) -> np.ndarray:
    """Compute spectral centroid from normalised weights.

    Parameters
    ----------
    p : np.ndarray
        Normalised spectral weights ``(N, F)``.
    bin_indices : np.ndarray
        1-D array ``(F,)`` of frequency bin indices.

    Returns
    -------
    np.ndarray
        Centroid values ``(N,)``.
    """
    return (p * bin_indices[np.newaxis, :]).sum(axis=1)


def spectral_variance_batch(
    p: np.ndarray,
    bin_indices: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Compute spectral variance around *center*.

    Parameters
    ----------
    p : np.ndarray
        Normalised spectral weights ``(N, F)``.
    bin_indices : np.ndarray
        1-D array ``(F,)`` of frequency bin indices.
    center : np.ndarray
        1-D array ``(N,)`` of centre values.

    Returns
    -------
    np.ndarray
        Variance values ``(N,)``.
    """
    diff = bin_indices[np.newaxis, :] - center[:, np.newaxis]
    return (p * diff ** 2).sum(axis=1)
