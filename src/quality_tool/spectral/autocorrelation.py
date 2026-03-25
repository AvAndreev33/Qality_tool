"""Normalized autocorrelation computation for signal inspection.

Provides a standalone function used by the signal inspector to display
the autocorrelation of a pixel signal with expected-period guidance.
"""

from __future__ import annotations

import numpy as np


def compute_normalized_autocorrelation(
    signal: np.ndarray,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the normalized autocorrelation of a 1-D signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D real-valued signal of length M.
    max_lag : int or None
        Maximum lag to compute.  Defaults to ``M // 2`` which is the
        range where the autocorrelation is statistically meaningful.

    Returns
    -------
    lags : np.ndarray
        1-D array of lag values ``[0, 1, ..., max_lag]``.
    autocorr : np.ndarray
        1-D array of normalized autocorrelation values.
        ``autocorr[0] == 1.0`` by construction.
    """
    m = len(signal)
    if max_lag is None:
        max_lag = m // 2
    max_lag = min(max_lag, m - 1)

    lags = np.arange(max_lag + 1, dtype=float)

    r0 = float(np.dot(signal, signal))
    if r0 == 0.0:
        return lags, np.zeros(max_lag + 1)

    autocorr = np.empty(max_lag + 1)
    for tau in range(max_lag + 1):
        r_tau = float(np.dot(signal[: m - tau], signal[tau:]))
        autocorr[tau] = r_tau / r0

    return lags, autocorr


def find_autocorrelation_peak(
    autocorr: np.ndarray,
    tau_min: int,
    tau_max: int,
) -> int | None:
    """Find the dominant autocorrelation peak within a lag window.

    Parameters
    ----------
    autocorr : np.ndarray
        1-D normalized autocorrelation array.
    tau_min, tau_max : int
        Inclusive lag search bounds.

    Returns
    -------
    int or None
        Lag of the peak, or ``None`` if the window is invalid.
    """
    tau_min = max(1, tau_min)
    tau_max = min(tau_max, len(autocorr) - 1)
    if tau_min > tau_max:
        return None

    window = autocorr[tau_min: tau_max + 1]
    best = int(np.argmax(window)) + tau_min
    return best
