"""Hilbert-based analytic envelope method.

Computes the envelope of a 1-D signal using the analytic signal
(via :func:`scipy.signal.hilbert`).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


class AnalyticEnvelopeMethod:
    """Envelope via the analytic signal (Hilbert transform).

    The envelope is computed as ``|hilbert(signal)|``, i.e. the modulus
    of the analytic signal.
    """

    name: str = "analytic"

    def compute(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        context: dict | None = None,
    ) -> np.ndarray:
        """Compute the Hilbert-based envelope of *signal*.

        Parameters
        ----------
        signal : np.ndarray
            1-D real-valued input signal.
        z_axis : np.ndarray | None
            Unused; accepted for interface compatibility.
        context : dict | None
            Unused; accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            1-D envelope array of the same length as *signal*.

        Raises
        ------
        ValueError
            If *signal* is not 1-D or is empty.
        """
        if signal.ndim != 1:
            raise ValueError(
                f"signal must be 1-D, got shape {signal.shape}"
            )
        if signal.size == 0:
            raise ValueError("signal must not be empty")

        analytic_signal = hilbert(signal)
        return np.abs(analytic_signal)

    def compute_batch(
        self,
        signals: np.ndarray,
        z_axis: np.ndarray | None = None,
        context: dict | None = None,
    ) -> np.ndarray:
        """Compute the Hilbert envelope for a batch of signals.

        Parameters
        ----------
        signals : np.ndarray
            2-D array of shape ``(N, M)``.

        Returns
        -------
        np.ndarray
            2-D envelope array of shape ``(N, M)``.
        """
        if signals.ndim != 2:
            raise ValueError(
                f"signals must be 2-D, got shape {signals.shape}"
            )
        if signals.shape[1] == 0:
            raise ValueError("signals must not be empty along axis 1")

        # scipy.signal.hilbert supports batch along axis=-1 by default
        analytic_signals = hilbert(signals, axis=1)
        return np.abs(analytic_signals)
