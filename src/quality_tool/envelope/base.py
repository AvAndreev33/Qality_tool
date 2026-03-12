"""Base envelope interface for Quality_tool.

Defines the protocol that every envelope method must satisfy.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class BaseEnvelopeMethod(Protocol):
    """Protocol for envelope computation methods.

    Every envelope method must expose a ``name`` attribute and a ``compute``
    method that returns an envelope array of the same length as the input
    signal.
    """

    name: str

    def compute(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        context: dict | None = None,
    ) -> np.ndarray:
        """Compute the envelope of *signal*.

        Parameters
        ----------
        signal : np.ndarray
            1-D input signal.
        z_axis : np.ndarray | None
            Optional physical z-axis of the same length as *signal*.
        context : dict | None
            Optional additional context (e.g. metadata).

        Returns
        -------
        np.ndarray
            1-D envelope array with the same length as *signal*.
        """
        ...
