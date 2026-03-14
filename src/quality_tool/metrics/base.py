"""Base metric interface for Quality_tool.

Defines the protocol that every quality metric must satisfy.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from quality_tool.core.models import MetricResult


class BaseMetric(Protocol):
    """Protocol for quality metric implementations.

    Every metric must expose a ``name`` attribute, an ``input_policy``
    attribute, and an ``evaluate`` method that returns a
    :class:`MetricResult`.
    """

    name: str

    input_policy: str
    """Which signal representation this metric must receive.

    Supported values:

    * ``"raw"`` — the metric must be evaluated on the original raw
      signal, ignoring any preprocessing or ROI extraction settings.
    * ``"processed"`` — the metric uses the signal after the currently
      configured preprocessing chain and optional ROI extraction.

    The evaluator reads this attribute to select the correct effective
    signal for each metric during multi-metric computation.
    """

    def evaluate(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelope: np.ndarray | None = None,
        context: dict | None = None,
    ) -> MetricResult:
        """Evaluate the metric on a single 1-D signal.

        Parameters
        ----------
        signal : np.ndarray
            1-D input signal.
        z_axis : np.ndarray | None
            Optional physical z-axis of the same length as *signal*.
        envelope : np.ndarray | None
            Optional precomputed envelope of the same length as *signal*.
        context : dict | None
            Optional additional context (e.g. precomputed spectral data).

        Returns
        -------
        MetricResult
            Scalar score, diagnostic features, and validity flag.
        """
        ...
