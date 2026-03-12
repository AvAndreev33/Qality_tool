"""Fringe visibility metric for Quality_tool.

Computes Michelson fringe visibility::

    V = (I_max - I_min) / (I_max + I_min)

This is a simple, interpretable measure of fringe contrast.  Values
range from 0 (no fringes) to 1 (maximum contrast) for non-negative
signals.
"""

from __future__ import annotations

import numpy as np

from quality_tool.core.models import MetricResult


class FringeVisibility:
    """Michelson fringe-visibility metric.

    Formula::

        V = (I_max - I_min) / (I_max + I_min)

    Returns ``valid=False`` when the denominator is zero or the signal
    is too short for meaningful evaluation.
    """

    name: str = "fringe_visibility"

    def evaluate(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelope: np.ndarray | None = None,
        context: dict | None = None,
    ) -> MetricResult:
        if signal.ndim != 1 or signal.size < 2:
            return MetricResult(
                score=0.0,
                features={},
                valid=False,
                notes="signal must be 1-D with at least 2 samples",
            )

        i_max = float(np.max(signal))
        i_min = float(np.min(signal))
        denominator = i_max + i_min

        if denominator == 0.0:
            return MetricResult(
                score=0.0,
                features={"i_max": i_max, "i_min": i_min},
                valid=False,
                notes="I_max + I_min is zero",
            )

        visibility = (i_max - i_min) / denominator

        return MetricResult(
            score=float(visibility),
            features={"i_max": i_max, "i_min": i_min},
        )
