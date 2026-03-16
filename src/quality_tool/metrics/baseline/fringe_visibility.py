"""Fringe visibility metric for Quality_tool.

Computes Michelson fringe visibility::

    V = (I_max - I_min) / (I_max + I_min)

This is a simple, interpretable measure of fringe contrast.  Values
range from 0 (no fringes) to 1 (maximum contrast) for non-negative
signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quality_tool.core.models import MetricResult

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class FringeVisibility:
    """Michelson fringe-visibility metric.

    Formula::

        V = (I_max - I_min) / (I_max + I_min)

    Returns ``valid=False`` when the denominator is zero or the signal
    is too short for meaningful evaluation.
    """

    name: str = "fringe_visibility"

    # Michelson visibility relies on absolute intensity values (I_max,
    # I_min).  Preprocessing steps such as baseline subtraction or
    # normalisation destroy the physical meaning of these values, so
    # this metric must always be evaluated on the raw signal.
    input_policy: str = "raw"

    needs_spectral: bool = False

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

        if i_min < 0.0:
            return MetricResult(
                score=0.0,
                features={"i_max": i_max, "i_min": i_min},
                valid=False,
                notes="signal contains negative values; "
                      "Michelson visibility requires non-negative intensity",
            )

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

    def evaluate_batch(
        self,
        signals: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelopes: np.ndarray | None = None,
        context: dict | None = None,
    ) -> BatchMetricArrays:
        """Vectorised evaluation over a chunk of signals.

        Parameters
        ----------
        signals : np.ndarray
            2-D array of shape ``(N, M)``.

        Returns
        -------
        BatchMetricArrays
        """
        from quality_tool.metrics.batch_result import BatchMetricArrays

        n = signals.shape[0]
        i_max = np.max(signals, axis=1)
        i_min = np.min(signals, axis=1)
        denom = i_max + i_min

        scores = np.full(n, np.nan)
        valid = np.ones(n, dtype=bool)

        # Invalid: any signal with negative values
        neg_mask = i_min < 0.0
        valid[neg_mask] = False

        # Invalid: zero denominator
        zero_mask = denom == 0.0
        valid[zero_mask] = False

        ok = valid
        with np.errstate(divide="ignore", invalid="ignore"):
            scores[ok] = (i_max[ok] - i_min[ok]) / denom[ok]

        features = {"i_max": i_max, "i_min": i_min}
        return BatchMetricArrays(scores=scores, valid=valid, features=features)
