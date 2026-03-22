"""Zero crossing stability metric for Quality_tool.

Measures the regularity of upward zero-crossing spacings filtered
to the expected fringe period::

    ZCS = MAD(d_i) / (median(d_i) + ε)

Lower scores indicate more stable zero-crossing intervals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quality_tool.core.models import MetricResult
from quality_tool.evaluation.recipe import (
    ROI_MEAN_SUBTRACTED_LINEAR_DETRENDED,
    RecipeBinding,
    SignalRecipe,
)
from quality_tool.metrics.regularity._regularity_helpers import (
    find_upward_zero_crossings,
)

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class ZeroCrossingStability:
    """Robust jitter of upward zero-crossing spacings.

    Score meaning: lower is better.
    """

    name: str = "zero_crossing_stability"
    category: str = "regularity"
    display_name: str = "Zero Crossing Stability"
    score_direction: str = "lower_better"
    score_scale: str = "positive_unbounded"
    signal_recipe: SignalRecipe = ROI_MEAN_SUBTRACTED_LINEAR_DETRENDED
    recipe_binding: RecipeBinding = "fixed"

    # ------------------------------------------------------------------

    def evaluate(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelope: np.ndarray | None = None,
        context: dict | None = None,
    ) -> MetricResult:
        if signal.ndim != 1 or signal.size < 4:
            return MetricResult(score=0.0, valid=False,
                                notes="signal too short")

        ctx = (context or {}).get("analysis_context")
        eps = getattr(ctx, "epsilon", 1e-12)
        t_exp = getattr(ctx, "expected_period_samples", 4)
        delta_t = getattr(ctx, "period_search_tolerance_fraction", 0.3)

        score, features, valid, notes = _compute_zcs(
            signal, t_exp, delta_t, eps,
        )
        return MetricResult(score=score, features=features,
                            valid=valid, notes=notes)

    def evaluate_batch(
        self,
        signals: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelopes: np.ndarray | None = None,
        context: dict | None = None,
    ) -> BatchMetricArrays:
        from quality_tool.metrics.batch_result import BatchMetricArrays

        ctx = (context or {}).get("analysis_context")
        eps = getattr(ctx, "epsilon", 1e-12)
        t_exp = getattr(ctx, "expected_period_samples", 4)
        delta_t = getattr(ctx, "period_search_tolerance_fraction", 0.3)

        n, m = signals.shape
        scores = np.full(n, np.nan)
        valid = np.zeros(n, dtype=bool)
        n_crossings = np.zeros(n)

        lo = (1 - delta_t) * t_exp
        hi = (1 + delta_t) * t_exp

        # Batch crossing detection — precompute all positions at once.
        neg = signals[:, :-1] < 0
        non_neg = signals[:, 1:] >= 0
        crossing_mask = neg & non_neg

        # Interpolated positions for ALL crossings in one shot.
        rows, cols = np.where(crossing_mask)
        if rows.size > 0:
            s0_vals = signals[rows, cols]
            s1_vals = signals[rows, cols + 1]
            positions_all = cols + (-s0_vals) / (s1_vals - s0_vals + eps)

            # Group by signal using counts.
            counts = np.bincount(rows, minlength=n)
            offsets = np.empty(n + 1, dtype=np.intp)
            offsets[0] = 0
            np.cumsum(counts, out=offsets[1:])

            for i in range(n):
                a, b = int(offsets[i]), int(offsets[i + 1])
                if b - a < 2:
                    continue

                distances = np.diff(positions_all[a:b])
                plausible = distances[(distances >= lo) & (distances <= hi)]
                if plausible.size < 2:
                    continue

                med = float(np.median(plausible))
                mad = float(np.median(np.abs(plausible - med)))
                scores[i] = mad / (med + eps)
                valid[i] = True
                n_crossings[i] = float(plausible.size)

        return BatchMetricArrays(
            scores=scores, valid=valid,
            features={"n_crossings": n_crossings},
        )


def _compute_zcs(
    signal: np.ndarray,
    t_exp: int,
    delta_t: float,
    eps: float,
) -> tuple[float, dict, bool, str]:
    """Core computation shared by scalar and batch paths."""
    crossings = find_upward_zero_crossings(signal, eps)

    if crossings.size < 2:
        return 0.0, {"n_crossings": float(crossings.size)}, False, "too few crossings"

    # Distances between consecutive upward crossings.
    distances = np.diff(crossings)

    # Filter to plausible distances near expected period.
    lo = (1 - delta_t) * t_exp
    hi = (1 + delta_t) * t_exp
    plausible = distances[(distances >= lo) & (distances <= hi)]

    if plausible.size < 2:
        return (
            0.0,
            {"n_crossings": float(crossings.size)},
            False,
            "too few plausible crossing intervals",
        )

    med = float(np.median(plausible))
    mad = float(np.median(np.abs(plausible - med)))
    zcs = mad / (med + eps)

    return float(zcs), {"n_crossings": float(plausible.size)}, True, ""
