"""Jitter of extrema metric for Quality_tool.

Measures cycle spacing stability using the robust coefficient of
variation of inter-peak distances::

    J_ext = MAD(d_i) / (median(d_i) + ε)

Lower scores indicate more regular extrema spacing.
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
from quality_tool.metrics.regularity._regularity_helpers import find_local_maxima

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class JitterOfExtrema:
    """Robust jitter of inter-peak distances.

    Score meaning: lower is better.
    """

    name: str = "jitter_of_extrema"
    category: str = "regularity"
    display_name: str = "Jitter of Extrema"
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
        d_frac = getattr(ctx, "peak_min_distance_fraction", 0.6)

        score, features, valid, notes = _compute_jitter(
            signal, t_exp, d_frac, eps,
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
        d_frac = getattr(ctx, "peak_min_distance_fraction", 0.6)

        n, m = signals.shape
        scores = np.full(n, np.nan)
        valid = np.zeros(n, dtype=bool)
        n_peaks = np.zeros(n)

        d_min = max(1, int(d_frac * t_exp))

        # Batch candidate detection — precompute (N, M-2) mask and
        # group candidates by signal via offsets.
        cand_mask = (
            (signals[:, 1:-1] > signals[:, :-2])
            & (signals[:, 1:-1] > signals[:, 2:])
        )
        cand_rows, cand_cols = np.where(cand_mask)
        cand_cols = cand_cols + 1  # adjust for 1:-1 offset
        cand_counts = np.bincount(cand_rows, minlength=n)
        cand_offsets = np.empty(n + 1, dtype=np.intp)
        cand_offsets[0] = 0
        np.cumsum(cand_counts, out=cand_offsets[1:])

        for i in range(n):
            a, b = int(cand_offsets[i]), int(cand_offsets[i + 1])
            if a == b:
                continue

            candidates = cand_cols[a:b]
            sig_i = signals[i]

            # Greedy peak selection — pure-Python list to avoid
            # numpy overhead on tiny (d_min*2+1)-element checks.
            order = np.argsort(-sig_i[candidates])
            sel = [False] * m
            keep: list[int] = []
            for idx_np in candidates[order]:
                idx = int(idx_np)
                lo_idx = max(0, idx - d_min)
                hi_idx = min(m, idx + d_min + 1)
                if not any(sel[lo_idx:hi_idx]):
                    sel[idx] = True
                    keep.append(idx)

            if len(keep) < 3:
                continue

            peaks = np.sort(np.array(keep, dtype=int))
            distances = np.diff(peaks).astype(float)
            med = float(np.median(distances))
            mad = float(np.median(np.abs(distances - med)))
            scores[i] = mad / (med + eps)
            valid[i] = True
            n_peaks[i] = float(len(keep))

        return BatchMetricArrays(
            scores=scores, valid=valid,
            features={"n_peaks": n_peaks},
        )


def _compute_jitter(
    signal: np.ndarray,
    t_exp: int,
    d_frac: float,
    eps: float,
) -> tuple[float, dict, bool, str]:
    """Core computation shared by scalar and batch paths."""
    d_min = max(1, int(d_frac * t_exp))
    peaks = find_local_maxima(signal, d_min)

    if peaks.size < 3:
        return 0.0, {"n_peaks": float(peaks.size)}, False, "too few peaks"

    distances = np.diff(peaks).astype(float)
    med = float(np.median(distances))
    mad = float(np.median(np.abs(distances - med)))
    jitter = mad / (med + eps)

    return float(jitter), {"n_peaks": float(peaks.size)}, True, ""
