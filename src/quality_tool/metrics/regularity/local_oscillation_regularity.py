"""Local oscillation regularity metric for Quality_tool.

Measures cycle-to-cycle waveform consistency by resampling each cycle
to a fixed length, normalising, and computing the median cosine
similarity of neighbouring cycles::

    LOR = median_i  dot(c_i, c_{i+1})

Higher scores indicate more consistent oscillation shape.
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
    find_local_maxima,
    resample_normalize_cycle,
)

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class LocalOscillationRegularity:
    """Cycle-shape consistency via resampled normalised waveforms.

    Score meaning: higher is better.
    """

    name: str = "local_oscillation_regularity"
    category: str = "regularity"
    display_name: str = "Local Oscillation Regularity"
    score_direction: str = "higher_better"
    score_scale: str = "bounded_01"
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
        l_cycle = getattr(ctx, "cycle_resample_length", 64)

        score, features, valid, notes = _compute_lor(
            signal, t_exp, d_frac, l_cycle, eps,
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
        l_cycle = getattr(ctx, "cycle_resample_length", 64)

        n, m = signals.shape
        scores = np.full(n, np.nan)
        valid = np.zeros(n, dtype=bool)
        n_cycles_arr = np.zeros(n)

        d_min = max(1, int(d_frac * t_exp))

        # Batch candidate detection — precompute and group by signal.
        cand_mask = (
            (signals[:, 1:-1] > signals[:, :-2])
            & (signals[:, 1:-1] > signals[:, 2:])
        )
        cand_rows, cand_cols = np.where(cand_mask)
        cand_cols = cand_cols + 1
        cand_counts = np.bincount(cand_rows, minlength=n)
        cand_offsets = np.empty(n + 1, dtype=np.intp)
        cand_offsets[0] = 0
        np.cumsum(cand_counts, out=cand_offsets[1:])

        # Pre-compute resampling grid (shared by all cycles).
        x_new = np.linspace(0, 1, l_cycle)

        # Cache x_old grids for common cycle lengths to avoid
        # repeated np.linspace calls (most cycles ≈ t_exp samples).
        _x_old_cache: dict[int, np.ndarray] = {}

        def _get_x_old(size: int) -> np.ndarray:
            grid = _x_old_cache.get(size)
            if grid is None:
                grid = np.linspace(0, 1, size)
                _x_old_cache[size] = grid
            return grid

        for i in range(n):
            a, b = int(cand_offsets[i]), int(cand_offsets[i + 1])
            if a == b:
                continue

            sig = signals[i]
            candidates = cand_cols[a:b]

            # Greedy peak selection — pure-Python list to avoid
            # numpy overhead on tiny (d_min*2+1)-element checks.
            order = np.argsort(-sig[candidates])
            sel = [False] * m
            keep: list[int] = []
            for idx_np in candidates[order]:
                idx = int(idx_np)
                lo_idx = max(0, idx - d_min)
                hi_idx = min(m, idx + d_min + 1)
                if not any(sel[lo_idx:hi_idx]):
                    sel[idx] = True
                    keep.append(idx)

            if len(keep) < 4:
                continue

            keep.sort()

            # --- resample and normalise each cycle (inlined) ---
            cycle_list: list[np.ndarray] = []
            for j in range(len(keep) - 1):
                seg = sig[keep[j]:keep[j + 1]]
                seg_n = seg.size
                if seg_n < 2:
                    continue
                resampled = np.interp(x_new, _get_x_old(seg_n), seg)
                resampled -= resampled.mean()
                norm = float(np.dot(resampled, resampled))
                if norm < eps:
                    continue
                resampled *= 1.0 / np.sqrt(norm)
                cycle_list.append(resampled)

            nc = len(cycle_list)
            if nc < 3:
                n_cycles_arr[i] = float(nc)
                continue

            # Cosine similarity of neighbouring cycles via matrix dot.
            cycle_arr = np.array(cycle_list)  # (K, l_cycle)
            sims = np.sum(cycle_arr[:-1] * cycle_arr[1:], axis=1)
            scores[i] = float(np.median(sims))
            valid[i] = True
            n_cycles_arr[i] = float(nc)

        return BatchMetricArrays(
            scores=scores, valid=valid,
            features={"n_cycles": n_cycles_arr},
        )


def _compute_lor(
    signal: np.ndarray,
    t_exp: int,
    d_frac: float,
    l_cycle: int,
    eps: float,
) -> tuple[float, dict, bool, str]:
    """Core computation shared by scalar and batch paths."""
    d_min = max(1, int(d_frac * t_exp))
    peaks = find_local_maxima(signal, d_min)

    # Need at least 4 peaks for 3 cycles (K-1 cycles, K-2 similarities).
    if peaks.size < 4:
        return 0.0, {"n_cycles": 0.0}, False, "too few peaks for cycle comparison"

    # Resample and normalise each cycle.
    cycles: list[np.ndarray] = []
    for j in range(len(peaks) - 1):
        c = resample_normalize_cycle(signal, peaks[j], peaks[j + 1], l_cycle, eps)
        if c is not None:
            cycles.append(c)

    if len(cycles) < 3:
        return 0.0, {"n_cycles": float(len(cycles))}, False, "too few valid cycles"

    # Cosine similarity of neighbouring cycles.
    similarities = np.array([
        float(np.dot(cycles[j], cycles[j + 1]))
        for j in range(len(cycles) - 1)
    ])

    score = float(np.median(similarities))
    return score, {"n_cycles": float(len(cycles))}, True, ""
