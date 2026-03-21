"""Normalization layer for metric comparison.

Maps native metric scores to a common ``[0, 1]`` higher-is-better
representation.  This is a **read-only** layer — native scores are
never modified.

Normalization uses two pieces of score-semantics metadata declared on
each metric:

* ``score_direction`` — ``"higher_better"`` or ``"lower_better"``
* ``score_scale`` — ``"bounded_01"``, ``"positive_unbounded"``, or
  ``"db_like"``

Normalization policies
----------------------
* **bounded_01** — already in ``[0, 1]``.  If ``lower_better``, flip
  via ``1 - x``.
* **positive_unbounded** — clamp to ``[p5, p95]`` of valid values,
  rescale to ``[0, 1]``.  If ``lower_better``, flip after rescaling.
* **db_like** — may be negative.  Same percentile clamping as
  ``positive_unbounded``.  If ``lower_better``, flip after rescaling.
"""

from __future__ import annotations

import numpy as np

from quality_tool.metrics.base import ScoreDirection, ScoreScale


def normalize_score_map(
    score_map: np.ndarray,
    valid_map: np.ndarray,
    score_direction: ScoreDirection,
    score_scale: ScoreScale,
) -> np.ndarray:
    """Normalize a 2-D score map to ``[0, 1]`` higher-is-better.

    Parameters
    ----------
    score_map : np.ndarray
        2-D array of native metric scores.
    valid_map : np.ndarray
        2-D boolean array — ``True`` for valid pixels.
    score_direction : ScoreDirection
        ``"higher_better"`` or ``"lower_better"``.
    score_scale : ScoreScale
        ``"bounded_01"``, ``"positive_unbounded"``, or ``"db_like"``.

    Returns
    -------
    np.ndarray
        2-D array of normalized scores in ``[0, 1]``.  Invalid pixels
        remain ``np.nan``.
    """
    valid_scores = score_map[valid_map]
    if valid_scores.size == 0:
        return np.full_like(score_map, np.nan, dtype=float)

    ref_min, ref_max = _reference_range(valid_scores, score_scale)
    result = np.full_like(score_map, np.nan, dtype=float)
    result[valid_map] = _normalize_values(
        score_map[valid_map], ref_min, ref_max, score_direction, score_scale,
    )
    return result


def normalize_single(
    score: float,
    score_direction: ScoreDirection,
    score_scale: ScoreScale,
    ref_min: float,
    ref_max: float,
) -> float:
    """Normalize a single native score to ``[0, 1]`` higher-is-better.

    ``ref_min`` and ``ref_max`` are the reference range endpoints
    (typically derived from the full score map via
    :func:`reference_range_from_map`).
    """
    arr = _normalize_values(
        np.array([score]),
        ref_min,
        ref_max,
        score_direction,
        score_scale,
    )
    return float(arr[0])


def reference_range_from_map(
    score_map: np.ndarray,
    valid_map: np.ndarray,
    score_scale: ScoreScale,
) -> tuple[float, float]:
    """Compute the ``(ref_min, ref_max)`` reference range for a score map.

    This allows callers to compute the range once and reuse it for
    many calls to :func:`normalize_single`.
    """
    valid_scores = score_map[valid_map]
    if valid_scores.size == 0:
        return 0.0, 1.0
    return _reference_range(valid_scores, score_scale)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _reference_range(
    valid_scores: np.ndarray,
    score_scale: ScoreScale,
) -> tuple[float, float]:
    """Determine the min/max reference range for rescaling."""
    if score_scale == "bounded_01":
        return 0.0, 1.0

    # positive_unbounded / db_like — use robust percentiles.
    lo = float(np.nanpercentile(valid_scores, 5))
    hi = float(np.nanpercentile(valid_scores, 95))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _normalize_values(
    values: np.ndarray,
    ref_min: float,
    ref_max: float,
    score_direction: ScoreDirection,
    score_scale: ScoreScale,
) -> np.ndarray:
    """Core normalization: clamp, rescale, optionally flip."""
    if score_scale == "bounded_01":
        normed = np.clip(values, 0.0, 1.0).astype(float)
    else:
        clamped = np.clip(values, ref_min, ref_max)
        span = ref_max - ref_min
        normed = (clamped - ref_min) / span

    if score_direction == "lower_better":
        normed = 1.0 - normed

    return normed
