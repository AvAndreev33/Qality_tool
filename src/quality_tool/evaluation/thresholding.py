"""Thresholding support for Quality_tool.

Applies a simple keep/reject rule to a :class:`MetricMapResult` and
produces a :class:`ThresholdResult`.

Invalid-pixel handling
----------------------
Pixels that are marked invalid in ``valid_map`` are **always rejected**
(``mask == False``), regardless of the thresholding rule.  Threshold
comparisons are applied only to valid pixels.
"""

from __future__ import annotations

import numpy as np

from quality_tool.core.models import MetricMapResult, ThresholdResult

_KEEP_RULES = {"above", "below"}


def apply_threshold(
    metric_map_result: MetricMapResult,
    threshold: float,
    keep_rule: str = "above",
) -> ThresholdResult:
    """Apply a scalar threshold to a metric map.

    Parameters
    ----------
    metric_map_result : MetricMapResult
        Aggregated metric output with ``score_map`` and ``valid_map``.
    threshold : float
        Scalar threshold value.
    keep_rule : str
        ``"above"`` keeps pixels where ``score >= threshold``.
        ``"below"`` keeps pixels where ``score <= threshold``.

    Returns
    -------
    ThresholdResult
        Binary mask and summary statistics.

    Raises
    ------
    ValueError
        If *keep_rule* is not ``"above"`` or ``"below"``.
    """
    if keep_rule not in _KEEP_RULES:
        raise ValueError(
            f"unsupported keep_rule {keep_rule!r}, "
            f"choose from {sorted(_KEEP_RULES)}"
        )

    score_map = metric_map_result.score_map
    valid_map = metric_map_result.valid_map

    # Start with all pixels rejected.
    mask = np.zeros_like(valid_map, dtype=bool)

    if keep_rule == "above":
        mask[valid_map] = score_map[valid_map] >= threshold
        rule_label = f"score >= {threshold}"
    else:  # "below"
        mask[valid_map] = score_map[valid_map] <= threshold
        rule_label = f"score <= {threshold}"

    # Summary statistics.
    total_pixels = int(valid_map.size)
    valid_pixels = int(np.sum(valid_map))
    kept_pixels = int(np.sum(mask))
    rejected_pixels = total_pixels - kept_pixels

    kept_fraction = kept_pixels / valid_pixels if valid_pixels > 0 else 0.0

    stats: dict = {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "kept_pixels": kept_pixels,
        "rejected_pixels": rejected_pixels,
        "kept_fraction": float(kept_fraction),
    }

    return ThresholdResult(
        threshold=threshold,
        keep_rule=rule_label,
        mask=mask,
        stats=stats,
    )
