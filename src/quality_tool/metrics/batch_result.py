"""Lightweight batch result container for vectorised metric evaluation.

Replaces per-pixel ``MetricResult`` objects with flat NumPy arrays to
eliminate the massive Python object overhead discovered during profiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BatchMetricArrays:
    """Arrays returned by ``evaluate_batch`` for a chunk of N signals.

    Attributes
    ----------
    scores : np.ndarray
        1-D float array of length N.  Invalid entries are ``np.nan``.
    valid : np.ndarray
        1-D bool array of length N.
    features : dict[str, np.ndarray]
        Each value is a 1-D float array of length N.
    """

    scores: np.ndarray
    valid: np.ndarray
    features: dict[str, np.ndarray] = field(default_factory=dict)
