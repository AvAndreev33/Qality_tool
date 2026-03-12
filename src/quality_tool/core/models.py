"""Core data models for Quality_tool.

Defines the canonical internal representations used throughout the project:
- SignalSet: loaded dataset with signals in (H, W, M) format
- MetricResult: result of evaluating one signal
- MetricMapResult: aggregated metric output for a full image
- ThresholdResult: binary mask from thresholding a metric map
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SignalSet:
    """Canonical representation of a loaded WLI dataset.

    Attributes:
        signals: 3-D array of shape (H, W, M) — one correlogram per pixel.
        width: image width (W).
        height: image height (H).
        z_axis: 1-D array of length M — physical or index-based signal axis.
        metadata: normalized acquisition metadata; ``None`` if unavailable.
        source_type: label for the input source (e.g. ``"image_stack"``, ``"txt_matrix"``).
        source_path: path to the primary data file.
        info_path: path to the sidecar acquisition-info file, if any.
        z_axis_path: path to the z-axis file, if any.
    """

    signals: np.ndarray
    width: int
    height: int
    z_axis: np.ndarray
    metadata: dict | None = None
    source_type: str = "unknown"
    source_path: str | None = None
    info_path: str | None = None
    z_axis_path: str | None = None

    def __post_init__(self) -> None:
        if self.signals.ndim != 3:
            raise ValueError(
                f"signals must be 3-D (H, W, M), got shape {self.signals.shape}"
            )
        h, w, m = self.signals.shape
        if h != self.height:
            raise ValueError(
                f"signals.shape[0] ({h}) != height ({self.height})"
            )
        if w != self.width:
            raise ValueError(
                f"signals.shape[1] ({w}) != width ({self.width})"
            )
        if self.z_axis.ndim != 1:
            raise ValueError(
                f"z_axis must be 1-D, got shape {self.z_axis.shape}"
            )
        if len(self.z_axis) != m:
            raise ValueError(
                f"z_axis length ({len(self.z_axis)}) != signal length ({m})"
            )


@dataclass
class MetricResult:
    """Result of evaluating a quality metric on a single signal.

    Attributes:
        score: scalar metric value.
        features: optional diagnostic outputs.
        valid: whether the evaluation was successful.
        notes: optional explanation for invalid or special cases.
    """

    score: float
    features: dict = field(default_factory=dict)
    valid: bool = True
    notes: str = ""


@dataclass
class MetricMapResult:
    """Aggregated metric output for an entire image.

    Attributes:
        metric_name: name of the evaluated metric.
        score_map: 2-D array of shape (H, W) with per-pixel scores.
        valid_map: 2-D boolean array of shape (H, W) with validity flags.
        feature_maps: optional extra 2-D feature maps.
        metadata: evaluation metadata (settings, timings, etc.).
    """

    metric_name: str
    score_map: np.ndarray
    valid_map: np.ndarray
    feature_maps: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class ThresholdResult:
    """Result of applying a threshold to a metric map.

    Attributes:
        threshold: threshold value used.
        keep_rule: human-readable threshold rule (e.g. ``"score >= threshold"``).
        mask: 2-D boolean array of shape (H, W).
        stats: optional summary statistics.
    """

    threshold: float
    keep_rule: str
    mask: np.ndarray
    stats: dict | None = None
