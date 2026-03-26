"""Optional CUDA acceleration backend for Quality_tool.

Provides GPU-accelerated metric evaluation via CuPy.  Falls back
gracefully when CuPy is not installed or no CUDA GPU is available.

Public API
----------
- ``is_available()`` — check if GPU compute is possible
- ``get_device_info()`` — return a dict with GPU name, memory, etc.
- ``evaluate_metric_maps_gpu()`` — GPU-accelerated metric evaluation
- ``GPU_METRIC_NAMES`` — set of metric names supported on GPU
"""

from __future__ import annotations

from quality_tool.cuda._backend import (
    GPU_METRIC_NAMES,
    get_device_info,
    is_available,
)
from quality_tool.cuda._evaluator import evaluate_metric_maps_gpu

__all__ = [
    "GPU_METRIC_NAMES",
    "evaluate_metric_maps_gpu",
    "get_device_info",
    "is_available",
]
