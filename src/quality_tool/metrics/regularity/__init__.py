"""Regularity quality metrics for Quality_tool.

Importing this module registers all regularity metrics with the
default registry.
"""

from quality_tool.metrics.regularity.autocorrelation_peak_strength import (
    AutocorrelationPeakStrength,
)
from quality_tool.metrics.regularity.jitter_of_extrema import JitterOfExtrema
from quality_tool.metrics.regularity.local_oscillation_regularity import (
    LocalOscillationRegularity,
)
from quality_tool.metrics.regularity.zero_crossing_stability import (
    ZeroCrossingStability,
)
from quality_tool.metrics.registry import default_registry

ALL_REGULARITY_METRICS = [
    AutocorrelationPeakStrength(),
    LocalOscillationRegularity(),
    JitterOfExtrema(),
    ZeroCrossingStability(),
]

for _m in ALL_REGULARITY_METRICS:
    default_registry.register(_m)
