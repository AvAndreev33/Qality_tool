"""Correlation / reference-model quality metrics for Quality_tool.

Importing this module registers all correlation metrics with the
default registry.
"""

from quality_tool.metrics.correlation.best_phase_reference_correlation import (
    BestPhaseReferenceCorrelation,
)
from quality_tool.metrics.correlation.centered_reference_correlation import (
    CenteredReferenceCorrelation,
)
from quality_tool.metrics.correlation.phase_relaxation_gain import PhaseRelaxationGain
from quality_tool.metrics.correlation.reference_envelope_correlation import (
    ReferenceEnvelopeCorrelation,
)
from quality_tool.metrics.registry import default_registry

ALL_CORRELATION_METRICS = [
    CenteredReferenceCorrelation(),
    BestPhaseReferenceCorrelation(),
    ReferenceEnvelopeCorrelation(),
    PhaseRelaxationGain(),
]

for _m in ALL_CORRELATION_METRICS:
    default_registry.register(_m)
