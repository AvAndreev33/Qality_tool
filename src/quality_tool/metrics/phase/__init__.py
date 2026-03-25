"""Phase quality metrics for Quality_tool.

Importing this module registers all phase metrics with the
default registry.
"""

from quality_tool.metrics.phase.phase_curvature_index import PhaseCurvatureIndex
from quality_tool.metrics.phase.phase_jump_fraction import PhaseJumpFraction
from quality_tool.metrics.phase.phase_linear_fit_residual import PhaseLinearFitResidual
from quality_tool.metrics.phase.phase_monotonicity_score import PhaseMonotonicityScore
from quality_tool.metrics.phase.phase_slope_stability import PhaseSlopeStability
from quality_tool.metrics.registry import default_registry

ALL_PHASE_METRICS = [
    PhaseSlopeStability(),
    PhaseLinearFitResidual(),
    PhaseCurvatureIndex(),
    PhaseMonotonicityScore(),
    PhaseJumpFraction(),
]

for _m in ALL_PHASE_METRICS:
    default_registry.register(_m)
