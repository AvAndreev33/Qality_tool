"""Envelope quality metrics for Quality_tool.

Importing this module registers all envelope metrics with the
default registry.
"""

from quality_tool.metrics.envelope.envelope_area import EnvelopeArea
from quality_tool.metrics.envelope.envelope_height import EnvelopeHeight
from quality_tool.metrics.envelope.envelope_sharpness import EnvelopeSharpness
from quality_tool.metrics.envelope.envelope_symmetry import EnvelopeSymmetry
from quality_tool.metrics.envelope.envelope_width import EnvelopeWidth
from quality_tool.metrics.envelope.main_peak_to_sidelobe_ratio import (
    MainPeakToSidelobeRatio,
)
from quality_tool.metrics.envelope.single_peakness import SinglePeakness
from quality_tool.metrics.registry import default_registry

ALL_ENVELOPE_METRICS = [
    EnvelopeHeight(),
    EnvelopeArea(),
    EnvelopeWidth(),
    EnvelopeSharpness(),
    EnvelopeSymmetry(),
    SinglePeakness(),
    MainPeakToSidelobeRatio(),
]

for _m in ALL_ENVELOPE_METRICS:
    default_registry.register(_m)
