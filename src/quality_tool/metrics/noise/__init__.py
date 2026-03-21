"""Noise quality metrics for Quality_tool.

Importing this module registers all noise metrics with the default
registry.
"""

from quality_tool.metrics.noise.envelope_peak_to_background_ratio import (
    EnvelopePeakToBackgroundRatio,
)
from quality_tool.metrics.noise.high_frequency_noise_level import (
    HighFrequencyNoiseLevel,
)
from quality_tool.metrics.noise.local_snr import LocalSNR
from quality_tool.metrics.noise.low_frequency_drift_level import (
    LowFrequencyDriftLevel,
)
from quality_tool.metrics.noise.noise_floor_level import NoiseFloorLevel
from quality_tool.metrics.noise.residual_noise_energy import ResidualNoiseEnergy
from quality_tool.metrics.noise.spectral_snr import SpectralSNR
from quality_tool.metrics.registry import default_registry

ALL_NOISE_METRICS = [
    SpectralSNR(),
    LocalSNR(),
    EnvelopePeakToBackgroundRatio(),
    NoiseFloorLevel(),
    ResidualNoiseEnergy(),
    HighFrequencyNoiseLevel(),
    LowFrequencyDriftLevel(),
]

for _m in ALL_NOISE_METRICS:
    default_registry.register(_m)
