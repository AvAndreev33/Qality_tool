"""Spectral quality metrics for Quality_tool.

Importing this module registers all spectral metrics with the
default registry.
"""

from quality_tool.metrics.spectral.presence_of_expected_carrier_frequency import (
    PresenceOfExpectedCarrierFrequency,
)
from quality_tool.metrics.spectral.dominant_spectral_peak_prominence import (
    DominantSpectralPeakProminence,
)
from quality_tool.metrics.spectral.carrier_to_background_spectral_ratio import (
    CarrierToBackgroundSpectralRatio,
)
from quality_tool.metrics.spectral.energy_concentration_in_working_band import (
    EnergyConcentrationInWorkingBand,
)
from quality_tool.metrics.spectral.spectral_centroid_offset import (
    SpectralCentroidOffset,
)
from quality_tool.metrics.spectral.spectral_spread import SpectralSpread
from quality_tool.metrics.spectral.spectral_entropy import SpectralEntropy
from quality_tool.metrics.spectral.spectral_kurtosis import SpectralKurtosis
from quality_tool.metrics.spectral.spectral_peak_sharpness import (
    SpectralPeakSharpness,
)
from quality_tool.metrics.spectral.envelope_spectrum_consistency import (
    EnvelopeSpectrumConsistency,
)
from quality_tool.metrics.registry import default_registry

ALL_SPECTRAL_METRICS = [
    PresenceOfExpectedCarrierFrequency(),
    DominantSpectralPeakProminence(),
    CarrierToBackgroundSpectralRatio(),
    EnergyConcentrationInWorkingBand(),
    SpectralCentroidOffset(),
    SpectralSpread(),
    SpectralEntropy(),
    SpectralKurtosis(),
    SpectralPeakSharpness(),
    EnvelopeSpectrumConsistency(),
]

for _m in ALL_SPECTRAL_METRICS:
    default_registry.register(_m)
