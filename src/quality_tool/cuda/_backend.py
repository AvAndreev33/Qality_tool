"""CUDA availability check and device info for Quality_tool.

Thin wrapper around CuPy that provides a clean interface for the rest
of the application.  All CuPy imports are lazy — this module is safe
to import even without CuPy installed.
"""

from __future__ import annotations

# Set of metric names that the GPU backend supports.
GPU_METRIC_NAMES: frozenset[str] = frozenset({
    # baseline
    "snr",
    "fringe_visibility",
    "power_band_ratio",
    # envelope
    "envelope_height",
    "envelope_area",
    "envelope_width",
    "envelope_sharpness",
    "envelope_symmetry",
    "single_peakness",
    "main_peak_to_sidelobe_ratio",
    # spectral
    "spectral_entropy",
    "spectral_centroid_offset",
    "dominant_spectral_peak_prominence",
    "energy_concentration_in_working_band",
    "carrier_to_background_spectral_ratio",
    "presence_of_expected_carrier_frequency",
    "spectral_peak_sharpness",
    "spectral_kurtosis",
    "spectral_spread",
    "envelope_spectrum_consistency",
    # noise
    "spectral_snr",
    "local_snr",
    "low_frequency_drift_level",
    "high_frequency_noise_level",
    "residual_noise_energy",
    "envelope_peak_to_background_ratio",
    # phase
    "phase_monotonicity_score",
    "phase_linear_fit_residual",
    "phase_slope_stability",
    "phase_curvature_index",
    "phase_jump_fraction",
    # correlation
    "centered_reference_correlation",
    "best_phase_reference_correlation",
    "reference_envelope_correlation",
    "phase_relaxation_gain",
    # regularity
    "autocorrelation_peak_strength",
    "zero_crossing_stability",
    "jitter_of_extrema",
    "local_oscillation_regularity",
})


def is_available() -> bool:
    """Return True if CuPy is installed and at least one GPU is visible."""
    try:
        import cupy as cp  # noqa: F401

        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def get_device_info() -> dict:
    """Return a dict describing the current CUDA device.

    Keys: ``name``, ``total_memory_mb``, ``cupy_version``.
    Returns an empty dict if CUDA is unavailable.
    """
    if not is_available():
        return {}
    try:
        import cupy as cp

        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        return {
            "name": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
            "total_memory_mb": round(props["totalGlobalMem"] / (1024 ** 2)),
            "cupy_version": cp.__version__,
        }
    except Exception:
        return {}
