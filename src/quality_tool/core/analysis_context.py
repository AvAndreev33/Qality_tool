"""Shared analysis context for Quality_tool.

Centralises numeric constants, spectral defaults, and group-level
heuristics that would otherwise be scattered across individual metric
implementations.

The context is frozen and immutable during evaluation.  It can be
built from dataset metadata via :func:`build_analysis_context` so
that oversampling-dependent parameters are resolved centrally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quality_tool.core.models import SignalSet


@dataclass(frozen=True)
class AnalysisContext:
    """Shared constants and heuristics for metric evaluation.

    Attributes
    ----------
    epsilon : float
        Small positive value for numeric stability (division guards).
    dc_exclude : bool
        Whether spectral computations should exclude the DC bin by
        default when computing band-based quantities.
    default_low_freq : float
        Default lower bound for signal-band selection (normalised
        frequency, used when no metadata-derived prior is available).
    default_high_freq : float
        Default upper bound for signal-band selection.
    noise_quarter_fraction : float
        Fraction of the signal used from each end for noise estimation.
    band_half_width_bins : int
        Half-width (Δk) of the carrier band around the dominant
        frequency bin, used by spectral noise metrics.
    drift_window : int
        Window length for moving-average drift estimation, used by
        the low-frequency drift metric.  Should be much larger than
        the fringe period.
    default_segment_size : int
        Fallback ROI segment size for fixed recipes when neither the
        recipe nor the active session provides one.
    expected_period_samples : int
        Approximate expected fringe period in samples (T_exp).
        Used by regularity metrics for autocorrelation search window,
        peak distance constraints, and zero-crossing filtering.
    peak_min_distance_fraction : float
        Minimum inter-peak distance as a fraction of
        ``expected_period_samples``.  Used for extrema detection in
        regularity metrics.
    period_search_tolerance_fraction : float
        Fractional tolerance (δ_T) around ``expected_period_samples``
        defining the search window for autocorrelation and
        zero-crossing metrics.
    cycle_resample_length : int
        Fixed length to which each detected cycle is resampled for
        shape-comparison in ``local_oscillation_regularity``.
    alpha_main_support : float
        Fraction of the envelope peak used to define the main-peak
        support region: ``W_main = {n : e[n] >= alpha * e_peak}``.
        Used by envelope metrics for single-peakness and sidelobe ratio.
    secondary_peak_min_distance : int
        Minimum sample distance between detected secondary peaks
        outside the main-peak support.
    secondary_peak_min_prominence : float
        Minimum prominence for secondary-peak detection outside the
        main-peak support.
    prominence_window_bins : int
        Neighbourhood half-width for the local background estimate
        used by ``dominant_spectral_peak_prominence``.
    prominence_exclusion_half_width_bins : int
        Inner exclusion half-width around the peak bin when
        estimating the local spectral background.
    local_spectrum_window_cycles : float
        Window length for local spectral analysis, expressed in
        multiples of the expected fringe period.
    local_spectrum_hop_cycles : float
        Hop length for local spectral analysis, expressed in
        multiples of the expected fringe period.
    coherence_to_bandwidth_scale : float
        Scale factor used when converting coherence-width-in-samples
        to expected spectral band half-width.
    phase_support_threshold_fraction : float
        Fraction (α_phase) of envelope peak for phase-support
        definition.
    phase_guard_samples : int
        Samples trimmed from each edge of phase support.
    phase_weight_power : float
        Exponent for envelope weighting in phase fits.
    phase_monotonicity_tolerance_fraction : float
        Fractional tolerance (τ_mon) for monotonicity inlier test.
    phase_jump_tolerance_fraction : float
        Fractional tolerance (τ_jump) for jump detection.
    minimum_phase_support_samples : int
        Minimum valid support length for phase metrics.
    minimum_phase_support_periods : float
        Minimum apparent carrier periods in phase support.
    reference_support_threshold_fraction : float
        Fraction (α_ref) of reference envelope for reference support.
    reference_carrier_period_nm : float or None
        Expected carrier period in nm, derived from metadata.
    reference_envelope_scale_nm : float or None
        Gaussian 1/e envelope scale in nm, derived from metadata.
    minimum_reference_support_samples : int
        Minimum samples in reference support.
    wavelength_nm : float or None
        Source wavelength in nanometres, from metadata.  ``None``
        when metadata is unavailable.
    coherence_length_nm : float or None
        Source coherence length in nanometres, from metadata.
        ``None`` when metadata is unavailable.
    z_step_nm : float or None
        Effective z-step in nanometres, from metadata.  ``None``
        when metadata is unavailable.
    """

    epsilon: float = 1e-12
    dc_exclude: bool = True
    default_low_freq: float = 0.05
    default_high_freq: float = 0.45
    noise_quarter_fraction: float = 0.25
    band_half_width_bins: int = 5
    drift_window: int = 31
    default_segment_size: int = 128
    expected_period_samples: int = 4
    peak_min_distance_fraction: float = 0.6
    period_search_tolerance_fraction: float = 0.3
    cycle_resample_length: int = 64
    alpha_main_support: float = 0.1
    secondary_peak_min_distance: int = 3
    secondary_peak_min_prominence: float = 0.0

    # Spectral-metric parameters.
    prominence_window_bins: int = 20
    prominence_exclusion_half_width_bins: int = 3
    local_spectrum_window_cycles: float = 4.0
    local_spectrum_hop_cycles: float = 2.0
    coherence_to_bandwidth_scale: float = 0.5

    # Phase-metric parameters.
    phase_support_threshold_fraction: float = 0.1
    phase_guard_samples: int = 2
    phase_weight_power: float = 2.0
    phase_monotonicity_tolerance_fraction: float = 0.5
    phase_jump_tolerance_fraction: float = 2.0
    minimum_phase_support_samples: int = 8
    minimum_phase_support_periods: float = 1.5

    # Correlation / reference-model metric parameters.
    reference_support_threshold_fraction: float = 0.05
    reference_carrier_period_nm: float | None = None
    reference_envelope_scale_nm: float | None = None
    minimum_reference_support_samples: int = 8

    # Metadata-derived fields (None when metadata is unavailable).
    wavelength_nm: float | None = None
    coherence_length_nm: float | None = None
    z_step_nm: float | None = None


# ------------------------------------------------------------------
# Project defaults (base values before oversampling scaling)
# ------------------------------------------------------------------
_BASE_BAND_HALF_WIDTH_BINS = 5
_BASE_DEFAULT_SEGMENT_SIZE = 128
_BASE_EXPECTED_PERIOD_SAMPLES = 4


def default_analysis_context() -> AnalysisContext:
    """Return the default analysis context with sensible fixed values."""
    return AnalysisContext()


def _effective_oversampling(metadata: dict | None) -> int:
    """Extract and validate the oversampling factor from metadata.

    Returns 1 when the factor is missing, NaN, or equal to 1.
    Otherwise returns the integer-converted factor.
    """
    if metadata is None:
        return 1
    raw = metadata.get("oversampling_factor")
    if raw is None:
        return 1
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1
    if math.isnan(value) or value <= 1.0:
        return 1
    return int(value)


def build_analysis_context(signal_set: SignalSet) -> AnalysisContext:
    """Build an effective :class:`AnalysisContext` from dataset metadata.

    Applies the canonical oversampling scaling rule:

    - If ``oversampling_factor`` is missing, NaN, or 1:
      ``band_half_width_bins = 5``, ``default_segment_size = 128``,
      ``expected_period_samples = 4``.
    - If ``oversampling_factor > 1``: each base value is multiplied
      by the oversampling factor.

    Also propagates ``wavelength_nm`` and ``coherence_length_nm``
    from metadata when available.

    Parameters
    ----------
    signal_set : SignalSet
        Loaded dataset whose ``metadata`` dict is inspected.

    Returns
    -------
    AnalysisContext
        Resolved context with oversampling-scaled values and
        metadata-derived optical parameters.
    """
    md = signal_set.metadata
    osf = _effective_oversampling(md)

    band_half_width_bins = _BASE_BAND_HALF_WIDTH_BINS * osf
    default_segment_size = _BASE_DEFAULT_SEGMENT_SIZE * osf
    expected_period_samples = _BASE_EXPECTED_PERIOD_SAMPLES * osf

    # Optical metadata — propagate when available.
    wavelength_nm: float | None = None
    coherence_length_nm: float | None = None
    z_step_nm: float | None = None
    if md is not None:
        wl = md.get("wavelength_nm")
        if wl is not None:
            try:
                wl_f = float(wl)
                if not math.isnan(wl_f):
                    wavelength_nm = wl_f
            except (TypeError, ValueError):
                pass
        cl = md.get("coherence_length_nm")
        if cl is not None:
            try:
                cl_f = float(cl)
                if not math.isnan(cl_f):
                    coherence_length_nm = cl_f
            except (TypeError, ValueError):
                pass
        zs = md.get("z_step_nm")
        if zs is not None:
            try:
                zs_f = float(zs)
                if not math.isnan(zs_f) and zs_f > 0:
                    z_step_nm = zs_f
            except (TypeError, ValueError):
                pass

    # Derive reference-model constants from metadata when possible.
    # For normal-incidence WLI the carrier period equals the source
    # wavelength and the Gaussian 1/e envelope scale is half the
    # coherence length.
    ref_carrier: float | None = None
    ref_envelope: float | None = None
    if wavelength_nm is not None and wavelength_nm > 0:
        ref_carrier = wavelength_nm
    if coherence_length_nm is not None and coherence_length_nm > 0:
        ref_envelope = coherence_length_nm / 2.0

    return AnalysisContext(
        band_half_width_bins=band_half_width_bins,
        default_segment_size=default_segment_size,
        expected_period_samples=expected_period_samples,
        reference_carrier_period_nm=ref_carrier,
        reference_envelope_scale_nm=ref_envelope,
        wavelength_nm=wavelength_nm,
        coherence_length_nm=coherence_length_nm,
        z_step_nm=z_step_nm,
    )
