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
    wavelength_nm : float or None
        Source wavelength in nanometres, from metadata.  ``None``
        when metadata is unavailable.
    coherence_length_nm : float or None
        Source coherence length in nanometres, from metadata.
        ``None`` when metadata is unavailable.
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

    # Metadata-derived fields (None when metadata is unavailable).
    wavelength_nm: float | None = None
    coherence_length_nm: float | None = None


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

    return AnalysisContext(
        band_half_width_bins=band_half_width_bins,
        default_segment_size=default_segment_size,
        expected_period_samples=expected_period_samples,
        wavelength_nm=wavelength_nm,
        coherence_length_nm=coherence_length_nm,
    )
