"""Shared analysis context for Quality_tool.

Centralises numeric constants, spectral defaults, and group-level
heuristics that would otherwise be scattered across individual metric
implementations.

The context is frozen and immutable during evaluation.  Future
iterations may allow GUI-configurable or metadata-derived values;
the current version uses sensible fixed defaults.
"""

from __future__ import annotations

from dataclasses import dataclass


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


def default_analysis_context() -> AnalysisContext:
    """Return the default analysis context with sensible fixed values."""
    return AnalysisContext()
