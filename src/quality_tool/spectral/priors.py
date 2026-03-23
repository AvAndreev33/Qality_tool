"""Metadata-derived spectral priors for Quality_tool.

Computes expected carrier location, expected working-band edges, and
related quantities from :class:`AnalysisContext` metadata and the
actual signal length.  These priors are computed once and injected
into the evaluation context so that individual metrics never need to
recompute them.

Expected-period rule
--------------------
- If ``oversampling_factor`` is missing, NaN, or 1:
  ``expected_period_samples = 4``
- If ``oversampling_factor > 1``:
  ``expected_period_samples = 4 * oversampling_factor``

This rule is already applied inside :func:`build_analysis_context`.

Expected carrier bin
--------------------
``expected_carrier_bin = round(M / expected_period_samples)``
clipped to the usable positive-frequency range ``[1, M//2 - 1]``.

Expected band width
-------------------
When ``coherence_length_nm`` and ``z_step_nm`` are available:

1. ``packet_width_samples = coherence_length_nm / z_step_nm``
2. ``half_width = round(coherence_to_bandwidth_scale * M / packet_width_samples)``

Otherwise falls back to ``analysis_context.band_half_width_bins``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quality_tool.core.analysis_context import AnalysisContext


@dataclass(frozen=True)
class SpectralPriors:
    """Resolved spectral priors for a given signal length.

    Attributes
    ----------
    expected_period_samples : float
        Expected fringe period in samples (from context).
    expected_carrier_bin : int
        Expected carrier bin in the one-sided positive spectrum.
    expected_band_half_width_bins : int
        Half-width of the expected working band.
    expected_band_low_bin : int
        Lower edge of the expected working band (inclusive).
    expected_band_high_bin : int
        Upper edge of the expected working band (inclusive).
    signal_length : int
        Signal length M used to compute these priors.
    num_positive_bins : int
        Number of one-sided frequency bins ``M // 2 + 1``.
    """

    expected_period_samples: float
    expected_carrier_bin: int
    expected_band_half_width_bins: int
    expected_band_low_bin: int
    expected_band_high_bin: int
    signal_length: int
    num_positive_bins: int


def compute_spectral_priors(
    signal_length: int,
    ctx: AnalysisContext,
) -> SpectralPriors:
    """Compute spectral priors from context metadata and signal length.

    Parameters
    ----------
    signal_length : int
        Current signal length M (after any ROI extraction).
    ctx : AnalysisContext
        Shared analysis context with metadata-derived parameters.

    Returns
    -------
    SpectralPriors
        Resolved priors for the given signal length.
    """
    m = signal_length
    num_pos = m // 2 + 1  # rfft output length

    t_exp = float(ctx.expected_period_samples)

    # Expected carrier bin.
    if t_exp > 0:
        k_exp = round(m / t_exp)
    else:
        k_exp = round(m / 4.0)

    # Clip to usable positive-frequency range [1, num_pos - 2].
    max_bin = max(1, num_pos - 2)
    k_exp = max(1, min(k_exp, max_bin))

    # Expected band half-width.
    half_w = _compute_expected_half_width(m, ctx)

    # Band edges clipped to [1, num_pos - 1].
    k_low = max(1, k_exp - half_w)
    k_high = min(num_pos - 1, k_exp + half_w)

    return SpectralPriors(
        expected_period_samples=t_exp,
        expected_carrier_bin=k_exp,
        expected_band_half_width_bins=half_w,
        expected_band_low_bin=k_low,
        expected_band_high_bin=k_high,
        signal_length=m,
        num_positive_bins=num_pos,
    )


def _compute_expected_half_width(
    signal_length: int,
    ctx: AnalysisContext,
) -> int:
    """Derive expected spectral band half-width from metadata.

    Falls back to ``ctx.band_half_width_bins`` when metadata is
    missing or invalid.
    """
    cl = ctx.coherence_length_nm
    zs = ctx.z_step_nm
    scale = ctx.coherence_to_bandwidth_scale

    if cl is not None and zs is not None and cl > 0 and zs > 0:
        packet_width = cl / zs
        if packet_width > 0:
            half_w = round(scale * signal_length / packet_width)
            half_w = max(1, half_w)
            return int(half_w)

    # Fallback to the context default.
    return ctx.band_half_width_bins


# ------------------------------------------------------------------
# Band-mask helpers
# ------------------------------------------------------------------

def build_expected_band_mask(
    num_positive_bins: int,
    priors: SpectralPriors,
) -> np.ndarray:
    """Boolean mask over positive-frequency bins for the expected band.

    Parameters
    ----------
    num_positive_bins : int
        Length of the one-sided spectrum (``M // 2 + 1``).
    priors : SpectralPriors
        Resolved spectral priors.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(num_positive_bins,)``.
    """
    mask = np.zeros(num_positive_bins, dtype=bool)
    lo = max(0, priors.expected_band_low_bin)
    hi = min(num_positive_bins - 1, priors.expected_band_high_bin)
    if lo <= hi:
        mask[lo:hi + 1] = True
    return mask


def build_harmonic_band_masks(
    num_positive_bins: int,
    priors: SpectralPriors,
) -> list[np.ndarray]:
    """Boolean masks for the 2nd and 3rd harmonic bands.

    Returns up to two masks.  A mask is omitted if the harmonic
    centre lies outside the positive spectrum.

    Parameters
    ----------
    num_positive_bins : int
        Length of the one-sided spectrum.
    priors : SpectralPriors
        Resolved spectral priors.

    Returns
    -------
    list of np.ndarray
        Each element is a boolean mask of shape ``(num_positive_bins,)``.
    """
    k_exp = priors.expected_carrier_bin
    dw = priors.expected_band_half_width_bins
    masks: list[np.ndarray] = []

    for harmonic in (2, 3):
        k_h = harmonic * k_exp
        if k_h >= num_positive_bins:
            continue
        lo = max(1, k_h - dw)
        hi = min(num_positive_bins - 1, k_h + dw)
        m = np.zeros(num_positive_bins, dtype=bool)
        if lo <= hi:
            m[lo:hi + 1] = True
        masks.append(m)

    return masks


def build_low_freq_mask(
    num_positive_bins: int,
    priors: SpectralPriors,
) -> np.ndarray:
    """Boolean mask for the low-frequency region below the expected band.

    Covers bins ``[1, k_low)`` — excludes DC and the expected band.

    Parameters
    ----------
    num_positive_bins : int
        Length of the one-sided spectrum.
    priors : SpectralPriors
        Resolved spectral priors.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(num_positive_bins,)``.
    """
    mask = np.zeros(num_positive_bins, dtype=bool)
    k_low = priors.expected_band_low_bin
    if k_low > 1:
        mask[1:k_low] = True
    return mask


def positive_freq_mask(num_positive_bins: int, dc_exclude: bool = True) -> np.ndarray:
    """Boolean mask for usable positive-frequency bins.

    Parameters
    ----------
    num_positive_bins : int
        Length of the one-sided spectrum.
    dc_exclude : bool
        Whether to exclude the DC bin (index 0).

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(num_positive_bins,)``.
    """
    mask = np.ones(num_positive_bins, dtype=bool)
    if dc_exclude and num_positive_bins > 1:
        mask[0] = False
    return mask
