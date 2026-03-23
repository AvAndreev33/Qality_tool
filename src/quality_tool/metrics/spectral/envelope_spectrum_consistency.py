"""Envelope–spectrum consistency metric.

Cross-representation metric comparing the envelope-width × spectral-spread
product to a metadata-derived reference::

    ESC = |EW_obs * SS_obs - C_ref| / (C_ref + eps)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quality_tool.core.models import MetricResult
from quality_tool.evaluation.recipe import (
    ROI_MEAN_SUBTRACTED_LINEAR_DETRENDED,
    RecipeBinding,
    SignalRecipe,
)
from quality_tool.metrics.base import RepresentationNeeds
from quality_tool.metrics.spectral._spectral_batch_helpers import (
    normalized_spectral_weights,
    spectral_variance_batch,
)
from quality_tool.spectral.priors import SpectralPriors

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


def _envelope_fwhm(envelope: np.ndarray) -> float | None:
    """FWHM-like width of a 1-D envelope."""
    peak = np.max(envelope)
    if peak <= 0:
        return None
    half = 0.5 * peak
    above = envelope >= half
    idx = np.where(above)[0]
    if len(idx) < 2:
        return None
    return float(idx[-1] - idx[0])


def _build_reference(
    signal_length: int,
    priors: SpectralPriors,
    ctx,
) -> tuple[float, float] | None:
    """Build reference EW and SS from metadata.

    Returns (EW_ref, SS_ref) or None if metadata is insufficient.
    """
    wl = getattr(ctx, "wavelength_nm", None)
    cl = getattr(ctx, "coherence_length_nm", None)
    zs = getattr(ctx, "z_step_nm", None)

    if wl is None or cl is None or zs is None:
        return None
    if wl <= 0 or cl <= 0 or zs <= 0:
        return None

    # Reference envelope width in samples.
    ew_ref = cl / zs

    # Reference spectral spread: for a Gaussian packet the spread
    # is approximately M / (2 * pi * ew_ref).
    m = signal_length
    ss_ref = m / (2.0 * np.pi * ew_ref) if ew_ref > 0 else 0.0

    if ew_ref <= 0 or ss_ref <= 0:
        return None

    return (ew_ref, ss_ref)


class EnvelopeSpectrumConsistency:
    """Envelope–spectrum consistency.

    Score meaning: lower is better.
    """

    name: str = "envelope_spectrum_consistency"
    category: str = "spectral"
    display_name: str = "Envelope–Spectrum Consistency"
    score_direction: str = "lower_better"
    score_scale: str = "positive_unbounded"
    signal_recipe: SignalRecipe = ROI_MEAN_SUBTRACTED_LINEAR_DETRENDED
    recipe_binding: RecipeBinding = "fixed"
    representation_needs: RepresentationNeeds = RepresentationNeeds(
        envelope=True, power=True,
    )

    def evaluate(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelope: np.ndarray | None = None,
        context: dict | None = None,
    ) -> MetricResult:
        ctx = (context or {}).get("analysis_context")
        priors: SpectralPriors | None = (context or {}).get("spectral_priors")
        eps = getattr(ctx, "epsilon", 1e-12)

        spectral = (context or {}).get("spectral_result")
        if spectral is None or spectral.power is None or priors is None:
            return MetricResult(score=0.0, valid=False, notes="missing data")
        if envelope is None or envelope.size == 0:
            return MetricResult(score=0.0, valid=False, notes="no envelope")

        m = len(signal)
        ref = _build_reference(m, priors, ctx)
        if ref is None:
            return MetricResult(score=0.0, valid=False, notes="metadata insufficient")
        ew_ref, ss_ref = ref
        c_ref = ew_ref * ss_ref

        ew_obs = _envelope_fwhm(envelope)
        if ew_obs is None or ew_obs <= 0:
            return MetricResult(score=0.0, valid=False, notes="bad envelope width")

        power = spectral.power[np.newaxis, :]
        k_exp = float(priors.expected_carrier_bin)
        bins = np.arange(power.shape[1], dtype=float)
        p = normalized_spectral_weights(power, eps)
        var = spectral_variance_batch(p, bins, np.array([k_exp]))
        ss_obs = float(np.sqrt(var[0]))

        c_obs = ew_obs * ss_obs
        score = abs(c_obs - c_ref) / (c_ref + eps)
        return MetricResult(
            score=float(score),
            features={"ew_obs": ew_obs, "ss_obs": ss_obs,
                       "c_obs": c_obs, "c_ref": c_ref},
        )

    def evaluate_batch(
        self,
        signals: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelopes: np.ndarray | None = None,
        context: dict | None = None,
    ) -> BatchMetricArrays:
        """Semi-vectorised: spectral spread is batch, FWHM is per-signal."""
        from quality_tool.metrics.batch_result import BatchMetricArrays

        n = signals.shape[0]
        scores = np.full(n, np.nan)
        valid_arr = np.zeros(n, dtype=bool)

        ctx = (context or {}).get("analysis_context")
        priors: SpectralPriors | None = (context or {}).get("spectral_priors")
        eps = getattr(ctx, "epsilon", 1e-12)
        power = (context or {}).get("batch_power")

        if power is None or priors is None or envelopes is None:
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        m = signals.shape[1]
        ref = _build_reference(m, priors, ctx)
        if ref is None:
            return BatchMetricArrays(scores=scores, valid=valid_arr)
        ew_ref, ss_ref = ref
        c_ref = ew_ref * ss_ref

        # Batch spectral spread around expected carrier.
        k_exp = float(priors.expected_carrier_bin)
        bins = np.arange(power.shape[1], dtype=float)
        p = normalized_spectral_weights(power, eps)
        center = np.full(n, k_exp)
        var = spectral_variance_batch(p, bins, center)
        ss_obs = np.sqrt(var)

        # Per-signal envelope FWHM.
        ew_obs = np.full(n, np.nan)
        for i in range(n):
            w = _envelope_fwhm(envelopes[i])
            if w is not None and w > 0:
                ew_obs[i] = w

        good = np.isfinite(ew_obs) & (ew_obs > 0)
        c_obs = ew_obs * ss_obs
        scores[good] = np.abs(c_obs[good] - c_ref) / (c_ref + eps)
        valid_arr = good

        return BatchMetricArrays(scores=scores, valid=valid_arr)
