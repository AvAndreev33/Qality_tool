"""Presence of expected carrier frequency metric.

Measures whether the dominant spectral feature is located where
the carrier is expected::

    PECF = P[k_*] / (P[k_0] + eps)

where k_* is the strongest bin inside B_exp and k_0 is the global
dominant bin.
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
from quality_tool.spectral.priors import SpectralPriors, build_expected_band_mask

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class PresenceOfExpectedCarrierFrequency:
    """Presence of expected carrier frequency.

    Score meaning: higher is better (1.0 = peak is in expected band).
    """

    name: str = "presence_of_expected_carrier_frequency"
    category: str = "spectral"
    display_name: str = "Presence of Expected Carrier"
    score_direction: str = "higher_better"
    score_scale: str = "bounded_01"
    signal_recipe: SignalRecipe = ROI_MEAN_SUBTRACTED_LINEAR_DETRENDED
    recipe_binding: RecipeBinding = "fixed"
    representation_needs: RepresentationNeeds = RepresentationNeeds(power=True)

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
        if spectral is None or spectral.power is None:
            return MetricResult(score=0.0, valid=False, notes="no power spectrum")
        if priors is None:
            return MetricResult(score=0.0, valid=False, notes="no spectral priors")

        power = spectral.power
        f = len(power)
        band_mask = build_expected_band_mask(f, priors)
        if not np.any(band_mask):
            return MetricResult(score=0.0, valid=False, notes="empty expected band")

        # Global dominant (exclude DC).
        search = power.copy()
        if f > 1:
            search[0] = -np.inf
        k_0 = int(np.argmax(search))
        p_global = power[k_0]

        # Best in expected band.
        band_power = np.where(band_mask, power, -np.inf)
        k_star = int(np.argmax(band_power))
        p_band = power[k_star]

        score = float(p_band / (p_global + eps))
        return MetricResult(
            score=score,
            features={"k_star": float(k_star), "k_0": float(k_0)},
        )

    def evaluate_batch(
        self,
        signals: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelopes: np.ndarray | None = None,
        context: dict | None = None,
    ) -> BatchMetricArrays:
        from quality_tool.metrics.batch_result import BatchMetricArrays

        n = signals.shape[0]
        scores = np.full(n, np.nan)
        valid = np.zeros(n, dtype=bool)

        ctx = (context or {}).get("analysis_context")
        priors: SpectralPriors | None = (context or {}).get("spectral_priors")
        eps = getattr(ctx, "epsilon", 1e-12)

        power = (context or {}).get("batch_power")
        if power is None or priors is None:
            return BatchMetricArrays(scores=scores, valid=valid)

        f = power.shape[1]
        band_mask = build_expected_band_mask(f, priors)
        if not np.any(band_mask):
            return BatchMetricArrays(scores=scores, valid=valid)

        # Global dominant (exclude DC).
        search = power.copy()
        if f > 1:
            search[:, 0] = -np.inf
        k_0 = np.argmax(search, axis=1)
        p_global = power[np.arange(n), k_0]

        # Best in expected band.
        band_power = np.where(band_mask[np.newaxis, :], power, -np.inf)
        k_star = np.argmax(band_power, axis=1)
        p_band = power[np.arange(n), k_star]

        scores = p_band / (p_global + eps)
        valid = np.ones(n, dtype=bool)

        return BatchMetricArrays(
            scores=scores, valid=valid,
            features={"k_star": k_star.astype(float), "k_0": k_0.astype(float)},
        )
