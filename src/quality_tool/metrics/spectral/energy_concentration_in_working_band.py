"""Energy concentration in working band metric.

Fraction of total spectral power inside the expected band::

    ECWB = sum(P[B_exp]) / (sum(P) + eps)
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


class EnergyConcentrationInWorkingBand:
    """Energy concentration in the expected working band.

    Score meaning: higher is better (1.0 = all energy in band).
    """

    name: str = "energy_concentration_in_working_band"
    category: str = "spectral"
    display_name: str = "Energy Concentration in Band"
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
        if spectral is None or spectral.power is None or priors is None:
            return MetricResult(score=0.0, valid=False, notes="missing data")

        power = spectral.power
        f = len(power)
        band = build_expected_band_mask(f, priors)
        if not np.any(band):
            return MetricResult(score=0.0, valid=False, notes="empty band")

        score = float(np.sum(power[band]) / (np.sum(power) + eps))
        return MetricResult(score=score)

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
        valid_arr = np.zeros(n, dtype=bool)

        ctx = (context or {}).get("analysis_context")
        priors: SpectralPriors | None = (context or {}).get("spectral_priors")
        eps = getattr(ctx, "epsilon", 1e-12)
        power = (context or {}).get("batch_power")

        if power is None or priors is None:
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        f = power.shape[1]
        band = build_expected_band_mask(f, priors)
        if not np.any(band):
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        e_band = power[:, band].sum(axis=1)
        e_total = power.sum(axis=1) + eps
        scores = e_band / e_total
        valid_arr = np.ones(n, dtype=bool)

        return BatchMetricArrays(scores=scores, valid=valid_arr)
