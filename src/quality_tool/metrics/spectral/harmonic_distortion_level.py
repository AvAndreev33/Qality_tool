"""Harmonic distortion level metric.

Compares energy in 2nd and 3rd harmonic bands to the carrier band::

    HDL = E_harm / (E_car + eps)
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
from quality_tool.spectral.priors import (
    SpectralPriors,
    build_expected_band_mask,
    build_harmonic_band_masks,
)

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class HarmonicDistortionLevel:
    """Harmonic distortion level.

    Score meaning: lower is better.
    """

    name: str = "harmonic_distortion_level"
    category: str = "spectral"
    display_name: str = "Harmonic Distortion Level"
    score_direction: str = "lower_better"
    score_scale: str = "positive_unbounded"
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
        car_mask = build_expected_band_mask(f, priors)
        if not np.any(car_mask):
            return MetricResult(score=0.0, valid=False, notes="empty carrier band")

        harm_masks = build_harmonic_band_masks(f, priors)
        e_harm = sum(float(np.sum(power[hm])) for hm in harm_masks)
        e_car = float(np.sum(power[car_mask]))

        score = float(e_harm / (e_car + eps))
        return MetricResult(score=score, features={"e_harm": e_harm, "e_car": e_car})

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
        car_mask = build_expected_band_mask(f, priors)
        if not np.any(car_mask):
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        harm_masks = build_harmonic_band_masks(f, priors)
        e_harm = np.zeros(n)
        for hm in harm_masks:
            e_harm += power[:, hm].sum(axis=1)
        e_car = power[:, car_mask].sum(axis=1)

        scores = e_harm / (e_car + eps)
        valid_arr = np.ones(n, dtype=bool)

        return BatchMetricArrays(
            scores=scores, valid=valid_arr,
            features={"e_harm": e_harm, "e_car": e_car},
        )
