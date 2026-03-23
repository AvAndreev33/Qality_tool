"""Carrier-to-background spectral ratio metric.

Compares average in-band power to out-of-band median::

    CBSR = mean(P[B_exp]) / (median(P[~B_exp]) + eps)
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
    positive_freq_mask,
)

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class CarrierToBackgroundSpectralRatio:
    """Carrier-to-background spectral ratio.

    Score meaning: higher is better.
    """

    name: str = "carrier_to_background_spectral_ratio"
    category: str = "spectral"
    display_name: str = "Carrier / Background Ratio"
    score_direction: str = "higher_better"
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
        in_band = build_expected_band_mask(f, priors)
        pos = positive_freq_mask(f)
        out_band = pos & ~in_band

        if not np.any(in_band) or not np.any(out_band):
            return MetricResult(score=0.0, valid=False, notes="empty band")

        p_car = float(np.mean(power[in_band]))
        p_bg = float(np.median(power[out_band]))
        score = float(p_car / (p_bg + eps))
        return MetricResult(score=score, features={"p_car": p_car, "p_bg": p_bg})

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
        in_band = build_expected_band_mask(f, priors)
        pos = positive_freq_mask(f)
        out_band = pos & ~in_band

        if not np.any(in_band) or not np.any(out_band):
            return BatchMetricArrays(scores=scores, valid=valid_arr)

        p_car = power[:, in_band].mean(axis=1)
        # Median of out-of-band per signal.
        p_bg = np.median(power[:, out_band], axis=1)

        scores = p_car / (p_bg + eps)
        valid_arr = np.ones(n, dtype=bool)

        return BatchMetricArrays(
            scores=scores, valid=valid_arr,
            features={"p_car": p_car, "p_bg": p_bg},
        )
