"""Autocorrelation peak strength metric for Quality_tool.

Measures global periodicity by finding the maximum of the normalised
autocorrelation within a search window around the expected fringe
period::

    APS = max_{τ in W_T} r_norm[τ]

Higher scores indicate stronger periodic structure.
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

if TYPE_CHECKING:
    from quality_tool.metrics.batch_result import BatchMetricArrays


class AutocorrelationPeakStrength:
    """Normalised autocorrelation peak strength near the expected period.

    Score meaning: higher is better.
    """

    name: str = "autocorrelation_peak_strength"
    category: str = "regularity"
    display_name: str = "Autocorrelation Peak Strength"
    score_direction: str = "higher_better"
    score_scale: str = "bounded_01"
    signal_recipe: SignalRecipe = ROI_MEAN_SUBTRACTED_LINEAR_DETRENDED
    recipe_binding: RecipeBinding = "fixed"

    # ------------------------------------------------------------------

    def evaluate(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelope: np.ndarray | None = None,
        context: dict | None = None,
    ) -> MetricResult:
        if signal.ndim != 1 or signal.size < 4:
            return MetricResult(score=0.0, valid=False,
                                notes="signal too short")

        ctx = (context or {}).get("analysis_context")
        eps = getattr(ctx, "epsilon", 1e-12)
        t_exp = getattr(ctx, "expected_period_samples", 4)
        delta_t = getattr(ctx, "period_search_tolerance_fraction", 0.3)

        score, features, valid, notes = _compute_aps(
            signal, t_exp, delta_t, eps,
        )
        return MetricResult(score=score, features=features,
                            valid=valid, notes=notes)

    def evaluate_batch(
        self,
        signals: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelopes: np.ndarray | None = None,
        context: dict | None = None,
    ) -> BatchMetricArrays:
        from quality_tool.metrics.batch_result import BatchMetricArrays

        ctx = (context or {}).get("analysis_context")
        eps = getattr(ctx, "epsilon", 1e-12)
        t_exp = getattr(ctx, "expected_period_samples", 4)
        delta_t = getattr(ctx, "period_search_tolerance_fraction", 0.3)

        n, m = signals.shape

        tau_min = max(1, round((1 - delta_t) * t_exp))
        tau_max = round((1 + delta_t) * t_exp)
        max_lag = m - 1

        scores = np.full(n, np.nan)
        valid = np.zeros(n, dtype=bool)
        best_lag_arr = np.zeros(n)

        if tau_min > max_lag or tau_min > tau_max:
            return BatchMetricArrays(
                scores=scores, valid=valid,
                features={"best_lag": best_lag_arr},
            )

        tau_max = min(tau_max, max_lag)

        # Vectorised energy — shape (N,).
        r0 = np.sum(signals * signals, axis=1)
        energy_ok = r0 >= eps

        # Vectorised autocorrelation search over all taus at once.
        best_scores = np.full(n, -np.inf)
        best_lags = np.full(n, tau_min, dtype=float)

        for tau in range(tau_min, tau_max + 1):
            r_tau = np.sum(
                signals[:, :m - tau] * signals[:, tau:], axis=1,
            )
            r_norm = r_tau / (r0 + eps)
            better = r_norm > best_scores
            best_scores = np.where(better, r_norm, best_scores)
            best_lags = np.where(better, float(tau), best_lags)

        valid = energy_ok
        scores = np.where(valid, best_scores, np.nan)
        best_lag_arr = best_lags

        return BatchMetricArrays(
            scores=scores, valid=valid,
            features={"best_lag": best_lag_arr},
        )


def _compute_aps(
    signal: np.ndarray,
    t_exp: int,
    delta_t: float,
    eps: float,
) -> tuple[float, dict, bool, str]:
    """Core computation shared by scalar and batch paths."""
    m = signal.size
    tau_min = max(1, round((1 - delta_t) * t_exp))
    tau_max = round((1 + delta_t) * t_exp)

    # Clamp to usable lag range.
    max_lag = m - 1
    if tau_min > max_lag or tau_min > tau_max:
        return 0.0, {}, False, "search interval empty or outside lag range"

    tau_max = min(tau_max, max_lag)

    # Normalised autocorrelation via direct summation.
    r0 = float(np.dot(signal, signal))
    if r0 < eps:
        return 0.0, {}, False, "zero-energy signal"

    best_score = -np.inf
    best_lag = tau_min
    for tau in range(tau_min, tau_max + 1):
        r_tau = float(np.dot(signal[:m - tau], signal[tau:]))
        r_norm = r_tau / (r0 + eps)
        if r_norm > best_score:
            best_score = r_norm
            best_lag = tau

    return float(best_score), {"best_lag": float(best_lag)}, True, ""
