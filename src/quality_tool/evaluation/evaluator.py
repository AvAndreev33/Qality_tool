"""Full-image metric evaluator for Quality_tool.

Runs one or more metrics over every pixel signal in a :class:`SignalSet`
and assembles the results into :class:`MetricMapResult` objects.

Signal recipe model
-------------------
Each metric declares a ``signal_recipe`` and ``recipe_binding``.  The
evaluator resolves the effective recipe for each metric, groups metrics
that share the same recipe via the planner, prepares signals once per
recipe group, and dispatches metrics onto the correct prepared bundle.

``raw`` is a recipe (the identity recipe), not a special case.

Representation bundles
----------------------
For each recipe group and each chunk the evaluator builds a
:class:`RepresentationBundle` containing the prepared signal and any
derived representations (envelope, spectral data) required by the
group's metrics.  Each representation is computed at most once per
recipe per chunk.

Batch / chunked evaluation
--------------------------
The evaluator operates in **chunked batch mode**: it reshapes the
canonical ``(H, W, M)`` signal array to ``(N, M)`` (where ``N = H*W``),
processes signals in configurable chunks, and writes results directly
into preallocated output arrays.

Metrics that implement ``evaluate_batch(signals, z_axis, envelopes,
context) -> BatchMetricArrays`` are evaluated in vectorised mode.
Metrics without ``evaluate_batch`` fall back to a per-signal loop
within each chunk.

Invalid-score convention
------------------------
When a pixel's evaluation is invalid:

* ``score_map[r, c]`` is set to ``np.nan``
* ``valid_map[r, c]`` is set to ``False``
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from quality_tool.core.analysis_context import AnalysisContext, default_analysis_context
from quality_tool.core.models import MetricMapResult, MetricResult, SignalSet
from quality_tool.envelope.analytic import AnalyticEnvelopeMethod
from quality_tool.envelope.base import BaseEnvelopeMethod
from quality_tool.evaluation.bundle import RepresentationBundle
from quality_tool.evaluation.planner import EvaluationPlan, RecipeGroup, build_plan
from quality_tool.evaluation.recipe import (
    RAW,
    SignalRecipe,
    resolve_effective_recipe,
)
from quality_tool.metrics.base import BaseMetric, resolve_representation_needs
from quality_tool.metrics.batch_result import BatchMetricArrays
from quality_tool.preprocessing.batch import (
    detrend_linear_batch,
    extract_roi_batch,
    normalize_amplitude_batch,
    smooth_batch,
    subtract_baseline_batch,
)
from quality_tool.preprocessing.basic import (
    normalize_amplitude,
    smooth,
    subtract_baseline,
)
from quality_tool.preprocessing.roi import extract_roi
from quality_tool.spectral.fft import (
    SpectralResult,
    compute_spectrum_batch,
)

# Default chunk size — balances vectorisation benefit against memory.
_DEFAULT_CHUNK = 50_000


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def evaluate_metric_map(
    signal_set: SignalSet,
    metric: BaseMetric,
    *,
    active_recipe: SignalRecipe | None = None,
    envelope_method: BaseEnvelopeMethod | None = None,
    analysis_context: AnalysisContext | None = None,
    chunk_size: int = _DEFAULT_CHUNK,
) -> MetricMapResult:
    """Evaluate *metric* on every pixel signal in *signal_set*.

    Convenience wrapper around :func:`evaluate_metric_maps` for
    single-metric evaluation.

    Parameters
    ----------
    signal_set : SignalSet
        Loaded dataset with ``signals.shape == (H, W, M)``.
    metric : BaseMetric
        Quality metric to evaluate.
    active_recipe : SignalRecipe | None
        The current session processing pipeline.  Used to resolve
        ``recipe_binding="active"`` metrics.  Ignored for fixed-recipe
        metrics.
    envelope_method : BaseEnvelopeMethod | None
        If given, the envelope is computed on the recipe-prepared signal
        and passed to the metric.
    analysis_context : AnalysisContext | None
        Shared constants and heuristics.  Uses defaults if ``None``.
    chunk_size : int
        Number of signals per batch chunk.

    Returns
    -------
    MetricMapResult
    """
    results = evaluate_metric_maps(
        signal_set,
        [metric],
        active_recipe=active_recipe,
        envelope_method=envelope_method,
        analysis_context=analysis_context,
        chunk_size=chunk_size,
    )
    return results[metric.name]


def evaluate_metric_maps(
    signal_set: SignalSet,
    metrics: Sequence[BaseMetric],
    *,
    active_recipe: SignalRecipe | None = None,
    envelope_method: BaseEnvelopeMethod | None = None,
    analysis_context: AnalysisContext | None = None,
    chunk_size: int = _DEFAULT_CHUNK,
) -> dict[str, MetricMapResult]:
    """Evaluate multiple metrics on every pixel signal in *signal_set*.

    Uses the planner to group metrics by effective recipe so that
    signal preparation and derived representations are computed once
    per recipe group.

    Parameters
    ----------
    signal_set : SignalSet
        Loaded dataset with ``signals.shape == (H, W, M)``.
    metrics : sequence of BaseMetric
        Quality metrics to evaluate.
    active_recipe : SignalRecipe | None
        The current session processing pipeline.
    envelope_method : BaseEnvelopeMethod | None
        If given, the envelope is computed per recipe group.
    analysis_context : AnalysisContext | None
        Shared constants and heuristics.  Uses defaults if ``None``.
    chunk_size : int
        Number of signals per batch chunk.

    Returns
    -------
    dict[str, MetricMapResult]
        Mapping from metric name to result.
    """
    if analysis_context is None:
        analysis_context = default_analysis_context()

    plan = build_plan(
        metrics,
        active_recipe=active_recipe,
        has_envelope=envelope_method is not None,
    )

    h, w, m = signal_set.signals.shape
    n_total = h * w
    z_axis = signal_set.z_axis

    # Flatten to (N, M) — this is a view, no copy.
    signals_2d = signal_set.signals.reshape(n_total, m)

    results: dict[str, MetricMapResult] = {}

    # Fallback envelope method: when a group's metrics declare that
    # they need envelope but no envelope method was provided by the
    # session, use the default analytic envelope.
    _fallback_envelope: BaseEnvelopeMethod | None = None

    for group in plan.groups:
        if group.needs_envelope:
            group_envelope = envelope_method
            if group_envelope is None:
                if _fallback_envelope is None:
                    _fallback_envelope = AnalyticEnvelopeMethod()
                group_envelope = _fallback_envelope
        else:
            group_envelope = None

        group_results = _evaluate_recipe_group(
            group,
            signals_2d=signals_2d,
            z_axis=z_axis,
            h=h,
            w=w,
            n_total=n_total,
            envelope_method=group_envelope,
            analysis_context=analysis_context,
            chunk_size=chunk_size,
            active_recipe=active_recipe,
        )
        results.update(group_results)

    return results


# ------------------------------------------------------------------
# Internal: per-recipe-group evaluation
# ------------------------------------------------------------------

def _evaluate_recipe_group(
    group: RecipeGroup,
    *,
    signals_2d: np.ndarray,
    z_axis: np.ndarray,
    h: int,
    w: int,
    n_total: int,
    envelope_method: BaseEnvelopeMethod | None,
    analysis_context: AnalysisContext,
    chunk_size: int,
    active_recipe: SignalRecipe | None,
) -> dict[str, MetricMapResult]:
    """Evaluate all metrics in a single recipe group."""
    recipe = group.recipe
    is_raw = recipe == RAW

    has_envelope_batch = (
        envelope_method is not None
        and hasattr(envelope_method, "compute_batch")
    )

    # Build batch preprocessing chain for this recipe.
    batch_preprocess = _recipe_to_batch_preprocess(recipe) if not is_raw else None

    # Determine ROI segment size.  Fixed recipes with roi_enabled=True
    # but no explicit segment_size inherit it from the active recipe,
    # then fall back to the analysis context default.
    segment_size: int | None = None
    if recipe.roi_enabled:
        segment_size = recipe.segment_size
        if segment_size is None and active_recipe is not None:
            segment_size = active_recipe.segment_size
        if segment_size is None:
            segment_size = analysis_context.default_segment_size

    # Merged representation needs for this group.
    group_needs = group.needs

    # Per-metric accumulators.
    metric_accum: dict[str, _MetricAccum] = {
        m.name: _MetricAccum(n_total) for m in group.metrics
    }

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk_n = end - start

        # --- 1) Prepare signals for this recipe ---
        if is_raw:
            chunk_signals = signals_2d[start:end]
            chunk_z = z_axis
        else:
            chunk_signals = signals_2d[start:end].copy()

            if batch_preprocess is not None:
                for fn in batch_preprocess:
                    chunk_signals = fn(chunk_signals)

            chunk_z = z_axis
            if segment_size is not None:
                chunk_signals = extract_roi_batch(chunk_signals, segment_size)
                chunk_z = None  # z_axis no longer matches

        # --- 2) Build representation bundle for this chunk ---
        bundle = _build_bundle(
            chunk_signals=chunk_signals,
            chunk_z=chunk_z,
            recipe=recipe,
            group_needs=group_needs,
            envelope_method=envelope_method,
            has_envelope_batch=has_envelope_batch,
            chunk_n=chunk_n,
            analysis_context=analysis_context,
        )

        # Context dict from the bundle (backward-compatible).
        context = bundle.to_context_dict()

        # --- 3) Evaluate each metric in this group ---
        for metric in group.metrics:
            accum = metric_accum[metric.name]
            has_batch = hasattr(metric, "evaluate_batch")

            if has_batch:
                batch_result = metric.evaluate_batch(
                    chunk_signals, chunk_z, bundle.envelope, context,
                )
                accum.score_flat[start:end] = batch_result.scores
                accum.valid_flat[start:end] = batch_result.valid
                accum.feature_chunks.append(batch_result.features)
            else:
                _evaluate_per_signal(
                    metric=metric,
                    chunk_signals=chunk_signals,
                    chunk_z=chunk_z,
                    bundle=bundle,
                    context=context,
                    accum=accum,
                    start=start,
                    chunk_n=chunk_n,
                )

            accum.chunk_ranges.append((start, end))

    # --- Assemble final results ---
    out: dict[str, MetricMapResult] = {}
    for metric in group.metrics:
        accum = metric_accum[metric.name]

        score_map = accum.score_flat.reshape(h, w)
        valid_map = accum.valid_flat.reshape(h, w)
        feature_maps = _merge_feature_chunks(
            accum.feature_chunks, accum.chunk_ranges, n_total, h, w,
        )

        effective_recipe = recipe

        metadata: dict = {
            "metric_name": metric.name,
            "recipe_binding": getattr(metric, "recipe_binding", "active"),
            "effective_recipe": effective_recipe,
            "envelope_method": (
                envelope_method.name if envelope_method is not None else None
            ),
            "image_shape": (h, w),
        }

        out[metric.name] = MetricMapResult(
            metric_name=metric.name,
            score_map=score_map,
            valid_map=valid_map,
            feature_maps=feature_maps,
            metadata=metadata,
        )

    return out


# ------------------------------------------------------------------
# Internal: bundle construction
# ------------------------------------------------------------------

def _build_bundle(
    *,
    chunk_signals: np.ndarray,
    chunk_z: np.ndarray | None,
    recipe: SignalRecipe,
    group_needs,
    envelope_method: BaseEnvelopeMethod | None,
    has_envelope_batch: bool,
    chunk_n: int,
    analysis_context: AnalysisContext,
) -> RepresentationBundle:
    """Build a RepresentationBundle for one chunk."""

    # --- Envelope (once per recipe group if needed) ---
    chunk_envelopes: np.ndarray | None = None
    if envelope_method is not None:
        if has_envelope_batch:
            chunk_envelopes = envelope_method.compute_batch(
                chunk_signals, chunk_z, None,
            )
        else:
            chunk_envelopes = np.empty_like(chunk_signals)
            for i in range(chunk_n):
                chunk_envelopes[i] = envelope_method.compute(
                    chunk_signals[i], chunk_z, None,
                )

    # --- Spectral (once per recipe group if needed) ---
    batch_spectral = None
    if group_needs.needs_spectral:
        batch_spectral = compute_spectrum_batch(
            chunk_signals,
            chunk_z,
            include_amplitude=group_needs.amplitude,
            include_power=group_needs.power,
            include_complex=group_needs.complex_fft,
        )

    return RepresentationBundle(
        signals=chunk_signals,
        z_axis=chunk_z,
        recipe=recipe,
        analysis_context=analysis_context,
        envelope=chunk_envelopes,
        spectral=batch_spectral,
    )


# ------------------------------------------------------------------
# Internal: per-signal fallback
# ------------------------------------------------------------------

def _evaluate_per_signal(
    *,
    metric: BaseMetric,
    chunk_signals: np.ndarray,
    chunk_z: np.ndarray | None,
    bundle: RepresentationBundle,
    context: dict,
    accum: _MetricAccum,
    start: int,
    chunk_n: int,
) -> None:
    """Evaluate a metric per-signal when no ``evaluate_batch`` exists."""
    metric_needs = resolve_representation_needs(metric)
    chunk_features: dict[str, list[float]] = {}

    for i in range(chunk_n):
        sig = chunk_signals[i]
        env = bundle.envelope[i] if bundle.envelope is not None else None

        # Build per-signal context.
        sig_ctx: dict = {
            "analysis_context": bundle.analysis_context,
        }
        if metric_needs.needs_spectral and bundle.spectral is not None:
            # Build a single-signal SpectralResult from the batch data.
            amp = (
                bundle.spectral.amplitude[i]
                if bundle.spectral.amplitude is not None
                else None
            )
            sig_ctx["spectral_result"] = SpectralResult(
                frequencies=bundle.spectral.frequencies,
                amplitude=amp if amp is not None else np.array([]),
                power=(
                    bundle.spectral.power[i]
                    if bundle.spectral.power is not None
                    else None
                ),
                complex_fft=(
                    bundle.spectral.complex_fft[i]
                    if bundle.spectral.complex_fft is not None
                    else None
                ),
            )

        result = metric.evaluate(sig, chunk_z, env, sig_ctx)
        accum.score_flat[start + i] = (
            result.score if result.valid else np.nan
        )
        accum.valid_flat[start + i] = result.valid
        for k, v in result.features.items():
            if k not in chunk_features:
                chunk_features[k] = [np.nan] * i
            chunk_features[k].append(
                float(v) if result.valid else np.nan
            )
        for k in chunk_features:
            if k not in result.features:
                chunk_features[k].append(np.nan)

    accum.feature_chunks.append(
        {k: np.array(v) for k, v in chunk_features.items()}
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

class _MetricAccum:
    """Per-metric accumulator for chunked evaluation."""

    __slots__ = ("score_flat", "valid_flat", "feature_chunks", "chunk_ranges")

    def __init__(self, n_total: int) -> None:
        self.score_flat = np.full(n_total, np.nan, dtype=float)
        self.valid_flat = np.zeros(n_total, dtype=bool)
        self.feature_chunks: list[dict[str, np.ndarray]] = []
        self.chunk_ranges: list[tuple[int, int]] = []


def _recipe_to_batch_preprocess(
    recipe: SignalRecipe,
) -> list[Callable[[np.ndarray], np.ndarray]] | None:
    """Convert a recipe's preprocessing flags to batch callables."""
    fns: list[Callable] = []
    if recipe.baseline:
        fns.append(subtract_baseline_batch)
    if recipe.normalize:
        fns.append(normalize_amplitude_batch)
    if recipe.smooth:
        fns.append(smooth_batch)
    if recipe.detrend:
        fns.append(detrend_linear_batch)
    return fns if fns else None


def _merge_feature_chunks(
    chunks: list[dict[str, np.ndarray]],
    ranges: list[tuple[int, int]],
    n_total: int,
    h: int,
    w: int,
) -> dict[str, np.ndarray]:
    """Merge per-chunk feature dicts into (H, W) feature maps."""
    if not chunks:
        return {}

    all_keys: set[str] = set()
    for ch in chunks:
        all_keys.update(ch.keys())

    if not all_keys:
        return {}

    feature_maps: dict[str, np.ndarray] = {}
    for key in all_keys:
        flat = np.full(n_total, np.nan, dtype=float)
        for ch, (start, end) in zip(chunks, ranges):
            if key in ch:
                flat[start:end] = ch[key]
        feature_maps[key] = flat.reshape(h, w)

    return feature_maps
