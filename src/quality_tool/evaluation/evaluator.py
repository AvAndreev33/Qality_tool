"""Full-image metric evaluator for Quality_tool.

Runs a single metric over every pixel signal in a :class:`SignalSet` and
assembles the results into a :class:`MetricMapResult`.

Batch / chunked evaluation
--------------------------
The evaluator operates in **chunked batch mode**: it reshapes the
canonical ``(H, W, M)`` signal array to ``(N, M)`` (where ``N = H*W``),
processes signals in configurable chunks, and writes results directly
into preallocated output arrays.  This avoids the per-pixel Python loop
and the huge transient list of ``MetricResult`` objects that dominated
runtime and RSS in the original implementation.

Metrics that implement ``evaluate_batch(signals, z_axis, envelopes,
context) -> BatchMetricArrays`` are evaluated in vectorised mode.
Metrics without ``evaluate_batch`` fall back to a per-signal loop
within each chunk.

Conditional FFT
---------------
The evaluator inspects ``metric.needs_spectral``.  When ``False``
(the default), no FFT is computed.  When ``True``, a batch FFT is
performed once per chunk and passed to the metric via ``context``.

Metric input policy
-------------------
Each metric declares an ``input_policy`` attribute:

* ``"raw"`` — the metric receives the original raw signal from the
  dataset.  Preprocessing and ROI extraction are **skipped**.
* ``"processed"`` — the metric receives the signal after the currently
  configured preprocessing chain and optional ROI extraction.

Invalid-score convention
------------------------
When a pixel's evaluation is invalid:

* ``score_map[r, c]`` is set to ``np.nan``
* ``valid_map[r, c]`` is set to ``False``
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from quality_tool.core.models import MetricMapResult, MetricResult, SignalSet
from quality_tool.envelope.base import BaseEnvelopeMethod
from quality_tool.metrics.base import BaseMetric
from quality_tool.metrics.batch_result import BatchMetricArrays
from quality_tool.preprocessing.batch import (
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
from quality_tool.spectral.fft import compute_spectrum

# Default chunk size — balances vectorisation benefit against memory.
_DEFAULT_CHUNK = 50_000

# Mapping from per-signal preprocessing functions to their batch
# equivalents.  Used to auto-detect batchable preprocessing chains.
_BATCH_PREPROCESS_MAP: dict[Callable, Callable] = {
    subtract_baseline: subtract_baseline_batch,
    normalize_amplitude: normalize_amplitude_batch,
}


def evaluate_metric_map(
    signal_set: SignalSet,
    metric: BaseMetric,
    *,
    preprocess: Sequence[Callable[[np.ndarray], np.ndarray]] | None = None,
    segment_size: int | None = None,
    envelope_method: BaseEnvelopeMethod | None = None,
    chunk_size: int = _DEFAULT_CHUNK,
) -> MetricMapResult:
    """Evaluate *metric* on every pixel signal in *signal_set*.

    Parameters
    ----------
    signal_set : SignalSet
        Loaded dataset with ``signals.shape == (H, W, M)``.
    metric : BaseMetric
        Quality metric to evaluate.
    preprocess : sequence of callables, optional
        Preprocessing functions applied **in order** to each 1-D signal
        before ROI extraction.  Each callable must accept and return a
        1-D ``np.ndarray``.
    segment_size : int, optional
        If given, a ROI of this length is extracted (centred on ``raw_max``)
        after preprocessing and before envelope / metric evaluation.
    envelope_method : BaseEnvelopeMethod, optional
        If given, the envelope is computed on the (possibly ROI-cropped)
        signal and passed to ``metric.evaluate``.
    chunk_size : int
        Number of signals per batch chunk.  Controls the memory / speed
        trade-off.  Default is 50 000.

    Returns
    -------
    MetricMapResult
        Aggregated per-pixel scores, validity flags, feature maps, and
        evaluation metadata.
    """
    h, w, m = signal_set.signals.shape
    n_total = h * w
    z_axis = signal_set.z_axis

    input_policy = getattr(metric, "input_policy", "processed")
    use_raw = input_policy == "raw"
    needs_spectral = getattr(metric, "needs_spectral", False)
    has_batch = hasattr(metric, "evaluate_batch")
    has_envelope_batch = (
        envelope_method is not None
        and hasattr(envelope_method, "compute_batch")
    )

    # Flatten to (N, M) for batch processing — this is a view, no copy.
    signals_2d = signal_set.signals.reshape(n_total, m)

    # Preallocate output arrays.
    score_flat = np.full(n_total, np.nan, dtype=float)
    valid_flat = np.zeros(n_total, dtype=bool)
    # Feature maps are collected per-chunk and merged at the end.
    feature_chunks: list[dict[str, np.ndarray]] = []
    # Track chunk boundaries for feature assembly.
    chunk_ranges: list[tuple[int, int]] = []

    # Build batch-capable preprocessing chain for processed metrics.
    batch_preprocess = _resolve_batch_preprocess(preprocess) if not use_raw else None

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk_n = end - start

        # --- 1) Get effective signals for this chunk ---
        if use_raw:
            # Raw metrics: direct view, no copy, no preprocessing.
            chunk_signals = signals_2d[start:end]
            chunk_z = z_axis
        else:
            # Processed metrics: copy then preprocess in batch.
            chunk_signals = signals_2d[start:end].copy()

            if batch_preprocess is not None:
                for fn in batch_preprocess:
                    chunk_signals = fn(chunk_signals)

            # ROI extraction
            chunk_z = z_axis
            if segment_size is not None:
                chunk_signals = extract_roi_batch(
                    chunk_signals, segment_size,
                )
                chunk_z = None  # z_axis no longer matches

        # --- 2) Spectral context (only if metric needs it) ---
        context: dict = {}
        if needs_spectral:
            chunk_m = chunk_signals.shape[1]
            if chunk_z is not None and len(chunk_z) >= 2:
                spacing = float(np.mean(np.diff(chunk_z)))
                if spacing <= 0:
                    spacing = 1.0
            else:
                spacing = 1.0
            fft_coeffs = np.fft.rfft(chunk_signals, axis=1)
            frequencies = np.fft.rfftfreq(chunk_m, d=spacing)
            amplitude = np.abs(fft_coeffs)
            context["batch_frequencies"] = frequencies
            context["batch_amplitude"] = amplitude

        # --- 3) Envelope (only if requested) ---
        chunk_envelopes: np.ndarray | None = None
        if envelope_method is not None:
            if has_envelope_batch:
                chunk_envelopes = envelope_method.compute_batch(
                    chunk_signals, chunk_z, context,
                )
            else:
                # Fallback: per-signal envelope within chunk
                chunk_envelopes = np.empty_like(chunk_signals)
                for i in range(chunk_n):
                    chunk_envelopes[i] = envelope_method.compute(
                        chunk_signals[i], chunk_z, context,
                    )

        # --- 4) Metric evaluation ---
        if has_batch:
            batch_result = metric.evaluate_batch(
                chunk_signals, chunk_z, chunk_envelopes, context,
            )
            score_flat[start:end] = batch_result.scores
            valid_flat[start:end] = batch_result.valid
            feature_chunks.append(batch_result.features)
        else:
            # Fallback: per-signal evaluation within chunk
            chunk_features: dict[str, list[float]] = {}
            for i in range(chunk_n):
                sig = chunk_signals[i]
                env = chunk_envelopes[i] if chunk_envelopes is not None else None
                # Build per-signal context — always a dict, populated
                # with spectral data only when the metric needs it.
                sig_ctx: dict = {}
                if needs_spectral:
                    from quality_tool.spectral.fft import SpectralResult
                    sig_ctx["spectral_result"] = SpectralResult(
                        frequencies=context["batch_frequencies"],
                        amplitude=context["batch_amplitude"][i],
                    )
                result = metric.evaluate(sig, chunk_z, env, sig_ctx)
                score_flat[start + i] = result.score if result.valid else np.nan
                valid_flat[start + i] = result.valid
                for k, v in result.features.items():
                    if k not in chunk_features:
                        chunk_features[k] = [np.nan] * i
                    chunk_features[k].append(float(v) if result.valid else np.nan)
                # Pad missing keys
                for k in chunk_features:
                    if k not in result.features:
                        chunk_features[k].append(np.nan)
            # Convert to arrays
            feature_chunks.append(
                {k: np.array(v) for k, v in chunk_features.items()}
            )

        chunk_ranges.append((start, end))

    # --- Assemble final result ---
    score_map = score_flat.reshape(h, w)
    valid_map = valid_flat.reshape(h, w)

    # Merge feature maps across chunks.
    feature_maps = _merge_feature_chunks(feature_chunks, chunk_ranges, n_total, h, w)

    # Build evaluation metadata.
    preprocess_names: list[str] = []
    if preprocess is not None:
        preprocess_names = [
            getattr(fn, "__name__", repr(fn)) for fn in preprocess
        ]

    metadata: dict = {
        "metric_name": metric.name,
        "input_policy": input_policy,
        "preprocess": preprocess_names if input_policy != "raw" else [],
        "segment_size": segment_size if input_policy != "raw" else None,
        "envelope_method": (
            envelope_method.name if envelope_method is not None else None
        ),
        "image_shape": (h, w),
    }

    return MetricMapResult(
        metric_name=metric.name,
        score_map=score_map,
        valid_map=valid_map,
        feature_maps=feature_maps,
        metadata=metadata,
    )


def _resolve_batch_preprocess(
    preprocess: Sequence[Callable[[np.ndarray], np.ndarray]] | None,
) -> list[Callable[[np.ndarray], np.ndarray]] | None:
    """Convert per-signal preprocessing callables to batch equivalents.

    Known functions (subtract_baseline, normalize_amplitude) are replaced
    by their batch counterparts.  ``smooth`` is wrapped.  Unknown
    callables are wrapped in a per-row loop.
    """
    if preprocess is None:
        return None

    batch_fns: list[Callable] = []
    for fn in preprocess:
        if fn in _BATCH_PREPROCESS_MAP:
            batch_fns.append(_BATCH_PREPROCESS_MAP[fn])
        elif fn is smooth:
            # smooth has a default window_size=5
            batch_fns.append(smooth_batch)
        else:
            # Generic fallback: apply per-row
            batch_fns.append(_make_row_wrapper(fn))
    return batch_fns


def _make_row_wrapper(
    fn: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a per-signal function to work row-wise on (N, M)."""
    def wrapper(signals: np.ndarray) -> np.ndarray:
        out = np.empty_like(signals)
        for i in range(signals.shape[0]):
            out[i] = fn(signals[i])
        return out
    wrapper.__name__ = getattr(fn, "__name__", repr(fn))
    return wrapper


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

    # Discover all feature keys across chunks.
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
                arr = ch[key]
                flat[start:end] = arr
        feature_maps[key] = flat.reshape(h, w)

    return feature_maps
