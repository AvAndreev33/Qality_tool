"""Full-image metric evaluator for Quality_tool.

Runs a single metric over every pixel signal in a :class:`SignalSet` and
assembles the results into a :class:`MetricMapResult`.

Metric input policy
-------------------
Each metric declares an ``input_policy`` attribute:

* ``"raw"`` — the metric receives the original raw signal from the
  dataset.  Preprocessing and ROI extraction are **skipped**.
* ``"processed"`` — the metric receives the signal after the currently
  configured preprocessing chain and optional ROI extraction.

The evaluator reads ``metric.input_policy`` and selects the effective
signal accordingly.  This keeps the decision close to the metric
definition and away from the GUI or caller code.

Evaluation order per pixel (processed metrics)
-----------------------------------------------
1. Extract 1-D signal from ``signal_set.signals[row, col, :]``
2. Apply preprocessing functions **in the order given**
3. Optionally extract ROI (``segment_size``)
4. Build ``context`` dict — precompute :class:`SpectralResult`
5. Optionally compute envelope
6. Call ``metric.evaluate(signal, z_axis, envelope, context)``

For raw-only metrics steps 2 and 3 are skipped; the raw signal is used
directly from step 1.

Invalid-score convention
------------------------
When ``MetricResult.valid`` is ``False``:

* ``score_map[r, c]`` is set to ``np.nan``
* ``valid_map[r, c]`` is set to ``False``

``np.nan`` is chosen because it propagates through arithmetic, is
trivially detectable with :func:`numpy.isnan`, and cannot be confused
with a legitimate score of ``0.0``.

Feature-map assembly
--------------------
The evaluator collects the **union** of all feature keys seen across
all pixels.  Each key becomes an ``(H, W)`` array in
``MetricMapResult.feature_maps``.  Missing or invalid entries are filled
with ``np.nan``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from quality_tool.core.models import MetricMapResult, MetricResult, SignalSet
from quality_tool.envelope.base import BaseEnvelopeMethod
from quality_tool.metrics.base import BaseMetric
from quality_tool.preprocessing.roi import extract_roi
from quality_tool.spectral.fft import compute_spectrum


def evaluate_metric_map(
    signal_set: SignalSet,
    metric: BaseMetric,
    *,
    preprocess: Sequence[Callable[[np.ndarray], np.ndarray]] | None = None,
    segment_size: int | None = None,
    envelope_method: BaseEnvelopeMethod | None = None,
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

    Returns
    -------
    MetricMapResult
        Aggregated per-pixel scores, validity flags, feature maps, and
        evaluation metadata.
    """
    h, w, _ = signal_set.signals.shape
    z_axis = signal_set.z_axis

    # Determine effective signal path from the metric's declared policy.
    # Metrics with input_policy="raw" receive the unmodified raw signal;
    # metrics with input_policy="processed" go through the full
    # preprocessing + ROI pipeline.
    use_raw = getattr(metric, "input_policy", "processed") == "raw"

    # Collect per-pixel results in a flat list; reshape later.
    results: list[MetricResult] = []

    for row in range(h):
        for col in range(w):
            raw_signal = signal_set.signals[row, col, :].copy()

            if use_raw:
                # Raw-only metric: skip preprocessing and ROI.
                signal = raw_signal
                sig_z_axis: np.ndarray | None = z_axis
            else:
                # Processed metric: apply full pipeline.
                signal = raw_signal

                # 1) Preprocessing
                if preprocess is not None:
                    for fn in preprocess:
                        signal = fn(signal)

                # 2) ROI extraction
                sig_z_axis = z_axis
                if segment_size is not None:
                    signal = extract_roi(signal, segment_size)
                    # z_axis no longer matches the cropped signal length;
                    # pass None so downstream code uses index-based spacing.
                    sig_z_axis = None

            # 3) Context — precompute spectral data on the effective signal
            spectral_result = compute_spectrum(signal, sig_z_axis)
            context: dict = {"spectral_result": spectral_result}

            # 4) Envelope (computed on the effective signal)
            envelope: np.ndarray | None = None
            if envelope_method is not None:
                envelope = envelope_method.compute(signal, sig_z_axis, context)

            # 5) Metric evaluation
            result = metric.evaluate(signal, sig_z_axis, envelope, context)
            results.append(result)

    return _assemble_map_result(results, h, w, metric, preprocess,
                                segment_size, envelope_method)


def _assemble_map_result(
    results: list[MetricResult],
    h: int,
    w: int,
    metric: BaseMetric,
    preprocess: Sequence[Callable[[np.ndarray], np.ndarray]] | None,
    segment_size: int | None,
    envelope_method: BaseEnvelopeMethod | None,
) -> MetricMapResult:
    """Build a :class:`MetricMapResult` from flat per-pixel results."""
    score_map = np.full((h, w), np.nan, dtype=float)
    valid_map = np.zeros((h, w), dtype=bool)

    # Discover the union of all feature keys.
    all_feature_keys: set[str] = set()
    for r in results:
        all_feature_keys.update(r.features.keys())

    # Pre-allocate feature-map arrays filled with NaN.
    feature_maps: dict[str, np.ndarray] = {
        key: np.full((h, w), np.nan, dtype=float) for key in all_feature_keys
    }

    for idx, result in enumerate(results):
        row = idx // w
        col = idx % w

        valid_map[row, col] = result.valid

        if result.valid:
            score_map[row, col] = result.score
            for key, value in result.features.items():
                feature_maps[key][row, col] = value
        # Invalid pixels keep np.nan in score_map and feature_maps.

    # Build evaluation metadata.
    preprocess_names: list[str] = []
    if preprocess is not None:
        preprocess_names = [
            getattr(fn, "__name__", repr(fn)) for fn in preprocess
        ]

    input_policy = getattr(metric, "input_policy", "processed")

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
