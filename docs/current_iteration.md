# Current iteration

## Iteration name

Evaluator + metric map + thresholding

## Goal

Implement the first full evaluation pipeline for the project.

This iteration should make it possible to:
- take a loaded `SignalSet`
- optionally preprocess each signal
- optionally extract ROI
- optionally compute envelope
- evaluate one metric over the full dataset
- collect results into `MetricMapResult`
- apply thresholding
- obtain a 2D binary mask

This is the first iteration where the system becomes a working quality-analysis pipeline.

---

## Why this iteration matters

The previous iterations implemented:
- data loading
- preprocessing scaffold
- ROI extraction
- envelope support
- spectral support
- baseline metrics

But these pieces are still separate.

This iteration connects them into one coherent evaluation flow.

It is the first real end-to-end pipeline for quality analysis on WLI signals.

---

## In scope

### Evaluator

Implement the main evaluator module.

Location:
- `src/quality_tool/evaluation/evaluator.py`

Responsibilities:
- accept a `SignalSet`
- accept one metric instance
- optionally accept preprocessing settings
- optionally accept ROI settings
- optionally accept envelope method
- iterate over all pixel signals
- evaluate the metric for each signal
- return `MetricMapResult`

Requirements:
- preserve image layout
- produce `score_map` with shape `(H, W)`
- produce `valid_map` with shape `(H, W)`
- collect feature maps where possible
- return metadata describing evaluation settings

The evaluator should be explicit and readable.
Do not overengineer it.

---

### Preprocessing integration

The evaluator must support optional preprocessing before metric evaluation.

For this iteration, support integration with the preprocessing functions already implemented.

The evaluator should make preprocessing order explicit.

Recommended order for this iteration:
1. start from raw signal
2. optionally apply preprocessing
3. optionally apply ROI extraction
4. optionally compute envelope
5. evaluate metric

This order should be documented in code.

---

### ROI integration

The evaluator must support optional ROI extraction.

Requirements:
- if ROI extraction is enabled, apply it before metric evaluation
- if ROI extraction is enabled and envelope is also enabled, compute envelope on the ROI signal that is actually passed to the metric
- make this behavior explicit in metadata / evaluator docstring

---

### Envelope integration

The evaluator must support optional envelope computation.

Requirements:
- envelope method should be optional
- if provided, compute envelope for each evaluated signal
- pass envelope into `metric.evaluate(...)`
- if not provided, pass `None`

Do not add automatic hidden envelope behavior inside the evaluator beyond this.

---

### Shared spectral reuse through context

The evaluator should support simple context-based reuse of derived signal representations.

For this iteration:
- if a metric may use spectral data, the evaluator may precompute FFT-derived data once per signal and pass it through `context`
- keep this simple
- do not add a heavy caching framework

A lightweight pattern is enough, for example:
- build a `context` dict per signal
- include precomputed `spectral_result` when needed

This should remain readable and local.

---

### MetricMapResult assembly

The evaluator must assemble full-image outputs into `MetricMapResult`.

Requirements:
- `score_map.shape == (H, W)`
- `valid_map.shape == (H, W)`
- `feature_maps` should contain `(H, W)` arrays where possible
- invalid metric results must be reflected in `valid_map`
- choose a clear convention for invalid score values and keep it consistent

Document that convention in code.

---

### Thresholding

Implement thresholding support.

Location:
- `src/quality_tool/evaluation/thresholding.py`

Requirements:
- accept a `MetricMapResult`
- accept a scalar threshold
- accept a keep rule such as:
  - `score >= threshold`
  - `score <= threshold`
- return `ThresholdResult`

Behavior:
- output mask must have shape `(H, W)`
- invalid pixels from `valid_map` must be handled explicitly and consistently
- compute simple summary stats, e.g.:
  - number of valid pixels
  - number of kept pixels
  - kept fraction

Keep thresholding logic simple and explicit.

---

### Tests

Add tests for:
- evaluator output shape
- evaluator on a small synthetic `SignalSet`
- evaluator with and without preprocessing
- evaluator with and without ROI
- evaluator with and without envelope
- evaluator with metric that uses spectral context
- proper assembly of `MetricMapResult`
- thresholding with both keep rules
- thresholding behavior with invalid pixels

Tests should stay synthetic, small, and deterministic.

---

## Out of scope

Do not implement in this iteration:
- visualization
- export
- multi-metric comparison
- experiment manifests
- synthetic signals
- benchmark workflows
- automatic optimization of thresholds

Keep this iteration focused on one-metric full-dataset evaluation.

---

## File targets

Expected modules:

- `src/quality_tool/evaluation/evaluator.py`
- `src/quality_tool/evaluation/thresholding.py`

Expected tests:

- `tests/test_evaluation/test_evaluator.py`
- `tests/test_evaluation/test_thresholding.py`

You may add small helper utilities inside `evaluation/` if truly needed, but avoid unnecessary abstraction.

---

## Implementation preferences

- keep the evaluator explicit and easy to read
- prefer a straightforward nested loop over premature optimization
- keep evaluation order explicit
- keep context construction local and understandable
- keep invalid-result handling explicit
- avoid hidden auto-magic behavior

---

## Definition of done

This iteration is complete when:
- evaluator exists
- evaluator can process a full `SignalSet`
- evaluator returns valid `MetricMapResult`
- thresholding exists
- thresholding returns valid `ThresholdResult`
- tests exist and pass
- code remains aligned with `docs/architecture.md`

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs
2. summarize understanding of this iteration
3. propose a short implementation plan
4. implement only this iteration
5. add tests
6. summarize changes and open follow-up items