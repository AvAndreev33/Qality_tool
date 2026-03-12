# Current iteration

## Iteration name

Metric interface + spectral support + first baseline metrics

## Goal

Introduce the metric layer of the system together with the first shared spectral representation.

This iteration implements:
- metric interface
- metric registry
- shared FFT / spectrum support
- first baseline metrics:
  - fringe visibility
  - SNR
  - power band ratio
- tests for all of the above

This iteration establishes the first real quality-analysis methods and prepares the project for full-image evaluation in the next step.

---

## Why this iteration matters

The project is intended to evolve into a modular WLI research workbench.

That means metrics should not be implemented as isolated one-off functions.
Before building the full evaluation pipeline, the project needs:

- a common metric interface
- a registry for metrics
- a shared way to compute and reuse spectral representations
- a first small set of meaningful baseline quality criteria

This iteration creates the first real metric layer while keeping future extensibility in mind.

---

## In scope

### Metric interface

Implement the common metric interface.

Location:
- `src/quality_tool/metrics/base.py`

Requirements:
- define a common protocol or lightweight base class
- input:
  - `signal`
  - optional `z_axis`
  - optional `envelope`
  - optional `context`
- output:
  - `MetricResult`

Rules:
- metrics must not modify input arrays in-place
- metrics must not silently apply unrelated preprocessing
- metrics should fail gracefully using `valid=False` when appropriate

---

### Metric registry

Implement a simple metric registry.

Location:
- `src/quality_tool/metrics/registry.py`

Capabilities:
- register metrics by name
- retrieve metric by name
- list available metrics

Keep the registry simple and explicit.

---

### Shared FFT / spectrum support

Implement a shared spectral helper module.

Recommended location:
- `src/quality_tool/spectral/fft.py`

Purpose:
- compute FFT-derived representations used by multiple metrics
- keep FFT logic out of individual metric implementations
- prepare for future reuse by evaluator and visualization code

Requirements:
- provide a simple function or small API to compute spectral representation from a 1D signal
- return a consistent result structure, for example:
  - frequencies
  - complex spectrum
  - magnitude / power spectrum
- keep it simple, readable, and testable

Important architectural note:
- this iteration does not need a full caching system
- but the design should allow future reuse of precomputed FFT results through `context`

Metrics that need spectral information should be able to:
- compute it internally if absent
- or use precomputed spectral data from `context` if available

---

### Baseline metrics

Implement the following baseline quality metrics.

#### 1. Fringe visibility

Implement a signal-quality metric representing fringe visibility.

Location suggestion:
- `src/quality_tool/metrics/baseline/fringe_visibility.py`

Requirements:
- define clearly how fringe visibility is computed
- document the chosen formula in code docstring
- return `MetricResult`

The first version should prefer a simple and interpretable definition.

---

#### 2. SNR

Implement a signal-to-noise ratio metric.

Location suggestion:
- `src/quality_tool/metrics/baseline/snr.py`

Requirements:
- define clearly what is treated as signal and what is treated as noise
- document the chosen approximation in code docstring
- return `MetricResult`

The first version should prefer a simple and stable heuristic rather than a complex estimator.

---

#### 3. Power band ratio

Implement a spectral metric based on power ratio between selected frequency bands.

Location suggestion:
- `src/quality_tool/metrics/baseline/power_band_ratio.py`

Requirements:
- use the shared FFT / spectrum support
- define clearly which band is considered numerator and denominator
- allow future configurability of band limits
- document the chosen formula in code docstring
- return `MetricResult`

This metric should be designed so that future variants can reuse the same spectral helper.

---

### Metric outputs

All metrics must return `MetricResult`.

Fields:
- `score`
- optional `features`
- `valid`
- optional `notes`

Useful intermediate values may be placed into `features` when appropriate.

---

### Tests

Add tests for:
- metric registry behavior
- FFT / spectrum helper behavior
- each baseline metric on small synthetic signals
- correct `MetricResult` output structure
- graceful handling of invalid or ambiguous cases
- use of precomputed spectral data from `context` where applicable

Tests should remain small, synthetic, and deterministic.

---

## Out of scope

Do not implement in this iteration:
- full-image evaluation
- metric map creation
- thresholding
- visualization
- export
- synthetic signals
- experiment configs
- full FFT caching/orchestration across the whole pipeline

A full shared-computation strategy will be handled later in evaluator-level logic.

---

## File targets

Expected modules:

- `src/quality_tool/metrics/base.py`
- `src/quality_tool/metrics/registry.py`

- `src/quality_tool/spectral/__init__.py`
- `src/quality_tool/spectral/fft.py`

- `src/quality_tool/metrics/baseline/fringe_visibility.py`
- `src/quality_tool/metrics/baseline/snr.py`
- `src/quality_tool/metrics/baseline/power_band_ratio.py`

Expected tests:

- `tests/test_metrics/test_registry.py`
- `tests/test_metrics/test_baseline_metrics.py`
- `tests/test_spectral/test_fft.py`

Add `__init__.py` files where needed.

---

## Implementation preferences

- keep formulas explicit and documented
- keep code simple and readable
- prefer clear heuristics over premature sophistication
- separate shared FFT logic from metric implementations
- design metrics so they can later consume precomputed derived data through `context`
- avoid introducing a heavy caching framework in this iteration

---

## Definition of done

This iteration is complete when:
- metric interface exists
- metric registry works
- shared FFT / spectrum helper exists
- fringe visibility metric exists
- SNR metric exists
- power band ratio metric exists
- spectral helper is tested
- baseline metrics are tested
- metrics return `MetricResult`
- code remains aligned with `docs/architecture.md`

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs
2. summarize understanding of this iteration
3. propose a short implementation plan
4. implement only this iteration
5. add tests
6. summarize changes and open follow-up items