# Current iteration

## Iteration name

First batch metric implementation — noise metrics + metric grouping in GUI

## Goal

Implement the first larger batch of new metrics using the new architecture:
- signal recipes
- recipe binding
- representation bundles
- shared analysis context
- refined spectral layer

This iteration should add the first **noise metric group** and integrate it cleanly into the GUI.

It should also make the GUI metric-selection dialog show metrics grouped by class/category, so that already implemented baseline metrics and the new noise metrics are visually separated.

---

## Why this iteration matters

The backend architecture is now ready for larger metric batches:
- recipe-based signal preparation exists
- derived representations can be reused per recipe
- analysis context exists
- spectral support is stronger
- multi-metric compute already works

The next logical step is to validate this architecture by implementing the first real metric batch that depends on:
- fixed prepared recipes
- spectral reuse
- envelope reuse
- shared constants from analysis context

The noise metric group is the best first test of this new architecture.

---

## In scope

### Noise metric batch implementation

Implement the metrics from `docs/metric_spec_noise.md`.

Target metrics:

- `spectral_snr`
- `local_snr`
- `envelope_peak_to_background_ratio`
- `noise_floor_level`
- `residual_noise_energy`
- `high_frequency_noise_level`
- `low_frequency_drift_level`

These metrics should be implemented according to the current canonical metric-batch spec, not by ad hoc reinterpretation.

---

### Use the new architecture correctly

The implementation must use the current backend architecture properly.

That means:
- use declared signal recipes
- use fixed recipe binding where specified
- use representation requirements instead of recomputing things inside each metric
- use `AnalysisContext` for shared constants/defaults
- reuse envelope/spectral representations from bundles when available
- do not duplicate FFT/envelope logic inside each metric unless truly required by a metric that needs a special form not yet provided by the shared layer

This iteration is partly a correctness test of the architecture.

---

### Shared analysis-context support for noise metrics

Add any small missing shared analysis-context fields needed by the noise metric batch.

Typical examples:
- `epsilon`
- `band_half_width_bins`
- `drift_window`

If a metric batch spec refers to a shared parameter that is not yet present in `AnalysisContext`, add it there rather than hardcoding the value inside the metric.

Keep these additions minimal and batch-driven.

---

### Small spectral/support refinements if needed

Small safe refinements are allowed if they are required for correct implementation of the noise metrics.

Examples:
- a helper for one-sided power-band selection
- a helper for symmetric complex-band reconstruction
- a small utility for excluding DC consistently

Do not redesign the spectral layer broadly in this iteration.
Only add what is needed for this metric batch.

---

### Metric metadata refinement if needed

If some small additions to metric metadata are needed for clean integration, they are allowed.

Examples:
- metric category/class label
- display name
- score direction metadata if not already explicit
- representation-needs declaration cleanup

Keep this minimal and useful.

Do not introduce a large new metadata framework.

---

### Batch + scalar consistency

Where practical, each metric should support the project’s batch-oriented evaluation path.

Requirements:
- implement behavior consistent with the current evaluator architecture
- preserve scalar semantics
- ensure batch behavior matches scalar behavior where both exist

The point is not only to add formulas, but to add them in the correct project style.

---

### Invalid-case handling

Each metric must handle invalid or unstable cases explicitly.

Examples:
- empty usable band
- unstable denominator
- too few samples outside peak support
- invalid drift estimate
- no stable carrier band

Return `valid=False` where appropriate instead of silently producing misleading values.

This must follow the current `MetricResult` / batch result semantics.

---

### GUI integration — metric grouping

Update the GUI metric-selection dialog so that metrics are shown grouped by category/class.

For this iteration, it is enough to support at least:
- **Baseline metrics**
- **Noise metrics**

Expected GUI behavior:
- already implemented baseline metrics remain visible first
- then a visible group label or section header appears for `Noise metrics`
- then the newly added metrics are listed

Keep this simple and readable.
It does not need to become a complex tree view unless that is naturally clean.

---

### GUI metric naming

The new metrics should appear in the GUI with readable names.

Requirements:
- keep internal metric identifiers stable
- allow a human-readable display label if needed
- do not expose confusing raw Python class names to the user

If a minimal `display_name` field is useful, it may be added.

Keep it lightweight.

---

## Out of scope

Do not implement in this iteration:
- other metric groups
- score normalization to `[0, 1]`
- GUI histogram redesign
- preprocessing/recipe GUI redesign
- CUDA backend
- broad spectral redesign beyond what is minimally needed
- benchmark workflows
- synthetic workflows

This iteration is specifically about the first real metric batch and clean GUI grouping.

---

## File targets

Expected modules to update:

- `src/quality_tool/metrics/base.py` if small metadata additions are needed
- `src/quality_tool/core/analysis_context.py`
- `src/quality_tool/evaluation/evaluator.py` only if small support changes are required
- `src/quality_tool/spectral/fft.py` only if small support changes are required
- `src/quality_tool/gui/dialogs/metrics_dialog.py`
- `src/quality_tool/gui/main_window.py` if metric grouping/display-name integration needs it

Expected new metric modules to create under:

- `src/quality_tool/metrics/noise/`  
  or another project-consistent location for the new batch

Suggested files:
- `spectral_snr.py`
- `local_snr.py`
- `envelope_peak_to_background_ratio.py`
- `noise_floor_level.py`
- `residual_noise_energy.py`
- `high_frequency_noise_level.py`
- `low_frequency_drift_level.py`

Also update metric registration in the appropriate registry location.

---

## Testing expectations

Add targeted tests for:
- each new metric on simple synthetic cases
- invalid-case handling
- scalar/batch consistency where applicable
- representation reuse correctness where practical
- analysis-context parameter usage where practical
- GUI metric-dialog grouping behavior
- presence of new metrics in the GUI selection flow

Keep tests focused and reliable.

---

## Implementation preferences

- implement metrics from the agreed batch spec
- keep the architecture-driven style
- use shared bundles and shared context instead of local duplication
- keep metadata additions minimal
- keep GUI grouping simple
- prefer correctness and clear semantics over premature cleverness
- do not overbuild abstractions during this batch

---

## Definition of done

This iteration is complete when:
- the noise metric batch is implemented
- the metrics are registered and usable
- they work with the current recipe/bundle/context architecture
- invalid cases are handled explicitly
- needed shared analysis-context parameters are centralized
- the GUI metric dialog shows baseline metrics and noise metrics as separate visible groups
- readable metric names are shown in the GUI
- tests exist and pass

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs, including `docs/current_iteration.md` and `docs/metric_spec_noise.md`
2. summarize the intended noise-metric batch implementation
3. propose a short implementation plan
4. implement only this iteration
5. add targeted tests
6. summarize created files, modified files, and any limitations