# Current iteration

## Iteration name

Second batch metric implementation — regularity metrics

## Goal

Implement the next metric batch using the current architecture:
- signal recipes
- recipe binding
- representation bundles
- shared analysis context
- normalized comparison layer
- GUI metric grouping

This iteration should add the first **regularity metric group** and integrate it cleanly into the backend and GUI.

---

## Why this iteration matters

The project already has:
- a stable backend architecture for recipe-based preparation and derived-representation reuse
- the first implemented batch of noise metrics
- GUI support for grouped metrics
- a comparison layer for per-pixel metric inspection

The next logical step is to add a metric group that tests a different type of signal-quality reasoning:
- periodicity
- cycle-to-cycle stability
- extrema spacing stability
- zero-crossing stability

The regularity group is a good next step because it exercises:
- recipe reuse
- shared analysis-context parameters
- reusable local-structure helpers
- explicit invalid-case handling

---

## In scope

### Regularity metric batch implementation

Implement the metrics from `docs/metric_spec_regulary.md`.

Target metrics:

- `autocorrelation_peak_strength`
- `local_oscillation_regularity`
- `jitter_of_extrema`
- `zero_crossing_stability`

These metrics should be implemented according to the current canonical batch spec, not by ad hoc reinterpretation.

---

### Use the current architecture correctly

The implementation must use the existing architecture properly.

That means:
- use declared signal recipes
- use fixed recipe binding where specified
- use representation bundles and shared analysis context
- avoid duplicating reusable helper logic inside each metric
- keep metric semantics explicit
- keep native score computation separate from the normalization/comparison layer

This iteration should validate that the current architecture works well not only for noise/spectral-envelope metrics but also for local-structure/periodicity metrics.

---

### Shared analysis-context support for regularity metrics

Add any small missing shared analysis-context fields needed by the regularity batch.

Typical examples:
- `epsilon`
- `expected_period_samples`
- `smoothing_window`
- `peak_min_distance_fraction`
- `period_search_tolerance_fraction`
- `cycle_resample_length`

If a required shared parameter is referenced by the batch spec and is not yet present in `AnalysisContext`, add it there rather than hardcoding it inside metric code.

Keep these additions minimal and batch-driven.

---

### Small reusable support helpers if needed

Small safe support helpers are allowed if they are needed for clean implementation of the regularity batch.

Examples:
- normalized autocorrelation helper
- extrema detection helper
- cycle resampling helper
- zero-crossing interpolation helper

These helpers should be reusable and backend-side.

Do not build a large signal-analysis framework in this iteration.
Only add what is needed for this batch.

---

### Representation requirements

Use the new representation/dependency model correctly.

For this batch:
- `autocorrelation_peak_strength` should use the current prepared signal directly
- `local_oscillation_regularity` and `jitter_of_extrema` may need reusable extrema-related support
- `zero_crossing_stability` should use the prepared signal directly, with clear invalid-case handling

If a small reusable representation/helper layer for extrema is useful and minimal, it may be introduced.
Keep it lightweight and compatible with the current representation-bundle direction.

---

### Batch + scalar consistency

Where practical, each metric should support the project’s batch-oriented evaluation path.

Requirements:
- preserve scalar semantics
- keep batch behavior consistent with scalar behavior
- use the current evaluator style rather than introducing a parallel execution path

The goal is not just to add formulas, but to add them in the correct project style.

---

### Invalid-case handling

Each metric must handle invalid or unstable cases explicitly.

Examples:
- expected-period search range empty
- too few extrema
- too few valid cycles
- no robust crossings
- unstable denominator
- invalid local similarity statistics

Return `valid=False` where appropriate instead of producing misleading scores.

This must follow the current `MetricResult` / batch-result semantics.

---

### GUI integration — metric grouping

Update the GUI metric-selection dialog so that metrics remain shown in visible groups.

For this iteration, it should support at least:
- **Baseline metrics**
- **Noise metrics**
- **Regularity metrics**

Expected GUI behavior:
- baseline metrics remain visible first
- noise metrics remain visible after that
- then a visible group label or section header appears for `Regularity metrics`
- then the newly added regularity metrics are listed

Keep this simple and readable.

---

### GUI metric naming

The new metrics should appear in the GUI with readable names.

Requirements:
- keep internal metric identifiers stable
- allow or use human-readable display labels if needed
- do not expose confusing raw Python class names to the user

Keep this lightweight and consistent with the current GUI grouping behavior.

---

## Out of scope

Do not implement in this iteration:
- other metric groups
- score normalization redesign
- GUI comparison redesign
- new histogram features
- CUDA backend
- benchmark workflows
- synthetic workflows
- broad new representation families beyond what is minimally needed for this batch

This iteration is specifically about the second real metric batch and clean GUI grouping continuity.

---

## File targets

Expected modules to update:

- `src/quality_tool/metrics/base.py` if small metadata additions are needed
- `src/quality_tool/core/analysis_context.py`
- `src/quality_tool/evaluation/evaluator.py` only if small support changes are required
- `src/quality_tool/gui/dialogs/metrics_dialog.py`
- `src/quality_tool/gui/main_window.py` if grouping/display integration needs it

Expected new metric modules to create under:

- `src/quality_tool/metrics/regularity/`

Suggested files:
- `autocorrelation_peak_strength.py`
- `local_oscillation_regularity.py`
- `jitter_of_extrema.py`
- `zero_crossing_stability.py`

Also update metric registration in the appropriate registry location.

Optional small helper modules may be added if truly needed, but keep structure minimal.

---

## Testing expectations

Add targeted tests for:
- each new metric on simple synthetic cases
- invalid-case handling
- scalar/batch consistency where applicable
- shared analysis-context parameter usage where practical
- any reusable extrema / crossing helper behavior if introduced
- GUI metric-dialog grouping behavior
- presence of new metrics in the GUI selection flow

Keep tests focused and reliable.

---

## Implementation preferences

- implement metrics from the agreed batch spec
- keep the architecture-driven style
- use shared bundles/context/helpers instead of local duplication
- keep metadata additions minimal
- keep GUI grouping simple
- prefer correctness and clear semantics over premature cleverness
- do not overbuild abstractions during this batch

---

## Definition of done

This iteration is complete when:
- the regularity metric batch is implemented
- the metrics are registered and usable
- they work with the current recipe/bundle/context architecture
- invalid cases are handled explicitly
- needed shared analysis-context parameters are centralized
- the GUI metric dialog shows baseline, noise, and regularity metrics as separate visible groups
- readable metric names are shown in the GUI
- tests exist and pass

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs, including `docs/current_iteration.md` and `docs/metric_spec_regulary.md`
2. summarize the intended regularity-metric batch implementation
3. propose a short implementation plan
4. implement only this iteration
5. add targeted tests
6. summarize created files, modified files, and any limitations