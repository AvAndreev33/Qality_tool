# Current iteration

## Iteration name

Third batch metric implementation — envelope metrics with batch-oriented execution

## Goal

Implement the next metric batch using the current architecture:
- signal recipes
- recipe binding
- representation bundles
- shared analysis context
- normalized comparison layer
- GUI metric grouping

This iteration should add the first **envelope metric group** and integrate it cleanly into the backend and GUI.

A key requirement of this iteration is:
**all computations that can be implemented efficiently in batch form should be implemented in batch form.**

---

## Why this iteration matters

The project already has:
- a stable backend architecture for recipe-based preparation and derived-representation reuse
- implemented baseline metrics
- implemented noise metrics
- implemented regularity metrics
- GUI support for grouped metrics
- a comparison layer for per-pixel metric inspection

The next logical step is to add a metric group centered on:
- envelope amplitude
- envelope width
- envelope compactness
- envelope symmetry
- single-peak dominance
- secondary-peak behavior

This group is especially suitable now because:
- envelope support already exists
- representation bundles already support reusable envelope computation
- many envelope metrics naturally share the same envelope-derived intermediates
- this is a good place to strengthen batch-oriented reuse and avoid repeated per-metric work

---

## In scope

### Envelope metric batch implementation

Implement the metrics from `docs/metric_spec_envelope.md`.

Target metrics:

- `envelope_height`
- `envelope_area`
- `envelope_width`
- `envelope_sharpness`
- `envelope_symmetry`
- `single_peakness`
- `main_peak_to_sidelobe_ratio`
- `num_significant_secondary_peaks`

These metrics should be implemented according to the current canonical batch spec, not by ad hoc reinterpretation.

---

### Use the current architecture correctly

The implementation must use the existing architecture properly.

That means:
- use declared signal recipes
- use fixed recipe binding where specified
- use representation bundles and shared analysis context
- reuse the envelope already computed for the relevant recipe
- avoid duplicating envelope-related helper logic inside each metric
- keep metric semantics explicit
- keep native score computation separate from the normalization/comparison layer

This iteration should validate that the current architecture works well for a metric group with strong dependence on a shared derived representation.

---

### Batch-oriented execution requirement

This iteration must explicitly prefer batch-oriented implementation where practical.

Requirements:
- if a computation can be implemented over a batch of signals/envelopes without changing semantics, do so
- if several metrics need the same intermediate envelope-derived structures, compute them once per batch or once per signal batch helper where practical
- avoid unnecessary Python per-signal loops when a vectorized or batch form is straightforward
- keep scalar and batch semantics consistent

This does **not** mean forcing unnatural vectorization everywhere.
But the default engineering choice in this iteration should be:
**batch when practical, scalar fallback only when truly needed.**

---

### Shared analysis-context support for envelope metrics

Add any small missing shared analysis-context fields needed by the envelope metric batch.

Typical examples:
- `epsilon`
- `alpha_main_support`
- `beta_secondary_peak`
- `secondary_peak_min_distance`
- `secondary_peak_min_prominence`

If a required shared parameter is referenced by the batch spec and is not yet present in `AnalysisContext`, add it there rather than hardcoding it inside metric code.

Keep these additions minimal and batch-driven.

---

### Reusable envelope support helpers if needed

Small safe support helpers are allowed if needed for clean implementation of the envelope batch.

Examples:
- half-maximum crossing helper
- main-support mask helper
- secondary-peak detection helper
- symmetry-comparison helper

These helpers should be reusable and backend-side.

Do not build a large new envelope-analysis framework in this iteration.
Only add what is needed for this batch.

---

### Reuse of shared envelope-derived intermediates

Because many envelope metrics share common intermediate notions, the implementation should avoid recomputing them independently in every metric where practical.

Examples of shared per-signal concepts:
- `e_peak`
- `n0 = argmax(e)`
- main-peak support mask
- half-maximum level
- secondary peaks outside main support

If a minimal reusable helper or compact shared intermediate computation improves correctness and performance, it should be used.

Keep this lightweight and clear.

---

### Batch + scalar consistency

Where practical, each metric should support the project’s batch-oriented evaluation path.

Requirements:
- preserve scalar semantics
- keep batch behavior consistent with scalar behavior
- use the current evaluator style rather than introducing a parallel execution path

The point is not only to add formulas, but to add them in the correct project style and in a performance-aware way.

---

### Invalid-case handling

Each metric must handle invalid or unstable cases explicitly.

Examples:
- empty or non-finite envelope
- missing half-maximum crossings
- empty main-support region
- too few samples outside the main support
- unstable secondary-peak detection
- near-zero total envelope mass

Return `valid=False` where appropriate instead of producing misleading scores.

This must follow the current `MetricResult` / batch-result semantics.

---

### GUI integration — metric grouping

Update the GUI metric-selection dialog so that metrics remain shown in visible groups.

For this iteration, it should support at least:
- **Baseline metrics**
- **Noise metrics**
- **Regularity metrics**
- **Envelope metrics**

Expected GUI behavior:
- baseline metrics remain visible first
- noise metrics remain visible after that
- regularity metrics remain visible after that
- then a visible group label or section header appears for `Envelope metrics`
- then the newly added envelope metrics are listed

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

This iteration is specifically about the third real metric batch and clean batch-oriented envelope-metric execution.

---

## File targets

Expected modules to update:

- `src/quality_tool/metrics/base.py` if small metadata additions are needed
- `src/quality_tool/core/analysis_context.py`
- `src/quality_tool/evaluation/evaluator.py` only if small support changes are required
- `src/quality_tool/gui/dialogs/metrics_dialog.py`
- `src/quality_tool/gui/main_window.py` if grouping/display integration needs it

Expected new metric modules to create under:

- `src/quality_tool/metrics/envelope/`

Suggested files:
- `envelope_height.py`
- `envelope_area.py`
- `envelope_width.py`
- `envelope_sharpness.py`
- `envelope_symmetry.py`
- `single_peakness.py`
- `main_peak_to_sidelobe_ratio.py`
- `num_significant_secondary_peaks.py`

Also update metric registration in the appropriate registry location.

Optional small helper modules may be added if truly needed, but keep structure minimal.

---

## Testing expectations

Add targeted tests for:
- each new metric on simple synthetic cases
- invalid-case handling
- scalar/batch consistency where applicable
- shared analysis-context parameter usage where practical
- any reusable envelope helper behavior if introduced
- GUI metric-dialog grouping behavior
- presence of new metrics in the GUI selection flow

Add tests or checks that make it clear batch-oriented execution is actually used where practical.

Keep tests focused and reliable.

---

## Implementation preferences

- implement metrics from the agreed batch spec
- keep the architecture-driven style
- use shared bundles/context/helpers instead of local duplication
- batch-optimize all computations that can be cleanly batch-optimized
- keep scalar fallbacks only where batch form is not practical
- keep metadata additions minimal
- keep GUI grouping simple
- prefer correctness and clear semantics over premature cleverness
- do not overbuild abstractions during this batch

---

## Definition of done

This iteration is complete when:
- the envelope metric batch is implemented
- the metrics are registered and usable
- they work with the current recipe/bundle/context architecture
- envelope-related shared computations are reused cleanly
- computations that can be practically batch-optimized are implemented in batch-oriented form
- invalid cases are handled explicitly
- needed shared analysis-context parameters are centralized
- the GUI metric dialog shows baseline, noise, regularity, and envelope metrics as separate visible groups
- readable metric names are shown in the GUI
- tests exist and pass

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs, including `docs/current_iteration.md` and `docs/metric_spec_envelope.md`
2. summarize the intended envelope-metric batch implementation
3. propose a short implementation plan
4. implement only this iteration
5. add targeted tests
6. summarize created files, modified files, and any limitations