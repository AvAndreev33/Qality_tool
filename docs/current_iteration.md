# Current iteration

## Iteration name

Metadata-aware analysis context + stricter derived-representation reuse per recipe

## Goal

Refine the backend so that:
- `AnalysisContext` is no longer only a container of fixed defaults
- key analysis parameters are resolved from dataset metadata where appropriate
- shared derived representations such as envelope and spectrum are computed once per recipe and reused consistently across metrics

This iteration should improve correctness, consistency, and computational efficiency before the next metric batches are implemented.

---

## Why this iteration matters

The project already supports:
- metadata parsing into normalized `SignalSet.metadata`
- signal recipes
- recipe planning
- representation bundles
- envelope support
- spectral support
- metric groups that increasingly share the same prepared signal

The next upcoming metric groups depend more strongly on:
- oversampling-aware defaults
- metadata-derived analysis parameters
- reuse of envelope and spectral representations across multiple metrics using the same recipe

Without this refinement, the system risks:
- scattering metadata-dependent heuristics across metrics
- inconsistent band/segment/period assumptions
- repeated envelope/FFT computation for the same recipe
- avoidable performance loss before future CUDA work

---

## Core design direction

The architecture should move toward:

### 1. Metadata-aware analysis context
`AnalysisContext` should be built from:
- stable project defaults
- metadata-derived resolved values
- later optional overrides

### 2. Centralized oversampling-aware scaling
For the current project, oversampling scaling is a canonical rule and should be applied consistently in one place.

### 3. Stricter per-recipe derived-representation reuse
If multiple selected metrics use the same signal recipe and need envelope and/or spectrum:
- prepared signal should be computed once
- envelope should be computed once for that recipe
- spectrum should be computed once for that recipe

Metrics must consume these shared representations rather than recomputing them locally.

---

## In scope

### Metadata-aware `AnalysisContext`

Refine the current `AnalysisContext` so that it supports both:
- fixed project defaults
- resolved values derived from dataset metadata

At minimum, `AnalysisContext` should now include:
- `band_half_width_bins`
- `default_segment_size`
- `expected_period_samples`
- `wavelength_nm`
- `coherence_length_nm`

The context must remain backend-side and must not depend on GUI logic.

---

### Canonical oversampling scaling rule

Implement the agreed current project rule for oversampling-aware scaling.

For the current project:

- if `oversampling_factor` is missing, `NaN`, or equal to `1`:
  - `band_half_width_bins = 5`
  - `default_segment_size = 128`
  - `expected_period_samples = 4`

- if `oversampling_factor > 1`:
  - `band_half_width_bins = 5 * oversampling_factor`
  - `default_segment_size = 128 * oversampling_factor`
  - `expected_period_samples = 4 * oversampling_factor`

This rule should be implemented in one centralized backend location.
Do not duplicate it across metrics or GUI code.

If integer conversion rules are needed, define them explicitly and keep them consistent.

---

### Central builder for `AnalysisContext`

Add a backend-side way to build the effective analysis context from:
- a `SignalSet`
- project defaults
- optional future overrides

This builder should:
- read normalized metadata from `SignalSet.metadata`
- apply the oversampling scaling rule
- populate wavelength and coherence-length fields when available
- fall back cleanly when metadata is missing

Keep this builder minimal and explicit.

---

### Preserve explicit defaults

Even after metadata-aware refinement, the system must still have clear project defaults.

That means:
- default values should remain visible and centralized
- metadata-derived values should be resolved from those defaults, not replace the idea of defaults completely

This keeps the system understandable and testable.

---

### Stricter reuse of envelope per recipe

Strengthen the current representation-bundle/evaluator behavior so that envelope reuse is guaranteed per recipe.

Requirements:
- if multiple metrics in the same evaluation run use the same effective recipe and require envelope
- the envelope for that recipe must be computed once per chunk and reused for all those metrics

This should be explicit in evaluator/bundle logic, not accidental.

Metrics should not recompute the same envelope locally unless they require a special envelope form not provided by the shared layer.

---

### Stricter reuse of spectrum per recipe

Strengthen the current representation-bundle/evaluator behavior so that spectrum reuse is guaranteed per recipe.

Requirements:
- if multiple metrics in the same evaluation run use the same effective recipe and require spectrum
- the spectrum for that recipe must be computed once per chunk and reused for all those metrics

This includes whichever spectral forms are already supported by the current refined spectral layer.

Metrics should not recompute the same FFT locally unless they require a very special form not yet provided by the shared layer.

---

### Planner/evaluator consistency

Update planner/evaluator behavior if needed so that:
- recipe grouping remains correct
- representation requirements remain correct
- `AnalysisContext` is passed in resolved form
- envelope/spectrum reuse per recipe is explicit and reliable

Keep this readable.
Do not redesign the whole evaluator.

---

### Metric implementation discipline

Ensure that the current and future metric implementations use:
- shared `AnalysisContext`
- shared bundle-provided envelope
- shared bundle-provided spectrum

This iteration may include small safe cleanup where metrics still rely on local constants or locally repeated derived computations.

Keep such cleanup minimal and architecture-driven.

---

## Out of scope

Do not implement in this iteration:
- new metric groups
- GUI controls for editing analysis-context parameters
- metadata-derived expected-band models beyond the agreed oversampling rule
- normalization redesign
- CUDA backend
- broad evaluator rewrite
- advanced caching framework
- experiment manifests

Keep this iteration focused on context resolution and reuse semantics.

---

## File targets

Expected modules to update:

- `src/quality_tool/core/analysis_context.py`
- `src/quality_tool/evaluation/evaluator.py`
- `src/quality_tool/evaluation/bundle.py`
- `src/quality_tool/evaluation/planner.py` only if needed
- metric modules only if small cleanup is needed to stop local recomputation or local constants

Possible new small helper module if useful:
- a builder/resolver for analysis context from `SignalSet`

Keep structure minimal.

---

## Testing expectations

Add targeted tests for:
- oversampling-aware scaling rule
- fallback behavior when metadata is missing or `NaN`
- correct propagation of `wavelength_nm` and `coherence_length_nm`
- central `AnalysisContext` builder behavior
- envelope reuse per recipe
- spectrum reuse per recipe
- evaluator correctness under multiple metrics sharing the same recipe
- no regression of current metric behavior

Keep tests focused and reliable.

---

## Implementation preferences

- keep the design minimal and explicit
- centralize all oversampling-aware scaling
- keep metadata-derived context backend-side
- preserve clear defaults
- make envelope/spectrum reuse per recipe explicit
- avoid local metric-specific recomputation where the shared bundle already provides the needed representation
- keep evaluator readable
- do not overbuild a context/override system yet

---

## Definition of done

This iteration is complete when:
- `AnalysisContext` can be built from dataset metadata and defaults
- the agreed oversampling rule is implemented centrally
- wavelength and coherence length are available in the resolved context
- envelope reuse per recipe is explicit and reliable
- spectrum reuse per recipe is explicit and reliable
- current metrics continue to work
- the system is better prepared for further metric batches and future performance work

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs
2. summarize the intended context/reuse refinement
3. propose a short implementation plan
4. implement only this iteration
5. add targeted tests
6. summarize created files, modified files, and any limitations