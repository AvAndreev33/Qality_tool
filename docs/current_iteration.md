# Current iteration

## Iteration name

Signal recipe planning — unified metric input preparation and reuse

## Goal

Replace the current simplistic metric input selection approach with a more explicit and reusable signal-preparation model.

This iteration should introduce a **signal recipe** concept and a lightweight **planner** that determines:
- which signal variants must be prepared for the selected metrics,
- which derived representations must be computed on top of those prepared signals,
- how those prepared results are reused across multiple metrics in one evaluation run.

The goal is to make metric execution semantically correct, reusable, and ready for future metric growth and later CUDA acceleration.

---

## Why this iteration matters

The project now supports:
- multiple metrics in one run,
- preprocessing,
- ROI extraction,
- envelope computation,
- spectral computation,
- result reuse,
- GUI-driven experimentation.

The current `raw` / `processed` distinction is no longer expressive enough.

Different metrics may require:
- the raw signal,
- a specific prepared signal recipe,
- the currently active GUI/experiment recipe,
- derived representations such as envelope or spectrum built on top of that prepared signal.

If this is not made explicit now, future metric growth will lead to:
- duplicated preprocessing,
- duplicated envelope/FFT computation,
- incorrect semantics,
- fragile GUI/backend coupling,
- difficult CUDA integration later.

---

## Core design direction

The architecture should move to the following model:

### 1. Signal recipe
A metric should declare which **signal recipe** it uses.

Examples:
- `raw`
- `roi_only`
- `roi_detrended`
- `active_pipeline`

The important point is:
- `raw` is also a recipe,
- not a special case outside the recipe system.

### 2. Recipe binding
A metric should also declare whether its recipe is:
- **fixed** — metric always uses the same declared recipe
- **active** — metric uses the current active processing pipeline selected in the GUI / session

This preserves both:
- strict physically constrained metrics,
- flexible exploratory metrics.

### 3. Derived representations
Envelope and spectrum must be treated as representations derived from a specific prepared signal recipe.

That means:
- envelope for `raw` and envelope for `roi_detrended` are different objects
- spectrum for `raw` and spectrum for `roi_detrended` are different objects

### 4. Planner
Before evaluation, a planner should:
- collect the selected metrics,
- determine which unique recipes are needed,
- determine which derived representations are needed per recipe,
- prepare them once,
- make them reusable during evaluation.

---

## In scope

### Signal recipe model

Introduce a minimal, explicit signal-recipe abstraction.

Requirements:
- represent raw signal as a recipe
- represent processed signal variants as recipes
- make recipes comparable / reusable
- make recipes suitable for caching/reuse logic

Keep the design minimal and codebase-friendly.
Do not build a large generic pipeline framework.

The recipe model should be strong enough for:
- fixed metric recipes
- active GUI/session recipe
- reuse of prepared signals

---

### Recipe binding model

Introduce a minimal way for metrics to declare how they obtain their signal recipe.

At minimum, support:
- `fixed`
- `active`

Meaning:
- `fixed` = metric always uses its declared recipe
- `active` = metric uses the current active processing pipeline from session/evaluator context

This should replace the need for ad hoc `raw` / `processed` special-casing.

---

### Metric metadata refinement

Update metric-side declarations so that a metric can declare:
- its `signal_recipe`
- its `recipe_binding`
- whether it needs envelope
- whether it needs spectral data

Keep this close to metric definitions / metadata.
Do not move this logic into the GUI.

For this iteration, migrate current metrics to this new scheme.

---

### Planner introduction

Add a lightweight planning layer before batch evaluation.

Responsibilities:
- inspect selected metrics
- resolve effective recipes
- group metrics by recipe
- determine whether envelope is needed per recipe
- determine whether spectral data is needed per recipe
- produce a plan that evaluator can execute efficiently

This planner should remain simple and explicit.

It does not need to be a large framework.
A small helper/module is enough.

---

### Evaluator refactor

Update the evaluator so that it no longer thinks in terms of only `raw` vs `processed`.

Instead, it should:
- receive or construct the execution plan
- prepare each required recipe once per chunk
- compute derived representations once per recipe when needed
- dispatch metrics onto the correct prepared signal bundle

Requirements:
- preserve current metric semantics
- preserve batch evaluation where available
- preserve correctness of thresholding and result assembly
- keep code readable

---

### Reuse / cache semantics refinement

Update reuse logic so that results are keyed by semantically correct execution inputs.

At minimum, reuse must now depend on:
- dataset identity
- metric identity
- effective signal recipe
- derived representation requirements where relevant

Important behavior:
- metrics with fixed raw recipe should remain reusable across GUI preprocessing changes
- metrics with active recipe should depend on the active processing pipeline configuration

Keep this simple and correct.
Do not build a heavyweight cache system.

---

### GUI integration

The GUI must remain thin.

Requirements:
- GUI should continue selecting active processing pipeline as it does now
- GUI should not decide metric recipe semantics itself
- current multi-metric compute flow should continue to work
- current session-state model should remain compatible
- if needed, GUI may pass active processing settings into the evaluator/planner, but not interpret them semantically

No major GUI redesign is needed in this iteration.

---

### Documentation and clarity

Document the new semantics clearly in code.

It should be easy to understand:
- what a signal recipe is
- what fixed vs active binding means
- how derived representations depend on recipe
- how reuse works under the new model

This is important because future metric batches will rely on this structure.

---

## Out of scope

Do not implement in this iteration:
- a broad experiment-manifest system
- a user-facing pipeline builder
- new GUI workflows
- CUDA backend
- major performance optimization beyond what naturally falls out of recipe reuse
- new metrics unrelated to validating the new architecture
- synthetic workflows

Keep this iteration architectural and focused.

---

## File targets

Expected modules to update:

- `src/quality_tool/metrics/base.py`
- baseline metric modules as needed
- `src/quality_tool/evaluation/evaluator.py`

Likely new module(s):
- a small planning/recipe module under `evaluation/` or another appropriate backend location

Possible GUI/session update only if needed:
- `src/quality_tool/gui/main_window.py`

Add only what is necessary.
Keep structure minimal.

---

## Testing expectations

Add targeted tests for:
- recipe equality / reuse behavior
- fixed recipe vs active recipe behavior
- mixed metric runs using different recipes
- envelope/spectral reuse per recipe
- evaluator correctness under multiple recipe groups
- GUI/session recompute semantics where practical
- no regression of current behavior

Tests should focus on semantics and reuse, not on GUI cosmetics.

---

## Implementation preferences

- keep the model minimal and explicit
- treat `raw` as a recipe, not a separate special case
- keep recipe semantics backend-side
- keep metric declarations close to metric definitions
- keep planner lightweight
- keep evaluator readable
- prefer correctness and reuse clarity over abstract generality
- avoid introducing a large framework

---

## Definition of done

This iteration is complete when:
- metrics declare signal recipes instead of relying on the old raw/processed split
- fixed vs active recipe binding exists
- evaluator executes through a lightweight recipe plan
- prepared signals are reused per recipe
- envelope/spectral representations are reused per recipe
- current multi-metric workflow still works
- reuse semantics become more correct and explicit
- the system is better prepared for future metric expansion and later CUDA backend work

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs
2. summarize the intended signal-recipe refinement
3. propose a short implementation plan
4. implement only this iteration
5. add targeted tests
6. summarize created files, modified files, and any limitations