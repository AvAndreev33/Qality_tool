# Current iteration

## Iteration name

Metric input policy refinement — raw vs processed signal contract

## Goal

Introduce an explicit metric input policy so that each metric can declare which signal representation it must use.

This iteration should make the pipeline robust for metrics that:
- must use the raw signal,
- may use the processed signal,
- may later require envelope-based inputs or other constrained representations.

The immediate practical case is:
- `fringe_visibility` must be evaluated on the **raw signal only**

This should be implemented as a general architectural refinement, not as a one-off special case.

---

## Why this iteration matters

The project now supports:
- preprocessing
- ROI extraction
- envelope computation
- multi-metric computation
- GUI settings that affect processing
- reuse of computed metric results in a session

This means different metrics may need different effective inputs.

Examples:
- `fringe_visibility` should use the raw signal because detrending / signed values destroy its physical meaning
- other metrics may legitimately use processed signals
- future metrics may require envelope-derived inputs

Without an explicit contract, metric evaluation becomes fragile and difficult to reason about.

This iteration introduces the first clean version of that contract.

---

## In scope

### Metric input policy

Introduce an explicit declared policy for metric input selection.

At minimum, each metric should be able to declare whether it uses:
- `raw`
- `processed`

This policy should live with the metric definition or metric metadata, not in ad hoc GUI conditionals.

Keep this minimal and simple for now.

---

### Immediate policy assignment

For this iteration:
- `fringe_visibility` must be declared as `raw`
- existing processed-signal metrics may remain `processed` if that matches current behavior
- if any current metric is ambiguous, choose the simplest documented policy and state it clearly

Do not overbuild a large policy system yet.

---

### Evaluator refinement

Update the evaluator so that it can correctly choose the effective signal per metric.

Requirements:
- evaluator must have access to the raw signal
- evaluator must be able to produce the processed signal when needed
- evaluator must select the correct effective signal according to the metric input policy
- evaluator must keep this logic explicit and readable

This should work correctly during multi-metric computation.

Example:
- in one compute run, one metric may use raw
- another metric may use processed

---

### Processing pipeline interaction

Clarify how preprocessing / ROI interact with the effective signal.

For this iteration:
- raw-only metrics must ignore preprocessing and ROI settings for their signal input
- processed-signal metrics should continue using the current processing pipeline as already intended
- if envelope is used by a metric, keep existing behavior consistent with the currently selected effective signal path

Keep this as close as possible to the current architecture and behavior.

---

### Computed result reuse / caching behavior

Refine result reuse logic so that it respects metric input policy.

Requirements:
- raw-only metric results should not be invalidated by preprocessing changes that do not affect raw input
- processed-signal metric results should still depend on the relevant processing configuration
- the reuse / cache key logic should remain simple but correct

Do not introduce a large caching framework.
Just make current reuse logic semantically correct.

---

### GUI integration

The GUI must remain thin, but it should correctly reflect the new behavior.

Requirements:
- GUI compute flow should still work with multi-metric computation
- GUI should not decide raw-vs-processed behavior itself
- GUI may display useful information about the effective input mode if simple and clean
- no major GUI redesign is needed in this iteration

If helpful and minimal, expose the effective input mode in:
- info dialog
- status text
- or result/session metadata

But keep this optional if it remains clean.

---

### Documentation and code clarity

Document the new metric-input behavior clearly in code where appropriate.

At minimum, it should be clear:
- why `fringe_visibility` is raw-only
- how evaluator chooses effective signal input
- how reuse behavior differs for raw vs processed metrics

This should reduce future confusion when adding new metrics.

---

## Out of scope

Do not implement in this iteration:
- a large generalized policy engine
- envelope-only / ROI-required policy types unless trivially needed
- new metrics
- GUI redesign
- new preprocessing methods
- CUDA / performance work
- experiment manifests
- synthetic workflows

Keep this iteration narrow and architectural.

---

## File targets

Expected modules to update:

- `src/quality_tool/metrics/base.py`
- `src/quality_tool/metrics/baseline/fringe_visibility.py`
- `src/quality_tool/evaluation/evaluator.py`
- `src/quality_tool/gui/main_window.py` if needed for reuse/session-state correctness
- any small supporting modules that currently hold metric/session metadata if needed

Expected tests to update/add:

- tests covering raw-only metric behavior
- tests covering multi-metric runs with mixed input policies
- tests covering reuse behavior for raw vs processed metrics
- tests confirming preprocessing changes do not incorrectly affect raw-only metric evaluation

Keep tests targeted and lightweight.

---

## Testing expectations

Add tests for:
- `fringe_visibility` always using raw signal
- evaluator correctly mixing raw-only and processed metrics in one compute run
- preprocessing changes not invalidating raw-only results unnecessarily
- processed metrics still depending on the current processing configuration
- GUI/session-level recompute behavior remaining correct where practical

Keep tests reliable and focused on semantics.

---

## Implementation preferences

- keep the solution minimal
- avoid hardcoded GUI-side metric special cases
- keep metric policy close to metric definitions
- keep evaluator logic explicit and readable
- keep reuse logic simple and correct
- prefer a small clear contract now over a complex future-proof abstraction

---

## Definition of done

This iteration is complete when:
- metrics can declare whether they use raw or processed signal input
- `fringe_visibility` is enforced as raw-only
- evaluator respects metric input policy during multi-metric computation
- reuse behavior is correct for raw-only vs processed metrics
- current GUI workflow still works
- the behavior is clearly documented in code
- tests cover the new semantics

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs
2. summarize the intended metric-input-policy refinement
3. propose a short implementation plan
4. implement only this iteration
5. add lightweight targeted tests
6. summarize created files, modified files, and any limitations
