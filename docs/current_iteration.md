# Current iteration

## Iteration name

Metric comparison layer — normalized scores + spectral band overlay + per-pixel metric inspector

## Goal

Add a comparison-focused layer on top of the already implemented metrics and GUI workflow.

This iteration should introduce:
- a normalized score layer for comparing different metrics in a common `0..1, higher is better` form
- useful-band visualization in the spectrum display
- a per-pixel metric inspector opened from the signal tools panel
- two comparison views for the selected pixel:
  - a table with all computed metrics
  - a horizontal normalized bar chart

The goal is to make already implemented metrics much easier to interpret, compare, and debug before implementing the next metric groups.

---

## Why this iteration matters

The project now has:
- a working backend with multiple metric groups beginning to grow
- a usable GUI with map viewing, signal inspection, thresholding, and histogram support
- the first batch of new metrics already implemented

The next practical need is not another architectural rewrite, but a stronger comparison layer.

To evaluate which quality criteria are actually useful, the user needs:
- a common comparison representation
- visual feedback for spectral band selection
- a convenient way to inspect all metric values for the currently selected pixel

This iteration improves interpretability and comparison quality before the next metric batches.

---

## In scope

### Normalized comparison layer

Introduce a separate normalization/comparison layer for metric scores.

Requirements:
- normalization must be a separate layer on top of native metric scores
- native metric values must remain preserved and accessible
- normalized values must be represented as:
  - range `[0, 1]`
  - larger = better
- metrics that are already naturally in `[0,1]` and higher-better should remain unchanged
- metrics with other semantics should be mapped according to an explicit normalization policy

This iteration should not redesign metric math itself.
It should only add a comparison representation.

---

### Metric score semantics metadata

Add minimal metadata needed for normalization.

At minimum, each metric should be able to declare:
- score direction:
  - `higher_better`
  - `lower_better`
- score scale type, for example:
  - `bounded_01`
  - `positive_unbounded`
  - `ratio_like`
  - `log_ratio`
  - another small minimal set if needed

Keep this metadata lightweight and close to metric definitions.

Do not implement normalization through hardcoded metric-name checks inside the GUI.

---

### Normalization policies

Implement a minimal normalization system that can map native scores to normalized comparison scores.

Requirements:
- keep the implementation simple and explicit
- use metric score semantics metadata
- avoid overcomplicated statistical normalization
- keep room for future refinement

For this iteration, a rough but stable normalization is acceptable.

Important:
- invalid native metric values should remain invalid in normalized form
- normalization must not overwrite native metric results

---

### Spectrum useful-band visualization

Extend the spectrum display in the signal inspector so the user can see the useful spectral band that was used by the corresponding spectral metric logic.

Requirements:
- show the spectrum as already supported
- visually mark:
  - dominant carrier bin / selected peak if available
  - useful band around that carrier
- the band should correspond to the current shared band-width logic / analysis context
- the display should make it easy to understand what region is treated as “signal” and what is treated as outside-band content

A shaded highlighted interval is preferred over barely visible markers if that remains simple.

This iteration is about visualizing the current band selection logic, not redesigning band-selection algorithms.

---

### Signal tools panel integration

Use the existing signal tools panel.

Add a button in the signal tools panel for the selected pixel, for example:
- `Pixel metrics`

This button should open metric-comparison windows for the currently selected pixel.

Keep this integration simple and aligned with the existing signal tools panel workflow.

---

### Per-pixel metric inspector — table window

Implement a window showing all currently computed metrics for the selected pixel in tabular form.

Requirements:
- it should use the currently selected pixel from the main viewer
- it should include all metrics already computed in the current session
- it should show at least:
  - metric name
  - metric group/class if available
  - native score
  - normalized score
  - valid / invalid state
- it should behave as a snapshot for the currently selected pixel when opened, unless a clearly cleaner live-updating design is already natural in the current codebase

Keep the table readable and simple.

---

### Per-pixel metric inspector — normalized bar chart window

Implement a second comparison window for the selected pixel.

This window should display the normalized scores of all currently computed metrics as a horizontal bar chart.

Requirements:
- one row per metric
- metric name on the left
- normalized score represented as a horizontal bar from `0` to `1`
- invalid metrics should be handled explicitly and clearly
- the display should make visual comparison fast and intuitive

This chart should be based on normalized scores, not native scores.

---

### Relationship between the two pixel-inspector windows

The two windows should complement each other:

- **table window** → exact values
- **horizontal bar chart window** → fast visual comparison

Both should use the same selected pixel and the same currently available computed metrics.

Keep the implementation simple and consistent.

---

### GUI workflow preservation

The existing workflow must remain stable:
- loading datasets
- computing metrics
- switching maps
- thresholding
- histogram view
- signal inspection

This iteration adds a comparison layer; it must not destabilize the current GUI.

---

## Out of scope

Do not implement in this iteration:
- new metric groups
- score normalization based on dataset-wide learned statistics
- radar/spider charts
- scatter plots between metrics
- automatic “best metric” recommendation
- normalization parameter editing in the GUI
- CUDA backend
- large GUI redesign
- advanced session persistence

Keep the iteration focused on comparison usability.

---

## File targets

Expected modules to update:

- `src/quality_tool/metrics/base.py` if score-semantics metadata is added there
- `src/quality_tool/gui/main_window.py`
- `src/quality_tool/gui/widgets/signal_inspector.py`
- metric modules if minimal score-semantics metadata must be declared there

Expected new modules to create:

- a small normalization/comparison helper module in backend or GUI-support layer
- `src/quality_tool/gui/windows/pixel_metrics_table_window.py`
- `src/quality_tool/gui/windows/pixel_metrics_chart_window.py`

Optional small helper modules may be added if truly needed, but keep structure minimal.

---

## Testing expectations

Add targeted tests for:
- normalization behavior for the currently implemented metrics
- preservation of native score values
- invalid-value handling in normalized form
- spectrum useful-band overlay logic where practical
- per-pixel metric table data generation
- per-pixel normalized bar chart data generation
- signal tools panel button wiring
- no regression of current GUI workflow

Keep tests focused and reliable.

---

## Implementation preferences

- keep normalization as a separate layer above native scores
- keep metric semantics metadata lightweight
- avoid hardcoded GUI-side metric-specific logic
- keep spectrum-band overlay simple and readable
- keep pixel comparison windows technical and minimal
- prefer correctness and interpretability over flashy UI
- preserve the current workflow and architecture

---

## Definition of done

This iteration is complete when:
- native metric scores remain preserved
- normalized comparison scores exist as a separate layer
- the spectrum view can show the useful band
- the signal tools panel includes a button for pixel metric inspection
- a table window for all computed metrics of the selected pixel exists
- a horizontal normalized bar chart window for the selected pixel exists
- both windows work on already computed session metrics
- the current GUI workflow remains stable

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs
2. summarize the intended comparison-layer iteration
3. propose a short implementation plan
4. implement only this iteration
5. add targeted tests
6. summarize created files, modified files, and any limitations