# Current iteration

## Iteration name

GUI extension — multi-metric selection + map session state + threshold controls

## Goal

Extend the first GUI slice into a more useful analysis interface.

This iteration should add:
- selection of multiple metrics through a dedicated dialog
- computation of multiple selected metrics in one run
- storage of computed metric results in the current GUI session
- map selector behavior based on actually available computed maps
- threshold controls tied to the currently displayed map
- masked map display without overwriting the original score map

This iteration should keep the GUI minimal while making it much more useful for real comparison work.

---

## Why this iteration matters

The current GUI already supports:
- loading a dataset
- selecting one metric
- computing one result
- displaying one map
- selecting a pixel
- displaying its signal
- opening comparison windows
- showing info on demand

That is a good first slice.

The next practical need is to make the GUI suitable for comparing multiple metrics and interactively thresholding the currently displayed map.

This is a natural step toward the intended engineering workflow of the project.

---

## In scope

### Metric selection dialog

Replace the single metric dropdown workflow with a dedicated metric-selection dialog.

Requirements:
- add a `Metrics...` action or button
- open a dialog listing all available metrics
- allow selecting multiple metrics using checkboxes
- preserve the currently selected metrics across dialog reopenings during the same session

Expected behavior:
- after confirming the dialog, the GUI remembers which metrics are selected
- the selected metrics become the set to compute when `Compute` is pressed

This dialog should be simple and read-only with respect to metric parameters.
No metric-specific parameter editing is required in this iteration.

---

### Multi-metric computation

Update the compute flow so that it can compute all selected metrics in one run.

Requirements:
- when `Compute` is pressed, compute all currently selected metrics
- store their results in session state
- do not discard previously computed results unless the dataset changes or recomputation replaces them
- after computation, display the first selected metric by default

Keep the implementation simple and explicit.
No background worker system is required in this iteration unless absolutely necessary.

---

### Session state for computed maps

Introduce explicit GUI session state for computed map results.

At minimum, the GUI should keep track of:
- the currently loaded dataset
- the selected metric names
- computed metric results by metric name
- the currently displayed map name
- threshold state for the currently displayed metric when applicable

A simple structure is enough.
Do not introduce a large state-management framework.

---

### Map selector behavior

Redefine the `Map` selector so that it controls **which available computed map is displayed in the main viewer**.

Requirements:
- the map selector should list the names of metrics that have already been computed in the current session
- selecting an item should switch the main viewer to that metric's score map
- the selector must reflect actual session results, not hardcoded generic map labels

Examples:
- `fringe_visibility`
- `snr`
- `power_band_ratio`

Do not use `score_map` and `threshold_mask` as the primary selector entries anymore.

---

### Threshold controls

Add explicit threshold controls for the currently displayed map.

Requirements:
- add a vertical threshold slider positioned near the colorbar / right side of the map area
- add an `Apply` control below the slider
- add a `Reset` control below the slider
- the slider range should correspond to the current displayed score map range:
  - minimum = map minimum
  - maximum = map maximum

Behavior:
- the threshold always applies to the currently displayed metric map
- moving the slider updates the current threshold value
- pressing `Apply` computes and stores a threshold result for the currently displayed map
- pressing `Reset` removes the active threshold presentation for the currently displayed map

---

### Threshold display model

The original score map must remain unchanged.

Thresholding must be treated as a display/filter layer, not as destructive modification of the underlying metric result.

This means:
- the raw score map remains stored as-is
- the threshold mask is stored separately
- the map viewer displays either:
  - the raw map,
  - a masked version of the raw map,
  - or the mask itself, depending on the chosen display mode

Do not overwrite the original computed metric data.

---

### Masked map display

Implement display support for thresholded visualization of the current map.

Requirements:
- after threshold `Apply`, the main viewer should show the thresholded view of the currently displayed map
- the masking should always be applied relative to the original score map
- changing the threshold and applying again should recompute from the original score map, not from a previously masked display

Keep the display logic simple and explicit.

---

### Optional mask-only display

If simple and clean to add in this iteration, support a lightweight way to view:
- masked score map
- binary mask only

This may be implemented through:
- a small display-mode selector,
- a checkbox,
- or another minimal UI control

Only add this if it remains clean and does not overcomplicate the current iteration.

If it is not clean, defer mask-only display to a later iteration.

---

### Main viewer updates

Update the main map viewer integration so that it can:
- show raw metric maps
- show masked metric maps
- optionally show binary masks if implemented
- keep selected-pixel interaction working under all display modes

Pixel selection behavior must remain stable and continue updating the signal inspector.

---

### Signal inspector continuity

The signal inspector should continue to work from the loaded dataset and selected pixel.

Requirements:
- selected pixel signal display must continue working regardless of which metric map is shown
- threshold display must not break the signal inspector
- pixel selection must remain tied to image coordinates, not to threshold state

---

### Status and info updates

Update status / info behavior so that it reflects the new GUI state where useful.

Helpful examples:
- current displayed metric
- whether a threshold is currently active
- current threshold value
- number of computed maps currently available

Keep this lightweight.

---

## Out of scope

Do not implement in this iteration:
- metric-specific parameter editing
- advanced threshold rules beyond the current backend behavior
- histogram view
- multi-map dashboard in the main window
- experiment manager
- batch comparison tables
- background task system
- height-map workflow
- synthetic-data workflow
- benchmark UI
- persistent workspace/session saving

Keep this iteration focused on multi-metric usability and threshold interaction.

---

## File targets

Expected existing modules to update:

- `src/quality_tool/gui/main_window.py`
- `src/quality_tool/gui/widgets/map_viewer.py`

Expected new modules to create:

- `src/quality_tool/gui/dialogs/metrics_dialog.py`

Optional helper module may be added if truly needed for simple GUI session state, but avoid building a full framework.

---

## Testing expectations

Add lightweight tests where practical.

Preferred focus:
- metrics dialog selection behavior
- map selector population from computed results
- repeated compute with multiple metrics
- threshold apply/reset logic on session state
- masked display logic if testable without heavy GUI complexity

Keep automated tests reliable and lightweight.

The main priority is a clean, working GUI behavior.

---

## Implementation preferences

- keep the GUI thin and backend-driven
- keep code simple and explicit
- do not duplicate backend metric or thresholding logic
- keep GUI state handling lightweight
- prefer small helper methods over heavy controller abstractions
- keep the main map viewer generic
- keep the original metric results immutable from the GUI point of view
- treat thresholding as display state, not data mutation

---

## Definition of done

This iteration is complete when:
- multiple metrics can be selected through a dialog
- pressing `Compute` computes all selected metrics
- computed metric results are stored in current GUI session state
- the `Map` selector lists actual computed metric maps
- selecting a map updates the main viewer correctly
- threshold slider + apply/reset work on the currently displayed map
- thresholding does not overwrite original score maps
- signal inspector still updates correctly on pixel selection
- the GUI remains aligned with `docs/gui_spec.md` and backend architecture

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs, including `docs/gui_spec.md`
2. summarize the planned GUI extension
3. propose a short implementation plan
4. implement only this iteration
5. add lightweight tests where practical
6. summarize created files, modified files, and any limitations