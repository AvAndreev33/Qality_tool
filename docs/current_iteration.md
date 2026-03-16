# Current iteration

## Iteration name

Histogram and threshold analysis — distribution view for current map

## Goal

Extend the GUI with a histogram-based analysis view for the currently displayed map.

This iteration should add:
- a histogram window for the current displayed map
- a threshold indicator line in the histogram
- basic descriptive statistics for the current map
- kept/rejected statistics based on the current threshold state
- support for opening multiple histogram windows as fixed comparison snapshots

The goal is to improve the workflow of selecting and comparing quality criteria by making the distribution of map values visible and directly connected to thresholding.

---

## Why this iteration matters

The current GUI already supports:
- loading datasets
- multi-metric computation
- map switching
- threshold application
- masked display
- signal inspection
- map comparison windows

But thresholding is still less interpretable than it should be.

To choose good thresholds and compare criteria properly, the user should be able to see:
- how the current map values are distributed,
- where the current threshold lies in that distribution,
- how many pixels are kept or rejected.

This is directly useful for the engineering task of comparing signal-quality criteria.

---

## In scope

### Histogram window for current displayed map

Add the ability to open a histogram window for the map currently shown in the main viewer.

Requirements:
- the histogram should be based on the currently displayed metric map
- the histogram should open in a separate window
- the histogram window should behave as a fixed snapshot of the current map state at the moment it is opened
- multiple histogram windows may be opened for comparison

This should mirror the current compare-window philosophy.

---

### Histogram content

The histogram should show the distribution of values of the current displayed map.

Requirements:
- use the raw underlying score map of the currently selected metric as the histogram source
- do not build the histogram from a destructively masked map
- handle invalid pixels consistently and explicitly
- if valid/invalid filtering is already naturally available, use only valid values for the histogram; otherwise document the chosen behavior clearly

Keep the implementation simple and explicit.

---

### Threshold indicator line

The histogram window should visually indicate the current threshold.

Requirements:
- draw a clear vertical threshold line on the histogram
- if the threshold for the currently displayed metric/map is active, the line should reflect its value
- if no threshold is currently active, the histogram may either:
  - show no line, or
  - show the current slider value if that is already cleanly available

Choose the simplest behavior that stays consistent with the current GUI state model.

The histogram and thresholded map should remain visually consistent.

---

### Basic map statistics

Display simple descriptive statistics for the currently displayed map.

Recommended values:
- min
- max
- mean
- median
- standard deviation

These may be shown:
- inside the histogram window,
- in a compact info area within that window,
- or in another simple and readable form

Keep it lightweight and clear.

---

### Threshold outcome statistics

Display simple threshold-related statistics for the current map.

Recommended values:
- number of valid pixels
- number of kept pixels
- number of rejected pixels
- kept fraction (or kept percentage)

Requirements:
- if a threshold is active, these values should reflect the current threshold result
- if no threshold is active, display a clear neutral state such as “threshold not applied”

Keep the logic consistent with the current threshold/session state model.

---

### Multiple histogram windows

The user should be able to open multiple histogram windows as fixed snapshots.

Requirements:
- opening a histogram should not replace a previous histogram window
- each histogram window should represent the map state at the moment it was opened
- later map switching in the main window should not retroactively change already opened histogram windows

This is important for visual comparison between criteria.

---

### Toolbar / action integration

Add a simple GUI action to open the histogram of the current displayed map.

This may be:
- a toolbar action
- a button
- or another high-level action consistent with the current GUI style

Keep it minimal.

---

### Session-state integration

Integrate histogram behavior cleanly with current GUI session state.

Requirements:
- histogram opening must use the current displayed metric/map
- threshold line and threshold statistics must reflect the current threshold state for that map
- histogram logic must not duplicate backend metric computation
- histogram should use already computed session results

Keep this GUI-side orchestration simple and explicit.

---

## Out of scope

Do not implement in this iteration:
- histogram overlays of multiple maps in one window
- correlation plots
- scatter plots between metrics
- automatic threshold suggestion
- log-scale histogram counts
- valid-only checkbox if it complicates the design
- histogram-based export
- major GUI redesign
- profiling / CUDA work

Keep this iteration focused on histogram-based threshold analysis.

---

## File targets

Expected modules to update:

- `src/quality_tool/gui/main_window.py`

Expected new modules to create:

- `src/quality_tool/gui/windows/histogram_window.py`

Optional helper module may be added if truly needed, but keep structure minimal.

---

## Testing expectations

Add lightweight tests where practical.

Preferred focus:
- histogram window creation
- histogram uses the currently displayed map
- threshold line presence/absence behavior
- map statistics calculation
- threshold statistics calculation
- multiple histogram windows behaving as fixed snapshots

Keep tests reliable and lightweight.

---

## Implementation preferences

- keep the GUI thin and backend-driven
- use already computed session results
- do not duplicate metric computation logic
- keep histogram windows simple and technical
- keep statistics compact and readable
- keep snapshot behavior explicit
- preserve the current compare-window style

---

## Definition of done

This iteration is complete when:
- the user can open a histogram window for the current displayed map
- the histogram shows the value distribution of that map
- a threshold indicator line is shown when appropriate
- basic map statistics are shown
- threshold-related kept/rejected statistics are shown when appropriate
- multiple histogram windows can be opened as snapshots
- the current GUI workflow remains stable

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs, including `docs/gui_spec.md`
2. summarize the intended histogram/threshold-analysis extension
3. propose a short implementation plan
4. implement only this iteration
5. add lightweight tests where practical
6. summarize created files, modified files, and any limitations