# Current iteration

## Iteration name

GUI research controls — preprocessing + ROI + envelope + signal overlays

## Goal

Extend the GUI so that it exposes more of the existing research pipeline already available in the backend.

This iteration should add:
- GUI access to preprocessing settings
- GUI access to ROI settings
- GUI access to envelope method selection
- GUI access to threshold keep rule selection
- signal-inspector display modes and overlays for better per-pixel inspection

The goal is to move the GUI from a simple map viewer into the first real interactive WLI signal-quality workbench, while still keeping it minimal and backend-driven.

---

## Why this iteration matters

The backend already supports:
- preprocessing
- ROI extraction
- envelope computation
- spectral support
- evaluator
- thresholding

But the GUI still exposes only a small part of that capability.

The next logical step is to let the user actually control and inspect these pipeline stages from the GUI.

This will make the tool much more useful for real engineering experimentation:
- trying preprocessing variants
- changing ROI behavior
- trying envelope methods
- comparing results
- inspecting how the selected pixel signal changes

---

## In scope

### Processing settings dialog

Add a dedicated processing/settings dialog.

This dialog should allow the user to configure the currently supported processing-related options in a simple and explicit way.

At minimum, it should support:
- preprocessing enable/disable
- baseline subtraction enable/disable, if supported by backend
- normalization enable/disable, if supported by backend
- smoothing enable/disable, if supported by backend
- ROI enable/disable
- ROI `segmentSize`
- ROI centering mode

For this iteration:
- only expose options that already exist in the backend
- do not invent new backend behavior
- if the backend currently supports only `raw_max` centering, expose only that option or keep it fixed and visible

The dialog should be simple and use the current backend semantics.

---

### Envelope settings integration

Add GUI access to envelope method selection.

Requirements:
- provide a simple way to enable or disable envelope usage
- allow selecting among currently available envelope methods
- store the selected envelope method in GUI session state
- ensure evaluator calls use the selected method when enabled

If there is only one meaningful envelope method right now, still expose it through a simple selection mechanism that can grow later.

---

### Threshold keep rule control

Add GUI control for threshold keep rule.

Requirements:
- support at least:
  - `score >= threshold`
  - `score <= threshold`
- keep this control simple
- integrate it with the existing threshold slider/apply/reset workflow
- the selected keep rule must affect threshold application for the currently displayed map

This should remain GUI-driven configuration of existing backend threshold behavior, not new backend logic.

---

### Signal inspector display modes

Extend the signal inspector so that it can show more than only the raw signal.

At minimum, add signal display modes such as:
- raw signal
- raw + envelope
- ROI signal, when ROI is enabled and available
- spectrum, if simple to support using existing spectral backend

The exact UI can be lightweight:
- a small control group in the signal tools panel
- a selector
- checkboxes where appropriate

The signal inspector should still stay compact and clear.

---

### Signal overlays and display logic

The signal inspector should use the currently selected pixel and reflect the current GUI processing configuration.

Expected behavior:
- raw mode always shows the original signal from the loaded dataset
- raw + envelope shows the signal plus the current envelope when envelope is enabled
- ROI mode shows the extracted ROI signal if ROI is enabled
- spectrum mode shows the spectral representation of the selected pixel signal if supported in this iteration

Keep this display logic explicit and backend-driven.

Do not duplicate signal-processing logic in the plot widget itself.
The widget should only display data prepared by higher-level GUI/backend integration.

---

### GUI session state extension

Extend the current GUI session state so it also tracks:
- current preprocessing settings
- current ROI settings
- current envelope enable/disable state
- current envelope method
- current threshold keep rule
- current signal display mode

Keep this state handling simple.
Do not introduce a large state-management framework.

---

### Evaluator integration

Update GUI compute flow so that evaluation uses the currently selected settings.

This means the GUI should pass into the backend evaluator:
- selected metric
- processing settings
- ROI settings
- envelope method if enabled
- threshold keep rule only when threshold is applied, not during metric computation

All such integration must go through existing backend APIs or minimal safe extensions if required.

---

### Signal tools panel usage

Use the existing signal tools panel introduced in the previous iteration.

This panel should now host the signal-related controls for this iteration, such as:
- signal display mode selector
- envelope display toggle if appropriate
- ROI-related display toggle if appropriate
- spectrum display toggle or selector if included

Keep the controls minimal and focused.

---

### Map tools panel extension

Keep threshold controls in the map tools panel and extend them with:
- keep rule selector (`>=` / `<=` or equivalent readable labels)

Do not overload the panel with unrelated controls.

---

## Out of scope

Do not implement in this iteration:
- metric-specific parameter editors
- advanced preprocessing pipelines not already supported in backend
- new envelope algorithms
- new metrics
- histogram window
- experiment manager
- synthetic-data GUI
- benchmark GUI
- CUDA / performance backend work
- session persistence
- complex multi-panel compare dashboards

Keep this iteration focused on exposing the existing backend research controls.

---

## File targets

Expected modules to update:

- `src/quality_tool/gui/main_window.py`
- `src/quality_tool/gui/dialogs/info_dialog.py` if needed
- `src/quality_tool/gui/widgets/signal_inspector.py`

Expected new modules to create:

- `src/quality_tool/gui/dialogs/processing_dialog.py`

Optional small helper module may be added if truly needed for lightweight GUI state/config handling, but avoid heavy abstractions.

---

## Testing expectations

Add lightweight tests where practical.

Preferred focus:
- processing dialog state behavior
- GUI state updates after settings changes
- keep rule switching
- signal inspector display mode switching
- compute flow using selected settings where practical
- no regression of existing metric/session/map selection behavior

Keep tests reliable and lightweight.

---

## Implementation preferences

- keep the GUI thin and backend-driven
- expose only backend functionality that already exists or is trivially supported
- do not duplicate backend preprocessing logic in GUI code
- keep session state simple and explicit
- keep widgets focused on display, not processing
- keep controls minimal and engineer-friendly
- preserve the current layout and workflow while extending capability

---

## Definition of done

This iteration is complete when:
- preprocessing settings can be configured from the GUI
- ROI settings can be configured from the GUI
- envelope method can be enabled/selected from the GUI
- threshold keep rule can be selected from the GUI
- signal inspector can show more than just raw signal
- compute flow respects the selected processing/envelope settings
- current workflow remains stable
- the GUI becomes a more useful interactive research workbench while staying minimal

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs, including `docs/gui_spec.md`
2. summarize the intended GUI research-control extension
3. propose a short implementation plan
4. implement only this iteration
5. add lightweight tests where practical
6. summarize created files, modified files, and any limitations