# Current iteration

## Iteration name

First GUI slice — main window + map viewer + signal inspector

## Goal

Implement the first usable GUI slice for the project.

This iteration should create a minimal desktop GUI that can:
- load a dataset,
- compute a selected metric,
- display one main 2D result map,
- let the user click a pixel,
- display the signal of that pixel in the lower plot,
- switch between currently available map types,
- open the current map in a separate comparison window,
- show dataset and run information on demand.

This should be the first thin GUI layer over the existing backend.

---

## Why this iteration matters

The backend now has a working end-to-end pipeline:
- data loading
- metadata parsing
- z-axis handling
- preprocessing / ROI / envelope / spectral support
- metrics
- evaluator
- thresholding

The next useful step is to make this pipeline conveniently inspectable through a minimal interactive GUI.

This iteration is not about building the full future research workbench.
It is about building the first clean and extensible visual shell around the current backend.

---

## GUI framework

Use the existing project direction based on **Python + Qt**.

Preferred implementation:
- PySide6 if already available or easy to add cleanly
- PyQt is acceptable only if that is the existing preferred stack in the repository

Keep the GUI implementation simple and explicit.

Do not build a custom framework around Qt.

---

## In scope

### Main application window

Implement one main application window.

Responsibilities:
- provide the central GUI layout
- host the main map viewer
- host the signal inspector
- provide the top toolbar / action bar
- provide a status bar
- connect UI actions to backend calls

The layout should follow `docs/gui_spec.md`.

---

### Main layout

The main window must contain:
- a top toolbar / action bar
- a large central 2D map viewer
- a lower signal inspector plot
- a status bar

No permanent left-side information or control panel should be added in this iteration.

---

### Dataset loading

Implement GUI-driven dataset loading.

Requirements:
- user can trigger dataset loading from the GUI
- loaded data must go through the existing backend loaders
- after loading, the GUI must store the active `SignalSet`
- status bar should reflect successful load or loading failure

Keep this simple and backend-driven.

---

### Metric selection

Provide a visible metric selector in the toolbar.

Requirements:
- populate it from the currently available metric registry or the current set of baseline metrics
- allow the user to choose the active metric before computation

Do not add advanced parameter editing yet.

---

### Compute action

Provide a `Compute` action in the toolbar.

Requirements:
- use the current dataset
- use the currently selected metric
- run the existing backend evaluator
- store the resulting `MetricMapResult`
- make the resulting map available to the main map viewer
- allow threshold mask display if thresholding is already available and practical to expose

Keep the action explicit.

No background worker system is required in this iteration unless absolutely necessary.

---

### Main map viewer widget

Implement a dedicated map viewer widget.

Responsibilities:
- display one 2D map
- support mouse click pixel selection
- visually indicate the selected pixel
- notify the main window about the selected pixel coordinates

Requirements:
- support at least:
  - quality map
  - threshold mask
- be implemented as a generic 2D map viewer rather than as a hardcoded “quality map widget”

Do not overengineer zoom/pan tools in this iteration unless they come almost for free.

---

### Signal inspector widget

Implement a lower signal plot widget.

Responsibilities:
- show the signal corresponding to the selected pixel
- update when the selected pixel changes
- use the active dataset and its z-axis
- show at least the raw signal in this iteration

Requirements:
- it should work directly with the currently loaded `SignalSet`
- it should not duplicate signal-processing logic from the backend
- it should stay simple and ready for future overlays

No envelope / ROI / spectrum overlays are required in this iteration.

---

### Map type switching

Add a map type selector to the toolbar.

For this iteration, it should support switching between available currently computed maps such as:
- metric score map
- threshold mask, if available

The switching logic should remain simple and explicit.

---

### Comparison window

Implement the ability to open the currently displayed map in a separate window.

Requirements:
- the comparison window should display a fixed snapshot of the current map
- it should not replace the main window
- it should allow the user to keep one result visible while continuing work in the main window

Keep this as a lightweight viewer window.

---

### Info action

Implement an `Info` action.

Requirements:
- open a dialog or secondary window on demand
- show useful dataset and current run information, such as:
  - dataset source
  - width / height
  - signal length
  - whether metadata was found
  - z-axis mode
  - active metric
  - current map type
- keep it read-only

This replaces the idea of a permanent info panel.

---

### Status bar

Implement a simple status bar.

It should display compact information such as:
- load / compute status
- selected pixel coordinates
- value at the selected pixel
- current metric
- current map type
- simple error messages where appropriate

Keep it lightweight.

---

### Backend integration

The GUI must use existing backend logic for:
- loading datasets
- evaluating metrics
- thresholding
- retrieving selected-pixel signals

Do not reimplement backend logic inside GUI widgets.

The GUI must remain a thin orchestration and visualization layer.

---

## Out of scope

Do not implement in this iteration:
- advanced settings dialogs for preprocessing / ROI / envelope
- histogram window
- multi-metric dashboard
- experiment manager
- batch comparison UI
- synthetic-data UI
- height map workflow
- advanced export UI
- background job system
- docking system
- full workspace persistence

Keep the first GUI iteration narrow and usable.

---

## File targets

Expected modules to create:

- `src/quality_tool/gui/__init__.py`
- `src/quality_tool/gui/app.py`
- `src/quality_tool/gui/main_window.py`
- `src/quality_tool/gui/widgets/__init__.py`
- `src/quality_tool/gui/widgets/map_viewer.py`
- `src/quality_tool/gui/widgets/signal_inspector.py`
- `src/quality_tool/gui/windows/__init__.py`
- `src/quality_tool/gui/windows/compare_window.py`
- `src/quality_tool/gui/dialogs/__init__.py`
- `src/quality_tool/gui/dialogs/info_dialog.py`

Optional helper modules may be added if truly needed, but keep structure minimal.

---

## Testing expectations

Add at least minimal tests where practical.

Preferred focus:
- widget creation smoke tests
- basic map viewer pixel-selection logic
- signal inspector update logic for selected pixel
- main-window wiring for load / compute flow where feasible without overcomplicating tests

If full GUI interaction testing requires extra heavy infrastructure, keep automated tests minimal and reliable.

The priority is a working, clean first GUI slice.

---

## Implementation preferences

- keep the GUI thin
- keep the code simple and readable
- prefer explicit signal-slot connections
- avoid large controller abstractions
- avoid duplicating backend logic
- keep widgets focused on display and interaction
- keep the main map viewer generic
- keep the signal inspector general enough for later overlays
- prefer a stable main layout over feature richness

---

## Definition of done

This iteration is complete when:
- the GUI application can be launched
- a dataset can be loaded through the GUI
- a metric can be selected
- the current metric can be computed
- the resulting map can be displayed in the main map viewer
- clicking a pixel updates the lower signal inspector
- current map type can be switched
- the current map can be opened in a separate comparison window
- an info dialog can be opened on demand
- the GUI uses the existing backend rather than duplicating logic

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs, including `docs/gui_spec.md`
2. summarize the planned first GUI slice
3. propose a short implementation plan
4. implement only this iteration
5. add lightweight tests where practical
6. summarize created files, modified files, and any limitations