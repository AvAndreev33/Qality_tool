# Current iteration

## Iteration name

GUI visual polish — modern minimal styling and UX refinement

## Goal

Refine the current GUI so that it looks clean, modern, and intentional while preserving the existing workflow and backend behavior.

This iteration should:
- improve the visual design of the current Qt GUI
- make the interface feel more polished and coherent
- keep the layout simple and engineering-oriented
- avoid flashy or bloated UI
- preserve the current functionality and workflow

The goal is not a redesign of product behavior.
The goal is to turn the current functional GUI into a cleaner and more professional-looking engineering desktop app.

---

## Why this iteration matters

The project already has:
- a strong backend
- multiple implemented metric groups
- a usable GUI with map viewer, signal inspector, thresholding, comparison, histogram, and grouped metric selection

Functionally, the tool is already useful.

The next improvement is visual and ergonomic:
- better spacing
- better visual hierarchy
- cleaner controls
- more intentional panel styling
- more readable status and labels
- more coherent modern desktop-app appearance

This should make the tool more pleasant to use without changing its core logic.

---

## In scope

### Visual benchmark research

Before editing the GUI, review a small set of modern minimal desktop UI references suitable for:
- scientific tools
- engineering tools
- data-inspection tools
- dark-theme desktop applications
- clean Qt-style applications

Requirements:
- do lightweight visual research
- gather a few simple reference directions
- prefer minimal, technical, uncluttered interfaces
- do not copy any one design literally
- synthesize a clean style direction appropriate for this project

The result should be:
- a short visual direction summary before implementation
- then one coherent style applied to the app

Keep the visual direction practical, not trendy for the sake of trendiness.

---

### Preserve the current layout and workflow

The current workflow must remain intact:
- dataset loading
- metric selection
- grouped metric dialog
- map view
- signal inspection
- threshold controls
- histogram windows
- comparison windows
- pixel inspection windows
- normalized score / masking modes

This iteration is not about changing what the GUI does.
It is about improving how it looks and feels.

---

### Main window styling

Refine the main window so it feels more structured and polished.

Focus areas:
- spacing and margins
- section alignment
- visual hierarchy
- panel balance
- typography hierarchy
- button consistency
- cleaner toolbar presentation
- cleaner tool-panel presentation

The window should look intentional and professionally assembled, not like a default raw Qt app.

---

### Map section polish

Refine the visual presentation of the map section.

Possible areas:
- cleaner framing of the map viewer
- cleaner visual relationship between map viewer and map tools panel
- improved labels / titles where useful
- more polished control grouping
- more coherent spacing for controls around the viewer

Do not overload the map section with decoration.

---

### Signal section polish

Refine the signal section so it looks as intentional as the map section.

Possible areas:
- cleaner visual framing of the signal plot
- cleaner signal-tools panel styling
- better grouping of controls
- more readable labels and toggles
- more coherent balance between plot and tools

Keep this technical and minimal.

---

### Control styling consistency

Make controls visually consistent across the application.

Focus on consistency for:
- push buttons
- combo boxes
- check boxes
- section labels
- dialogs
- status indicators
- table windows
- chart windows

The GUI should feel like one application, not a set of separately built widgets.

---

### Dialog and window polish

Refine the look of secondary windows and dialogs, including where applicable:
- metric selection dialog
- info dialog
- histogram windows
- per-pixel metric table window
- per-pixel metric chart window
- compare windows
- 3D map window

Requirements:
- keep them visually consistent with the main window
- preserve their current functionality
- improve readability and modern desktop feel

---

### Styling approach

Use a clean Qt-appropriate styling approach.

Possible acceptable methods:
- improved widget layout and spacing
- Qt stylesheet cleanup
- palette refinement
- small icon / label polish if already available or trivial to add
- lightweight reusable style helpers if needed

Do not build a large custom theming framework.
Do not introduce a large dependency just for cosmetics unless already natural and minimal.

---

### Status/readability polish

Improve the readability of current state information.

Possible areas:
- status bar clarity
- active mode labels
- current metric/map visibility
- clearer group labels
- clearer visual distinction between primary and secondary actions

Keep it subtle and useful.

---

### Metric grouping dialog polish

The metric-selection dialog already supports grouped metrics.

Improve its presentation so that:
- groups are clearer visually
- spacing is better
- labels are easier to scan
- the dialog feels modern and organized

Do not redesign the selection logic.
Only improve presentation and clarity.

---

### Minimalism requirement

This iteration must stay minimal.

Do:
- clean
- structured
- modern
- technical
- readable

Do not:
- add flashy gradients everywhere
- add decorative clutter
- make it look like a consumer app
- introduce unnecessary animation
- replace clarity with visual effects

The target style is:
**simple, modern, minimal engineering desktop software**

---

## Out of scope

Do not implement in this iteration:
- backend changes
- new metric logic
- new analysis features
- workflow redesign
- new data-processing controls
- large animation systems
- major layout rearchitecture
- web-style UI overhaul
- unrelated feature additions

This iteration is visual polish and ergonomic refinement only.

---

## File targets

Expected modules to update may include:

- `src/quality_tool/gui/main_window.py`
- `src/quality_tool/gui/widgets/map_viewer.py`
- `src/quality_tool/gui/widgets/signal_inspector.py`
- `src/quality_tool/gui/dialogs/metrics_dialog.py`
- `src/quality_tool/gui/dialogs/info_dialog.py`
- `src/quality_tool/gui/windows/*.py`

A small shared GUI styling helper/module may be added if useful, but keep it minimal.

---

## Testing expectations

Automated tests do not need to prove aesthetics.

Add or update tests only where needed to ensure:
- no regression of existing window creation
- no regression of current controls and workflow
- no breakage from renamed/restructured GUI code

The priority is preserving behavior while improving appearance.

---

## Implementation preferences

- keep functionality unchanged
- improve visual hierarchy
- improve spacing and consistency
- prefer clean dark-theme engineering aesthetics
- use lightweight reference research before implementation
- synthesize a coherent style instead of copying one example
- avoid overengineering theming
- keep the GUI technical and minimal

---

## Definition of done

This iteration is complete when:
- the GUI looks noticeably cleaner and more modern
- the current workflow remains unchanged
- windows and dialogs feel visually consistent
- map and signal sections are better balanced and presented
- grouped metric selection looks cleaner
- controls look more coherent across the app
- the result feels like a polished engineering desktop tool rather than a default raw Qt interface

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs
2. inspect the current GUI code and current app appearance
3. do lightweight visual benchmark research
4. summarize the chosen visual direction
5. implement only this visual-polish iteration
6. verify current behavior still works
7. summarize what was visually improved and any remaining limitations