# Current iteration

## Iteration name

GUI layout refinement — map tools panel + signal tools panel

## Goal

Refine the GUI layout so that the main visual areas become more structured and extensible.

This iteration should:
- move threshold controls out of the top toolbar,
- introduce a dedicated right-side tools panel for the main map viewer,
- introduce a dedicated right-side tools panel for the signal inspector,
- preserve the current GUI workflow and backend integration,
- prepare the layout for future signal-related and map-related controls.

This is a layout and interaction refinement iteration.
It should not change the core backend logic.

---

## Why this iteration matters

The current GUI already supports:
- dataset loading
- multi-metric selection
- multi-metric computation
- map switching
- threshold apply/reset
- compare window
- info dialog
- signal inspection by pixel click

That is already useful.

The next practical improvement is to make the window layout more suitable for future growth:
- threshold controls should live near the map they affect,
- future map-related controls should have a dedicated area,
- future signal-related controls should have a dedicated area.

This makes the GUI cleaner and more extensible without changing the project architecture.

---

## In scope

### Main layout refinement

Refine the central window layout so that it has two stacked visual sections:

1. **Map section**
2. **Signal section**

Each section should contain:
- a main viewer area on the left
- a narrow tools panel on the right

The main window should still contain:
- top toolbar / action bar
- central content
- status bar

The overall layout should remain simple and readable.

---

### Map section

The upper section should contain:
- the existing main map viewer
- a new dedicated right-side **map tools panel**

The map tools panel should be approximately the same height as the map viewer and visually associated with it.

This panel is intended for controls that affect the currently displayed map.

#### For this iteration, move into the map tools panel:
- threshold slider
- threshold spinbox if currently present
- `Apply`
- `Reset`

Keep the threshold functionality exactly as it currently works.
Only move and reorganize the controls.

### Design requirement
The map viewer itself should remain generic and focused on display and pixel interaction.
The tools panel should contain the controls.

---

### Signal section

The lower section should contain:
- the existing signal inspector
- a new dedicated right-side **signal tools panel**

For this iteration, this panel may initially contain:
- placeholder structure
- section title / label if helpful
- optional disabled or future-facing control placeholders only if they remain clean

The main purpose of this panel in this iteration is layout preparation.

It should clearly reserve space for future signal-related controls such as:
- envelope display controls
- spectrum display controls
- ROI-related signal inspection controls

Do not implement those future features now.

---

### Toolbar cleanup

Update the top toolbar so that it no longer contains threshold controls.

The toolbar should remain focused on high-level actions such as:
- load
- metrics selection
- compute
- map selection
- compare
- info
- export

This should make the toolbar cleaner and less overloaded.

---

### Preserve current behavior

The following behavior must remain unchanged:
- dataset loading
- metric selection dialog
- multi-metric compute
- session state
- map switching
- threshold application and reset
- compare window
- info dialog
- selected pixel handling
- signal inspector update

This iteration is about layout refinement, not feature redesign.

---

### Threshold interaction continuity

Threshold controls must continue to operate on the currently displayed metric map.

Requirements:
- the slider range must still track the currently displayed map range
- apply/reset must still work per current map / metric
- masked display behavior must remain non-destructive
- status updates should remain correct

The control relocation must not break the current thresholding behavior.

---

### Pixel and signal continuity

Pixel selection must continue to work exactly as before.

Requirements:
- clicking the map still selects a pixel
- signal inspector still updates with the selected pixel signal
- thresholded display modes must not break pixel selection
- moving controls to side panels must not affect signal inspection logic

---

### Visual cleanliness

The new layout should feel deliberate and uncluttered.

Requirements:
- map tools panel should visually belong to the map section
- signal tools panel should visually belong to the signal section
- no oversized empty control groups
- no decorative complexity
- the main focus should still remain the map and signal plots

---

## Out of scope

Do not implement in this iteration:
- new backend logic
- new threshold algorithms
- signal overlays
- envelope controls
- spectrum visualization
- ROI controls in the GUI
- histogram view
- advanced compare layout
- docking system
- background workers
- workspace persistence

This iteration is only about layout refinement and control relocation.

---

## File targets

Expected modules to update:

- `src/quality_tool/gui/main_window.py`

Potential widget updates if needed:

- `src/quality_tool/gui/widgets/map_viewer.py`
- `src/quality_tool/gui/widgets/signal_inspector.py`

Only modify other files if truly necessary.

---

## Testing expectations

Add lightweight tests where practical.

Preferred focus:
- threshold controls still connected correctly after relocation
- main window layout still initializes correctly
- map switching and signal selection still work
- apply/reset still work after control relocation

Keep tests reliable and lightweight.

---

## Implementation preferences

- keep the GUI thin and backend-driven
- preserve current behavior
- make only layout-oriented changes
- avoid introducing unnecessary abstractions
- keep the main viewers generic
- use the new side panels as simple, future-ready containers
- prefer clear Qt layouts over custom geometry hacks

---

## Definition of done

This iteration is complete when:
- threshold controls are no longer in the top toolbar
- the map section has a right-side tools panel
- the signal section has a right-side tools panel
- threshold controls work from the map tools panel
- the existing workflow continues to work
- signal inspector still updates correctly
- layout is cleaner and more extensible
- no backend logic was duplicated or redesigned

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs, including `docs/gui_spec.md`
2. summarize the intended layout refinement
3. propose a short implementation plan
4. implement only this iteration
5. add lightweight tests where practical
6. summarize created files, modified files, and any limitations