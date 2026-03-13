# Quality_tool — GUI Specification

## 1. Purpose

This document defines the GUI concept for `Quality_tool`.

The GUI should provide a minimal but effective desktop interface for:
- loading real WLI datasets,
- computing and viewing result maps,
- inspecting pixel-wise signals,
- switching between available maps,
- opening result maps in separate windows for visual comparison.

The GUI must remain a thin layer over the existing backend and must not duplicate core logic inside the interface.

---

## 2. GUI philosophy

The GUI should be designed as:

**a generic 2D map viewer + signal inspector**

This is the core idea of the interface.

The GUI must not be treated as a special-purpose “quality map window” only.  
The same main visual structure should later support:
- quality maps,
- threshold masks,
- height maps,
- future feature maps.

The GUI should stay:
- minimal,
- technically clear,
- visually focused on data,
- easy to extend,
- easy to use during engineering inspection.

---

## 3. Scope of GUI v0.1

GUI v0.1 is intentionally minimal.

It should support:
- loading a dataset,
- selecting a metric,
- computing a result,
- displaying one main 2D map,
- displaying the signal of the selected pixel,
- switching between available map types,
- opening the current map in a separate comparison window,
- showing dataset and run information on demand,
- exporting available outputs through the backend when supported.

GUI v0.1 should not try to expose the full future research-platform complexity.

---

## 4. Main window concept

The GUI should use one main application window.

Its structure should be centered around two visual areas:

1. **Main map viewer**
2. **Signal inspector**

The main map viewer is the primary analysis area.  
The signal inspector is the linked per-pixel inspection area.

### Main layout idea

    +--------------------------------------------------------------+
    | Toolbar / action bar                                         |
    +--------------------------------------------------------------+
    |                                                              |
    |                      Main map viewer                         |
    |                                                              |
    |                                                              |
    +--------------------------------------------------------------+
    |                      Signal inspector                        |
    +--------------------------------------------------------------+
    | Status bar                                                   |
    +--------------------------------------------------------------+

This should remain the dominant layout of the GUI.

No permanent left-side control or information panel is required in v0.1.

---

## 5. Main map viewer

The main map viewer is the central visual element of the application.

### Responsibilities
- display one currently selected 2D map
- support pixel selection by mouse click
- visually indicate the selected pixel
- allow switching between map types
- allow opening the current map in a separate window

### Expected map types in v0.1
At minimum, the viewer should be ready to display:
- quality map
- threshold mask

The viewer should be designed so that future map types can be displayed in the same place without redesigning the GUI.

### Interaction
When the user clicks a pixel in the map:
- that pixel becomes the active selection,
- the signal inspector updates to show the corresponding signal,
- the status bar updates with pixel coordinates and current value.

### Visual requirements
- the selected pixel should be clearly marked
- the map area should remain large and easy to inspect
- the viewer should prioritize readability over decorative UI elements

---

## 6. Signal inspector

The lower plot area is the signal inspector.

Its purpose is to show the signal corresponding to the currently selected pixel in the main map viewer.

### Responsibilities
- display the signal of the selected pixel
- update when the selected pixel changes
- use the current dataset z-axis representation
- support future overlays without redesigning the widget

### Initial v0.1 behavior
At minimum, it should show:
- raw signal of the selected pixel

It should be designed so that future versions can also show:
- envelope
- ROI segment
- processed signal
- spectrum

### Visual behavior
- the plot should span the same width as the main map viewer
- the plot should update immediately after pixel selection
- axis labels should be simple and clear

---

## 7. Toolbar / action bar

The GUI should provide a compact top toolbar or action bar.

This should contain the primary user actions and high-level selections.

### Recommended controls for v0.1

#### File / dataset actions
- `Load dataset`

#### Compute actions
- `Compute`

#### Map selection
- `Map type selector`
  - for example: quality map / threshold mask / future map types

#### Metric selection
- `Metric selector`
  - select which metric is currently used for computation or display

#### Comparison actions
- `Open current map in new window`

#### Information
- `Info`

#### Export
- `Export current result`
  - only for outputs already supported by the backend

### Design rule
The toolbar should expose only high-level actions.  
It should not become a dense control panel.

---

## 8. Information display policy

Dataset info, run settings, and technical details should **not** occupy permanent space in the main window.

Instead, GUI v0.1 should provide an **Info** action.

When requested, the GUI should show an information dialog or secondary panel containing relevant details such as:
- dataset source
- dimensions
- signal length
- whether metadata was found
- z-axis mode
- current metric
- current threshold settings
- current processing settings

This keeps the main window visually clean while preserving access to technical information.

---

## 9. Map switching

The user must be able to switch which map is shown in the main map viewer.

Examples:
- show one metric map
- switch to another metric map
- switch to threshold mask
- later switch to height map

This switching behavior is important because the same viewer is intended to serve as a generic 2D map viewer.

The GUI should avoid creating a separate primary window for each map type.  
One central viewer with user-controlled switching is preferred.

---

## 10. Comparison windows

The GUI should support opening the currently displayed map in a separate window.

This is the primary comparison mechanism for v0.1.

### Purpose
It should allow the user to:
- keep one result visible,
- switch the main window to another result,
- compare maps visually side by side.

### Behavior
- the separate comparison window should display a fixed snapshot of the currently selected map
- it should not replace the main interactive window
- multiple comparison windows may be opened if useful

### Design principle
Comparison should be implemented through additional viewer windows, not by overloading the main window with too many simultaneous panels.

---

## 11. Status bar

The GUI should include a simple status bar.

Its purpose is to provide compact live information without cluttering the main interface.

### Recommended contents
- selected pixel coordinates
- current displayed value at that pixel
- current map type
- current metric name
- simple state messages such as loading / computing / done / error

The status bar should remain lightweight.

---

## 12. Processing and parameter controls

GUI v0.1 should avoid a large permanent parameter sidebar.

However, the user still needs access to processing and evaluation settings.

The preferred approach is:
- simple essential selections in the toolbar,
- more detailed settings in dialogs when needed.

### Suggested policy
For v0.1:
- keep directly visible controls minimal
- open configuration dialogs for more detailed settings

This is important because the intended users are engineers who usually do not need a constant reminder panel for parameters they already set.

---

## 13. GUI workflows

### Workflow A — basic map inspection
1. Load dataset
2. Select metric
3. Press Compute
4. View the resulting map
5. Click a pixel
6. Inspect its signal in the lower plot

### Workflow B — switch displayed result
1. Compute one map
2. Change metric or map type
3. Compute again or switch to another available result
4. Inspect the updated map in the same main viewer

### Workflow C — compare maps
1. Compute a result
2. Open current map in a new window
3. Return to the main window
4. Compute or select another map
5. Compare visually

### Workflow D — inspect technical details
1. Load or compute data
2. Click `Info`
3. Review dataset and run information
4. Close the info dialog and continue work

---

## 14. Out of scope for GUI v0.1

GUI v0.1 should not include:
- permanent left-side information/control panel
- histogram panel
- experiment manager
- pipeline builder
- multi-metric dashboard in the main window
- benchmark controls
- synthetic-data workflow controls
- advanced workspace management
- embedded scripting interface

These can be added later if needed.

---

## 15. Backend integration principles

The GUI must remain a thin layer over the existing backend.

It should call backend functionality for:
- loading datasets
- computing metrics
- thresholding
- retrieving signals for selected pixels
- exporting results

The GUI must not duplicate backend logic for:
- signal processing
- metric computation
- thresholding
- data parsing

All such logic should stay in the core backend modules.

---

## 16. Extensibility principles

The GUI should be easy to extend in later versions.

It should be designed so that the same main structure can support:
- height maps
- additional feature maps
- envelope visualization
- ROI visualization
- spectrum visualization
- richer processing configuration
- comparison workflows

The key extensibility rule is:

**keep the main structure stable and extend behavior around it**

That means:
- keep the main map viewer generic
- keep the lower plot as a general signal inspector
- add dialogs and secondary windows rather than redesigning the main window repeatedly

---

## 17. Visual design principles

The GUI should feel:
- technical
- clean
- focused
- minimal

Priority should go to:
- large readable visualization areas
- simple controls
- clear interactions
- easy interpretation

The GUI should not attempt to look flashy.  
Its value comes from clarity and speed of engineering inspection.

---

## 18. Summary

GUI v0.1 is a minimal desktop interface built around one central concept:

**generic 2D map viewer + signal inspector**

Core elements:
- one main map viewer
- one lower signal plot
- one compact toolbar
- one status bar
- one on-demand information dialog
- optional separate map comparison windows

This gives a simple but strong foundation for the first usable GUI while keeping future extension straightforward.