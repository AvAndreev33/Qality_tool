# Quality_tool — GUI Specification

## 1. Purpose

This document defines the GUI specification for `Quality_tool`.

The GUI provides a desktop interface for:
- loading real WLI datasets,
- selecting and computing quality metrics,
- viewing 2D result maps with threshold overlays,
- inspecting pixel-wise signals in multiple display modes,
- switching between computed metric maps,
- comparing maps and distributions through standalone windows.

The GUI is a thin layer over the backend and must not duplicate core logic.

---

## 2. GUI philosophy

The GUI is designed as:

**a generic 2D map viewer + signal inspector**

The same main visual structure supports:
- metric score maps,
- threshold masks,
- masked score maps,
- future feature maps and height maps.

The GUI should remain:
- technically clear,
- visually focused on data,
- easy to extend,
- easy to use during engineering inspection.

---

## 3. Main window layout

The GUI uses one main application window structured around two paired sections, each with a right-side tool panel:

    +--------------------------------------------------------------+
    | Toolbar                                                       |
    | (Load | Metrics… Settings… Compute | Map View | Compare       |
    |  Histogram Info Export)                                        |
    +--------------------------------------------------------------+
    |                                    |                          |
    |         Map viewer                 |   Map tools panel        |
    |                                    |   (threshold controls)   |
    |                                    |                          |
    +--------------------------------------------------------------+
    |                                    |                          |
    |         Signal inspector           |   Signal tools panel     |
    |                                    |   (display mode,         |
    |                                    |    envelope toggle)      |
    +--------------------------------------------------------------+
    | Status bar                                                    |
    +--------------------------------------------------------------+

The map section and signal section are separated by a vertical splitter with a default 3:1 ratio (map section larger). Each section is a horizontal pair of the main viewer/inspector and a narrow tool panel.

---

## 4. Map viewer

The map viewer is the central visual element of the application.

### Responsibilities
- display one currently selected 2D result map
- support pixel selection by mouse click
- visually indicate the selected pixel with a red crosshair marker
- support three display modes for the same metric map

### Display modes

The toolbar View selector controls how the current metric map is rendered:

- **score** — raw score map with viridis colormap and colorbar
- **masked** — score map with rejected pixels shown in neutral gray, kept pixels retain the viridis colormap with color range anchored to the full original score map
- **mask_only** — binary mask rendered as green (kept) / red (rejected), no colorbar

Display mode switching does not recompute anything — it only changes the rendering of existing data.

### Interaction
When the user clicks a pixel:
- the pixel is marked with a crosshair
- the signal inspector updates to show the corresponding signal
- the status bar shows pixel coordinates and the current map value at that pixel

### Visual requirements
- the selected pixel is clearly marked with a red crosshair
- the map area should remain large and easy to inspect

---

## 5. Signal inspector

The signal inspector is the lower plot area showing the signal corresponding to the currently selected pixel.

### Display modes

The signal tools panel controls which representation is shown:

- **Raw** — original signal from the dataset, unprocessed
- **Processed** — signal after applying the current preprocessing pipeline and optional ROI extraction
- **Spectrum** — amplitude spectrum computed via FFT, displayed with logarithmic y-scale

### Envelope overlay

When the envelope overlay checkbox is enabled, the envelope (computed via the currently selected envelope method with baseline subtraction) is drawn as a dashed orange line over the signal plot. This overlay is available in Raw and Processed modes.

### Behavior
- the plot updates immediately when the selected pixel changes or display mode changes
- all processing is done by the main window using backend functions — the inspector widget only receives pre-computed data
- axis labels are simple and clear (z/intensity for signals, frequency/amplitude for spectra)

---

## 6. Map tools panel

A narrow fixed-width panel (170px) beside the map viewer containing threshold controls.

### Controls

- **Mask source** — combo box selecting which computed metric provides the score map for threshold evaluation. Populated from computed results.
- **Threshold slider** — horizontal slider mapped to the score range of the mask-source metric (1000 discrete steps)
- **Threshold spinbox** — numeric entry with 4 decimal places, bidirectionally synced with the slider
- **Apply** — computes a ThresholdResult from the mask-source metric's score map at the current threshold value, switches the view to "masked" mode
- **Reset** — clears the current threshold and switches back to "score" mode

The slider and spinbox range is automatically synced to the min/max of valid scores in the mask-source metric whenever the mask-source selection changes.

---

## 7. Signal tools panel

A narrow fixed-width panel (170px) beside the signal inspector containing display mode controls.

### Controls

- **Display mode combo** — Raw / Processed / Spectrum
- **Envelope overlay checkbox** — toggle envelope overlay on/off

---

## 8. Toolbar

A compact fixed toolbar at the top of the window.

### Controls (left to right)

#### Dataset
- **Load** — opens the load dialog

#### Computation
- **Metrics…** — opens the metric selection dialog
- **Settings…** — opens the processing settings dialog
- **Compute** — runs all selected metrics on the loaded dataset

#### Map selection
- **Map** combo — selects which computed metric map to display; populated from computed results
- **View** combo — selects display mode: score / masked / mask_only

#### Inspection
- **Compare** — opens the current map in a standalone comparison window
- **Histogram** — opens a histogram window for the current metric map
- **Info** — opens the information dialog
- **Export** — exports the current map view as `.txt`

The toolbar exposes only high-level actions. Detailed settings use dialogs.

---

## 9. Map switching

The Map combo in the toolbar is populated from the names of all computed metric results in the current session.

When the user selects a different metric name:
- the map viewer updates to show that metric's result
- the display mode (score/masked/mask_only) is preserved
- the selected pixel marker is preserved if possible

This is the primary mechanism for comparing different metrics — compute multiple metrics, then switch between them using the combo.

---

## 10. Comparison windows

The Compare button opens a standalone window showing a fixed snapshot of the currently displayed map.

### Behavior
- the window displays a static copy of the data at the moment it was opened
- it does not update when the main window changes
- multiple comparison windows may be opened simultaneously
- boolean mask data is rendered as green (kept) / red (rejected)
- non-boolean data uses viridis colormap with a colorbar
- windows auto-delete on close

### Purpose
Allows the user to keep one result visible while switching the main window to another result for side-by-side visual comparison.

---

## 11. Histogram windows

The Histogram button opens a standalone snapshot window for the currently displayed metric map.

### Content
- histogram plot of valid pixel score values (50 bins)
- if a threshold is active and built from the same metric, a red dashed vertical threshold line is drawn on the histogram
- descriptive statistics panel (min, max, mean, median, std)
- threshold statistics panel (valid pixels, kept, rejected, kept percentage) — shown only when a threshold is active for that metric

### Behavior
- the window captures a frozen snapshot of the metric map values at the time it is opened
- it does not update when the main window changes
- multiple histogram windows may be opened simultaneously
- windows auto-delete on close

### Threshold applicability
The threshold line and threshold statistics are only shown when the threshold was built from the same metric that is currently displayed. If the displayed metric is different from the mask-source metric, the histogram shows statistics without threshold information.

---

## 12. Dialogs

### 12.1 Metrics selection dialog

A checkbox-based dialog listing all registered metrics.

- each metric appears as a labeled checkbox
- previously selected metrics are pre-checked
- returns the list of selected metric names on accept

This supports multi-metric computation: the user selects which metrics to compute, and all selected metrics are evaluated when Compute is pressed.

### 12.2 Processing settings dialog

A modal dialog for configuring preprocessing, ROI, and envelope settings.

#### Preprocessing group
- **Baseline subtraction** — checkbox
- **Normalize amplitude** — checkbox
- **Smoothing** — checkbox

#### ROI group
- **Enable ROI** — checkbox
- **Segment size** — spinbox (range 3–100,000, default 128)
- **Centering mode** — fixed label "raw_max" (not user-selectable)

#### Envelope group
- **Enable envelope** — checkbox
- **Method** — combo populated from the envelope registry

Returns settings as a plain dict. When settings change, the main window invalidates cached results for processed-input metrics while preserving raw-input metric results.

### 12.3 Load dialog

A small dialog for choosing the loading method and parameters.

- **Source type** combo — image_stack / txt_matrix
- **Path** — browse button; directory selector for image_stack, file selector for txt_matrix
- **Width / Height** spinboxes — enabled only for txt_matrix (range 1–100,000)

Calls the appropriate backend loader on accept and returns a SignalSet.

### 12.4 Information dialog

A read-only scrollable dialog showing current session details as key-value pairs.

Contents include:
- source type, path, dimensions (H x W x M)
- z-axis mode (file or index-based)
- metadata presence and normalized fields
- current preprocessing settings
- ROI state and segment size
- envelope state and method
- list of computed metrics
- currently displayed metric
- mask-source metric
- threshold value, keep rule, and kept pixel count

---

## 13. Session state model

The main window maintains session state that drives all GUI behavior.

### Core session state
- **signal_set** — the loaded dataset (SignalSet or None)
- **selected_metrics** — list of metric names chosen via the Metrics dialog
- **computed_results** — dict mapping metric name to MetricMapResult
- **current_map_name** — name of the metric currently shown in the map viewer
- **display_mode** — current map display mode (score / masked / mask_only)
- **mask_source_metric** — which metric's scores drive the threshold
- **current_threshold** — ThresholdResult or None
- **processing** — dict of preprocessing/ROI/envelope settings
- **signal_display_mode** — Raw / Processed / Spectrum
- **envelope_overlay** — boolean
- **selected_pixel** — (row, col) or None

### Result caching and reuse
- computed metric results are cached in the session and reused across Compute actions
- when processing settings change, results for metrics with `input_policy="processed"` are invalidated, while `input_policy="raw"` results are preserved
- when a new dataset is loaded, all session state is cleared

### Map combo and mask-source combo
Both combos are populated from the keys of computed_results. When results are invalidated, the combos are refreshed and selections fall back to the first available result if the previous selection was removed.

---

## 14. Threshold model

Thresholding is non-destructive and decoupled from the displayed metric.

### Key concepts
- the **mask-source metric** determines which score map is used to compute the threshold
- the threshold produces a separate ThresholdResult with a binary mask
- the mask can be applied as a display overlay on any metric's score map via the View mode selector
- the original score maps are never modified

### Workflow
1. select a mask-source metric in the map tools panel
2. adjust the threshold value via slider or spinbox
3. press Apply — the threshold is computed and the view switches to "masked"
4. switch the Map combo to view other metrics — the same mask overlay applies
5. press Reset — the threshold is cleared and the view returns to "score"

### Slider range
The slider range auto-syncs to the min/max of valid scores in the mask-source metric. When the mask-source selection changes, the slider and spinbox are reset to the new range.

---

## 15. Export

The Export button saves the currently displayed view as a `.txt` file via file dialog.

- in **score** or **masked** mode: exports the score map as a 2D matrix
- in **mask_only** mode: exports the binary mask as a 2D integer matrix (0/1)

Export uses `np.savetxt` with `%.6g` formatting.

---

## 16. Status bar

A single-line status bar at the bottom of the window.

### Content
- pixel coordinates and metric value on pixel selection: `Pixel (row, col) metric_name=value`
- dataset info after loading: source type, dimensions, z-axis mode
- computation progress: `Computing metric_name…`
- computation summary: `N metric(s) available (X new, Y reused)`
- threshold info: `Threshold value on metric_name (N kept)`
- error and state messages

---

## 17. Backend integration

The GUI calls backend functions for all data operations:
- loading datasets (image_stack_loader, txt_matrix_loader)
- preprocessing (subtract_baseline, normalize_amplitude, smooth)
- ROI extraction (extract_roi)
- envelope computation (envelope registry methods)
- spectral computation (compute_spectrum)
- metric evaluation (evaluate_metric_map)
- thresholding (apply_threshold)

The GUI does not duplicate any of this logic. All processing for signal display is performed by the main window using backend functions before passing pre-computed data to the inspector widget.

---

## 18. Workflows

### Workflow A — basic multi-metric inspection
1. Load dataset
2. Open Metrics… dialog, select one or more metrics
3. Press Compute
4. View the first metric's score map
5. Switch between metrics using the Map combo
6. Click pixels to inspect their signals

### Workflow B — threshold and mask overlay
1. Compute one or more metrics
2. Select a mask-source metric in the map tools panel
3. Adjust threshold slider
4. Press Apply
5. View masked result — rejected pixels shown in gray
6. Switch Map combo to another metric — same mask applies
7. Switch View to mask_only to see the binary mask
8. Press Reset to clear the threshold

### Workflow C — compare maps
1. Compute a metric, view its map
2. Press Compare to snapshot the current view
3. Switch to another metric or apply a threshold
4. Press Compare again
5. Arrange the standalone windows for side-by-side comparison

### Workflow D — histogram inspection
1. Compute a metric
2. Optionally apply a threshold
3. Press Histogram
4. Inspect the value distribution, threshold line, and statistics
5. Open additional histograms for other metrics as needed

### Workflow E — signal inspection modes
1. Select a pixel in the map
2. View the raw signal in the signal inspector
3. Switch to Processed to see the preprocessed/ROI-extracted signal
4. Switch to Spectrum to see the amplitude spectrum
5. Enable envelope overlay to see the envelope on Raw or Processed view

### Workflow F — inspect session info
1. Load or compute data
2. Press Info
3. Review dataset properties, processing settings, and computed metrics
4. Close the dialog and continue work

---

## 19. Visual design principles

The GUI should feel:
- technical,
- clean,
- focused,
- minimal.

Priority should go to:
- large readable visualization areas,
- simple controls,
- clear interactions,
- easy interpretation.

The GUI's value comes from clarity and speed of engineering inspection.

---

## 20. Future extensions

The following are not currently implemented but the architecture supports them:
- height maps and additional feature map types in the same viewer
- additional centering modes for ROI extraction
- richer comparison workflows (linked pixel selection across windows)
- additional export formats

These should be added through new dialogs and secondary windows rather than by redesigning the main window.

---

## 21. Summary

The GUI is a desktop interface built around one central concept:

**generic 2D map viewer + signal inspector**

Core elements:
- one main map viewer with right-side threshold tools panel
- one signal inspector with right-side display mode panel
- one compact toolbar for actions and map/view switching
- one status bar for context
- modal dialogs for metrics selection, processing settings, loading, and info
- standalone snapshot windows for comparison and histogram inspection

The GUI remains a thin orchestration layer: all data processing is performed by the backend.
