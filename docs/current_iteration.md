# Current iteration

## Iteration name

Metric cleanup + normalized map view + canonical processed signal + autocorrelation inspection

## Goal

Refine the current analysis and inspection workflow before the next metric batches.

This iteration should:
- remove three spectral metrics that are no longer wanted,
- add a `normalized_score` map-view mode,
- add two new signal-inspector modes:
  - `Canonical processed`
  - `Autocorrelation`
- keep the current existing signal-display modes intact,
- make the envelope overlay cleaner and independent from whether an envelope happened to be computed during metric evaluation.

This is a semantic/inspection refinement iteration.
It should improve clarity and usability without redesigning the GUI.

---

## Why this iteration matters

The project already has:
- multiple implemented metric groups,
- normalized comparison scores,
- grouped metric selection in the GUI,
- signal inspection modes,
- processed-spectrum visualization,
- metric comparison windows.

Before adding more functionality, it is useful to:
- simplify the metric set slightly,
- make map viewing more comparison-friendly,
- expose autocorrelation directly in the signal inspector,
- clarify what "processed signal" means in inspection mode,
- decouple envelope viewing from metric-execution side effects.

This makes the tool cleaner and more interpretable for further development.

---

## In scope

### Metric cleanup

Remove the following metrics from the implemented metric set:

- `low_frequency_trend_energy_fraction`
- `harmonic_distortion_level`
- `spectral_correlation_score`

This removal should be reflected consistently in:
- metric registration,
- GUI metric selection,
- tests,
- any metric-group presentation logic,
- any supporting documentation/spec usage that depends on the currently implemented set.

Do not leave dead registration or GUI entries behind.

---

### Map view mode: `normalized_score`

Extend the current map-view mode selector so that it can show:

- `score`
- `normalized_score`
- `masked`
- `mask_only`

Requirements:
- `normalized_score` must use the already implemented normalization/comparison layer
- native metric score maps must remain preserved
- switching to `normalized_score` must only change visualization
- thresholding semantics must remain based on native metric scores, not normalized scores
- switching back to `score` must restore the original map view

This must be a clean display-layer feature, not a change to metric computation.

---

### Signal inspector — preserve current modes, add new ones

Do **not** remove existing signal-inspector modes that already exist in the implementation.

For this iteration:
- preserve the current modes already implemented,
- add the following new modes:

#### `Canonical processed`
This mode should show the canonical processed signal:

`roi_mean_subtracted_linear_detrended`

This should be treated as the standard backend-side canonical processed representation used by many metrics.

Important:
- this does **not** replace the existing processing/settings-based processed mode
- both should remain available
- the new mode is an additional explicit signal-inspection mode

#### `Autocorrelation`
This mode should display the normalized autocorrelation of the currently selected pixel signal.

The autocorrelation should be computed from the relevant currently displayed signal mode where appropriate, using clear and consistent logic.

---

### Autocorrelation expected-period visualization

In the new `Autocorrelation` mode, visualize expected-period guidance from `AnalysisContext`.

Requirements:
- show the normalized autocorrelation curve
- show the expected period location
- show the expected lag search interval/window around that period
- if practical, also show the detected dominant autocorrelation peak in that interval

The purpose is to make regularity-related behavior visually interpretable, similarly to how the useful band is shown in the spectral view.

Keep this simple and readable.

---

### Envelope overlay cleanup

Refine envelope-overlay behavior in the signal inspector.

New intended behavior:
- envelope overlay should be computed from the signal currently being displayed in the signal viewer
- it should not depend on whether an envelope happened to be computed earlier for some metric
- it should behave as a viewer-side inspection tool, not as a side effect of metric execution

Requirements:
- envelope overlay should work where it is meaningful
- if a displayed mode is not suitable for envelope overlay, the control should be disabled or ignored cleanly
- no hidden dependency on metric bundle state should remain

This change should make signal inspection cleaner and more predictable.

---

### Preserve existing workflow

The following behavior must remain stable:
- dataset loading
- metric computation
- grouped metric selection
- thresholding
- histogram windows
- per-pixel metric inspector
- processed-spectrum visualization
- current signal display modes already present in the implementation

This iteration should add/refine inspection behavior, not destabilize the existing workflow.

---

## Out of scope

Do not implement in this iteration:
- zoom/pan/reset for map viewer
- 3D map plotting
- new metric groups
- thresholding redesign
- normalization redesign
- CUDA backend
- major GUI layout changes

These belong to later iterations.

---

## File targets

Expected modules to update:

- `src/quality_tool/gui/main_window.py`
- `src/quality_tool/gui/widgets/signal_inspector.py`
- `src/quality_tool/gui/dialogs/metrics_dialog.py`
- metric registration modules
- normalization/view-handling code if needed

Metric modules/tests for the removed metrics may also need cleanup.

Only modify other files if truly necessary.

---

## Testing expectations

Add targeted tests for:
- removal of the three metrics from registration / GUI availability
- `normalized_score` view-mode behavior
- preservation of native score maps when normalized view is selected
- addition of `Canonical processed` mode
- addition of `Autocorrelation` mode
- expected-period / expected-window visualization data generation for autocorrelation
- envelope overlay behavior based on the currently displayed signal
- no regression of existing signal modes and current GUI workflow

Keep tests focused and reliable.

---

## Implementation preferences

- keep the GUI thin and backend-driven
- preserve existing implemented signal modes
- add new signal modes without removing current ones
- keep normalized-score view as a pure display-layer feature
- keep thresholding tied to native scores
- keep envelope overlay viewer-side and explicit
- keep autocorrelation visualization simple and readable
- avoid unrelated cleanup outside this iteration

---

## Definition of done

This iteration is complete when:
- the three specified spectral metrics are removed cleanly
- `normalized_score` is available as a map-view mode
- existing signal-inspector modes remain intact
- `Canonical processed` mode is added
- `Autocorrelation` mode is added
- expected-period guidance is shown in autocorrelation view
- envelope overlay is computed from the currently displayed signal instead of relying on prior metric execution state
- the current GUI workflow remains stable

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs
2. summarize the intended cleanup/inspection refinement
3. propose a short implementation plan
4. implement only this iteration
5. add targeted tests
6. summarize created files, modified files, and any limitations