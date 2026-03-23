# Current iteration

## Iteration name

Spectral priors + spectral metric batch + processed-spectrum band visualization

## Goal

Implement the spectral metric batch together with a metadata-aware spectral-prior layer.

This iteration should:
- compute expected spectral priors from dataset metadata
- integrate those priors into `AnalysisContext`
- implement the spectral metric batch
- expose both:
  - the empirically selected useful band around the detected dominant peak
  - the theoretically expected band derived from metadata
- show both bands in the processed-spectrum view in the GUI

This iteration should make spectral metrics both usable and interpretable.

---

## Why this iteration matters

The project already supports:
- signal recipes
- recipe planning
- representation bundles
- shared analysis context
- envelope/spectrum reuse per recipe
- baseline, noise, regularity, and envelope metric groups
- processed spectrum display in the GUI

The next metric group depends strongly on shared spectral assumptions:
- expected carrier location
- expected useful-band width
- expected working band in positive frequencies

These quantities should not be recomputed independently inside each metric.

At the same time, GUI inspection should make it visually clear:
- which band was selected empirically from the actual spectrum
- which band is expected from metadata/theory

This is useful both for debugging metrics and for tuning spectral assumptions.

---

## Core design direction

The architecture for this iteration should follow this model:

### 1. Metadata-derived spectral priors
Expected spectral quantities should be computed once from dataset metadata and signal length.

### 2. Shared analysis context
These priors should be stored in resolved `AnalysisContext`, not hardcoded inside metrics.

### 3. Shared spectral reuse
Spectral metrics should reuse the already computed processed spectrum from the relevant recipe bundle.

### 4. Dual-band visibility
The GUI should be able to show:
- detected/empirical useful band
- expected/theoretical band

on the same processed-spectrum view.

---

## In scope

### Metadata-aware spectral priors

Extend the resolved `AnalysisContext` so that it can include spectral priors derived from metadata.

At minimum, support:

- `expected_period_samples`
- `expected_carrier_bin`
- `expected_band_half_width_bins`
- `expected_band_low_bin`
- `expected_band_high_bin`

These priors should be derived centrally and stored in the resolved analysis context.

---

### Expected carrier computation

Use the agreed project rule:

#### Oversampling-based expected period
- if `oversampling_factor` is missing, `NaN`, or equal to `1`:
  - `expected_period_samples = 4`
- if `oversampling_factor > 1`:
  - `expected_period_samples = 4 * oversampling_factor`

#### Expected carrier bin
For current signal length `M`:
- `expected_carrier_bin = round(M / expected_period_samples)`

Use the current one-sided positive-frequency convention and clip the result safely into the usable spectral range.

This rule should be implemented centrally, not inside metrics.

---

### Expected band-width computation

Implement a metadata-aware expected-band-width heuristic.

Use:
- `coherence_length_nm`
- effective z-step from available metadata / context
- current signal length `M`

The implementation should:
1. convert coherence length into approximate packet width in samples
2. derive expected spectral band half-width from that packet width
3. store the resulting half-width and full expected band in `AnalysisContext`

If metadata is missing or invalid, fall back cleanly to the existing default band-width logic.

Important:
- the mapping from coherence-width-in-samples to spectral half-width may use a simple centralized scale factor / heuristic
- that factor must live in shared analysis context/defaults, not in metric code

---

### Shared spectral-band helpers

Add any small backend-side helpers needed for:

- expected band construction from priors
- empirical band construction around the detected dominant peak
- safe clipping to valid positive-frequency bins
- consistent exclusion of DC if applicable

Keep this minimal and backend-oriented.

---

### Spectral metric batch implementation

Implement the metrics from `docs/metric_spec_spectrum.md`.

Target metrics:

- `presence_of_expected_carrier_frequency`
- `dominant_spectral_peak_prominence`
- `carrier_to_background_spectral_ratio`
- `energy_concentration_in_working_band`
- `low_frequency_trend_energy_fraction`
- `harmonic_distortion_level`
- `spectral_centroid_offset`
- `spectral_spread`
- `spectral_entropy`
- `spectral_kurtosis`
- `spectral_peak_sharpness`
- `envelope_spectrum_consistency`
- `spectral_correlation_score`

These metrics should be implemented according to the current canonical batch spec, not by ad hoc reinterpretation.

---

### Use the current architecture correctly

The implementation must use the existing architecture properly.

That means:
- use declared signal recipes
- use fixed recipe binding where specified
- use representation bundles and shared analysis context
- reuse already computed spectral representations from the relevant recipe bundle
- reuse envelope where required by mixed metrics such as `envelope_spectrum_consistency`
- avoid duplicating spectral preparation logic inside each metric

This iteration should validate the current architecture on the strongest spectrum-dependent batch so far.

---

### Batch-oriented execution requirement

This iteration must explicitly prefer batch-oriented implementation where practical.

Requirements:
- if a spectral computation or statistic can be implemented over a batch without changing semantics, do so
- if several spectral metrics depend on the same shared band masks, carrier estimates, or normalized spectra, reuse them where practical
- avoid unnecessary Python per-signal loops when vectorized/batch form is clean

This does not require unnatural overengineering.
But the default engineering choice should be:
**batch when practical, scalar fallback only where needed.**

---

### Shared analysis-context support for spectral metrics

Add any small missing shared analysis-context fields needed by the spectral batch.

Typical examples:
- `epsilon`
- `expected_period_samples`
- `expected_carrier_bin`
- `expected_band_half_width_bins`
- `expected_band_low_bin`
- `expected_band_high_bin`
- `prominence_window_bins`
- `prominence_exclusion_half_width_bins`
- `local_spectrum_window_cycles`
- `local_spectrum_hop_cycles`
- any centralized scale factor needed for coherence-to-band-width mapping

If a required shared parameter is referenced by the batch spec and is not yet present in `AnalysisContext`, add it there rather than hardcoding it inside metric code.

Keep these additions minimal and batch-driven.

---

### Reusable spectral support helpers if needed

Small safe support helpers are allowed if needed for clean implementation of the spectral batch.

Examples:
- expected-band mask helper
- harmonic-band helper
- normalized positive-spectrum helper
- local-window spectral helper for spectral correlation score

These helpers should be reusable and backend-side.

Do not build a large new DSP framework in this iteration.
Only add what is needed for this batch.

---

### Invalid-case handling

Each metric must handle invalid or unstable cases explicitly.

Examples:
- empty expected band
- empty usable positive-frequency range
- no robust half-maximum crossings
- invalid local spectral windows
- unstable variance/entropy denominators
- missing metadata-derived reference for `envelope_spectrum_consistency`

Return `valid=False` where appropriate instead of producing misleading scores.

This must follow the current `MetricResult` / batch-result semantics.

---

### GUI integration — metric grouping

Update the GUI metric-selection dialog so that metrics remain shown in visible groups.

For this iteration, it should support at least:
- **Baseline metrics**
- **Noise metrics**
- **Regularity metrics**
- **Envelope metrics**
- **Spectral metrics**

Expected GUI behavior:
- previous groups remain visible in their current order
- then a visible group label or section header appears for `Spectral metrics`
- then the newly added spectral metrics are listed

Keep this simple and readable.

---

### GUI metric naming

The new metrics should appear in the GUI with readable names.

Requirements:
- keep internal metric identifiers stable
- allow or use human-readable display labels if needed
- do not expose confusing raw Python class names to the user

Keep this lightweight and consistent with current GUI grouping behavior.

---

### Processed-spectrum visualization — empirical vs expected band

Extend the processed-spectrum view so that the user can visually compare:

- the empirically selected useful band
- the expected/theoretical band derived from metadata priors

Requirements:
- the processed spectrum should still be shown normally
- mark the empirically selected band around the detected dominant peak
- mark the expected band from `AnalysisContext`
- make the two bands visually distinguishable
- if useful, also mark the expected carrier bin and/or detected dominant carrier bin

The goal is to make it immediately visible:
- what the current data-driven selection used
- what the metadata-derived expectation suggests

This must stay simple and readable.

---

## Out of scope

Do not implement in this iteration:
- new GUI parameter controls for editing spectral priors
- score normalization redesign
- new histogram features
- CUDA backend
- benchmark workflows
- synthetic workflows
- broad new spectral theory beyond the agreed metadata-derived heuristic

This iteration is specifically about metadata-aware spectral priors, spectral metrics, and spectrum-band interpretability.

---

## File targets

Expected modules to update:

- `src/quality_tool/core/analysis_context.py`
- `src/quality_tool/evaluation/evaluator.py` only if small support changes are required
- `src/quality_tool/spectral/fft.py` and/or related spectral helpers
- `src/quality_tool/gui/widgets/signal_inspector.py`
- `src/quality_tool/gui/main_window.py`
- `src/quality_tool/gui/dialogs/metrics_dialog.py`

Expected new metric modules to create under:

- `src/quality_tool/metrics/spectral/`

Suggested files:
- `presence_of_expected_carrier_frequency.py`
- `dominant_spectral_peak_prominence.py`
- `carrier_to_background_spectral_ratio.py`
- `energy_concentration_in_working_band.py`
- `low_frequency_trend_energy_fraction.py`
- `harmonic_distortion_level.py`
- `spectral_centroid_offset.py`
- `spectral_spread.py`
- `spectral_entropy.py`
- `spectral_kurtosis.py`
- `spectral_peak_sharpness.py`
- `envelope_spectrum_consistency.py`
- `spectral_correlation_score.py`

Also update metric registration in the appropriate registry location.

Optional small helper modules may be added if truly needed, but keep structure minimal.

---

## Testing expectations

Add targeted tests for:
- expected-period / expected-carrier / expected-band computation from metadata
- fallback behavior when metadata is missing or invalid
- each new spectral metric on simple synthetic cases
- invalid-case handling
- scalar/batch consistency where applicable
- shared analysis-context parameter usage where practical
- any reusable spectral helper behavior if introduced
- GUI metric-dialog grouping behavior
- processed-spectrum expected/empirical band overlay logic where practical
- presence of new metrics in the GUI selection flow

Keep tests focused and reliable.

---

## Implementation preferences

- implement metrics from the agreed spectral batch spec
- centralize metadata-derived spectral priors
- reuse shared bundles/context/helpers instead of local duplication
- batch-optimize all computations that can be cleanly batch-optimized
- keep scalar fallbacks only where batch form is not practical
- keep metadata additions minimal and explicit
- keep GUI grouping simple
- keep spectrum-band visualization clear and technical
- prefer correctness and interpretability over cleverness
- do not overbuild abstractions during this batch

---

## Definition of done

This iteration is complete when:
- metadata-derived spectral priors are resolved centrally
- expected carrier and expected band are available in `AnalysisContext`
- the spectral metric batch is implemented
- the metrics are registered and usable
- they work with the current recipe/bundle/context architecture
- computations that can be practically batch-optimized are implemented in batch-oriented form
- invalid cases are handled explicitly
- the processed-spectrum view can show both empirical and expected band
- the GUI metric dialog shows spectral metrics as a separate visible group
- readable metric names are shown in the GUI
- tests exist and pass

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs, including `docs/current_iteration.md` and `docs/metric_spec_spectrum.md`
2. summarize the intended spectral-priors + spectral-batch implementation
3. propose a short implementation plan
4. implement only this iteration
5. add targeted tests
6. summarize created files, modified files, and any limitations