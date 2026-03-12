# Current iteration

## Iteration name

Core models + IO scaffold

## Goal

Implement the first practical foundation of the project:
- core internal data structures
- acquisition metadata parsing
- z-axis loading
- image stack loading
- txt matrix loading
- basic validation tests

This iteration is about establishing a correct and stable real-data loading layer.

## Why this iteration matters

Everything downstream depends on correct loading and normalization.

Before metrics, thresholding, and comparison workflows, the project must reliably:
- read real input data
- normalize it to the canonical internal format
- preserve image geometry
- attach useful metadata
- define a valid z-axis

## In scope

### Core models
Implement core dataclasses for:
- `SignalSet`
- `MetricResult`
- `MetricMapResult`
- `ThresholdResult`

These should match the architecture document.

### Metadata parsing
Implement parsing of acquisition info text files.

Behavior:
- parse useful normalized fields only
- ignore irrelevant fields
- tolerate missing optional fields
- do not store the full raw text as internal metadata

Expected normalized fields include:
- `pixel_size_x_mm`
- `pixel_size_y_mm`
- `wavelength_nm`
- `coherence_length_nm`
- `objective_magnification`
- `z_step_nm`
- `z_start_mm`
- `z_end_mm`
- `oversampling_factor`
- `trigger_rate_fps`
- `scan_velocity_um_s`
- `exposure_time_us`
- `scan_distance_during_exposure_nm`
- `periods_during_exposure`
- `illumination_intensity`
- `x_axis_offset_mm`
- `y_axis_offset_mm`

### Z-axis loading
Implement support for:
- explicit `z_axis.txt`
- fallback to index-based `np.arange(M)`

`z_axis` must always exist in `SignalSet`.

### Image stack loader
Implement loader for `.tif` / `.tiff` stacks.

Required behavior:
- load stack data
- normalize to canonical shape `(H, W, M)`
- determine `width` and `height`
- attach metadata if sidecar info file exists
- attach explicit z-axis if `z_axis.txt` exists
- otherwise create index-based z-axis
- return `SignalSet`

### TXT matrix loader
Implement loader for `.txt` signal matrices.

Required behavior:
- read matrix of shape `(N, M)`
- require `width` and `height` as arguments if not otherwise available
- validate `N == width * height`
- reshape to `(H, W, M)`
- attach metadata if sidecar info file exists
- attach explicit z-axis if `z_axis.txt` exists
- otherwise create index-based z-axis
- return `SignalSet`

### Validation and tests
Add tests covering:
- correct `SignalSet` construction
- metadata parsing of representative fields
- fallback behavior when optional metadata is absent
- explicit z-axis loading
- index-based z-axis fallback
- txt reshape from `(N, M)` to `(H, W, M)`
- invalid txt shape mismatch against `width * height`

## Out of scope

Do not implement in this iteration:
- metrics registry
- actual quality metrics
- preprocessing pipeline
- ROI extraction
- envelope methods
- evaluator
- thresholding logic
- visualization
- export
- synthetic signals
- benchmark workflows

## File targets

Expected modules to create:

- `src/quality_tool/core/models.py`
- `src/quality_tool/io/metadata_parser.py`
- `src/quality_tool/io/z_axis_loader.py`
- `src/quality_tool/io/image_stack_loader.py`
- `src/quality_tool/io/txt_matrix_loader.py`

Expected tests to create:

- `tests/test_io/test_metadata_parser.py`
- `tests/test_io/test_z_axis_loader.py`
- `tests/test_io/test_txt_matrix_loader.py`

Add image stack tests too if practical for the current repository setup.

## Implementation preferences

- keep code simple and readable
- keep responsibilities separated
- use dataclasses for core models
- use typed functions where reasonable
- avoid premature abstraction
- avoid hidden behavior
- fail clearly on invalid shapes
- handle optional metadata gracefully

## Definition of done

This iteration is done when:
- the required modules exist
- loaders return valid `SignalSet`
- canonical shape `(H, W, M)` is enforced
- metadata parsing works for the targeted useful fields
- z-axis behavior is correct in both modes
- txt loader validates shape correctly
- tests exist for the implemented functionality
- code is aligned with `docs/architecture.md`

## Expected assistant workflow

1. read `CLAUDE.md` and the docs
2. summarize understanding of this iteration
3. propose a short implementation plan
4. implement only this iteration
5. add tests
6. summarize changes and open follow-up items