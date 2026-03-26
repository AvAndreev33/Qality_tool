# Quality_tool — Architecture

## 1. Purpose

This document defines the architecture of `Quality_tool`.

`Quality_tool` is a modular WLI research workbench with a practical real-data-first core.
Its current purpose is to:

- load real WLI signal data,
- normalize it into a unified internal representation,
- evaluate signal-quality criteria,
- build 2D result maps,
- apply threshold-based masking,
- support interactive inspection through a thin GUI layer,
- remain extensible toward broader experimentation workflows.

The architecture supports:
- stable reference CPU computation,
- optional CUDA-accelerated evaluation via CuPy (see `cuda_backend_spec.md`),
- rapid addition of new metrics and derived representations.

---

## 2. Core architectural principle

All supported input sources must be converted into one common internal representation.

After loading, the rest of the pipeline must work independently of the original source format.

High-level pipeline:

Input -> Load -> Normalize to SignalSet -> Optional preprocessing / ROI / envelope / spectral derivation -> Evaluate metric(s) -> Build result maps -> Threshold / mask -> Visualize / export

This gives:
- one stable internal data model,
- clean separation between IO and analysis,
- consistent behavior across input formats,
- direct spatial interpretation of results,
- a stable base for future acceleration backends.

---

## 3. Scope of the current architecture

This architecture currently covers:

- real-data loading,
- metadata parsing,
- z-axis handling,
- preprocessing,
- ROI extraction,
- envelope support,
- spectral support,
- metric evaluation,
- thresholding,
- histogram/statistical inspection support,
- a thin desktop GUI over the backend.

Out of scope for the current architecture:
- synthetic signal generation,
- benchmark orchestration,
- large-scale experiment management.

The architecture should remain compatible with those future directions.

---

## 4. Data representation

## 4.1 Canonical signal format

The canonical internal representation of loaded datasets is:

```python
signals.shape == (H, W, M)
```

Where:
- `H` = image height
- `W` = image width
- `M` = number of samples in each signal

This format must be used regardless of input source.

### Rationale

This is preferred over `(N, M)` because:
- signals are tied to image pixels,
- metric outputs are naturally spatial maps,
- threshold masks are naturally 2D,
- visualization becomes simpler,
- no separate mapping stage is required for most outputs.

For internal batch computation, temporary reshaping is allowed:

```python
signals_2d = signals.reshape(H * W, M)
```

But this is only a computation view, not the canonical stored format.

---

## 4.2 SignalSet

`SignalSet` is the central internal representation of loaded data.

Recommended definition:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class SignalSet:
    signals: np.ndarray                  # shape = (H, W, M)
    width: int
    height: int
    z_axis: np.ndarray                   # shape = (M,)
    metadata: dict | None = None
    source_type: str = "unknown"
    source_path: str | None = None
    info_path: str | None = None
    z_axis_path: str | None = None
```

### Rules

- `signals` must always be 3D.
- `width` and `height` are mandatory.
- `signals.shape[0] == height`
- `signals.shape[1] == width`
- `signals.shape[2] == len(z_axis)`
- `z_axis` must always exist.
- `metadata` contains only normalized useful fields.
- raw text of info files must not be stored inside `SignalSet`.

---

## 4.3 Width and height policy

`width` and `height` are always required.

They must come from one of the following:
- explicit user input,
- parsed acquisition metadata,
- frame dimensions in the case of image-stack loading.

Loading is invalid if width and height cannot be determined.

This simplifies:
- shape validation,
- 2D metric outputs,
- 2D threshold masks,
- export and visualization logic.

---

## 4.4 Z-axis policy

The system supports exactly two current modes:

### Mode A — explicit physical z-axis
If `z_axis.txt` exists, it is loaded as a 1D array of length `M` and used as the dataset z-axis.

### Mode B — index-based z-axis
If `z_axis.txt` does not exist, the z-axis is defined as:

```python
np.arange(M)
```

This means the signal is represented sample-by-sample without physical units.

### Rule
`z_axis` must always exist in `SignalSet`, even if it is only an index axis.

---

## 4.5 Normalized metadata policy

Sidecar acquisition info files may exist for both image-stack and txt input.

These files may contain many fields, but only useful normalized metadata should be stored.

Recommended fields:

### Geometry and scale
- `pixel_size_x_mm`
- `pixel_size_y_mm`

### Optics
- `wavelength_nm`
- `coherence_length_nm`
- `objective_magnification`

### Scanning
- `z_step_nm`
- `z_start_mm`
- `z_end_mm`
- `oversampling_factor`

### Motion and timing
- `trigger_rate_fps`
- `scan_velocity_um_s`
- `exposure_time_us`
- `scan_distance_during_exposure_nm`
- `periods_during_exposure`

### Additional useful fields
- `illumination_intensity`
- `x_axis_offset_mm`
- `y_axis_offset_mm`

### Important rule
The whole raw info file must not be copied into the internal representation.

---

## 5. Layered architecture

## 5.1 IO layer

The IO layer is responsible for reading input files and converting them into `SignalSet`.

Modules:
- `image_stack_loader.py`
- `txt_matrix_loader.py`
- `metadata_parser.py`
- `z_axis_loader.py`

Responsibilities:
- load image-stack or txt-matrix data,
- determine `width` and `height`,
- normalize signals to shape `(H, W, M)`,
- parse useful sidecar metadata,
- load or generate `z_axis`,
- return a valid `SignalSet`.

The IO layer must not evaluate metrics.

---

## 5.2 Input contracts

### Image stack input

For this project, image-stack input is defined as:

- a directory containing sequential TIFF frames,
- files named in a stable ordered pattern such as:
  - `Image_00001.tif`
  - `Image_00002.tif`
  - ...
- each file represents one z-slice / signal sample layer.

The image-stack loader must:
1. accept a directory path,
2. discover TIFF frame files,
3. sort them in numeric order,
4. validate that all frames have the same `(H, W)`,
5. stack them into `(H, W, M)`.

Associated files may exist in the same directory:
- `image_stack_info.txt`
- `z_axis.txt`

Single multipage TIFF is not the primary contract of this project.

### TXT matrix input

TXT input is defined as:
- a matrix of shape `(N, M)`
- where `N == width * height`

The txt loader must:
1. read the matrix,
2. validate `N == width * height`,
3. reshape into `(H, W, M)`,
4. attach metadata if available,
5. load or generate `z_axis`.

Associated files may exist beside the txt file:
- `image_stack_info.txt` or equivalent sidecar info file
- `z_axis.txt`

---

## 5.3 Core model layer

The core model layer defines the main internal dataclasses.

Main entities:
- `SignalSet`
- `MetricResult`
- `MetricMapResult`
- `ThresholdResult`

This layer should remain small, explicit, and stable.

---

## 5.4 Preprocessing layer

The preprocessing layer contains explicit operations applied to signals before metric evaluation.

Examples:
- baseline subtraction,
- normalization,
- smoothing,
- ROI extraction.

Preprocessing must be:
- explicit,
- configurable,
- reproducible,
- easy to extend.

No hidden preprocessing should be buried inside metrics unless clearly documented as part of the metric’s own definition.

The preprocessing layer provides both per-signal functions (`basic.py`) and vectorized batch equivalents (`batch.py`) for use by the evaluator.

---

## 5.5 ROI layer

ROI extraction is treated as an explicit part of signal preparation.

Purpose:
- crop a local segment from a signal,
- evaluate downstream operations on that segment.

Current main parameter:
- `segment_size`

Current centering behavior:
- the architecture must allow different centering modes,
- current default mode is based on the raw-signal maximum.

ROI is not implicitly applied to all metrics.
It is part of the processing pipeline and must be selected explicitly.

---

## 5.6 Envelope layer

Envelope computation is a separate extensible subsystem.

Reason:
- many metrics may use envelope information,
- envelope is also useful for signal inspection and broader WLI workflows,
- different envelope methods may need to be compared.

The layer should provide:
- a common interface for envelope methods,
- a registry of available methods,
- reusable envelope computation independent of any one metric.

Envelope must not be hardcoded inside one specific metric.

The current Hilbert-based implementation must expose the envelope itself, not the raw analytic signal.

Envelope methods may optionally provide a `compute_batch` method for vectorized evaluation over a 2-D `(N, M)` array of signals. The evaluator will prefer `compute_batch` when available and fall back to per-signal `compute` otherwise.

---

## 5.7 Spectral layer

Spectral computation is also a separate derived-representation layer.

Reason:
- multiple metrics may use FFT-derived information,
- spectral inspection is useful in the GUI,
- future performance work will likely target this layer.

The spectral layer should provide a simple shared API that returns a consistent spectral representation.

Current expected output is intentionally minimal:
- frequencies
- amplitude spectrum

The spectral layer must remain independent of any specific metric.

---

## 5.8 Metrics layer

The metrics layer contains signal-quality criteria implementations.

Responsibilities:
- evaluate a single signal,
- optionally use derived representations such as envelope or spectrum,
- return scalar score,
- return optional diagnostic features,
- handle invalid cases gracefully.

Metrics must remain easy to add and test.

---

## 5.9 Signal recipe and recipe binding

Each metric declares a `SignalRecipe` (what preprocessing to apply) and a `recipe_binding` that controls how the effective recipe is resolved:

- `"fixed"` — the metric always uses its declared recipe, regardless of the active session pipeline.
- `"active"` — the metric uses the current active processing pipeline from the GUI/session.

A `SignalRecipe` is a frozen dataclass describing preprocessing steps (baseline, detrend, normalize, smooth, ROI). Pre-defined recipes include `RAW` (identity), `ROI_ONLY`, and `ROI_MEAN_SUBTRACTED_LINEAR_DETRENDED`.

The evaluator groups metrics by effective recipe via a planner, prepares signals once per group, and builds a `RepresentationBundle` containing shared derived representations (envelope, spectral) for the group. Each representation is computed at most once per recipe per chunk.

---

## 5.10 Evaluation layer

The evaluation layer runs metrics on all signals of a `SignalSet`.

Responsibilities:
- iterate over all pixel signals,
- prepare raw signal and processed signal representations as needed,
- choose the correct effective signal according to the metric input policy,
- optionally compute envelope,
- optionally compute spectral representation when the metric declares `needs_spectral`,
- pass the correct context into the metric,
- collect results,
- return 2D metric outputs.

Because the canonical signal representation is `(H, W, M)`, metric outputs should naturally be `(H, W)`.

The evaluator must keep input-selection logic explicit and readable.

---

## 5.11 Batch evaluation architecture

The evaluator uses a chunked batch-evaluation strategy for performance.

### Execution model

Signals are reshaped from `(H, W, M)` to `(N, M)` and processed in configurable chunks (default chunk size: 50,000 signals). This bounds memory usage while enabling vectorized computation within each chunk.

### Batch metric interface

Metrics may optionally implement `evaluate_batch` for vectorized evaluation over a 2-D chunk of `(N, M)` signals. The evaluator will prefer `evaluate_batch` when available and fall back to the per-signal `evaluate` otherwise.

Batch evaluation returns a `BatchMetricArrays` container:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class BatchMetricArrays:
    scores: np.ndarray              # shape = (N,), invalid entries are np.nan
    valid: np.ndarray               # shape = (N,), boolean
    features: dict[str, np.ndarray] # each value is shape = (N,)
```

This flat-array representation avoids per-pixel object overhead and enables efficient aggregation into `(H, W)` result maps.

### Batch preprocessing

The preprocessing layer provides vectorized batch equivalents of per-signal functions in `preprocessing/batch.py`:
- `subtract_baseline_batch`
- `normalize_amplitude_batch`
- `smooth_batch`
- `extract_roi_batch`

The evaluator auto-resolves per-signal preprocessing functions to their batch equivalents when available.

### Conditional spectral computation

The evaluator only computes batch FFT when the metric declares `needs_spectral = True`. Spectral results are passed to the metric via the `context` dict.

---

## 5.12 Thresholding layer

The thresholding layer applies simple keep/reject logic to a metric map.

Responsibilities:
- apply threshold rule to a source metric map,
- produce binary mask,
- compute summary statistics,
- keep thresholding non-destructive.

Threshold output must naturally follow image layout.

### Current semantic model
- thresholding produces a separate `ThresholdResult`
- the original score map remains unchanged
- a mask may be produced from one metric and applied as a display/filter layer to another displayed map in the GUI

This distinction must remain explicit.

---

## 5.13 Histogram / distribution analysis layer

Histogram-based inspection is part of the analysis layer built on top of existing metric maps.

Responsibilities:
- visualize value distribution of a currently selected metric map,
- show threshold position on that distribution,
- provide compact descriptive statistics,
- provide kept/rejected statistics when thresholding is active.

Histogram windows operate on already computed session results.
They do not perform metric computation themselves.

---

## 5.14 Visualization / GUI layer

The GUI is a thin desktop layer over the backend.

Its purpose is to provide:
- dataset loading,
- map viewing,
- 3D surface viewing (hardware-accelerated OpenGL with LOD),
- signal inspection,
- metric selection,
- threshold interaction,
- histogram inspection,
- comparison windows.

The GUI must not duplicate backend logic for:
- loading,
- preprocessing,
- ROI,
- envelope,
- spectral computation,
- metric evaluation,
- thresholding.

The GUI should remain an orchestration and visualization layer only.

### 3D map viewer

The 3D surface viewer (`Map3DWindow`) uses pyqtgraph.opengl for hardware-accelerated rendering with matplotlib CPU fallback.

Key design decisions:
- **Reusable window**: the GL context is created once and persisted; `closeEvent` hides instead of destroying. This avoids pyqtgraph's global shader-cache invalidation on context destruction.
- **Interactive LOD**: during mouse interaction a decimated mesh (~120K vertices) is shown; full resolution (~500K vertices cap) restores on mouse release.
- **Color mapping**: computed on already-decimated arrays via matplotlib colormaps, passed as flat `(N, 4)` to work around pyqtgraph 0.14 per-face color indexing bug.
- **Z normalization**: geometry normalized to `[-0.5..+0.5]` for stable positioning; colors mapped from original values.
- **Controls**: left-drag = orbit, middle-drag = pan, wheel = zoom, Space = reset to home view.
- **Axes**: white `GLLinePlotItem` lines with `GLTextItem` labels (row, col, value) from origin.

Implementation: `src/quality_tool/gui/windows/map_3d_window.py`.

---

## 5.15 Export layer

The export layer remains intentionally simple.

Current required outputs:
- quality/result maps as `.txt`
- masks as `.txt`

Additional export functionality may be added later, but the architecture should keep export separate from GUI widgets and separate from metric computation logic.

---

## 6. Core result objects

## 6.1 MetricResult

Represents the result of evaluating one signal.

```python
from dataclasses import dataclass, field

@dataclass
class MetricResult:
    score: float
    features: dict = field(default_factory=dict)
    valid: bool = True
    notes: str = ""
```

### Meaning
- `score`: scalar metric value
- `features`: optional diagnostic outputs
- `valid`: whether evaluation was successful
- `notes`: optional explanation for invalid or special cases

---

## 6.2 MetricMapResult

Represents the aggregated metric output for a full image.

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class MetricMapResult:
    metric_name: str
    score_map: np.ndarray                # shape = (H, W)
    valid_map: np.ndarray                # shape = (H, W)
    feature_maps: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
```

### Meaning
- `metric_name`: evaluated metric
- `score_map`: scalar metric values for all pixels
- `valid_map`: pixel-wise validity flags
- `feature_maps`: optional extra feature maps
- `metadata`: evaluation metadata

---

## 6.3 ThresholdResult

Represents the thresholding output.

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ThresholdResult:
    threshold: float
    keep_rule: str                       # current logic remains explicit
    mask: np.ndarray                     # shape = (H, W)
    stats: dict | None = None
```

### Meaning
- `threshold`: threshold value
- `keep_rule`: threshold logic description
- `mask`: binary valid/invalid mask
- `stats`: optional summary statistics

---

## 7. Metric interface

Recommended metric interface:

```python
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class BaseMetric(Protocol):
    name: str
    input_policy: str

    def evaluate(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelope: np.ndarray | None = None,
        context: dict | None = None,
    ) -> MetricResult:
        ...
```

### Required attributes
- `name`: unique metric identifier
- `input_policy`: declares whether the metric uses `raw` or `processed` signal input

### Optional attributes
- `needs_spectral`: when `True`, the evaluator precomputes batch FFT and passes spectral data via the `context` dict. Defaults to `False` when absent.

### Optional methods
- `evaluate_batch`: vectorized evaluation over `(N, M)` signals, returning `BatchMetricArrays`. The evaluator prefers this when available.

### Rules
- metric must return one scalar score
- metric may use envelope if needed
- metric may use spectral/context data if needed
- metric must not modify input arrays in-place
- metric should fail gracefully through `valid=False` when possible

---

## 8. Envelope interface

Envelope computation should use a dedicated common interface.

Recommended idea:

```python
from typing import Protocol
import numpy as np

class BaseEnvelopeMethod(Protocol):
    name: str

    def compute(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        context: dict | None = None,
    ) -> np.ndarray:
        ...
```

### Rules
- output envelope must have the same length as the input signal
- the method must be deterministic
- the method must not modify input arrays in-place
- new methods must be easy to register

### Optional batch support
Envelope methods may implement `compute_batch(signals, z_axis=None, context=None)` for vectorized `(N, M)` input. The evaluator will use it when available.

---

## 9. Standard evaluation flow

The standard evaluation flow is:

1. Load data into `SignalSet`
2. Keep raw signal available
3. Optionally prepare processed signal path
4. Optionally compute envelope on the effective signal path where appropriate
5. Optionally compute spectral representation where appropriate
6. Evaluate one or more metrics, each using its declared input policy
7. Aggregate results into `MetricMapResult`
8. Optionally apply threshold to produce `ThresholdResult`
9. Visualize or export results

Because spatial structure is preserved from the start, metric results and masks are already image-shaped.

---

## 10. Result reuse / session semantics

Computed results may be reused within a session, but reuse must respect evaluation semantics.

### Rule
Result reuse must depend on:
- dataset identity,
- metric identity,
- effective signal path semantics.

### Practical implication
- raw-input metrics may remain reusable across preprocessing-setting changes that do not affect raw input
- processed-input metrics must depend on the relevant processing configuration

This should remain simple, explicit, and correct.
A large caching framework is not required at this stage.

---

## 11. GUI state semantics

The GUI may store session state such as:
- selected metrics,
- computed metric results,
- threshold results,
- active displayed map,
- active processing settings,
- active envelope settings,
- signal display mode,
- histogram snapshot windows.

However, GUI state must not become the source of truth for metric semantics.
Backend rules such as metric input policy must stay in backend-side architecture.

---

## 12. Repository structure

Current repository layout:

```text
quality_tool/
  docs/
    product_spec.md
    architecture.md
    roadmap.md
    current_iteration.md
    gui_spec.md                    # future
    performance_notes.md           # future
    metric_authoring.md            # future

  src/
    quality_tool/
      core/
        models.py

      io/
        image_stack_loader.py
        txt_matrix_loader.py
        metadata_parser.py
        z_axis_loader.py

      preprocessing/
        basic.py
        batch.py
        roi.py

      envelope/
        base.py
        registry.py
        analytic.py

      spectral/
        fft.py
        priors.py

      metrics/
        base.py
        registry.py
        batch_result.py
        baseline/
        envelope/
        spectral/
        noise/
        phase/
        correlation/
        regularity/

      evaluation/
        evaluator.py
        planner.py
        bundle.py
        recipe.py
        thresholding.py

      cuda/
        __init__.py
        _backend.py
        _evaluator.py

      gui/
        app.py
        main_window.py
        style.py
        widgets/
        dialogs/
        windows/
          map_3d_window.py
          compare_window.py
          histogram_window.py
          pixel_metrics_chart_window.py
          pixel_metrics_table_window.py

  tests/
    test_io/
    test_preprocessing/
    test_envelope/
    test_spectral/
    test_metrics/
    test_evaluation/
    test_gui/
```

---

## 13. Extension rules

### Adding a new metric
To add a new metric, it should be enough to:
1. create the metric module,
2. implement the metric interface,
3. declare its input policy,
4. register it,
5. add tests.

### Adding a new envelope method
To add a new envelope method, it should be enough to:
1. create the method module,
2. implement the envelope interface,
3. register the method,
4. add tests.

### Adding a new input source
To add a new source:
1. implement a loader,
2. normalize output to `SignalSet`,
3. keep downstream logic unchanged.

### Adding a new metadata field
To add a new metadata field:
1. update metadata parsing,
2. map it to a normalized key,
3. keep compatibility with missing-field cases.

### CUDA backend
The CUDA backend (`quality_tool.cuda`) provides GPU-accelerated evaluation for all 39 metrics via CuPy. It mirrors the CPU evaluator interface and returns identical result types. The CPU path remains the reference implementation; the GPU path is an optional acceleration layer with automatic fallback. See `cuda_backend_spec.md` for details.

---

## 14. Non-functional requirements

The system should satisfy the following requirements:
- code must stay simple and readable,
- every stage must be testable independently,
- no hidden transformations should happen silently,
- loaders should fail clearly on invalid shape assumptions,
- metadata parsing should be tolerant to missing optional fields,
- architecture must remain easy to extend,
- real-data workflow must stay the primary focus,
- CPU implementation should remain a reliable reference path,
- performance-oriented changes should preserve semantic correctness.

---

## 15. Current implementation priorities

The current practical priorities are:
1. maintain a stable working real-data workbench
2. preserve correctness of metric semantics
3. extend toward height-map computation workflows
4. improve usability of inspection and comparison
5. leverage CUDA backend for performance on supported hardware

---

## 16. Final summary

The architecture is built around one central idea:

**all datasets become `SignalSet(signals[H, W, M])`, and all further processing preserves spatial structure while explicitly controlling which signal representation each metric is allowed to use.**

That gives:
- a natural representation for WLI data,
- direct 2D metric outputs,
- simple threshold masks,
- clear separation between raw and processed metric semantics,
- easier visualization,
- a stable base for future performance work and future research expansion.