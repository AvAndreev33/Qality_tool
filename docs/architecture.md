# Quality_tool — Architecture

## 1. Purpose

`Quality_tool` is an internal tool for quality analysis of WLI correlogram signals.

The goal of the system is:
- to load real signal data from supported sources;
- to convert all data into one unified internal representation;
- to evaluate quality criteria on all pixel signals;
- to produce 2D quality maps and threshold masks;
- to support future extension toward additional metrics, envelope methods, and research workflows.

The architecture is intentionally real-data-first and minimal for v0.1.

---

## 2. Core architectural principle

All supported input sources must be converted into one common internal representation.

After loading, the rest of the pipeline must work independently of the original source format.

High-level pipeline:

Input -> Load -> Normalize to SignalSet -> Optional preprocessing / optional envelope -> Evaluate metric -> Threshold -> Visualize / Export

This gives:
- one stable internal data model;
- simple support for multiple input formats;
- easy addition of new metrics;
- easy addition of new envelope methods;
- direct spatial interpretation of results.

---

## 3. Scope of v0.1

This architecture targets a first practical version focused on real data only.

Supported input types:
1. image stack (`.tif`, `.tiff`)
2. text matrix (`.txt`)

Supported sidecar files:
- acquisition info file (for metadata)
- optional `z_axis.txt`

Out of scope for v0.1:
- synthetic signal generation
- benchmark framework
- metric correlation analysis
- advanced export formats
- CLI workflows
- complex agent orchestration

---

## 4. Data representation

## 4.1 Canonical signal format

The canonical internal representation of all loaded datasets is:

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
- signals are naturally tied to image pixels;
- metric results are naturally spatial maps;
- threshold masks are naturally 2D;
- visualization becomes simpler;
- no separate mapping stage is needed for most outputs.

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
- explicit user input;
- parsed acquisition metadata.

Loading is invalid if width and height cannot be determined.

This simplifies:
- shape validation;
- 2D metric outputs;
- 2D threshold masks;
- export and visualization logic.

---

## 4.4 Z-axis policy

The system supports exactly two modes:

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

Sidecar acquisition info files may exist for both image stack and txt input.

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
- load image stack or txt matrix
- determine `width` and `height`
- normalize signals to shape `(H, W, M)`
- parse useful sidecar metadata
- load or generate `z_axis`
- return a valid `SignalSet`

The IO layer must not evaluate metrics.

---

## 5.2 Core model layer

The core model layer defines the main internal dataclasses.

Main entities:
- `SignalSet`
- `MetricResult`
- `MetricMapResult`
- `ThresholdResult`

This layer should remain small and stable.

---

## 5.3 Preprocessing layer

The preprocessing layer contains explicit operations applied to signals before metric evaluation.

Examples:
- baseline subtraction
- normalization
- smoothing
- ROI extraction

Preprocessing must be:
- explicit
- configurable
- reproducible
- easy to extend

No hidden preprocessing should be buried inside metrics unless clearly documented.

---

## 5.4 Envelope layer

Envelope computation must be treated as a separate extendable subsystem.

Reason:
- many metrics will need an envelope;
- envelope is also relevant for WLI height-related workflows;
- different envelope computation methods may need to be compared.

This layer should provide:
- a common interface for envelope methods
- a registry of available methods
- easy addition of new envelope algorithms

Envelope must not be hardcoded inside one specific metric.

---

## 5.5 Metrics layer

The metrics layer contains quality criteria implementations.

Responsibilities:
- evaluate a single signal
- optionally use envelope information
- return scalar score
- return optional diagnostic features
- handle invalid cases gracefully

Metrics must be easy to extend.

---

## 5.6 Evaluation layer

The evaluation layer runs a chosen metric on all pixel signals.

Responsibilities:
- iterate over all signals
- apply preprocessing if configured
- compute envelope if required
- evaluate metric
- collect results
- return 2D metric outputs

Because the canonical signal representation is `(H, W, M)`, metric outputs should naturally be `(H, W)`.

---

## 5.7 Thresholding layer

The thresholding layer applies simple keep/reject logic to metric maps.

Responsibilities:
- apply threshold rule to a metric map
- produce binary mask
- compute basic statistics if needed

The threshold output must naturally follow image layout.

---

## 5.8 Visualization layer

The visualization layer is responsible for:
- plotting selected signals
- showing histograms of metric values
- displaying quality maps
- displaying binary masks

Visualization must remain thin and not perform hidden data transformations.

---

## 5.9 Export layer

The export layer is intentionally minimal in v0.1.

Supported export outputs:
- quality map as `.txt`
- threshold mask as `.txt`

Both exports are saved as 2D matrices.

No additional export formats are required in v0.1.

---

## 6. Core result objects

## 6.1 MetricResult

Represents the result of evaluating one signal.

```python
from dataclasses import dataclass

@dataclass
class MetricResult:
    score: float
    features: dict
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
from dataclasses import dataclass
import numpy as np

@dataclass
class MetricMapResult:
    metric_name: str
    score_map: np.ndarray                # shape = (H, W)
    valid_map: np.ndarray                # shape = (H, W)
    feature_maps: dict[str, np.ndarray]
    metadata: dict
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
    keep_rule: str                       # e.g. "score >= threshold"
    mask: np.ndarray                     # shape = (H, W)
    stats: dict | None = None
```

### Meaning
- `threshold`: threshold value
- `keep_rule`: threshold logic
- `mask`: binary valid/invalid mask
- `stats`: optional summary statistics

---

## 7. Input contracts

## 7.1 Image stack loader

### Input
- path to `.tif` / `.tiff`
- optional explicit path to info file
- optional explicit path to `z_axis.txt`

### Output
A valid `SignalSet`.

### Logic
1. load image stack
2. determine stack dimension order
3. interpret each pixel across stack depth as one signal
4. normalize data to `(H, W, M)`
5. determine `width` and `height`
6. parse sidecar info file if available
7. load `z_axis.txt` if available
8. otherwise create index-based z-axis
9. return `SignalSet`

---

## 7.2 TXT matrix loader

### Input
- path to `.txt`
- `width`
- `height`
- optional parsing settings
- optional explicit path to info file
- optional explicit path to `z_axis.txt`

### Output
A valid `SignalSet`.

### Logic
1. load matrix of shape `(N, M)`
2. validate that `N == width * height`
3. reshape signals to `(H, W, M)`
4. parse sidecar info file if available
5. load `z_axis.txt` if available
6. otherwise create index-based z-axis
7. return `SignalSet`

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

---

## 9. Metric interface

Recommended metric interface:

```python
from typing import Protocol
import numpy as np

class BaseMetric(Protocol):
    name: str

    def evaluate(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray | None = None,
        envelope: np.ndarray | None = None,
        context: dict | None = None,
    ) -> MetricResult:
        ...
```

### Rules
- metric must return one scalar score
- metric may use envelope if needed
- metric may return useful diagnostic features
- metric should fail gracefully through `valid=False` when possible
- metric must not modify input arrays in-place

---

## 10. ROI extraction policy

ROI extraction must be supported as an explicit preprocessing step.

Purpose:
- to crop a local segment from a longer signal
- to evaluate metrics only around the most relevant region

The main parameter is:
- `segmentSize`

The extracted segment must have length `segmentSize`.

### Centering policy
The architecture must allow different future centering methods.

Examples:
- `raw_max`
- `envelope_max`
- other future strategies

For v0.1, the default centering method is:
- `raw_max`

This means ROI is centered around the maximum of the raw signal.

---

## 11. Standard evaluation flow

The standard evaluation flow is:

1. Load data into `SignalSet`
2. Optionally preprocess signals
3. Optionally compute envelope
4. Evaluate one metric for each pixel signal
5. Aggregate results into a `MetricMapResult`
6. Apply threshold to produce `ThresholdResult`
7. Visualize or export results

Because spatial structure is preserved from the start, metric results and masks are already image-shaped.

---

## 12. Repository structure

Recommended minimal repository layout:

```text
quality_tool/
  docs/
    product_spec.md
    architecture.md
    roadmap.md

  src/
    quality_tool/
      core/
        models.py
        types.py

      io/
        image_stack_loader.py
        txt_matrix_loader.py
        metadata_parser.py
        z_axis_loader.py

      preprocessing/
        basic.py
        roi.py

      envelope/
        base.py
        registry.py

      metrics/
        base.py
        registry.py
        baseline/

      evaluation/
        evaluator.py
        thresholding.py

      visualization/
        plots.py
        maps.py

      export/
        txt_export.py

      utils/
        validation.py

  tests/
    test_io/
    test_preprocessing/
    test_envelope/
    test_metrics/
    test_evaluation/
```

---

## 13. Extension rules

### Adding a new metric
To add a new metric, it should be enough to:
1. create the metric module
2. implement the metric interface
3. register the metric
4. add tests

### Adding a new envelope method
To add a new envelope method, it should be enough to:
1. create the method module
2. implement the envelope interface
3. register the method
4. add tests

### Adding a new input source
To add a new source:
1. implement a loader
2. normalize output to `SignalSet`
3. keep downstream logic unchanged

### Adding a new metadata field
To add a new metadata field:
1. update metadata parsing
2. map it to a normalized key
3. keep compatibility with missing-field cases

---

## 14. Non-functional requirements

The system should satisfy the following requirements:
- code must stay simple and readable
- every stage must be testable independently
- no hidden transformations should happen silently
- loaders should fail clearly on invalid shape assumptions
- metadata parsing should be tolerant to missing optional fields
- architecture must remain easy to extend
- real-data workflow must stay the primary focus

---

## 15. v0.1 implementation priorities

Recommended implementation order:
1. core models
2. metadata parser
3. z-axis loader
4. image stack loader
5. txt matrix loader
6. preprocessing basics
7. ROI extraction
8. envelope interface and first method
9. metric interface and registry
10. first baseline metrics
11. evaluator
12. thresholding
13. visualization
14. txt export

---

## 16. Final summary

The architecture is built around one central idea:

**all datasets become `SignalSet(signals[H, W, M])`, and all further processing preserves spatial structure.**

This gives:
- a natural representation for WLI data
- direct 2D metric outputs
- simple threshold masks
- easier visualization
- a stable base for future growth