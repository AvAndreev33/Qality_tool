# Quality_tool — Product Specification

## 1. Purpose

`Quality_tool` is an internal tool for analysis of signal quality in white-light interferometry (WLI).

The first goal of the tool is practical and focused:

- load real correlogram data;
- evaluate selected quality criteria on all pixel signals;
- build quality maps;
- apply thresholding;
- see which pixels are accepted or rejected by a chosen criterion;
- compare the behavior of different criteria on the same real dataset.

The tool is not intended to start as a general research platform.  
Its first version must solve a concrete real-data workflow in a clear and reliable way.

---

## 2. Main user

The main user of v0.1 is:

- an engineer or researcher working with real WLI data;
- interested in understanding how different signal-quality criteria behave;
- interested in seeing which pixels are filtered out by a selected threshold;
- interested in fast visual and numerical inspection of results.

The tool is designed for technical users, not for general end users.

---

## 3. Product vision

The product should evolve in stages.

### Stage 1
A reliable and minimal real-data-first tool for:
- loading real signals;
- computing baseline quality metrics;
- visualizing metric maps and masks;
- exporting core results.

### Stage 2
A more mature comparison tool for:
- evaluating more metrics;
- comparing criteria more systematically;
- improving repeatability and usability.

### Stage 3
A broader experimental platform for:
- synthetic signals;
- controlled perturbations;
- benchmark workflows;
- development of new composite criteria.

The first implementation must focus only on Stage 1.

---

## 4. Scope of v0.1

v0.1 includes only the real-data workflow.

### v0.1 must support

- loading image stack data from `.tif` / `.tiff`;
- loading signal matrix data from `.txt`;
- reading useful acquisition metadata from sidecar info files when available;
- reading `z_axis.txt` when available;
- converting all datasets to a unified internal representation;
- evaluating baseline quality metrics;
- optionally using envelope-based processing where needed;
- optionally extracting a signal ROI using `segmentSize`;
- building 2D quality maps;
- applying thresholding to produce binary masks;
- visualizing signals, distributions, maps, and masks;
- exporting quality map and mask as `.txt`.

### v0.1 does not include

- synthetic signal generation;
- controlled perturbation framework;
- benchmark datasets and benchmark runner;
- metric correlation matrices;
- automatic selection of the best metric;
- machine-learning ranking;
- advanced export formats;
- CLI workflows as a primary mode;
- full research automation.

---

## 5. Supported input data

## 5.1 Image stack input

The tool must support image stacks in `.tif` / `.tiff` format.

Interpretation:
- each image corresponds to one sample position along the signal axis;
- each pixel across the stack defines one signal.

The resulting dataset must preserve spatial structure.

---

## 5.2 TXT matrix input

The tool must support `.txt` input containing a matrix of signals.

Input format:
- shape `(N, M)`
- `N` = number of signals
- `M` = number of samples per signal

The user must provide `width` and `height` if they are not available from metadata.

The loader must validate:

```text
N = width * height
```

Then the dataset must be reshaped to image-aware form.

---

## 5.3 Width and height policy

`width` and `height` are always required.

They must come from:
- acquisition metadata, or
- explicit user input.

A dataset is not valid for loading if width and height cannot be determined.

This is a product-level rule, not an optional convenience.

---

## 5.4 Sidecar info files

For both image stack and txt input, a sidecar acquisition info file may exist.

Examples:
- `image_stack_info.txt`
- similar info file next to txt data

The tool should:
- attempt to find such a file automatically;
- parse useful values from it;
- use them for interpretation and downstream processing.

The whole raw text file should not be treated as main product data.  
Only useful normalized metadata should be kept internally.

Useful metadata may include:
- wavelength;
- coherence length;
- objective magnification;
- pixel size;
- z-step information;
- scan velocity;
- trigger rate;
- exposure time;
- illumination intensity;
- axis offsets.

---

## 5.5 Z-axis input

The tool supports two z-axis modes.

### Mode A — explicit z-axis
If `z_axis.txt` exists, it must be loaded and used as the physical axis for the signal.

### Mode B — index axis
If `z_axis.txt` does not exist, the signal axis must be treated as sample index:

```text
0, 1, 2, ..., M-1
```

This allows the tool to work both with physically calibrated datasets and with sample-based datasets.

---

## 6. Unified internal representation

All loaded datasets must be converted to one common internal representation.

Product-level expectation:

```text
signals.shape = (H, W, M)
```

Where:
- `H` = image height
- `W` = image width
- `M` = signal length

This gives the product several important advantages:
- direct spatial interpretation;
- direct 2D metric outputs;
- direct 2D masks;
- simpler visualization;
- easier reasoning about pixels and maps.

This internal representation is not exposed to the user as a technical burden, but it defines consistent behavior across all workflows.

---

## 7. Main product workflow

The expected user workflow in v0.1 is:

1. Load a real dataset.
2. Confirm that width, height, and signal length are correct.
3. See whether acquisition metadata and z-axis were found.
4. Optionally inspect several example signals.
5. Choose a quality metric.
6. Optionally configure preprocessing.
7. Optionally configure envelope method if the metric requires it.
8. Optionally configure ROI extraction via `segmentSize`.
9. Evaluate the metric on all pixel signals.
10. View the resulting quality map.
11. Choose a threshold.
12. View the valid/invalid mask.
13. Export the quality map and/or mask as `.txt`.

This workflow should be simple, explicit, and easy to repeat.

---

## 8. Metrics in v0.1

The first version should support baseline signal-quality metrics.

The product requirement is not to maximize the number of metrics immediately.  
Instead, the product must provide:

- a clean infrastructure for adding metrics;
- a few initial baseline metrics;
- easy comparison of their outputs on real data.

The most important product requirement is that different metrics can be evaluated through one consistent workflow.

---

## 9. Envelope support

Envelope-related processing must be considered a first-class capability.

Reason:
- many metrics may require envelope information;
- envelope is also important in broader WLI workflows;
- different methods of envelope computation may need to be compared later.

Product requirement:
- the tool must allow envelope computation as an optional reusable part of the pipeline;
- it must be possible to add new envelope methods later without redesigning the product;
- the first version may start with a minimal set of methods, but the product structure must support growth.

Envelope support is not a hidden implementation detail.  
It is part of the intended product capability.

---

## 10. ROI extraction support

The tool must support optional extraction of a local signal segment before metric evaluation.

Main parameter:
- `segmentSize`

Behavior:
- a segment of length `segmentSize` is extracted from the original signal;
- this segment is then used for downstream evaluation.

Centering strategy:
- the architecture must allow different centering methods in the future;
- in v0.1, the default strategy is centering around the maximum of the raw signal.

This means ROI extraction is intentionally simple in the first version, but extensible.

---

## 11. Metric outputs

Metric evaluation must produce results that are naturally spatial.

Expected outputs:
- a 2D quality map with shape `(H, W)`
- a 2D validity map if needed
- optional additional feature maps for diagnostic outputs

The user should not need to think about linear indexing when working with results.

---

## 12. Thresholding behavior

Thresholding is a core product feature.

The user must be able to:
- choose a threshold value;
- apply a simple keep/reject rule;
- obtain a binary mask of accepted and rejected pixels.

Typical logic:
- keep pixels where metric score is above threshold
- or keep pixels where metric score is below threshold

The result must be a 2D mask aligned with image geometry.

Thresholding should be simple and explicit in v0.1.

---

## 13. Visualization requirements

The first version should provide enough visualization to support practical analysis.

Minimum useful visual outputs:

- several selected raw signals;
- histogram of metric values;
- 2D quality map;
- 2D threshold mask.

The product does not need an advanced visualization environment in v0.1.  
It needs clear and practical outputs.

---

## 14. Export requirements

The export scope of v0.1 should remain intentionally small.

Required exports:
- quality map as `.txt`
- threshold mask as `.txt`

Both must be saved as 2D matrices.

No additional export formats are required in v0.1.

This is enough for:
- quick inspection;
- downstream analysis;
- debugging and comparison;
- keeping the first version minimal.

---

## 15. Product requirements for robustness

The product should behave predictably and clearly.

Required behavior:
- fail clearly if width and height cannot be determined;
- fail clearly if txt data shape is inconsistent with `width * height`;
- tolerate missing optional metadata;
- tolerate absence of `z_axis.txt` by falling back to index-based z-axis;
- keep preprocessing and envelope handling explicit;
- avoid hidden transformations.

The product should prefer transparency over automation.

---

## 16. What success looks like in v0.1

v0.1 is successful if the user can:

- load a real dataset from image stack or txt;
- obtain a valid `(H, W, M)` signal representation internally;
- see that acquisition metadata and z-axis were handled correctly;
- compute one or more baseline metrics;
- obtain a meaningful 2D quality map;
- apply thresholding and obtain a meaningful 2D mask;
- export map and mask as `.txt`;
- repeat the workflow reliably on other datasets.

If those things work well, v0.1 has achieved its purpose.

---

## 17. Product priorities

### Highest priority
- real-data workflow
- correctness of loading
- consistent internal representation
- direct 2D outputs
- simple threshold workflow

### Medium priority
- envelope extensibility
- ROI extraction support
- easy addition of new metrics
- useful metadata handling

### Lower priority for now
- richer export
- broad automation
- synthetic experiments
- benchmark framework

---

## 18. Product summary

`Quality_tool` v0.1 is a focused internal analysis tool for real WLI signals.

Its job is simple:

- turn real input data into a consistent signal representation;
- evaluate practical quality criteria;
- show spatial quality behavior as maps;
- show which pixels survive thresholding;
- keep the workflow minimal, explicit, and extensible.

The first version is deliberately narrow in scope so that it is useful early and can later grow into a stronger experimental platform.