# Current iteration

## Iteration name

Profiling and performance analysis — bottlenecks before CUDA

## Goal

Profile the current backend and GUI-driven compute workflow on real data in order to identify the true bottlenecks before introducing a CUDA backend.

This iteration should:
- measure where time is spent during real workflow execution
- identify the slowest stages in the current pipeline
- identify unnecessary recomputation, copies, and memory-heavy paths
- produce a short actionable performance report
- implement only small safe fixes if they are trivial and clearly beneficial

The main goal is understanding and planning, not large optimization yet.

---

## Why this iteration matters

The project now has:
- real-data loaders
- preprocessing / ROI / envelope / spectral support
- metrics
- evaluator
- thresholding
- usable GUI
- histogram-based analysis

The system already works, but compute time is high on real WLI data.

Before building a CUDA backend, we need to know:
- which parts are truly the bottleneck
- what should be accelerated first
- what can be improved on CPU immediately
- what architectural changes would help CUDA integration later

This iteration creates that foundation.

---

## In scope

### Real-data profiling

Profile the current pipeline on real data from `testing_data`.

Use realistic workflows, for example:
- load real txt dataset
- load real image stack dataset if appropriate
- compute one metric
- compute multiple metrics
- compute with and without preprocessing
- compute with and without envelope where relevant
- apply thresholding
- update GUI-driven compute path if practical to measure cleanly

The main focus is compute path profiling, not GUI rendering speed.

---

### Stage-level timing breakdown

Measure the runtime contribution of the main stages.

At minimum, try to separate timings for:
- loading
- metadata parsing
- z-axis handling
- preprocessing
- ROI extraction
- envelope computation
- spectral / FFT computation
- metric evaluation
- thresholding
- GUI-triggered orchestration overhead if relevant

The result should make it clear which stages dominate total runtime.

---

### Metric-level profiling

Profile current metrics individually on realistic data.

At minimum, include:
- `fringe_visibility`
- `snr`
- `power_band_ratio`

Check:
- which metrics are cheap
- which metrics are expensive
- whether mixed raw/processed policies affect cost
- whether spectral metrics cause repeated FFT overhead

---

### Reuse / recomputation analysis

Inspect whether the current implementation performs unnecessary repeated work.

Examples to check:
- repeated preprocessing of the same signals
- repeated envelope computation
- repeated FFT computation
- repeated per-pixel conversions or allocations
- repeated recomputation of already available metric results
- repeated map rebuilds on GUI-side operations

The goal is to identify obvious CPU-side inefficiencies before CUDA.

---

### Memory and data-layout analysis

Inspect important memory-related behavior.

Examples to check:
- unnecessary `float64` use
- unnecessary array copies
- shape/layout transformations that may be expensive
- whether current `(H, W, M)` usage is causing overhead in hot loops
- whether temporary arrays are large and repeated

This does not require a complete redesign.
It is a diagnostic pass.

---

### Small safe performance fixes

Small safe improvements are allowed only if they are:
- local
- low risk
- clearly beneficial
- do not redesign architecture

Examples:
- removing obvious repeated computations
- avoiding unnecessary copies
- simple dtype cleanup where safe
- caching a derived representation inside one evaluation pass if already semantically correct

Do not implement major optimization or CUDA in this iteration.

---

### Performance report

Produce a short written report in the repository.

Recommended file:
- `docs/performance_notes.md`

The report should summarize:
- what was profiled
- which stages dominate runtime
- which metrics dominate runtime
- what obvious inefficiencies were found
- what should be optimized first on CPU
- what should be first candidates for CUDA backend implementation

Keep it practical and engineering-oriented.

---

## Out of scope

Do not implement in this iteration:
- CUDA backend
- large vectorization rewrite
- major evaluator redesign
- new metrics
- GUI redesign
- benchmark framework
- synthetic workflows
- broad caching framework
- large architecture changes

This iteration is for profiling and bottleneck discovery.

---

## File targets

Expected modules/files to update only if needed:

- backend modules that require small safe local fixes
- optional profiling helper scripts if useful
- `docs/performance_notes.md`

Optional new file if helpful:
- `scripts/profile_pipeline.py`

Keep additions minimal.

---

## Testing expectations

If any code is changed:
- keep changes safe and covered where practical
- do not weaken existing tests
- ensure the full test suite still passes

If profiling helpers/scripts are added:
- they do not need heavy automated testing
- they should be simple and reproducible

---

## Implementation preferences

- prefer measurement over guessing
- prefer simple instrumentation over heavy tooling if enough
- keep profiling code separate from production code where practical
- keep performance notes concrete
- distinguish clearly between:
  - current bottlenecks
  - safe immediate improvements
  - later CUDA candidates

---

## Definition of done

This iteration is complete when:
- realistic profiling has been performed on current real workflows
- major bottlenecks are identified
- obvious recomputation/copy issues are documented
- any small safe fixes are implemented if clearly justified
- a concise performance report exists
- the project has a clear next-step basis for CPU cleanup and CUDA design

---

## Expected assistant workflow

1. read `CLAUDE.md` and the docs
2. inspect the current backend and relevant GUI compute path
3. propose a short profiling plan
4. run profiling / instrumentation
5. implement only small safe fixes if clearly justified
6. write `docs/performance_notes.md`
7. summarize findings, changes, and recommended next optimization steps