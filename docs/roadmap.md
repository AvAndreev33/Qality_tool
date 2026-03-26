# Quality_tool — Roadmap

## 1. Strategic direction

`Quality_tool` is evolving toward a **modular WLI research workbench**.

Its long-term role is not limited to computing one quality map or testing one fixed criterion.  
The intended direction is to create a flexible engineering and research environment where an engineer can:

- load real WLI signal data;
- apply different preprocessing methods;
- apply different envelope methods;
- evaluate different quality metrics;
- combine methods into pipelines;
- compare multiple pipeline variants;
- quickly add and test new methods;
- later extend the same workflow to synthetic and benchmark data.

At the same time, the first implementation must remain narrow, useful, and practical.

The project therefore follows this principle:

**broad long-term architecture, narrow first implementation.**

---

## 2. Immediate practical focus

The current practical task is still the same as discussed before:

- analyze signal quality on real WLI data;
- compute and compare quality criteria;
- build quality maps;
- apply thresholding;
- inspect which pixels are accepted or rejected.

So the first implementation step does **not** try to build the whole research platform at once.

Instead, it builds the first usable slice of that future platform:
a strong and extensible real-data quality-analysis core.

---

## 3. Product evolution model

The project should evolve in layers.

### Layer A — Quality analysis core
A minimal but solid system for:
- loading real datasets;
- computing baseline quality metrics;
- visualizing maps and masks;
- supporting first comparisons.

### Layer B — Composable experimentation
A more flexible system where:
- preprocessing becomes modular;
- envelope computation becomes modular;
- metric pipelines become configurable;
- multiple variants can be compared systematically.

### Layer C — Research-grade evaluation
A broader experimentation framework with:
- synthetic signals;
- controlled perturbations;
- benchmark scenarios;
- robustness analysis;
- development of new composite methods.

This layered evolution keeps the project useful from the beginning while preserving a modern research-oriented direction.

---

## 4. Guiding philosophy

The project should be built around **experimentation with pipelines**, not only around individual metrics.

In practice, meaningful comparison is rarely just:

- metric A vs metric B

More often it is:

- metric A on raw signals
- metric A after preprocessing 1
- metric A with envelope method 1
- metric A with envelope method 2
- metric B with ROI extraction
- metric C with another signal representation

So the deeper long-term unit of analysis is:

```text
dataset
+ preprocessing chain
+ envelope method
+ metric
+ threshold / analysis settings
= experiment
```

This idea should shape the roadmap, even though v0.1 still implements only the first simple subset of it.

---

## 5. v1.0 — Real-data quality analysis core (done)

Delivered. The quality-analysis core is complete and working:

- two loaders (image stack, txt matrix), metadata, z-axis
- unified `SignalSet(H, W, M)` representation
- preprocessing: baseline, detrend, normalize, smooth, ROI
- envelope: Hilbert-based analytic envelope
- spectral: batch rFFT with priors
- 39 quality metrics across 7 categories (baseline, envelope, spectral, noise, phase, correlation, regularity)
- recipe-based evaluation planner with grouped chunked batch processing
- thresholding with flexible mask-source selection
- full desktop GUI: map viewer, signal inspector, compare, histogram, 3D, pixel-metrics
- CUDA backend: all 39 metrics accelerated via CuPy with auto-detect and CPU fallback
- export of maps and masks as `.txt`

---

## 6. Next direction — Height-map computation and research tooling

## Goal

Extend the workbench from quality analysis toward height-map computation using different algorithms, configurable via the existing Settings infrastructure.

Parallel track: research tooling — agent-assisted literature monitoring, classification of recent publications by relevance and method class, lightweight interest scoring to help prioritize what to read and potentially implement.

## Scope (to be refined)

- height-map computation from WLI signals (multiple algorithms)
- settings-driven algorithm configuration
- agent-assisted research workflow for WLI literature
- publication classification and interest scoring

---

## 7. v0.3 — Pipeline comparison and reproducible experiments

## Goal

Turn the modular processing system into a reproducible pipeline-comparison workbench.

This stage is about making experiments systematic.

## Scope

- define experiment configurations more explicitly
- compare multiple pipelines on the same dataset
- compare multiple datasets under the same pipeline
- save experiment settings in a reproducible way
- improve analysis of differences between methods
- make result comparison easier and more structured

## Deliverables

- experiment config / manifest concept
- reusable pipeline definitions
- systematic comparison utilities
- result aggregation helpers
- structured reports or summaries
- stronger separation between raw results and experiment interpretation

## Intended outcome

The tool becomes a real engineering workbench for iterative method development.

---

## 8. v0.4 — Synthetic signals and controlled perturbations

## Goal

Extend the platform beyond real data into controlled signal experimentation.

This stage introduces synthetic and semi-synthetic workflows to better study behavior of metrics and pipeline variants.

## Scope

- synthetic signal generation
- controlled perturbation methods
- noise injection
- baseline drift
- asymmetry
- contrast degradation
- clipping / saturation
- other artifact scenarios relevant to WLI

## Deliverables

- synthetic signal generator
- perturbation modules
- scenario definitions
- controlled experiment presets
- ability to compare real-data and synthetic-data behavior

## Intended outcome

The tool can now support hypothesis testing under controlled conditions, not only empirical inspection of real datasets.

---

## 9. v0.5 — Benchmark framework for WLI quality pipelines

## Goal

Create a systematic framework for evaluating and comparing quality-analysis pipelines.

At this stage, the project should support not only exploratory usage, but also structured benchmarking.

## Scope

- curated benchmark datasets or scenarios
- standardized evaluation protocol
- comparison across multiple metrics and pipeline variants
- robustness analysis
- reproducible benchmark runs
- summary comparisons across datasets and perturbations

## Deliverables

- benchmark runner
- benchmark manifests
- result aggregation format
- comparison reports
- ranking-ready summaries
- stronger reproducibility guarantees

## Intended outcome

The project becomes suitable for more rigorous engineering comparison and possibly publication-oriented evaluation.

---

## 10. v0.6 — New method development and composite criteria

## Goal

Use the accumulated infrastructure to create and test stronger new methods.

At this stage, the system is not only evaluating existing methods, but actively supporting creation of new ones.

## Scope

- composite metrics
- combinations of baseline features
- use of envelope-derived features
- use of preprocessing-derived features
- robust or adaptive criteria
- local criteria around coherence-peak neighborhoods
- comparison of newly designed methods against baselines

## Deliverables

- composite metric modules
- structured ablation-ready comparison workflow
- better support for engineering hypothesis testing
- stronger research reports and summaries

## Intended outcome

The platform becomes a practical environment for inventing and validating new WLI signal-quality methods.

---

## 11. Priority order

## Current
- height-map computation workflows
- settings infrastructure for algorithm configuration
- research literature monitoring tooling

## Next
- systematic pipeline comparison
- reproducible experiments
- synthetic data and benchmark scenarios

## Later
- composite method development
- broader research automation

---

## 12. Things to avoid too early

The roadmap should explicitly avoid several traps in early stages.

Do not overbuild too early:
- no oversized automation before the core workflow is stable
- no complex orchestration before basic experimentation works
- no benchmark framework before method comparison becomes mature
- no synthetic-first workflow before real-data workflow is solid
- no too-general “platform for everything” before there is a strong core

The project should grow by extending a stable core, not by trying to implement the final vision immediately.

---

## 13. Recommended next steps

1. design height-map computation interface and first algorithm
2. integrate with existing Settings dialog for algorithm configuration
3. prototype research-agent workflow for literature scanning
4. refine roadmap based on initial height-map results

---

## 14. Final roadmap summary

The project starts as a **focused quality-analysis tool for real WLI data**,  
but it is intentionally being designed to grow into a **modular WLI research workbench**.

The first implementation should stay narrow and practical.

The long-term direction should support:
- fast method addition,
- fast method comparison,
- pipeline experimentation,
- reproducible engineering research,
- and eventually systematic benchmark-driven development.

That is the intended trajectory of `Quality_tool`.