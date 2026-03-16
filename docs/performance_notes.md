# Quality_tool — Performance Profiling Report

Profiling date: 2026-03-16
Python 3.14.0, NumPy 2.4.3, Windows 11, single-threaded CPU evaluation.

---

## 1. Datasets profiled

| Dataset | Shape | Pixels | Signal length (M) | File size | dtype |
|---------|-------|--------|--------------------|-----------|-------|
| TXT matrix | (1200, 1920, 128) | 2,304,000 | 128 | 940 MB | float64 |
| Image stack | (1200, 1920, 418) | 2,304,000 | 418 | 418 TIFFs | float64 |

---

## 2. Loading performance

| Operation | Time | signals.nbytes | RSS delta |
|-----------|------|----------------|-----------|
| TXT load (`np.loadtxt` + reshape) | **23.0 s** | 2,359 MB | +2,361 MB |
| Image stack load (418 TIFFs) | **16.0 s** | 7,705 MB | — |

The TXT loader uses `np.loadtxt` which is slow on a 940 MB file (text parsing).
The image stack loader reads 418 individual TIFF frames sequentially via `tifffile.imread`.

---

## 3. Full-run timing — full TXT dataset (2.3M pixels)

Settings: `preprocess=[subtract_baseline]`, `segment_size=64`, no envelope unless noted.

| Workflow | Wall time | Per-pixel |
|----------|-----------|-----------|
| `fringe_visibility` (raw metric) | **65.8 s** | 28.6 us/px |
| `snr` (processed) | **114.1 s** | 49.5 us/px |
| `power_band_ratio` (processed, spectral) | **88.6 s** | 38.4 us/px |
| `snr` + envelope (Hilbert) | **240.1 s** | 104.2 us/px |
| Thresholding (after compute) | **9.8 ms** | negligible |

Key observation: even the simplest raw metric (`fringe_visibility`) takes 66 seconds
on 2.3M pixels due to the per-pixel Python loop and unconditional FFT computation.

---

## 4. Stage-level breakdown — 50x50 subset (2,500 pixels, M=128)

| Stage | Time | % of total | Per-pixel |
|-------|------|-----------|-----------|
| Signal copy | 3.2 ms | 1.0% | 1.3 us |
| Preprocessing | 19.7 ms | 6.4% | 7.9 us |
| ROI extraction | 10.5 ms | 3.4% | 4.2 us |
| **FFT / spectral context** | **38.2 ms** | **12.4%** | **15.3 us** |
| **Envelope (Hilbert)** | **106.6 ms** | **34.5%** | **42.6 us** |
| Metric: fringe_visibility | 24.4 ms | 7.9% | 9.8 us |
| Metric: snr | 67.6 ms | 21.9% | 27.0 us |
| Metric: power_band_ratio | 38.5 ms | 12.5% | 15.4 us |

When envelope is enabled, Hilbert transform dominates at 34.5%.
When envelope is disabled, FFT context + metric evaluation dominate.

---

## 5. FFT overhead analysis

### 5.1 Unconditional FFT on raw metrics

The evaluator always calls `compute_spectrum()` at line 134, regardless of whether the
metric uses spectral data. For raw metrics like `fringe_visibility`, this is pure waste.

| Scenario | Time (50x50) | Per-pixel |
|----------|-------------|-----------|
| `fringe_visibility` without FFT | 21.7 ms | 8.7 us |
| `fringe_visibility` with FFT (current) | 86.8 ms | 34.7 us |
| **FFT waste** | **65.1 ms** | **75% of runtime** |
| **Extrapolated to full dataset** | **~60 s wasted** | |

75% of `fringe_visibility` runtime is spent computing FFT that nobody reads.

### 5.2 FFT for SNR (does not use spectral data)

| Scenario | Time (50x50) |
|----------|-------------|
| SNR without FFT context | 69.0 ms |
| SNR with FFT context (current) | 109.9 ms |
| **FFT waste** | **40.9 ms (37%)** |
| **Extrapolated to full dataset** | **~38 s wasted** |

### 5.3 FFT for PowerBandRatio (uses spectral data)

| Scenario | Time (50x50) |
|----------|-------------|
| PBR with no precomputed context (own FFT) | 85.6 ms |
| PBR with precomputed context | 86.5 ms |

PBR correctly reuses the precomputed spectral context — the current pattern
works well here. When context is available, PBR avoids double-FFT.

### 5.4 Summary

FFT is computed unconditionally for every pixel, for every metric. This wastes:
- ~60 s for `fringe_visibility` (raw, never uses FFT)
- ~38 s for `snr` (processed, never uses FFT)
- 0 s for `power_band_ratio` (actually uses FFT)

**Total estimated FFT waste per full run: ~60–100 s depending on metric.**

---

## 6. Multi-metric recomputation cost

When computing multiple processed metrics sequentially (current GUI behavior),
preprocessing and FFT are recomputed independently for each metric.

| Approach | Time (50x50, SNR+PBR) | Est. full dataset |
|----------|----------------------|-------------------|
| Sequential (current) | 199.3 ms | ~184 s |
| Shared preprocess+FFT | 149.9 ms | ~138 s |
| **Savings** | **49.4 ms (24.8%)** | **~46 s** |

For two processed metrics, sharing preprocessing and FFT across metrics
would save ~25% of total compute time.

---

## 7. Memory analysis

### 7.1 Dataset memory

| Item | Size |
|------|------|
| TXT signals (float64) | 2,359 MB |
| Stack signals (float64) | 7,705 MB |
| z_axis | <0.01 MB |
| RSS after TXT load | 2,393 MB |

### 7.2 dtype observation

Signals are stored as `float64` (8 bytes per sample). If `float32` is sufficient:

| | float64 | float32 | Savings |
|--|---------|---------|---------|
| TXT | 2,359 MB | 1,180 MB | 1,180 MB |
| Stack | 7,705 MB | 3,852 MB | 3,852 MB |

Whether float32 is sufficient depends on the required precision for metric computation
and source data range. TXT data loaded via `np.loadtxt` defaults to float64. TIFF data
is converted via `.astype(float)` which also produces float64. This should be
investigated but is not a trivial change — metric accuracy needs verification.

### 7.3 Transient memory during evaluation

Per-pixel transient allocation (processed metric, segment_size=64):
- Raw signal copy: 1,024 B
- After preprocessing: 1,024 B
- After ROI: 512 B
- Spectral result: ~784 B (complex FFT + amplitude)
- Envelope: 512 B
- **Total per-pixel transient: ~3,864 B**

Since evaluation is sequential (one pixel at a time), transient memory is small.
However, the result list accumulates 2.3M `MetricResult` objects:

| Item | Estimated size |
|------|---------------|
| 2.3M MetricResult objects | significant Python object overhead |
| RSS during full FV compute | peaked at 4,674 MB vs 2,405 MB baseline |
| **Peak transient RSS** | **~2,269 MB above baseline** |

The 2.3 GB peak transient comes from the flat `results` list of MetricResult dataclass
instances, each containing a dict, a float, a bool, and a string. Python object overhead
dominates here — 2.3M small objects with dicts is expensive.

---

## 8. The dominant bottleneck: per-pixel Python loop

The most important finding from this profiling is not any individual stage, but the
**per-pixel Python for-loop** in `evaluator.py` (lines 108–144).

All metric evaluation currently runs as:
```
for row in range(h):
    for col in range(w):
        # ... per-pixel Python operations ...
```

This means 2.3M iterations of Python interpreter overhead per metric.

### Vectorization potential (measured)

| Operation | Per-pixel loop | Vectorized (numpy) | Speedup |
|-----------|---------------|-------------------|---------|
| fringe_visibility | 65.8 s | **0.37 s** | **180x** |
| SNR (no preprocess) | ~66 s (est.) | **1.4 s** | **~47x** |
| Batch FFT (all pixels) | ~35 s (est.) | **1.7 s** | **~21x** |

Vectorized numpy operations on the full (H*W, M) array are 20–180x faster than
per-pixel Python loops, depending on the operation. This is the single largest
optimization opportunity.

---

## 9. Bottleneck ranking

| Rank | Bottleneck | Impact | Fix complexity |
|------|-----------|--------|---------------|
| **1** | **Per-pixel Python loop** | **60–240 s per metric** | Medium — requires evaluator refactor to batch numpy ops |
| **2** | **Unconditional FFT** | **38–60 s wasted per non-spectral metric** | Low — conditional check on whether metric uses spectral |
| **3** | **Envelope (Hilbert) per-pixel** | **~100 s added when enabled** | Medium — batch Hilbert is straightforward |
| **4** | **Multi-metric recomputation** | **~46 s for 2 metrics** | Medium — shared preprocessing pass |
| **5** | **TXT loading (np.loadtxt)** | **23 s** | Low — switch to np.genfromtxt or pandas read_csv |
| **6** | **float64 memory** | **2.4 GB vs 1.2 GB** | Low risk but needs accuracy check |
| **7** | **MetricResult list overhead** | **~2.3 GB peak RSS** | Medium — direct array assembly |

---

## 10. Recommended next steps

### Immediate safe CPU improvements (no architecture change)

1. **Skip FFT for metrics that don't use spectral data.**
   The evaluator currently calls `compute_spectrum()` unconditionally (line 134).
   Adding a conditional check — without changing the metric interface — would save
   38–60 s per non-spectral metric on the full dataset. This is a 1–2 line change
   in `evaluator.py`.

2. **Skip signal copy for raw metrics that don't modify input.**
   `fringe_visibility` never modifies the signal, but the evaluator always copies
   at line 110. For raw metrics, using a view saves ~1.3 us/px (small but free).

### Near-term CPU optimizations (evaluator refactor)

3. **Vectorize metric evaluation.**
   Replace the per-pixel Python loop with batch numpy operations. This is the
   highest-impact change: measured 20–180x speedup depending on the metric.
   Requires refactoring the evaluator to operate on `(N, M)` arrays rather than
   individual signals, and metrics to support batch evaluation.

4. **Batch preprocessing.**
   Apply `subtract_baseline`, `normalize_amplitude`, `smooth` to the full
   `(H*W, M)` array in one numpy call instead of per-pixel.

5. **Batch envelope (Hilbert).**
   `scipy.signal.hilbert` already supports batch input along an axis.
   Switching from per-pixel to batch would eliminate per-call overhead.

6. **Direct array assembly instead of MetricResult list.**
   Instead of collecting 2.3M MetricResult objects, allocate `score_map` and
   `valid_map` arrays up front and fill them directly. This would eliminate
   the 2.3 GB peak RSS from Python object overhead.

### CUDA candidates (future iteration)

7. **Batch FFT** — `np.fft.rfft` on (N, M) is already fast on CPU (1.7 s for 2.3M
   signals) but would be even faster on GPU via cuFFT.

8. **Hilbert envelope** — FFT-based, naturally GPU-friendly.

9. **Metric evaluation kernels** — simple per-element arithmetic, ideal for GPU.

10. **Data loading to GPU** — if signals can stay on GPU through the pipeline,
    host-device transfers are minimized.

### Loading improvements

11. **Replace `np.loadtxt` with a faster parser** (e.g., `np.fromfile` for binary,
    or `pandas.read_csv` with `engine='c'` for text). The 23 s load time for a
    940 MB text file is significant for interactive use.

---

---

## 11. After optimization — batch/chunked evaluation

Optimization date: 2026-03-16.

All six recommended CPU optimizations (sections 10.1–10.6) were implemented in a
single iteration. The evaluator was rewritten to operate in chunked batch mode on
`(N, M)` arrays, metrics gained optional `evaluate_batch()` methods, preprocessing
and envelope computation were vectorized, and the `MetricResult` object list was
replaced by direct array assembly into preallocated numpy arrays.

### 11.1 Full-run timing — full TXT dataset (2.3M pixels, M=128)

Settings: `preprocess=[subtract_baseline]`, `segment_size=64`, `chunk_size=50000`.

| Workflow | BEFORE | AFTER | Speedup |
|----------|--------|-------|---------|
| `fringe_visibility` (raw) | 65.8 s | **0.53 s** | **124x** |
| `snr` (processed) | 114.1 s | **5.26 s** | **22x** |
| `power_band_ratio` (processed, spectral) | 88.6 s | **5.38 s** | **16x** |
| `snr` + envelope (Hilbert) | 240.1 s | **10.20 s** | **24x** |

`fringe_visibility` is fastest because it is a raw metric: no preprocessing, no
ROI extraction, no FFT, no signal copy — just vectorized min/max on a view of the
original array.

`snr` and `power_band_ratio` are slower due to batch preprocessing (baseline
subtraction), batch ROI extraction, and (for PBR) batch FFT. The envelope workflow
adds batch Hilbert transform cost.

### 11.2 Memory — transient RSS during evaluation

| Metric | BEFORE (RSS above baseline) | AFTER (RSS above baseline) |
|--------|----------------------------|---------------------------|
| `fringe_visibility` | ~2,269 MB | **~116 MB** |

The dominant transient memory source (2.3M `MetricResult` Python objects) was
eliminated by direct array assembly. The remaining ~116 MB is the preallocated
output arrays plus one chunk of working data.

**Note on peak working-set:** The Windows `peak_wset` metric reports the
*historical high-water mark* for the process lifetime, not the current or
transient peak. Because loading the TXT dataset alone consumes ~2.4 GB, and
numpy may request additional temporary buffers during `loadtxt`, the process
peak working set (~4.67 GB) is set during the *loading* phase, not during
evaluation. The optimization reduced transient evaluation RSS from ~2.3 GB to
~116 MB, but this does not lower the already-established peak from loading.

### 11.3 What got faster

| Bottleneck | Before | After | How |
|-----------|--------|-------|-----|
| Per-pixel Python loop | 60–240 s | 0.5–10 s | Batch numpy on `(N, M)` chunks |
| Unconditional FFT | 38–60 s wasted | 0 s wasted | `needs_spectral` flag, conditional FFT |
| MetricResult list (2.3 GB) | 2.3M Python objects | preallocated arrays | Direct array assembly |
| Per-pixel preprocessing | part of loop overhead | batch `(N, M)` ops | `subtract_baseline_batch`, etc. |
| Per-pixel envelope | ~100 s (Hilbert) | ~5 s (batch Hilbert) | `scipy.signal.hilbert(signals, axis=1)` |
| Signal copy for raw metrics | unnecessary copy | view, no copy | `input_policy="raw"` skip path |

### 11.4 What remains a bottleneck

| Item | Current time | Notes |
|------|-------------|-------|
| TXT loading (`np.loadtxt`) | 23 s | Text parsing, not addressed this iteration |
| Image stack loading (418 TIFFs) | 16 s | Sequential TIFF reads, not addressed |
| Multi-metric recomputation | ~25% overhead for 2 metrics | Preprocessing recomputed per metric; shared-pass not yet implemented |
| Dataset memory (float64) | 2.4–7.7 GB | dtype not changed; accuracy impact needs study |

### 11.5 What was intentionally not changed

- **No CUDA** — this iteration focuses on CPU vectorization only.
- **No float64→float32 conversion** — needs accuracy verification first.
- **No shared preprocessing across metrics** — would require evaluator API changes.
- **No loader optimization** — separate concern, not in iteration scope.
- **No metric interface redesign** — `evaluate_batch()` is optional; metrics without
  it fall back to a per-signal loop within each chunk (still faster than the old
  global per-pixel loop due to chunked overhead reduction).
- **No GUI changes** — `evaluate_metric_map()` signature is backward-compatible
  (new `chunk_size` parameter has a default value).

---

## 12. Key takeaway

The per-pixel Python loop that dominated all evaluation has been eliminated.
Batch/chunked numpy evaluation delivers **16–124x speedups** on the full 2.3M-pixel
TXT dataset, reducing wall time from minutes to seconds. Transient evaluation
memory dropped from ~2.3 GB to ~116 MB.

The remaining CPU bottlenecks are I/O (loading) and multi-metric recomputation.
The batch data-flow pattern established here (`(N, M)` chunks, `evaluate_batch`,
`BatchMetricArrays`) provides a natural foundation for future CUDA acceleration
via cupy/torch drop-in replacements.
