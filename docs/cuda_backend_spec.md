# CUDA Backend Specification

## Overview

Optional GPU-accelerated metric evaluation via CuPy.
Falls back transparently to CPU when CuPy is not installed or no GPU is available.

## Module structure

```
src/quality_tool/cuda/
├── __init__.py        # public API: is_available, evaluate_metric_maps_gpu, GPU_METRIC_NAMES
├── _backend.py        # availability check, device info, supported metric registry
└── _evaluator.py      # GPU preprocessing, representations, all metric implementations
```

## Supported metrics

All 39 project metrics are supported. Three execution modes:

| Mode | Description | Metrics |
|------|-------------|---------|
| Full GPU | All math on device | baseline (3), spectral entropy/centroid/kurtosis/spread/energy/presence/peak_sharpness, single_peakness, envelope height/area, local_snr, high_freq_noise, residual_noise, autocorrelation |
| GPU + host hybrid | FFT/envelope on GPU, per-signal logic on host | envelope width/sharpness/symmetry/sidelobe, phase (5), correlation (4), regularity zero_crossing/jitter/local_oscillation, noise drift/peak_to_background, carrier_to_background, peak_prominence, envelope_spectrum_consistency |

## Data flow

```
SignalSet (H,W,M)
  → flatten to (N,M)
  → chunk loop:
      → cp.asarray (upload)
      → GPU preprocessing (baseline, detrend, normalize, ROI)
      → GPU envelope (Hilbert via cuFFT)
      → GPU spectral (cuFFT rfft)
      → dispatch metrics (GPU arrays → scores)
      → cp.asnumpy (download scores)
  → reshape (N,) → (H,W)
  → return MetricMapResult (same as CPU)
```

## GUI integration

In `_on_compute`:
- auto-detect: `cuda.is_available()` checked at compute time
- GPU metrics dispatched to `evaluate_metric_maps_gpu`
- remaining metrics (if any unsupported) run on CPU
- silent fallback on GPU error
- status bar shows `[gpu]`, `[gpu+cpu]`, or `[cpu]`

## Dependencies

- `cupy-cuda12x` (or `cupy-cuda11x`) — optional, not in base requirements
- no other GPU-specific dependencies

## Limitations

- phase, correlation, regularity metrics download to host for per-signal logic (variable-length support regions, peak detection loops)
- no custom CUDA kernels — pure CuPy
- chunk size shared with CPU default (5000 in GUI, 50000 internal default)
- no float32 fast path yet
