"""Performance profiling script for Quality_tool.

Two profiling modes:
  1. Full-run timing   — measures end-to-end wall time for realistic workflows
                         on the full TXT dataset.
  2. Subset profiling  — uses a small representative slice for stage-level
                         breakdown and per-pixel cost estimation.

Usage:
    python scripts/profile_pipeline.py

Requires the testing_data directory to be present at the repository root.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the src directory is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from quality_tool.core.models import SignalSet
from quality_tool.envelope.analytic import AnalyticEnvelopeMethod
from quality_tool.evaluation.evaluator import evaluate_metric_map
from quality_tool.evaluation.thresholding import apply_threshold
from quality_tool.io.image_stack_loader import load_image_stack
from quality_tool.io.txt_matrix_loader import load_txt_matrix
from quality_tool.metrics.baseline.fringe_visibility import FringeVisibility
from quality_tool.metrics.baseline.power_band_ratio import PowerBandRatio
from quality_tool.metrics.baseline.snr import SNR
from quality_tool.preprocessing.basic import (
    normalize_amplitude,
    smooth,
    subtract_baseline,
)
from quality_tool.preprocessing.roi import extract_roi
from quality_tool.spectral.fft import compute_spectrum

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TXT_DATA = REPO_ROOT / "testing_data" / "real_data_txt"
TXT_FILE = TXT_DATA / "correlogram_segments_surface_0_0.txt"
STACK_DIR = REPO_ROOT / "testing_data" / "real_data_stack"

# TXT dataset dimensions (from info file: 1920 x 1200)
TXT_WIDTH = 1920
TXT_HEIGHT = 1200

# Representative subset for stage-level breakdown
SUBSET_ROWS = 50
SUBSET_COLS = 50

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_process_rss_mb() -> float:
    """Return current process RSS in MB (Windows or Unix)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    # Fallback: try /proc on Linux
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except (FileNotFoundError, OSError):
        pass
    return float("nan")


def fmt_time(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


def make_subset(signal_set: SignalSet, rows: int, cols: int) -> SignalSet:
    """Extract a small spatial subset for detailed profiling."""
    r = min(rows, signal_set.height)
    c = min(cols, signal_set.width)
    return SignalSet(
        signals=signal_set.signals[:r, :c, :].copy(),
        width=c,
        height=r,
        z_axis=signal_set.z_axis.copy(),
        metadata=signal_set.metadata,
        source_type=signal_set.source_type,
        source_path=signal_set.source_path,
    )


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_row(label: str, value: str) -> None:
    print(f"  {label:<45} {value:>12}")


# ===================================================================
# SECTION 1: TXT Loading
# ===================================================================

def profile_txt_loading() -> SignalSet:
    print_header("1. TXT Dataset Loading")
    rss_before = get_process_rss_mb()

    t0 = time.perf_counter()
    ss = load_txt_matrix(TXT_FILE, TXT_WIDTH, TXT_HEIGHT)
    t1 = time.perf_counter()

    rss_after = get_process_rss_mb()

    print_row("Load time", fmt_time(t1 - t0))
    print_row("signals.shape", str(ss.signals.shape))
    print_row("signals.dtype", str(ss.signals.dtype))
    print_row("signals.nbytes", f"{ss.signals.nbytes / 1e6:.1f} MB")
    print_row("z_axis.shape", str(ss.z_axis.shape))
    print_row("RSS before load", f"{rss_before:.0f} MB")
    print_row("RSS after load", f"{rss_after:.0f} MB")
    print_row("RSS delta", f"{rss_after - rss_before:.0f} MB")
    return ss


# ===================================================================
# SECTION 2: Image Stack Loading
# ===================================================================

def profile_stack_loading() -> SignalSet | None:
    print_header("2. Image Stack Loading")
    if not STACK_DIR.is_dir():
        print("  (skipped — stack directory not found)")
        return None

    rss_before = get_process_rss_mb()

    t0 = time.perf_counter()
    ss = load_image_stack(STACK_DIR)
    t1 = time.perf_counter()

    rss_after = get_process_rss_mb()

    print_row("Load time", fmt_time(t1 - t0))
    print_row("signals.shape", str(ss.signals.shape))
    print_row("signals.dtype", str(ss.signals.dtype))
    print_row("signals.nbytes", f"{ss.signals.nbytes / 1e6:.1f} MB")
    print_row("RSS before load", f"{rss_before:.0f} MB")
    print_row("RSS after load", f"{rss_after:.0f} MB")
    print_row("RSS delta", f"{rss_after - rss_before:.0f} MB")
    return ss


# ===================================================================
# SECTION 3: Full-Run Timing (end-to-end on full TXT dataset)
# ===================================================================

def profile_full_run(ss: SignalSet) -> dict[str, float]:
    print_header("3. Full-Run Timing — Full TXT Dataset")
    total_pixels = ss.height * ss.width
    print(f"  Total pixels: {total_pixels:,}")
    print()

    preprocess = [subtract_baseline]
    segment_size = 64
    envelope_method = AnalyticEnvelopeMethod()

    metrics = {
        "fringe_visibility": FringeVisibility(),
        "snr": SNR(),
        "power_band_ratio": PowerBandRatio(),
    }

    timings: dict[str, float] = {}

    # --- Individual metrics ---
    for name, metric in metrics.items():
        rss_before = get_process_rss_mb()
        t0 = time.perf_counter()
        result = evaluate_metric_map(
            ss, metric,
            preprocess=preprocess,
            segment_size=segment_size,
            envelope_method=None,
        )
        t1 = time.perf_counter()
        rss_after = get_process_rss_mb()
        elapsed = t1 - t0
        timings[name] = elapsed
        per_pixel_us = (elapsed / total_pixels) * 1e6
        print_row(f"{name}", f"{fmt_time(elapsed)}  ({per_pixel_us:.1f} us/px)")
        print_row(f"  RSS delta", f"{rss_after - rss_before:+.0f} MB")

    # --- With envelope ---
    print()
    t0 = time.perf_counter()
    result = evaluate_metric_map(
        ss, metrics["snr"],
        preprocess=preprocess,
        segment_size=segment_size,
        envelope_method=envelope_method,
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    timings["snr+envelope"] = elapsed
    per_pixel_us = (elapsed / total_pixels) * 1e6
    print_row("snr + envelope", f"{fmt_time(elapsed)}  ({per_pixel_us:.1f} us/px)")

    # --- Thresholding ---
    t0 = time.perf_counter()
    thr = apply_threshold(result, 5.0)
    t1 = time.perf_counter()
    timings["thresholding"] = t1 - t0
    print_row("thresholding (after compute)", fmt_time(t1 - t0))

    return timings


# ===================================================================
# SECTION 4: Stage-Level Breakdown (representative subset)
# ===================================================================

def profile_stages(ss: SignalSet) -> dict[str, float]:
    print_header("4. Stage-Level Breakdown — Subset Profiling")
    sub = make_subset(ss, SUBSET_ROWS, SUBSET_COLS)
    h, w, m = sub.signals.shape
    total = h * w
    print(f"  Subset: {h}x{w} = {total:,} pixels, M={m}")
    print()

    z_axis = sub.z_axis
    preprocess_fns = [subtract_baseline]
    segment_size = 64

    # Accumulators
    t_copy = 0.0
    t_preprocess = 0.0
    t_roi = 0.0
    t_fft = 0.0
    t_envelope = 0.0
    t_metric_fv = 0.0
    t_metric_snr = 0.0
    t_metric_pbr = 0.0

    envelope_method = AnalyticEnvelopeMethod()
    fv = FringeVisibility()
    snr = SNR()
    pbr = PowerBandRatio()

    for row in range(h):
        for col in range(w):
            # Copy
            t0 = time.perf_counter()
            raw_signal = sub.signals[row, col, :].copy()
            t_copy += time.perf_counter() - t0

            # Preprocessing
            t0 = time.perf_counter()
            signal = raw_signal.copy()
            for fn in preprocess_fns:
                signal = fn(signal)
            t_preprocess += time.perf_counter() - t0

            # ROI
            t0 = time.perf_counter()
            roi_signal = extract_roi(signal, segment_size)
            t_roi += time.perf_counter() - t0

            # FFT / spectral context
            t0 = time.perf_counter()
            spectral = compute_spectrum(roi_signal, None)
            t_fft += time.perf_counter() - t0

            context = {"spectral_result": spectral}

            # Envelope (Hilbert)
            t0 = time.perf_counter()
            envelope = envelope_method.compute(roi_signal, None, context)
            t_envelope += time.perf_counter() - t0

            # Metric: fringe_visibility (on raw)
            t0 = time.perf_counter()
            fv.evaluate(raw_signal, z_axis, None, context)
            t_metric_fv += time.perf_counter() - t0

            # Metric: snr (on processed/ROI)
            t0 = time.perf_counter()
            snr.evaluate(roi_signal, None, envelope, context)
            t_metric_snr += time.perf_counter() - t0

            # Metric: power_band_ratio (on processed/ROI, uses spectral)
            t0 = time.perf_counter()
            pbr.evaluate(roi_signal, None, envelope, context)
            t_metric_pbr += time.perf_counter() - t0

    stages = {
        "signal_copy": t_copy,
        "preprocessing": t_preprocess,
        "roi_extraction": t_roi,
        "fft_spectral": t_fft,
        "envelope_hilbert": t_envelope,
        "metric_fringe_visibility": t_metric_fv,
        "metric_snr": t_metric_snr,
        "metric_power_band_ratio": t_metric_pbr,
    }

    total_time = sum(stages.values())
    for name, elapsed in stages.items():
        pct = (elapsed / total_time * 100) if total_time > 0 else 0
        per_px = (elapsed / total) * 1e6
        print_row(f"{name}", f"{fmt_time(elapsed)}  {pct:5.1f}%  ({per_px:.1f} us/px)")

    print_row("TOTAL (all stages)", fmt_time(total_time))
    return stages


# ===================================================================
# SECTION 5: FFT Overhead Measurement
# ===================================================================

def profile_fft_overhead(ss: SignalSet) -> dict[str, float]:
    print_header("5. FFT Overhead — With vs Without Spectral Context")
    sub = make_subset(ss, SUBSET_ROWS, SUBSET_COLS)
    h, w, m = sub.signals.shape
    total = h * w
    print(f"  Subset: {h}x{w} = {total:,} pixels, M={m}")

    preprocess = [subtract_baseline]
    segment_size = 64
    snr_metric = SNR()

    # --- SNR without FFT overhead (manual loop, skip spectral) ---
    t0 = time.perf_counter()
    for row in range(h):
        for col in range(w):
            signal = sub.signals[row, col, :].copy()
            for fn in preprocess:
                signal = fn(signal)
            signal = extract_roi(signal, segment_size)
            # SNR does not use spectral — skip compute_spectrum
            snr_metric.evaluate(signal, None, None, {})
    t_without_fft = time.perf_counter() - t0

    # --- SNR with FFT overhead (as evaluator currently does) ---
    t0 = time.perf_counter()
    for row in range(h):
        for col in range(w):
            signal = sub.signals[row, col, :].copy()
            for fn in preprocess:
                signal = fn(signal)
            signal = extract_roi(signal, segment_size)
            spectral = compute_spectrum(signal, None)
            context = {"spectral_result": spectral}
            snr_metric.evaluate(signal, None, None, context)
    t_with_fft = time.perf_counter() - t0

    fft_overhead = t_with_fft - t_without_fft
    pct = (fft_overhead / t_with_fft * 100) if t_with_fft > 0 else 0

    print()
    print_row("SNR without FFT context", fmt_time(t_without_fft))
    print_row("SNR with FFT context (current)", fmt_time(t_with_fft))
    print_row("FFT overhead", f"{fmt_time(fft_overhead)} ({pct:.1f}%)")

    # Extrapolate to full dataset
    full_pixels = ss.height * ss.width
    scale = full_pixels / total
    est_overhead_full = fft_overhead * scale
    print_row("Est. FFT waste on full dataset", fmt_time(est_overhead_full))

    # --- Also measure: PBR with vs without precomputed spectral ---
    pbr_metric = PowerBandRatio()

    t0 = time.perf_counter()
    for row in range(h):
        for col in range(w):
            signal = sub.signals[row, col, :].copy()
            for fn in preprocess:
                signal = fn(signal)
            signal = extract_roi(signal, segment_size)
            # PBR computes its own FFT internally (no context)
            pbr_metric.evaluate(signal, None, None, {})
    t_pbr_own_fft = time.perf_counter() - t0

    t0 = time.perf_counter()
    for row in range(h):
        for col in range(w):
            signal = sub.signals[row, col, :].copy()
            for fn in preprocess:
                signal = fn(signal)
            signal = extract_roi(signal, segment_size)
            spectral = compute_spectrum(signal, None)
            context = {"spectral_result": spectral}
            pbr_metric.evaluate(signal, None, None, context)
    t_pbr_ctx_fft = time.perf_counter() - t0

    print()
    print_row("PBR own FFT (no context)", fmt_time(t_pbr_own_fft))
    print_row("PBR with precomputed context", fmt_time(t_pbr_ctx_fft))
    print_row("PBR double-FFT overhead",
              f"{fmt_time(t_pbr_own_fft - t_pbr_ctx_fft)} (context saves FFT)")

    return {
        "snr_without_fft": t_without_fft,
        "snr_with_fft": t_with_fft,
        "fft_overhead_snr": fft_overhead,
        "pbr_own_fft": t_pbr_own_fft,
        "pbr_ctx_fft": t_pbr_ctx_fft,
    }


# ===================================================================
# SECTION 6: Memory Analysis
# ===================================================================

def profile_memory(ss: SignalSet) -> None:
    print_header("6. Memory Analysis")

    signals = ss.signals
    h, w, m = signals.shape
    total_pixels = h * w

    print_row("signals.dtype", str(signals.dtype))
    print_row("signals.nbytes", f"{signals.nbytes / 1e6:.1f} MB")
    print_row("z_axis.nbytes", f"{ss.z_axis.nbytes / 1e6:.3f} MB")
    print_row("Total dataset memory", f"{(signals.nbytes + ss.z_axis.nbytes) / 1e6:.1f} MB")

    # Estimate per-pixel allocations in the evaluator hot loop
    # For a processed metric: copy + preprocessing + roi + spectral + envelope
    signal_bytes = m * signals.itemsize
    roi_m = 64
    roi_bytes = roi_m * signals.itemsize
    fft_n = roi_m // 2 + 1
    spectral_bytes = fft_n * 16 + fft_n * 8  # complex + amplitude
    envelope_bytes = roi_m * 8

    per_pixel_transient = (
        signal_bytes       # raw copy
        + signal_bytes     # after subtract_baseline
        + roi_bytes        # after ROI
        + spectral_bytes   # spectral result
        + envelope_bytes   # envelope
    )

    print()
    print_row("Signal length M", str(m))
    print_row("Per-signal raw bytes", f"{signal_bytes} B")
    print_row("Per-pixel transient alloc (est.)", f"{per_pixel_transient} B")
    print_row("If all pixels simultaneous (est.)",
              f"{total_pixels * per_pixel_transient / 1e9:.1f} GB")
    print_row("(actual: sequential, so only 1 pixel at a time)", "")

    # dtype check: would float32 help?
    print()
    print_row("If float32 signals.nbytes", f"{signals.size * 4 / 1e6:.1f} MB")
    savings_mb = (signals.nbytes - signals.size * 4) / 1e6
    print_row("Savings vs float64", f"{savings_mb:.1f} MB")

    # Process RSS
    rss = get_process_rss_mb()
    print_row("Current process RSS", f"{rss:.0f} MB")


# ===================================================================
# SECTION 7: Multi-Metric Sequential Overhead
# ===================================================================

def profile_multi_metric(ss: SignalSet) -> None:
    print_header("7. Multi-Metric Sequential — Recomputation Cost")
    sub = make_subset(ss, SUBSET_ROWS, SUBSET_COLS)
    h, w, _ = sub.signals.shape
    total = h * w
    print(f"  Subset: {h}x{w} = {total:,} pixels")

    preprocess = [subtract_baseline]
    segment_size = 64

    metrics = [SNR(), PowerBandRatio()]

    # Current approach: one evaluate_metric_map per metric (sequential)
    t0 = time.perf_counter()
    for metric in metrics:
        evaluate_metric_map(
            sub, metric,
            preprocess=preprocess,
            segment_size=segment_size,
        )
    t_sequential = time.perf_counter() - t0

    # Hypothetical: share preprocessing + FFT, evaluate both metrics per pixel
    t0 = time.perf_counter()
    for row in range(h):
        for col in range(w):
            signal = sub.signals[row, col, :].copy()
            for fn in preprocess:
                signal = fn(signal)
            signal = extract_roi(signal, segment_size)
            spectral = compute_spectrum(signal, None)
            context = {"spectral_result": spectral}
            for metric in metrics:
                metric.evaluate(signal, None, None, context)
    t_shared = time.perf_counter() - t0

    savings = t_sequential - t_shared
    pct = (savings / t_sequential * 100) if t_sequential > 0 else 0

    print()
    print_row("Sequential (current)", fmt_time(t_sequential))
    print_row("Shared preprocess+FFT", fmt_time(t_shared))
    print_row("Savings", f"{fmt_time(savings)} ({pct:.1f}%)")

    full_pixels = ss.height * ss.width
    scale = full_pixels / total
    print_row("Est. savings on full dataset", fmt_time(savings * scale))


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    print("=" * 60)
    print("  Quality_tool — Performance Profiling")
    print("=" * 60)
    print(f"  Python: {sys.version}")
    print(f"  NumPy:  {np.__version__}")
    print(f"  PID:    {os.getpid()}")
    print(f"  RSS at start: {get_process_rss_mb():.0f} MB")

    # 1. Load TXT dataset
    ss = profile_txt_loading()

    # 2. Image stack loading (time only, then free)
    stack_ss = profile_stack_loading()
    if stack_ss is not None:
        print_row("Stack signals.nbytes", f"{stack_ss.signals.nbytes / 1e6:.1f} MB")
        del stack_ss  # free memory for compute profiling

    # 3. Full-run timing on full TXT dataset
    full_timings = profile_full_run(ss)

    # 4. Stage-level breakdown on subset
    stage_timings = profile_stages(ss)

    # 5. FFT overhead
    fft_timings = profile_fft_overhead(ss)

    # 6. Memory analysis
    profile_memory(ss)

    # 7. Multi-metric recomputation cost
    profile_multi_metric(ss)

    print_header("PROFILING COMPLETE")
    print(f"  Final RSS: {get_process_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
