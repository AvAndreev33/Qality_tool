"""Minimal tests for the CUDA backend module."""

from __future__ import annotations

import pytest
import numpy as np

from quality_tool.cuda._backend import GPU_METRIC_NAMES, is_available, get_device_info


def test_gpu_metric_names_is_frozenset():
    assert isinstance(GPU_METRIC_NAMES, frozenset)


def test_all_metrics_registered():
    assert len(GPU_METRIC_NAMES) == 39


def test_is_available_returns_bool():
    assert isinstance(is_available(), bool)


def test_get_device_info_returns_dict():
    info = get_device_info()
    assert isinstance(info, dict)


def test_dispatch_table_matches_gpu_names():
    """Dispatch table keys must match GPU_METRIC_NAMES exactly."""
    from quality_tool.cuda._evaluator import _DISPATCH
    assert set(_DISPATCH.keys()) == GPU_METRIC_NAMES


_requires_cuda = pytest.mark.skipif(
    not is_available(), reason="CUDA / CuPy not available",
)


@_requires_cuda
def test_gpu_all_metrics_smoke():
    """Smoke test: evaluate all 42 GPU metrics on synthetic data."""
    from quality_tool.core.models import SignalSet
    from quality_tool.cuda import evaluate_metric_maps_gpu
    from quality_tool.metrics.registry import MetricRegistry

    # Build a registry with all metrics to get instances
    from quality_tool.metrics.baseline.snr import SNR
    from quality_tool.metrics.baseline.fringe_visibility import FringeVisibility
    from quality_tool.metrics.baseline.power_band_ratio import PowerBandRatio
    from quality_tool.metrics.envelope import ALL_ENVELOPE_METRICS
    from quality_tool.metrics.spectral import ALL_SPECTRAL_METRICS
    from quality_tool.metrics.noise import ALL_NOISE_METRICS
    from quality_tool.metrics.phase import ALL_PHASE_METRICS
    from quality_tool.metrics.correlation import ALL_CORRELATION_METRICS
    from quality_tool.metrics.regularity import ALL_REGULARITY_METRICS

    all_metrics = [SNR(), FringeVisibility(), PowerBandRatio()]
    for group in [ALL_ENVELOPE_METRICS, ALL_SPECTRAL_METRICS,
                  ALL_NOISE_METRICS, ALL_PHASE_METRICS,
                  ALL_CORRELATION_METRICS, ALL_REGULARITY_METRICS]:
        all_metrics.extend([m() if callable(m) else m for m in group])

    H, W, M = 4, 5, 128
    rng = np.random.default_rng(42)
    z = np.linspace(0, 20, M)
    carrier = np.cos(2 * np.pi * z / 2.5)
    env = np.exp(-((z - 10) ** 2) / 8)
    signals = rng.normal(100, 1, (H, W, M)) + 30 * env * carrier

    ss = SignalSet(
        signals=signals, width=W, height=H, z_axis=z,
        metadata={
            "wavelength_nm": 600.0,
            "coherence_length_nm": 3000.0,
            "z_step_nm": float(z[1] - z[0]) * 1000,
        },
    )

    results = evaluate_metric_maps_gpu(ss, all_metrics)

    # All GPU-supported metrics should have results
    for name, result in results.items():
        assert result.score_map.shape == (H, W), f"{name} shape"
        assert result.valid_map.shape == (H, W), f"{name} valid shape"


@_requires_cuda
def test_gpu_unsupported_skipped():
    from quality_tool.core.models import SignalSet
    from quality_tool.cuda import evaluate_metric_maps_gpu

    signals = np.ones((2, 3, 32))
    z = np.arange(32, dtype=float)
    ss = SignalSet(signals=signals, width=3, height=2, z_axis=z)

    class FakeMetric:
        name = "not_real"
        signal_recipe = None
        recipe_binding = "active"

    assert evaluate_metric_maps_gpu(ss, [FakeMetric()]) == {}
