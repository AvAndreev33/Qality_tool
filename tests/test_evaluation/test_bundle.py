"""Tests for quality_tool.evaluation.bundle — RepresentationBundle."""

from __future__ import annotations

import numpy as np

from quality_tool.core.analysis_context import AnalysisContext, default_analysis_context
from quality_tool.evaluation.bundle import RepresentationBundle
from quality_tool.evaluation.recipe import RAW, SignalRecipe
from quality_tool.spectral.fft import BatchSpectralResult


class TestRepresentationBundle:
    def test_minimal_bundle(self):
        signals = np.ones((4, 16))
        ctx = default_analysis_context()
        bundle = RepresentationBundle(
            signals=signals,
            z_axis=np.arange(16, dtype=float),
            recipe=RAW,
            analysis_context=ctx,
        )
        assert bundle.envelope is None
        assert bundle.spectral is None
        assert bundle.recipe == RAW

    def test_bundle_tied_to_recipe(self):
        recipe_a = SignalRecipe(baseline=True)
        recipe_b = RAW
        ctx = default_analysis_context()

        bundle_a = RepresentationBundle(
            signals=np.ones((2, 8)),
            z_axis=None,
            recipe=recipe_a,
            analysis_context=ctx,
        )
        bundle_b = RepresentationBundle(
            signals=np.ones((2, 8)),
            z_axis=None,
            recipe=recipe_b,
            analysis_context=ctx,
        )
        assert bundle_a.recipe != bundle_b.recipe

    def test_to_context_dict_minimal(self):
        ctx = default_analysis_context()
        bundle = RepresentationBundle(
            signals=np.ones((2, 8)),
            z_axis=None,
            recipe=RAW,
            analysis_context=ctx,
        )
        d = bundle.to_context_dict()
        assert d["analysis_context"] is ctx
        assert "spectral" not in d
        assert "batch_frequencies" not in d

    def test_to_context_dict_with_spectral(self):
        ctx = default_analysis_context()
        freqs = np.array([0.0, 0.1, 0.2])
        amp = np.ones((3, 3))
        spectral = BatchSpectralResult(
            frequencies=freqs, amplitude=amp, power=amp ** 2,
        )
        bundle = RepresentationBundle(
            signals=np.ones((3, 4)),
            z_axis=None,
            recipe=RAW,
            analysis_context=ctx,
            spectral=spectral,
        )
        d = bundle.to_context_dict()
        assert "spectral" in d
        assert d["spectral"] is spectral
        np.testing.assert_array_equal(d["batch_frequencies"], freqs)
        np.testing.assert_array_equal(d["batch_amplitude"], amp)
        np.testing.assert_array_equal(d["batch_power"], amp ** 2)

    def test_to_context_dict_amplitude_only(self):
        ctx = default_analysis_context()
        spectral = BatchSpectralResult(
            frequencies=np.array([0.0, 0.5]),
            amplitude=np.ones((2, 2)),
        )
        bundle = RepresentationBundle(
            signals=np.ones((2, 4)),
            z_axis=None,
            recipe=RAW,
            analysis_context=ctx,
            spectral=spectral,
        )
        d = bundle.to_context_dict()
        assert "batch_amplitude" in d
        assert "batch_power" not in d
