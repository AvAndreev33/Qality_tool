"""Tests for quality_tool.envelope.analytic."""

import numpy as np
import pytest

from quality_tool.envelope.analytic import AnalyticEnvelopeMethod


@pytest.fixture
def method():
    return AnalyticEnvelopeMethod()


class TestAnalyticEnvelopeMethod:

    def test_name(self, method):
        assert method.name == "analytic"

    def test_output_shape_matches_input(self, method):
        sig = np.random.default_rng(0).standard_normal(128)
        env = method.compute(sig)
        assert env.shape == sig.shape

    def test_sine_wave_envelope_is_approximately_constant(self, method):
        # A pure sine has a constant Hilbert envelope equal to its amplitude.
        t = np.linspace(0, 4 * np.pi, 500, endpoint=False)
        amplitude = 3.0
        sig = amplitude * np.sin(t)
        env = method.compute(sig)
        # Exclude edges where Hilbert artefacts are common.
        core = env[50:-50]
        np.testing.assert_allclose(core, amplitude, atol=0.1)

    def test_envelope_non_negative(self, method):
        sig = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0])
        env = method.compute(sig)
        assert np.all(env >= 0.0)

    def test_deterministic(self, method):
        sig = np.array([1.0, 3.0, 2.0, 5.0, 1.0])
        env1 = method.compute(sig)
        env2 = method.compute(sig)
        np.testing.assert_array_equal(env1, env2)

    def test_rejects_2d_input(self, method):
        with pytest.raises(ValueError, match="1-D"):
            method.compute(np.ones((3, 4)))

    def test_rejects_empty_input(self, method):
        with pytest.raises(ValueError, match="empty"):
            method.compute(np.array([]))
