"""Tests for quality_tool.core.analysis_context."""

from quality_tool.core.analysis_context import AnalysisContext, default_analysis_context


class TestAnalysisContext:
    def test_defaults(self):
        ctx = default_analysis_context()
        assert ctx.epsilon == 1e-12
        assert ctx.dc_exclude is True
        assert ctx.default_low_freq == 0.05
        assert ctx.default_high_freq == 0.45
        assert ctx.noise_quarter_fraction == 0.25

    def test_frozen(self):
        ctx = AnalysisContext()
        try:
            ctx.epsilon = 1e-6  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_custom_values(self):
        ctx = AnalysisContext(epsilon=1e-8, dc_exclude=False)
        assert ctx.epsilon == 1e-8
        assert ctx.dc_exclude is False

    def test_equality(self):
        a = AnalysisContext()
        b = AnalysisContext()
        assert a == b

    def test_inequality(self):
        a = AnalysisContext()
        b = AnalysisContext(epsilon=1e-6)
        assert a != b
