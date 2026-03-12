"""Tests for quality_tool.envelope.registry."""

import pytest

from quality_tool.envelope.registry import EnvelopeRegistry


class _DummyMethod:
    """Minimal envelope method stub for testing."""

    def __init__(self, name: str = "dummy") -> None:
        self.name = name

    def compute(self, signal, z_axis=None, context=None):
        return signal


class TestEnvelopeRegistry:

    def test_register_and_get(self):
        reg = EnvelopeRegistry()
        method = _DummyMethod("test")
        reg.register(method)
        assert reg.get("test") is method

    def test_list_methods(self):
        reg = EnvelopeRegistry()
        reg.register(_DummyMethod("alpha"))
        reg.register(_DummyMethod("beta"))
        assert set(reg.list_methods()) == {"alpha", "beta"}

    def test_empty_registry_list(self):
        reg = EnvelopeRegistry()
        assert reg.list_methods() == []

    def test_duplicate_registration_raises(self):
        reg = EnvelopeRegistry()
        reg.register(_DummyMethod("dup"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_DummyMethod("dup"))

    def test_get_missing_raises_key_error(self):
        reg = EnvelopeRegistry()
        with pytest.raises(KeyError, match="no envelope method"):
            reg.get("nonexistent")
