"""Envelope method registry for Quality_tool.

Provides a simple name-based registry for envelope methods.
"""

from __future__ import annotations

from quality_tool.envelope.base import BaseEnvelopeMethod


class EnvelopeRegistry:
    """Registry of named envelope methods.

    Usage::

        registry = EnvelopeRegistry()
        registry.register(my_method)
        method = registry.get("my_method")
    """

    def __init__(self) -> None:
        self._methods: dict[str, BaseEnvelopeMethod] = {}

    def register(self, method: BaseEnvelopeMethod) -> None:
        """Register *method* under its ``name``.

        Raises
        ------
        ValueError
            If a method with the same name is already registered.
        """
        if method.name in self._methods:
            raise ValueError(
                f"envelope method {method.name!r} is already registered"
            )
        self._methods[method.name] = method

    def get(self, name: str) -> BaseEnvelopeMethod:
        """Return the method registered under *name*.

        Raises
        ------
        KeyError
            If no method with that name exists.
        """
        try:
            return self._methods[name]
        except KeyError:
            raise KeyError(
                f"no envelope method named {name!r}; "
                f"available: {self.list_methods()}"
            ) from None

    def list_methods(self) -> list[str]:
        """Return the names of all registered methods."""
        return list(self._methods.keys())


default_registry = EnvelopeRegistry()
"""Module-level default registry for convenience."""
