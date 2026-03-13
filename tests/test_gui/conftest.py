"""Shared fixtures for GUI tests.

A QApplication instance is required before any QWidget can be created.
This module ensures one exists for the entire test session.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Provide a QApplication for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
