"""Smoke tests for the main window and helper dialogs."""

from __future__ import annotations

import numpy as np

from quality_tool.gui.dialogs.info_dialog import InfoDialog
from quality_tool.gui.main_window import MainWindow
from quality_tool.gui.windows.compare_window import CompareWindow


class TestMainWindow:
    def test_creation(self):
        window = MainWindow()
        assert window is not None

    def test_registry_populated(self):
        window = MainWindow()
        names = window._registry.list_metrics()
        assert "fringe_visibility" in names
        assert "snr" in names
        assert "power_band_ratio" in names


class TestCompareWindow:
    def test_creation_continuous(self):
        data = np.random.rand(8, 10)
        win = CompareWindow(data, title="test")
        assert win is not None

    def test_creation_bool(self):
        data = np.ones((8, 10), dtype=bool)
        win = CompareWindow(data, title="mask")
        assert win is not None


class TestInfoDialog:
    def test_creation(self):
        info = {"Key1": "value1", "Key2": "value2"}
        dlg = InfoDialog(info)
        assert dlg is not None
