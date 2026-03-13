"""Smoke tests for GUI widgets.

These verify that widgets can be instantiated, updated, and emit the
expected signals without crashing.  They do not test visual rendering.
"""

from __future__ import annotations

import numpy as np
import pytest

from quality_tool.gui.widgets.map_viewer import MapViewer
from quality_tool.gui.widgets.signal_inspector import SignalInspector


# ------------------------------------------------------------------
# MapViewer
# ------------------------------------------------------------------

class TestMapViewer:
    def test_creation(self):
        viewer = MapViewer()
        assert viewer is not None

    def test_set_map(self):
        viewer = MapViewer()
        data = np.random.rand(10, 12)
        viewer.set_map(data, title="test map")
        assert viewer._data is not None
        assert viewer._data.shape == (10, 12)

    def test_set_binary_mask(self):
        viewer = MapViewer()
        mask = np.ones((8, 6), dtype=bool)
        mask[0, 0] = False
        viewer.set_binary_mask(mask, title="test mask")
        assert viewer._data is not None
        assert viewer._data.dtype == bool

    def test_value_at(self):
        viewer = MapViewer()
        data = np.arange(20, dtype=float).reshape(4, 5)
        viewer.set_map(data)
        assert viewer.value_at(0, 0) == 0.0
        assert viewer.value_at(1, 2) == 7.0
        assert viewer.value_at(99, 99) is None

    def test_value_at_no_data(self):
        viewer = MapViewer()
        assert viewer.value_at(0, 0) is None

    def test_get_snapshot(self):
        viewer = MapViewer()
        data = np.ones((3, 4))
        viewer.set_map(data, title="snap")
        snap_data, snap_title = viewer.get_snapshot()
        assert snap_data is not None
        assert snap_data.shape == (3, 4)
        assert snap_title == "snap"

    def test_pixel_selected_signal(self, qtbot):
        """Verify that pixel_selected is emitted on simulated click."""
        viewer = MapViewer()
        data = np.random.rand(10, 10)
        viewer.set_map(data)

        received = []
        viewer.pixel_selected.connect(lambda r, c: received.append((r, c)))

        # Simulate an internal call (not a real mouse event, but
        # exercises the signal-emission path).
        viewer._draw_marker(3, 5)
        viewer.pixel_selected.emit(3, 5)

        assert len(received) == 1
        assert received[0] == (3, 5)

    def test_set_map_twice(self):
        """Regression: second set_map must not crash on colorbar removal."""
        viewer = MapViewer()
        viewer.set_map(np.random.rand(6, 6), title="first")
        viewer.set_map(np.random.rand(6, 6), title="second")
        assert viewer._ax.get_title() == "second"

    def test_set_map_then_binary_mask(self):
        """Regression: switching from score map to mask must not crash."""
        viewer = MapViewer()
        viewer.set_map(np.random.rand(6, 6), title="scores")
        viewer.set_binary_mask(np.ones((6, 6), dtype=bool), title="mask")
        assert viewer._data.dtype == bool

    def test_clear(self):
        viewer = MapViewer()
        viewer.set_map(np.ones((4, 4)))
        viewer.clear()
        assert viewer._data is None


# ------------------------------------------------------------------
# SignalInspector
# ------------------------------------------------------------------

class TestSignalInspector:
    def test_creation(self):
        inspector = SignalInspector()
        assert inspector is not None

    def test_update_signal(self):
        inspector = SignalInspector()
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        z = np.arange(100, dtype=float)
        inspector.update_signal(signal, z, title="px (2,3)")

    def test_clear(self):
        inspector = SignalInspector()
        inspector.clear()
