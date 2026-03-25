"""Targeted tests for the map viewer navigation + 3D map window iteration.

Covers:
- mouse-wheel zoom state changes
- reset-view restores original extent
- pixel selection still works after zoom
- home limits stored on set_map
- 3D map window creation with float data
- 3D map window with NaN data (masked mode)
- MapToolsPanel new buttons
- no regression of existing map-viewer behavior
"""

from __future__ import annotations

import numpy as np
import pytest


# ================================================================
# MapViewer zoom and reset
# ================================================================


class TestMapViewerZoom:
    """Tests for mouse-wheel zoom in the map viewer."""

    def test_home_limits_stored_on_set_map(self):
        from quality_tool.gui.widgets.map_viewer import MapViewer
        viewer = MapViewer()
        data = np.random.rand(10, 20)
        viewer.set_map(data, title="test")
        assert viewer._home_xlim is not None
        assert viewer._home_ylim is not None
        # For a 10x20 image: xlim ~ (-0.5, 19.5), ylim ~ (9.5, -0.5)
        xl, xr = viewer._home_xlim
        assert xl < xr
        assert xr == pytest.approx(19.5, abs=0.5)

    def test_home_limits_stored_on_set_binary_mask(self):
        from quality_tool.gui.widgets.map_viewer import MapViewer
        viewer = MapViewer()
        mask = np.ones((8, 12), dtype=bool)
        viewer.set_binary_mask(mask)
        assert viewer._home_xlim is not None
        assert viewer._home_ylim is not None

    def test_home_limits_stored_on_set_masked_map(self):
        from quality_tool.gui.widgets.map_viewer import MapViewer
        viewer = MapViewer()
        data = np.random.rand(6, 8)
        mask = data > 0.5
        viewer.set_masked_map(data, mask)
        assert viewer._home_xlim is not None

    def test_zoom_changes_limits(self):
        """Simulated zoom-in should shrink the axis limits."""
        from quality_tool.gui.widgets.map_viewer import MapViewer
        viewer = MapViewer()
        data = np.random.rand(100, 100)
        viewer.set_map(data)
        xl_before, xr_before = viewer._ax.get_xlim()
        span_before = xr_before - xl_before

        # Simulate a zoom-in scroll event via internal method.
        class FakeEvent:
            inaxes = viewer._ax
            button = "up"
            xdata = 50.0
            ydata = 50.0
        viewer._on_scroll(FakeEvent())

        xl_after, xr_after = viewer._ax.get_xlim()
        span_after = xr_after - xl_after
        assert span_after < span_before

    def test_zoom_out_expands_limits(self):
        """Simulated zoom-out should expand the axis limits."""
        from quality_tool.gui.widgets.map_viewer import MapViewer
        viewer = MapViewer()
        data = np.random.rand(100, 100)
        viewer.set_map(data)

        # First zoom in.
        class FakeZoomIn:
            inaxes = viewer._ax
            button = "up"
            xdata = 50.0
            ydata = 50.0
        viewer._on_scroll(FakeZoomIn())
        xl_in, xr_in = viewer._ax.get_xlim()
        span_in = xr_in - xl_in

        # Then zoom out.
        class FakeZoomOut:
            inaxes = viewer._ax
            button = "down"
            xdata = 50.0
            ydata = 50.0
        viewer._on_scroll(FakeZoomOut())
        xl_out, xr_out = viewer._ax.get_xlim()
        span_out = xr_out - xl_out
        assert span_out > span_in

    def test_reset_view_restores_home(self):
        """reset_view should restore the original full-map extent."""
        from quality_tool.gui.widgets.map_viewer import MapViewer
        viewer = MapViewer()
        data = np.random.rand(50, 60)
        viewer.set_map(data)
        home_xl = viewer._home_xlim
        home_yl = viewer._home_ylim

        # Zoom in.
        class FakeEvent:
            inaxes = viewer._ax
            button = "up"
            xdata = 30.0
            ydata = 25.0
        for _ in range(5):
            viewer._on_scroll(FakeEvent())

        # Limits should have changed.
        xl_zoomed = viewer._ax.get_xlim()
        assert xl_zoomed != home_xl

        # Reset.
        viewer.reset_view()
        assert viewer._ax.get_xlim() == pytest.approx(home_xl, abs=1e-10)
        assert viewer._ax.get_ylim() == pytest.approx(home_yl, abs=1e-10)

    def test_reset_view_no_data(self):
        """reset_view should be safe when no data is loaded."""
        from quality_tool.gui.widgets.map_viewer import MapViewer
        viewer = MapViewer()
        viewer.reset_view()  # should not crash

    def test_scroll_ignored_without_data(self):
        """Scroll events should be ignored when no map is loaded."""
        from quality_tool.gui.widgets.map_viewer import MapViewer
        viewer = MapViewer()

        class FakeEvent:
            inaxes = viewer._ax
            button = "up"
            xdata = 5.0
            ydata = 5.0
        viewer._on_scroll(FakeEvent())  # should not crash

    def test_scroll_ignored_outside_axes(self):
        """Scroll events outside the map axes should be ignored."""
        from quality_tool.gui.widgets.map_viewer import MapViewer
        viewer = MapViewer()
        data = np.random.rand(10, 10)
        viewer.set_map(data)
        xl_before = viewer._ax.get_xlim()

        class FakeEvent:
            inaxes = None  # not on the map axes
            button = "up"
            xdata = 5.0
            ydata = 5.0
        viewer._on_scroll(FakeEvent())
        assert viewer._ax.get_xlim() == xl_before

    def test_pixel_selection_works_after_zoom(self):
        """Pixel selection should still work correctly after zooming."""
        from quality_tool.gui.widgets.map_viewer import MapViewer
        viewer = MapViewer()
        data = np.random.rand(50, 50)
        viewer.set_map(data)

        # Zoom in.
        class FakeScroll:
            inaxes = viewer._ax
            button = "up"
            xdata = 25.0
            ydata = 25.0
        for _ in range(3):
            viewer._on_scroll(FakeScroll())

        # Simulate a click at data coordinates (25, 25).
        received = []
        viewer.pixel_selected.connect(lambda r, c: received.append((r, c)))

        class FakeClick:
            inaxes = viewer._ax
            xdata = 25.0
            ydata = 25.0
        viewer._on_click(FakeClick())
        assert len(received) == 1
        assert received[0] == (25, 25)

    def test_clear_resets_home_limits(self):
        from quality_tool.gui.widgets.map_viewer import MapViewer
        viewer = MapViewer()
        viewer.set_map(np.random.rand(10, 10))
        assert viewer._home_xlim is not None
        viewer.clear()
        assert viewer._home_xlim is None
        assert viewer._home_ylim is None


# ================================================================
# MapToolsPanel new buttons
# ================================================================


class TestMapToolsPanelViewGroup:
    """Tests for the new View group in MapToolsPanel."""

    def test_has_reset_view_button(self):
        from quality_tool.gui.widgets.tool_panels import MapToolsPanel
        panel = MapToolsPanel()
        assert panel.btn_reset_view is not None

    def test_has_show_3d_button(self):
        from quality_tool.gui.widgets.tool_panels import MapToolsPanel
        panel = MapToolsPanel()
        assert panel.btn_show_3d is not None

    def test_reset_view_signal_emitted(self):
        from quality_tool.gui.widgets.tool_panels import MapToolsPanel
        panel = MapToolsPanel()
        received = []
        panel.reset_view_clicked.connect(lambda: received.append(True))
        panel.btn_reset_view.click()
        assert len(received) == 1

    def test_show_3d_signal_emitted(self):
        from quality_tool.gui.widgets.tool_panels import MapToolsPanel
        panel = MapToolsPanel()
        received = []
        panel.show_3d_clicked.connect(lambda: received.append(True))
        panel.btn_show_3d.click()
        assert len(received) == 1


# ================================================================
# 3D map window
# ================================================================


class TestMap3DWindow:
    """Tests for the 3D map snapshot window."""

    def test_creation_with_float_data(self):
        from quality_tool.gui.windows.map_3d_window import Map3DWindow
        data = np.random.rand(10, 15)
        win = Map3DWindow(data, title="test score")
        assert win is not None
        assert win.windowTitle() == "3D — test score"

    def test_creation_with_nan_data(self):
        """NaN values (masked pixels) should not crash the 3D window."""
        from quality_tool.gui.windows.map_3d_window import Map3DWindow
        data = np.random.rand(8, 10)
        data[0, :] = np.nan
        data[3, 5] = np.nan
        win = Map3DWindow(data, title="masked 3D")
        assert win is not None

    def test_delete_on_close_attribute(self):
        from quality_tool.gui.windows.map_3d_window import Map3DWindow
        from PySide6.QtCore import Qt
        data = np.ones((4, 4))
        win = Map3DWindow(data)
        assert win.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
