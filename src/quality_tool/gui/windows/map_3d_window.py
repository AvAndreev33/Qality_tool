"""3D surface view of a 2D map for Quality_tool.

Uses pyqtgraph.opengl (hardware-accelerated OpenGL) when available for
smooth interactive rotation/zoom.  Falls back to matplotlib software
renderer when pyqtgraph is not installed.

Interactive LOD: during mouse rotation/zoom a decimated mesh is shown
for instant feedback; full resolution is restored on mouse release.

The window is reused across invocations to avoid pyqtgraph's global
shader-cache invalidation on GL context destruction.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import QVBoxLayout, QWidget

try:
    import pyqtgraph.opengl as gl

    _HAS_GL = True
except ImportError:
    _HAS_GL = False

# Max vertex counts for full-res and interactive LOD meshes.
_FULL_MAX_VERTICES = 500_000
_LOD_MAX_VERTICES = 120_000


def _stride_for(data: np.ndarray, target: int) -> int:
    """Return stride factor to bring vertex count under *target*."""
    h, w = data.shape
    n = h * w
    if n <= target:
        return 1
    return max(1, int(math.ceil(math.sqrt(n / target))))


def _cmap_to_colors(data: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """Map 2D data to RGBA float array (N, 4) ready for pyqtgraph."""
    from matplotlib import colormaps

    cm = colormaps[cmap_name]
    flat = data.ravel().astype(np.float64)
    mask = np.isfinite(flat)
    normed = np.zeros_like(flat)
    if mask.any():
        lo, hi = float(np.nanmin(flat[mask])), float(np.nanmax(flat[mask]))
        if hi > lo:
            normed[mask] = (flat[mask] - lo) / (hi - lo)
        else:
            normed[mask] = 0.5
    rgba = cm(normed).astype(np.float32)  # (N, 4)
    rgba[~mask, 3] = 0.0
    return rgba


if _HAS_GL:

    class _LODViewWidget(gl.GLViewWidget):
        """GLViewWidget that swaps between full and decimated surface on interaction."""

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent=parent)
            self._full: gl.GLSurfacePlotItem | None = None
            self._lod: gl.GLSurfacePlotItem | None = None
            self._interacting = False

            self._restore_timer = QTimer(self)
            self._restore_timer.setSingleShot(True)
            self._restore_timer.setInterval(200)
            self._restore_timer.timeout.connect(self._show_full)

        def set_surfaces(
            self,
            full: gl.GLSurfacePlotItem,
            lod: gl.GLSurfacePlotItem | None,
        ) -> None:
            self._full = full
            self._lod = lod
            self._interacting = False

        # -- interaction hooks --

        def _show_lod(self) -> None:
            if self._lod is None or self._full is None:
                return
            if not self._interacting:
                self._interacting = True
                self._full.setVisible(False)
                self._lod.setVisible(True)
            self._restore_timer.stop()

        def _show_full(self) -> None:
            self._interacting = False
            if self._lod is not None:
                self._lod.setVisible(False)
            if self._full is not None:
                self._full.setVisible(True)

        def set_home(self, center: QVector3D, distance: float) -> None:
            """Store the default camera position for reset via Space."""
            self._home_center = center
            self._home_distance = distance
            self._home_elevation = 25
            self._home_azimuth = 45

        def mousePressEvent(self, ev):  # noqa: N802
            self._show_lod()
            super().mousePressEvent(ev)

        def keyPressEvent(self, ev):  # noqa: N802
            if ev.key() == Qt.Key.Key_Space:
                self.opts["center"] = QVector3D(self._home_center)
                self.setCameraPosition(
                    distance=self._home_distance,
                    elevation=self._home_elevation,
                    azimuth=self._home_azimuth,
                )
                self.update()
                return
            super().keyPressEvent(ev)

        def mouseReleaseEvent(self, ev):  # noqa: N802
            super().mouseReleaseEvent(ev)
            if self._lod is not None:
                self._restore_timer.start()

        def wheelEvent(self, ev):  # noqa: N802
            self._show_lod()
            super().wheelEvent(ev)
            if self._lod is not None:
                self._restore_timer.start()


class Map3DWindow(QWidget):
    """A standalone window showing a 3D surface of a 2D map.

    The window is designed to be reused: call ``update_data()`` to
    replace the displayed surface without destroying the GL context.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("3D")
        self.resize(700, 550)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)

        self._view: QWidget | None = None
        self._use_gl = _HAS_GL

    # ── public API ────────────────────────────────────────────────

    def update_data(
        self,
        data: np.ndarray,
        title: str = "",
        cmap: str = "viridis",
    ) -> None:
        """Replace the displayed surface with new data."""
        self.setWindowTitle(f"3D — {title}")

        if self._use_gl:
            self._update_gl(data, cmap)
        else:
            self._update_matplotlib(data, title, cmap)

    def closeEvent(self, ev) -> None:  # noqa: N802
        """Hide instead of destroying to preserve the GL context."""
        ev.ignore()
        self.hide()

    # ── pyqtgraph OpenGL (GPU) path ──────────────────────────────

    def _ensure_gl_view(self) -> _LODViewWidget:
        """Create the persistent GLViewWidget on first use."""
        if self._view is not None and isinstance(self._view, _LODViewWidget):
            return self._view

        # Remove old widget if switching backends.
        if self._view is not None:
            self._layout.removeWidget(self._view)
            self._view.deleteLater()

        view = _LODViewWidget()
        view.setBackgroundColor(45, 45, 48)
        self._layout.addWidget(view)
        self._view = view
        return view

    def _update_gl(self, data: np.ndarray, cmap: str) -> None:
        view = self._ensure_gl_view()

        # Clear all previous items from the view.
        for item in list(view.items):
            view.removeItem(item)

        h, w = data.shape
        z = data.astype(np.float32)
        z_safe = np.where(np.isfinite(z), z, 0.0)

        # Normalize Z geometry to [-0.5..+0.5] so mean level sits at Z=0.
        z_min = float(np.nanmin(z_safe))
        z_max = float(np.nanmax(z_safe))
        z_range = z_max - z_min if z_max > z_min else 1.0
        z_norm = (z_safe - z_min) / z_range - 0.5

        z_height = max(h, w) * 0.3

        def _make_surface(
            z_data: np.ndarray, src_data: np.ndarray, stride: int,
        ) -> gl.GLSurfacePlotItem:
            colors = _cmap_to_colors(src_data, cmap)
            s = gl.GLSurfacePlotItem(
                z=z_data,
                shader="shaded",
                smooth=False,
                glOptions="opaque",
            )
            s._meshdata.setVertexColors(colors)
            s.scale(stride, stride, z_height)
            return s

        # -- full-resolution surface (capped) --
        full_stride = _stride_for(data, _FULL_MAX_VERTICES)
        z_full = z_norm[::full_stride, ::full_stride] if full_stride > 1 else z_norm
        src_full = z[::full_stride, ::full_stride] if full_stride > 1 else z

        full_surface = _make_surface(z_full, src_full, full_stride)
        view.addItem(full_surface)

        # -- decimated LOD surface --
        lod_stride = _stride_for(data, _LOD_MAX_VERTICES)
        lod_stride = max(lod_stride, full_stride)

        if lod_stride > full_stride:
            z_lod = z_norm[::lod_stride, ::lod_stride]
            src_lod = z[::lod_stride, ::lod_stride]
            lod_surface = _make_surface(z_lod, src_lod, lod_stride)
            lod_surface.setVisible(False)
            view.addItem(lod_surface)
            view.set_surfaces(full_surface, lod_surface)
        else:
            view.set_surfaces(full_surface, None)

        # 3D axes — white lines from origin.
        ax_color = (200, 200, 200, 200)
        half_z = z_height * 0.5
        axes_data = [
            ("row",   [[0, 0, 0], [h, 0, 0]]),
            ("col",   [[0, 0, 0], [0, w, 0]]),
            ("value", [[0, 0, -half_z], [0, 0, half_z]]),
        ]
        for label, pts in axes_data:
            line = gl.GLLinePlotItem(
                pos=np.array(pts, dtype=np.float32),
                color=ax_color,
                width=1.5,
                antialias=True,
            )
            view.addItem(line)
            end = np.array(pts[1], dtype=np.float32) * 1.05
            lbl = gl.GLTextItem(
                pos=end, text=label, color=(180, 180, 180, 220),
            )
            view.addItem(lbl)

        # Camera — set home position and apply it.
        home_center = QVector3D(h / 2, w / 2, 0)
        home_distance = max(h, w) * 1.6
        view.set_home(home_center, home_distance)
        view.opts["center"] = QVector3D(home_center)
        view.setCameraPosition(
            distance=home_distance,
            elevation=25,
            azimuth=45,
        )

    # ── Matplotlib (CPU) fallback ────────────────────────────────

    def _update_matplotlib(
        self, data: np.ndarray, title: str, cmap: str,
    ) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from quality_tool.gui.style import apply_mpl_dark_style, create_dark_figure

        # Remove old canvas.
        if self._view is not None:
            self._layout.removeWidget(self._view)
            self._view.deleteLater()

        figure = create_dark_figure(tight_layout=True)
        canvas = FigureCanvasQTAgg(figure)
        ax = figure.add_subplot(111, projection="3d")
        apply_mpl_dark_style(figure)

        h, w = data.shape
        x = np.arange(w)
        y = np.arange(h)
        x_grid, y_grid = np.meshgrid(x, y)
        z = data.astype(float)

        max_side = 200
        if h > max_side or w > max_side:
            stride_r = max(1, h // max_side)
            stride_c = max(1, w // max_side)
            x_grid = x_grid[::stride_r, ::stride_c]
            y_grid = y_grid[::stride_r, ::stride_c]
            z = z[::stride_r, ::stride_c]

        ax.plot_surface(
            x_grid, y_grid, z,
            cmap=cmap, edgecolor="none", rstride=1, cstride=1,
        )
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        ax.set_zlabel("value")
        ax.set_title(title)

        pane_color = (0.18, 0.18, 0.19, 1.0)
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)

        self._layout.addWidget(canvas)
        self._view = canvas
