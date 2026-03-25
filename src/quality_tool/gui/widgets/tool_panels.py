"""Right-side tool panels for the map and signal sections.

MapToolsPanel — hosts threshold controls (including mask-source metric
selector) beside the map viewer.

SignalToolsPanel — hosts signal display mode controls and envelope
overlay toggle beside the signal inspector.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# Panel width shared by both panels for visual consistency.
_PANEL_WIDTH = 170


class MapToolsPanel(QWidget):
    """Narrow panel beside the map viewer for threshold controls.

    Emits signals when the user interacts with controls.  The parent
    window is responsible for connecting these to backend logic.
    """

    apply_clicked = Signal()
    reset_clicked = Signal()
    slider_moved = Signal(int)
    spin_changed = Signal(float)
    mask_source_changed = Signal(str)
    reset_view_clicked = Signal()
    show_3d_clicked = Signal()

    def __init__(self, slider_steps: int = 1000, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(_PANEL_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Threshold group ---
        group = QGroupBox("Threshold")
        group_layout = QVBoxLayout(group)

        # Mask-source metric selector
        group_layout.addWidget(QLabel("Mask source:"))
        self.mask_source_combo = QComboBox()
        self.mask_source_combo.currentTextChanged.connect(self.mask_source_changed)
        group_layout.addWidget(self.mask_source_combo)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, slider_steps)
        self.slider.valueChanged.connect(self.slider_moved)
        group_layout.addWidget(self.slider)

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setDecimals(4)
        self.spinbox.valueChanged.connect(self.spin_changed)
        group_layout.addWidget(self.spinbox)

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self.apply_clicked)
        group_layout.addWidget(self.btn_apply)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_clicked)
        group_layout.addWidget(self.btn_reset)

        layout.addWidget(group)

        # --- View group ---
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout(view_group)

        self.btn_reset_view = QPushButton("Reset view")
        self.btn_reset_view.clicked.connect(self.reset_view_clicked)
        view_layout.addWidget(self.btn_reset_view)

        self.btn_show_3d = QPushButton("Show 3D map")
        self.btn_show_3d.clicked.connect(self.show_3d_clicked)
        view_layout.addWidget(self.btn_show_3d)

        layout.addWidget(view_group)
        layout.addStretch()


class SignalToolsPanel(QWidget):
    """Narrow panel beside the signal inspector with display mode controls,
    envelope overlay toggle, and pixel metrics button."""

    display_mode_changed = Signal(str)
    envelope_toggled = Signal(bool)
    pixel_metrics_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(_PANEL_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        group = QGroupBox("Signal Display")
        group_layout = QVBoxLayout(group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Raw", "Processed", "Canonical processed",
            "Spectrum", "Processed spectrum", "Autocorrelation",
        ])
        self.mode_combo.currentTextChanged.connect(self.display_mode_changed)
        group_layout.addWidget(self.mode_combo)

        self.envelope_checkbox = QCheckBox("Envelope overlay")
        self.envelope_checkbox.toggled.connect(self.envelope_toggled)
        group_layout.addWidget(self.envelope_checkbox)

        self.btn_pixel_metrics = QPushButton("Pixel metrics")
        self.btn_pixel_metrics.setEnabled(False)
        self.btn_pixel_metrics.clicked.connect(self.pixel_metrics_clicked)
        group_layout.addWidget(self.btn_pixel_metrics)

        layout.addWidget(group)
        layout.addStretch()
