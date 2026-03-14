"""Right-side tool panels for the map and signal sections.

MapToolsPanel — hosts threshold controls beside the map viewer.
SignalToolsPanel — placeholder for future signal-related controls.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
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

    def __init__(self, slider_steps: int = 1000, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(_PANEL_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Threshold group ---
        group = QGroupBox("Threshold")
        group_layout = QVBoxLayout(group)

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
        layout.addStretch()


class SignalToolsPanel(QWidget):
    """Narrow placeholder panel beside the signal inspector.

    Reserves layout space for future signal-related controls
    (envelope, spectrum, ROI inspection, etc.).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(_PANEL_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        label = QLabel("Signal Tools")
        label.setStyleSheet("color: gray;")
        layout.addWidget(label)
        layout.addStretch()
