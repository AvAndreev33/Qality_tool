"""Processing settings dialog for Quality_tool.

Allows the user to configure preprocessing, ROI extraction, and
envelope method selection.  All settings are returned as a plain dict
so the caller can store them in session state without coupling to this
dialog class.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ProcessingDialog(QDialog):
    """Modal dialog for preprocessing, ROI, and envelope settings."""

    def __init__(
        self,
        envelope_methods: list[str],
        current: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Processing Settings")
        self.resize(360, 340)

        cur = current or {}

        layout = QVBoxLayout(self)

        # --- Preprocessing group ---
        pp_group = QGroupBox("Preprocessing")
        pp_layout = QVBoxLayout(pp_group)

        self._chk_baseline = QCheckBox("Baseline subtraction")
        self._chk_baseline.setChecked(cur.get("baseline", False))
        pp_layout.addWidget(self._chk_baseline)

        self._chk_normalize = QCheckBox("Normalize amplitude")
        self._chk_normalize.setChecked(cur.get("normalize", False))
        pp_layout.addWidget(self._chk_normalize)

        self._chk_smooth = QCheckBox("Smoothing")
        self._chk_smooth.setChecked(cur.get("smooth", False))
        pp_layout.addWidget(self._chk_smooth)

        layout.addWidget(pp_group)

        # --- ROI group ---
        roi_group = QGroupBox("ROI Extraction")
        roi_layout = QVBoxLayout(roi_group)

        self._chk_roi = QCheckBox("Enable ROI")
        self._chk_roi.setChecked(cur.get("roi_enabled", False))
        roi_layout.addWidget(self._chk_roi)

        seg_row = QHBoxLayout()
        seg_row.addWidget(QLabel("Segment size:"))
        self._spin_segment = QSpinBox()
        self._spin_segment.setRange(3, 100_000)
        self._spin_segment.setValue(cur.get("segment_size", 128))
        seg_row.addWidget(self._spin_segment)
        roi_layout.addLayout(seg_row)

        roi_layout.addWidget(QLabel("Centering: raw_max (fixed)"))

        layout.addWidget(roi_group)

        # --- Envelope group ---
        env_group = QGroupBox("Envelope")
        env_layout = QVBoxLayout(env_group)

        self._chk_envelope = QCheckBox("Enable envelope")
        self._chk_envelope.setChecked(cur.get("envelope_enabled", False))
        env_layout.addWidget(self._chk_envelope)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._combo_method = QComboBox()
        if envelope_methods:
            self._combo_method.addItems(envelope_methods)
            saved = cur.get("envelope_method", "")
            idx = self._combo_method.findText(saved)
            if idx >= 0:
                self._combo_method.setCurrentIndex(idx)
        method_row.addWidget(self._combo_method)
        env_layout.addLayout(method_row)

        layout.addWidget(env_group)

        # --- OK / Cancel ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

    def settings(self) -> dict:
        """Return the current dialog state as a plain dict."""
        return {
            "baseline": self._chk_baseline.isChecked(),
            "normalize": self._chk_normalize.isChecked(),
            "smooth": self._chk_smooth.isChecked(),
            "roi_enabled": self._chk_roi.isChecked(),
            "segment_size": self._spin_segment.value(),
            "envelope_enabled": self._chk_envelope.isChecked(),
            "envelope_method": self._combo_method.currentText(),
        }
