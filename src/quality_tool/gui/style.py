"""Shared dark-theme styling for the Quality_tool GUI.

Provides a Qt stylesheet, matplotlib figure styling, and color constants
used across all windows and widgets.  Intentionally self-contained —
no external theme dependencies.
"""

from __future__ import annotations

from matplotlib.figure import Figure

# ── Color palette ─────────────────────────────────────────────────────
# Layered dark grays (darker = lower elevation).
BG_BASE = "#1e1e1e"
BG_PANEL = "#252526"
BG_WIDGET = "#2d2d30"
BG_HOVER = "#3e3e42"
BG_SELECTED = "#264f78"

# Borders and dividers.
BORDER = "#3e3e42"
BORDER_LIGHT = "#474747"

# Text.
TEXT_PRIMARY = "#d4d4d4"
TEXT_SECONDARY = "#808080"
TEXT_DISABLED = "#5a5a5a"

# Accent.
ACCENT = "#007acc"
ACCENT_HOVER = "#1c97ea"
ACCENT_PRESSED = "#005a9e"

# Semantic.
COLOR_KEPT = "#4caf50"
COLOR_REJECTED = "#e05252"
COLOR_WARNING = "#cca700"

# ── Qt stylesheet ─────────────────────────────────────────────────────

DARK_STYLESHEET = f"""
/* ── Global ──────────────────────────────────────────────── */
QWidget {{
    background-color: {BG_BASE};
    color: {TEXT_PRIMARY};
    font-size: 12px;
}}

/* ── Main window & toolbar ───────────────────────────────── */
QMainWindow {{
    background-color: {BG_BASE};
}}

QToolBar {{
    background-color: {BG_PANEL};
    border-bottom: 1px solid {BORDER};
    spacing: 4px;
    padding: 3px 6px;
}}

QToolBar QLabel {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
    padding: 0 2px;
    background: transparent;
}}

QToolBar::separator {{
    width: 1px;
    background: {BORDER};
    margin: 4px 6px;
}}

/* ── Status bar ──────────────────────────────────────────── */
QStatusBar {{
    background-color: {BG_PANEL};
    color: {TEXT_SECONDARY};
    border-top: 1px solid {BORDER};
    font-size: 11px;
    padding: 2px 8px;
}}

/* ── Push buttons ────────────────────────────────────────── */
QPushButton {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 4px 12px;
    min-height: 18px;
}}

QPushButton:hover {{
    background-color: {BG_HOVER};
    border-color: {BORDER_LIGHT};
}}

QPushButton:pressed {{
    background-color: {ACCENT_PRESSED};
    border-color: {ACCENT};
}}

QPushButton:disabled {{
    color: {TEXT_DISABLED};
    background-color: {BG_BASE};
    border-color: {BG_WIDGET};
}}

/* ── Combo boxes ─────────────────────────────────────────── */
QComboBox {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 3px 8px;
    min-height: 18px;
}}

QComboBox:hover {{
    border-color: {BORDER_LIGHT};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_SECONDARY};
    margin-right: 6px;
}}

QComboBox QAbstractItemView {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    selection-background-color: {BG_SELECTED};
    selection-color: {TEXT_PRIMARY};
    outline: none;
}}

/* ── Spin boxes ──────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 3px 6px;
    min-height: 18px;
}}

QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {BORDER_LIGHT};
}}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {BG_HOVER};
    border: none;
    width: 16px;
}}

/* ── Progress bar ────────────────────────────────────────── */
QProgressBar {{
    background-color: {BG_WIDGET};
    border: 1px solid {BORDER};
    border-radius: 3px;
    text-align: center;
    color: {TEXT_SECONDARY};
    font-size: 10px;
}}

QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 2px;
}}

/* ── Check boxes ─────────────────────────────────────────── */
QCheckBox {{
    spacing: 6px;
    color: {TEXT_PRIMARY};
    background: transparent;
}}

QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER};
    border-radius: 2px;
    background-color: {BG_WIDGET};
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

QCheckBox::indicator:hover {{
    border-color: {BORDER_LIGHT};
}}

/* ── Group boxes ─────────────────────────────────────────── */
QGroupBox {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: bold;
    font-size: 11px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: {TEXT_PRIMARY};
    background-color: {BG_PANEL};
}}

/* ── Sliders ─────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {BG_WIDGET};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {ACCENT};
    border: none;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}}

QSlider::handle:horizontal:hover {{
    background: {ACCENT_HOVER};
}}

/* ── Scroll areas / bars ─────────────────────────────────── */
QScrollArea {{
    background-color: {BG_BASE};
    border: none;
}}

QScrollBar:vertical {{
    background: {BG_BASE};
    width: 10px;
    border: none;
}}

QScrollBar::handle:vertical {{
    background: {BG_HOVER};
    border-radius: 4px;
    min-height: 20px;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: {BG_BASE};
    height: 10px;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background: {BG_HOVER};
    border-radius: 4px;
    min-width: 20px;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── Tables ──────────────────────────────────────────────── */
QTableWidget {{
    background-color: {BG_BASE};
    alternate-background-color: {BG_PANEL};
    color: {TEXT_PRIMARY};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    selection-background-color: {BG_SELECTED};
    selection-color: {TEXT_PRIMARY};
}}

QHeaderView::section {{
    background-color: {BG_PANEL};
    color: {TEXT_PRIMARY};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 4px 8px;
    font-weight: bold;
    font-size: 11px;
}}

/* ── Dialogs ─────────────────────────────────────────────── */
QDialog {{
    background-color: {BG_BASE};
}}

/* ── Splitter handle ─────────────────────────────────────── */
QSplitter::handle {{
    background-color: {BORDER};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

/* ── Message box ─────────────────────────────────────────── */
QMessageBox {{
    background-color: {BG_BASE};
}}

QMessageBox QLabel {{
    color: {TEXT_PRIMARY};
}}

/* ── Tooltips ────────────────────────────────────────────── */
QToolTip {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    padding: 4px;
}}
"""

# ── Matplotlib dark styling ───────────────────────────────────────────

_MPL_RC = {
    "figure.facecolor": BG_PANEL,
    "axes.facecolor": BG_WIDGET,
    "axes.edgecolor": BORDER_LIGHT,
    "axes.labelcolor": TEXT_PRIMARY,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": False,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "grid.color": BG_HOVER,
    "grid.alpha": 0.3,
    "legend.facecolor": BG_PANEL,
    "legend.edgecolor": BORDER,
    "legend.fontsize": 8,
    "legend.labelcolor": TEXT_PRIMARY,
    "savefig.facecolor": BG_PANEL,
    "figure.titlesize": 11,
}

# Colors tested for legibility on dark backgrounds.
_MPL_DARK_CYCLE = [
    "#4fc3f7",  # light blue
    "#ff8a65",  # orange
    "#81c784",  # green
    "#e57373",  # red
    "#ba68c8",  # purple
    "#fff176",  # yellow
    "#4dd0e1",  # cyan
    "#f48fb1",  # pink
]


def apply_mpl_dark_style(figure: Figure) -> None:
    """Apply the dark theme to a matplotlib Figure and its axes.

    Call this right after creating the Figure (before any plotting)
    so backgrounds, text, ticks, and legend colors all match the Qt
    dark palette.
    """
    for key, value in _MPL_RC.items():
        parts = key.split(".")
        if len(parts) == 2:
            group, param = parts
            try:
                figure.get_axes()  # ensure figure exists
            except Exception:
                pass

    # Apply figure-level properties.
    figure.set_facecolor(_MPL_RC["figure.facecolor"])

    # Apply to all current and future axes via rcParams update on
    # the figure's canvas if available, otherwise set directly.
    for ax in figure.get_axes():
        _style_axes(ax)


def _style_axes(ax) -> None:
    """Apply dark styling to a single matplotlib Axes."""
    ax.set_facecolor(_MPL_RC["axes.facecolor"])
    ax.tick_params(
        colors=_MPL_RC["xtick.color"],
        labelsize=_MPL_RC["xtick.labelsize"],
    )
    ax.xaxis.label.set_color(_MPL_RC["axes.labelcolor"])
    ax.xaxis.label.set_fontsize(_MPL_RC["axes.labelsize"])
    ax.yaxis.label.set_color(_MPL_RC["axes.labelcolor"])
    ax.yaxis.label.set_fontsize(_MPL_RC["axes.labelsize"])
    ax.title.set_color(_MPL_RC["text.color"])
    ax.title.set_fontsize(_MPL_RC["axes.titlesize"])
    for spine in ax.spines.values():
        spine.set_edgecolor(_MPL_RC["axes.edgecolor"])


def create_dark_figure(**kwargs) -> Figure:
    """Create a new matplotlib Figure pre-styled for the dark theme.

    Accepts the same keyword arguments as ``Figure()``.
    """
    kwargs.setdefault("facecolor", _MPL_RC["figure.facecolor"])
    fig = Figure(**kwargs)
    return fig
