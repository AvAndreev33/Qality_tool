"""Histogram window for Quality_tool.

Displays a fixed snapshot of the value distribution of a metric map,
with optional threshold indicator and descriptive/threshold statistics.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from quality_tool.gui.style import apply_mpl_dark_style, create_dark_figure


def compute_map_statistics(values: np.ndarray) -> dict[str, float]:
    """Compute descriptive statistics for a 1-D array of valid map values.

    Returns a dict with keys: min, max, mean, median, std.
    Returns all NaN if the array is empty.
    """
    if values.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
        }
    return {
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "std": float(np.nanstd(values)),
    }


class HistogramWindow(QWidget):
    """A standalone window showing a fixed snapshot of a metric map histogram.

    Parameters
    ----------
    values:
        1-D array of valid pixel scores (already filtered from the score map).
    metric_name:
        Name of the metric, used for the window title.
    threshold_value:
        Current threshold value to draw as a vertical line, or None.
    keep_rule:
        Human-readable threshold rule string, or None.
    threshold_stats:
        Dict with keys total_pixels, valid_pixels, kept_pixels,
        rejected_pixels, kept_fraction — or None if no threshold is active.
    """

    def __init__(
        self,
        values: np.ndarray,
        metric_name: str = "",
        threshold_value: float | None = None,
        keep_rule: str | None = None,
        threshold_stats: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Histogram — {metric_name}")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(560, 420)

        # Store snapshot data for testing.
        self._values = values.copy()
        self._metric_name = metric_name
        self._threshold_value = threshold_value
        self._keep_rule = keep_rule
        self._threshold_stats = dict(threshold_stats) if threshold_stats else None

        # Compute descriptive statistics.
        self._map_stats = compute_map_statistics(self._values)

        # --- build UI ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Histogram plot.
        figure = create_dark_figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(figure)
        ax = figure.add_subplot(111)
        apply_mpl_dark_style(figure)

        if self._values.size > 0:
            ax.hist(self._values, bins=50, color="#4a86c8", edgecolor="#2a5a8c",
                    alpha=0.85)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_title(metric_name)

        # Threshold line.
        if threshold_value is not None:
            ax.axvline(threshold_value, color="#e05252", linewidth=1.5,
                       linestyle="--", label=f"threshold={threshold_value:.4g}")
            ax.legend(fontsize=8)

        layout.addWidget(self._canvas, stretch=1)

        # --- statistics labels ---
        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)

        _stats_style = (
            "font-size: 11px; font-family: monospace;"
            "color: #d4d4d4; background-color: #252526;"
            "border: 1px solid #3e3e42; border-radius: 3px;"
            "padding: 6px 8px;"
        )

        # Map statistics (left).
        map_text = self._format_map_stats()
        map_label = QLabel(map_text)
        map_label.setTextFormat(Qt.TextFormat.PlainText)
        map_label.setStyleSheet(_stats_style)
        stats_row.addWidget(map_label, stretch=1)

        # Threshold statistics (right).
        thresh_text = self._format_threshold_stats()
        thresh_label = QLabel(thresh_text)
        thresh_label.setTextFormat(Qt.TextFormat.PlainText)
        thresh_label.setStyleSheet(_stats_style)
        stats_row.addWidget(thresh_label, stretch=1)

        layout.addLayout(stats_row)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_map_stats(self) -> str:
        s = self._map_stats
        lines = [
            "Map statistics",
            f"  min:    {s['min']:.4g}",
            f"  max:    {s['max']:.4g}",
            f"  mean:   {s['mean']:.4g}",
            f"  median: {s['median']:.4g}",
            f"  std:    {s['std']:.4g}",
        ]
        return "\n".join(lines)

    def _format_threshold_stats(self) -> str:
        ts = self._threshold_stats
        if ts is None:
            return "Threshold\n  not applied"
        kept = ts.get("kept_pixels", "?")
        rejected = ts.get("rejected_pixels", "?")
        valid = ts.get("valid_pixels", "?")
        fraction = ts.get("kept_fraction", None)
        pct = f"{fraction * 100:.1f}%" if fraction is not None else "?"
        lines = [
            "Threshold statistics",
            f"  valid:    {valid}",
            f"  kept:     {kept}",
            f"  rejected: {rejected}",
            f"  kept %:   {pct}",
        ]
        return "\n".join(lines)
