"""Signal inspector widget for Quality_tool.

Displays the 1-D signal of a selected pixel.  Supports three display
modes (raw, processed, spectrum) and an optional envelope overlay.
The widget only renders data — all processing is done by the caller
before passing data in.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from quality_tool.gui.style import apply_mpl_dark_style, create_dark_figure


class SignalInspector(QWidget):
    """Lower plot showing the signal of the currently selected pixel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._figure = create_dark_figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        apply_mpl_dark_style(self._figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._ax.set_xlabel("z")
        self._ax.set_ylabel("intensity")
        self._ax.set_title("Signal")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_signal(
        self,
        signal: np.ndarray,
        z_axis: np.ndarray,
        *,
        label: str = "raw",
        title: str | None = None,
        envelope: np.ndarray | None = None,
    ) -> None:
        """Plot a 1-D signal with optional envelope overlay.

        Parameters
        ----------
        signal : np.ndarray
            1-D array of length M.
        z_axis : np.ndarray
            1-D array of length M — physical or index-based.
        label : str
            Legend label for the signal line.
        title : str or None
            Plot title.  Defaults to ``"Signal"``.
        envelope : np.ndarray or None
            If provided, overlay this 1-D envelope on the plot.
        """
        self._ax.clear()
        apply_mpl_dark_style(self._figure)
        self._ax.plot(z_axis, signal, linewidth=0.8, label=label, color="#4fc3f7")
        if envelope is not None:
            self._ax.plot(
                z_axis, envelope, linewidth=1.0, linestyle="--",
                color="tab:orange", label="envelope",
            )
        self._ax.set_xlabel("z")
        self._ax.set_ylabel("intensity")
        self._ax.set_title(title or "Signal")
        self._ax.legend(loc="upper right", fontsize="small")
        self._canvas.draw_idle()

    def update_spectrum(
        self,
        frequencies: np.ndarray,
        amplitude: np.ndarray,
        *,
        title: str | None = None,
        band_info: tuple[float, float, float] | None = None,
        expected_band_info: tuple[float, float, float] | None = None,
    ) -> None:
        """Plot the amplitude spectrum with logarithmic y-scale.

        Parameters
        ----------
        frequencies : np.ndarray
            1-D frequency array.
        amplitude : np.ndarray
            1-D amplitude array.
        title : str or None
            Plot title.
        band_info : tuple or None
            ``(carrier_freq, band_low, band_high)`` — empirically
            detected band.  Highlighted with a blue shaded span.
        expected_band_info : tuple or None
            ``(expected_carrier_freq, band_low, band_high)`` —
            metadata-derived expected band.  Highlighted with an
            orange shaded span.
        """
        self._ax.clear()
        apply_mpl_dark_style(self._figure)

        if expected_band_info is not None:
            e_carrier, e_lo, e_hi = expected_band_info
            self._ax.axvspan(
                e_lo, e_hi, alpha=0.12, color="tab:orange",
                label="expected band",
            )
            self._ax.axvline(
                e_carrier, color="tab:orange", linewidth=0.7,
                linestyle=":", alpha=0.7,
            )

        if band_info is not None:
            carrier, b_lo, b_hi = band_info
            self._ax.axvspan(
                b_lo, b_hi, alpha=0.15, color="tab:blue",
                label="empirical band",
            )
            self._ax.axvline(
                carrier, color="tab:blue", linewidth=0.7,
                linestyle="--", alpha=0.6,
            )

        self._ax.plot(frequencies, amplitude, linewidth=0.8, label="spectrum", color="#4fc3f7")
        self._ax.set_yscale("log")
        self._ax.set_xlabel("frequency")
        self._ax.set_ylabel("amplitude")
        self._ax.set_title(title or "Spectrum")
        self._ax.legend(loc="upper right", fontsize="small")
        self._canvas.draw_idle()

    def update_autocorrelation(
        self,
        lags: np.ndarray,
        autocorr: np.ndarray,
        *,
        title: str | None = None,
        expected_period: float | None = None,
        search_window: tuple[float, float] | None = None,
        detected_peak_lag: float | None = None,
    ) -> None:
        """Plot normalized autocorrelation with expected-period guidance.

        Parameters
        ----------
        lags : np.ndarray
            1-D array of lag values.
        autocorr : np.ndarray
            1-D array of normalized autocorrelation values.
        title : str or None
            Plot title.
        expected_period : float or None
            Expected fringe period in samples — shown as a vertical line.
        search_window : tuple or None
            ``(tau_min, tau_max)`` — shaded search interval.
        detected_peak_lag : float or None
            Lag of the dominant autocorrelation peak in the search window.
        """
        self._ax.clear()
        apply_mpl_dark_style(self._figure)

        if search_window is not None:
            tau_lo, tau_hi = search_window
            self._ax.axvspan(
                tau_lo, tau_hi, alpha=0.12, color="tab:orange",
                label="search window",
            )

        if expected_period is not None:
            self._ax.axvline(
                expected_period, color="tab:orange", linewidth=0.8,
                linestyle=":", alpha=0.7, label="expected period",
            )

        self._ax.plot(lags, autocorr, linewidth=0.8, label="autocorrelation", color="#4fc3f7")

        if detected_peak_lag is not None:
            # Find the autocorrelation value at the detected peak lag.
            idx = int(round(detected_peak_lag))
            idx = max(0, min(idx, len(autocorr) - 1))
            self._ax.plot(
                detected_peak_lag, autocorr[idx], "o",
                color="tab:red", markersize=5, label="detected peak",
            )

        self._ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        self._ax.set_yscale("linear")
        self._ax.set_ylim(-1.05, 1.05)
        self._ax.set_xlabel("lag (samples)")
        self._ax.set_ylabel("normalized autocorrelation")
        self._ax.set_title(title or "Autocorrelation")
        self._ax.legend(loc="upper right", fontsize="small")
        self._canvas.draw_idle()

    def clear(self) -> None:
        """Clear the signal plot."""
        self._ax.clear()
        apply_mpl_dark_style(self._figure)
        self._ax.set_xlabel("z")
        self._ax.set_ylabel("intensity")
        self._ax.set_title("Signal")
        self._canvas.draw_idle()
