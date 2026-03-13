"""Tests for the MetricsDialog."""

from __future__ import annotations

from quality_tool.gui.dialogs.metrics_dialog import MetricsDialog
from quality_tool.metrics.baseline.fringe_visibility import FringeVisibility
from quality_tool.metrics.baseline.snr import SNR
from quality_tool.metrics.baseline.power_band_ratio import PowerBandRatio
from quality_tool.metrics.registry import MetricRegistry


def _make_registry() -> MetricRegistry:
    reg = MetricRegistry()
    reg.register(FringeVisibility())
    reg.register(SNR())
    reg.register(PowerBandRatio())
    return reg


class TestMetricsDialog:
    def test_creation(self):
        dlg = MetricsDialog(_make_registry())
        assert dlg is not None

    def test_no_preselection(self):
        dlg = MetricsDialog(_make_registry())
        assert dlg.selected_metrics() == []

    def test_preselection_preserved(self):
        dlg = MetricsDialog(_make_registry(), selected=["snr"])
        selected = dlg.selected_metrics()
        assert "snr" in selected
        assert len(selected) == 1

    def test_all_selected(self):
        reg = _make_registry()
        all_names = reg.list_metrics()
        dlg = MetricsDialog(reg, selected=all_names)
        assert set(dlg.selected_metrics()) == set(all_names)

    def test_checkbox_toggle(self):
        reg = _make_registry()
        dlg = MetricsDialog(reg, selected=["snr"])
        # Uncheck snr, check fringe_visibility.
        dlg._checkboxes["snr"].setChecked(False)
        dlg._checkboxes["fringe_visibility"].setChecked(True)
        selected = dlg.selected_metrics()
        assert "snr" not in selected
        assert "fringe_visibility" in selected
