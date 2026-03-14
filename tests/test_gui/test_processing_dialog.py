"""Tests for the processing settings dialog."""

from __future__ import annotations

from quality_tool.gui.dialogs.processing_dialog import ProcessingDialog


class TestProcessingDialog:
    def test_creation_defaults(self):
        dlg = ProcessingDialog(envelope_methods=["analytic"])
        assert dlg is not None

    def test_settings_returns_dict(self):
        dlg = ProcessingDialog(envelope_methods=["analytic"])
        s = dlg.settings()
        assert isinstance(s, dict)
        assert "baseline" in s
        assert "normalize" in s
        assert "smooth" in s
        assert "roi_enabled" in s
        assert "segment_size" in s
        assert "envelope_enabled" in s
        assert "envelope_method" in s

    def test_defaults_all_off(self):
        dlg = ProcessingDialog(envelope_methods=["analytic"])
        s = dlg.settings()
        assert s["baseline"] is False
        assert s["normalize"] is False
        assert s["smooth"] is False
        assert s["roi_enabled"] is False
        assert s["envelope_enabled"] is False

    def test_respects_current_settings(self):
        current = {
            "baseline": True,
            "normalize": False,
            "smooth": True,
            "roi_enabled": True,
            "segment_size": 256,
            "envelope_enabled": True,
            "envelope_method": "analytic",
        }
        dlg = ProcessingDialog(
            envelope_methods=["analytic"], current=current,
        )
        s = dlg.settings()
        assert s["baseline"] is True
        assert s["smooth"] is True
        assert s["roi_enabled"] is True
        assert s["segment_size"] == 256
        assert s["envelope_enabled"] is True
        assert s["envelope_method"] == "analytic"

    def test_segment_size_default(self):
        dlg = ProcessingDialog(envelope_methods=[])
        s = dlg.settings()
        assert s["segment_size"] == 128

    def test_empty_envelope_methods(self):
        dlg = ProcessingDialog(envelope_methods=[])
        s = dlg.settings()
        assert s["envelope_method"] == ""

    def test_multiple_envelope_methods(self):
        dlg = ProcessingDialog(envelope_methods=["analytic", "other"])
        # Default should be first
        s = dlg.settings()
        assert s["envelope_method"] == "analytic"
