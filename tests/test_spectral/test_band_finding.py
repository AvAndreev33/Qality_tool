"""Tests for the carrier band detection helper."""

import numpy as np

from quality_tool.spectral.fft import find_carrier_band


class TestFindCarrierBand:
    """Tests for find_carrier_band."""

    def test_finds_peak(self):
        """Band should center on the dominant non-DC peak."""
        freqs = np.linspace(0.0, 0.5, 65)
        amp = np.ones(65) * 0.1
        # Place a strong peak at index 20.
        amp[20] = 10.0
        result = find_carrier_band(freqs, amp, band_half_width_bins=3)
        assert result is not None
        carrier, lo, hi = result
        assert carrier == freqs[20]
        assert lo == freqs[17]
        assert hi == freqs[23]

    def test_excludes_dc(self):
        """DC bin should not be picked as carrier."""
        freqs = np.linspace(0.0, 0.5, 65)
        amp = np.ones(65) * 0.1
        amp[0] = 100.0   # large DC
        amp[10] = 5.0     # smaller but real carrier
        result = find_carrier_band(freqs, amp, band_half_width_bins=2)
        assert result is not None
        carrier, _, _ = result
        assert carrier == freqs[10]

    def test_band_clamps_at_edges(self):
        """Band should not extend past array boundaries."""
        freqs = np.linspace(0.0, 0.5, 10)
        amp = np.ones(10) * 0.1
        amp[1] = 5.0  # peak near start
        result = find_carrier_band(freqs, amp, band_half_width_bins=5)
        assert result is not None
        _, lo, hi = result
        assert lo == freqs[0]

    def test_returns_none_for_tiny_signal(self):
        """Should return None for a signal shorter than 2."""
        freqs = np.array([0.0])
        amp = np.array([1.0])
        assert find_carrier_band(freqs, amp) is None

    def test_dc_include(self):
        """When dc_exclude=False, DC can be picked."""
        freqs = np.linspace(0.0, 0.5, 10)
        amp = np.ones(10) * 0.1
        amp[0] = 10.0
        result = find_carrier_band(freqs, amp, dc_exclude=False)
        assert result is not None
        carrier, _, _ = result
        assert carrier == freqs[0]
