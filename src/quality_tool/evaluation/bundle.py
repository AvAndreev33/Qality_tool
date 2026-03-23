"""Representation bundle for Quality_tool.

A bundle groups a prepared signal chunk with its derived
representations (envelope, spectral data).  Bundles are always tied
to the recipe that produced the prepared signal — representations
derived from different recipes are never mixed.

Bundles are short-lived: one is built per recipe group per chunk
inside the evaluator, used to dispatch metrics, and then discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quality_tool.core.analysis_context import AnalysisContext
from quality_tool.evaluation.recipe import SignalRecipe
from quality_tool.metrics.base import RepresentationNeeds
from quality_tool.spectral.fft import BatchSpectralResult
from quality_tool.spectral.priors import SpectralPriors, compute_spectral_priors


@dataclass
class RepresentationBundle:
    """Derived representations for one prepared-signal chunk.

    Attributes
    ----------
    signals : np.ndarray
        Prepared signal chunk, shape ``(N, M)``.
    z_axis : np.ndarray | None
        Shared z-axis (``None`` when ROI truncated it).
    recipe : SignalRecipe
        The recipe that produced ``signals``.
    analysis_context : AnalysisContext
        Shared constants and heuristics.
    envelope : np.ndarray | None
        Envelope of the prepared signal, shape ``(N, M)``.
        ``None`` when envelope was not requested or not available.
    spectral : BatchSpectralResult | None
        Batch spectral result.  ``None`` when no spectral
        representation was requested.
    """

    signals: np.ndarray
    z_axis: np.ndarray | None
    recipe: SignalRecipe
    analysis_context: AnalysisContext
    envelope: np.ndarray | None = None
    spectral: BatchSpectralResult | None = None
    spectral_priors: SpectralPriors | None = None

    def __post_init__(self) -> None:
        """Auto-compute spectral priors from signal length if needed."""
        if self.spectral_priors is None and self.signals.ndim == 2:
            m = self.signals.shape[1]
            object.__setattr__(
                self,
                "spectral_priors",
                compute_spectral_priors(m, self.analysis_context),
            )

    def to_context_dict(self) -> dict:
        """Build a ``context`` dict suitable for metric ``evaluate`` /
        ``evaluate_batch`` calls.

        This provides backward-compatible keys that existing metrics
        already consume (``batch_frequencies``, ``batch_amplitude``)
        alongside the new structured objects.
        """
        ctx: dict = {
            "analysis_context": self.analysis_context,
        }

        if self.spectral_priors is not None:
            ctx["spectral_priors"] = self.spectral_priors

        if self.spectral is not None:
            ctx["spectral"] = self.spectral
            # Backward-compatible keys used by existing metrics.
            ctx["batch_frequencies"] = self.spectral.frequencies
            if self.spectral.amplitude is not None:
                ctx["batch_amplitude"] = self.spectral.amplitude
            if self.spectral.power is not None:
                ctx["batch_power"] = self.spectral.power

        return ctx
