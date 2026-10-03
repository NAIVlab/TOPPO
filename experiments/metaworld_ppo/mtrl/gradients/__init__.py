"""Gradient manipulation methods for multi-task learning."""

from mtrl.gradients.cagrad import CAGradDiagnostics, cagrad_combine
from mtrl.gradients.fairgrad import FairGradDiagnostics, fairgrad_combine
from mtrl.gradients.pcgrad import PCGradDiagnostics, pcgrad_combine
from mtrl.gradients.per_side import PerSideGradientOptimizer, build_combine_fn
