"""GOAL utilities — feature extraction, analysis helpers, ASE calculator, and backbone wrappers."""

from __future__ import annotations

from goal.ml.registry import BACKBONE_REGISTRY
from goal.ml.utils.calculator import GOALCalculator
from goal.ml.utils.hardware import (
    detect_compute_backend,
    nvalchemi_available,
    nvalchemiops_available,
)

BACKBONE_REGISTRY.register_lazy(
    "mace-large-final",
    "goal.ml.utils.extraction:_build_mace_large_final",
)
BACKBONE_REGISTRY.register_lazy(
    "mace-large-multiscale",
    "goal.ml.utils.extraction:_build_mace_large_multiscale",
)
BACKBONE_REGISTRY.register_lazy(
    "mace-large-frozen",
    "goal.ml.utils.extraction:_build_mace_large_frozen",
)

__all__ = [
    "GOALCalculator",
    "detect_compute_backend",
    "nvalchemi_available",
    "nvalchemiops_available",
]
