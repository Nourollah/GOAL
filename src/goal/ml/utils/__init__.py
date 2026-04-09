"""GOAL utilities — feature extraction, analysis helpers, ASE calculator, and backbone wrappers."""

from __future__ import annotations

from goal.ml.registry import BACKBONE_REGISTRY
from goal.ml.utils.calculator import GOALCalculator

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
