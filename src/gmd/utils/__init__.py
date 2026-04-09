"""GMD utilities — feature extraction, analysis helpers, ASE calculator, and backbone wrappers."""

from __future__ import annotations

from gmd.registry import BACKBONE_REGISTRY
from gmd.utils.calculator import GMDCalculator

BACKBONE_REGISTRY.register_lazy(
    "mace-large-final",
    "gmd.utils.extraction:_build_mace_large_final",
)
BACKBONE_REGISTRY.register_lazy(
    "mace-large-multiscale",
    "gmd.utils.extraction:_build_mace_large_multiscale",
)
BACKBONE_REGISTRY.register_lazy(
    "mace-large-frozen",
    "gmd.utils.extraction:_build_mace_large_frozen",
)
