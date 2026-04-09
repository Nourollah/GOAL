"""Foundation model adapters — wrap pre-trained models as GOAL backbones."""

from __future__ import annotations

from goal.ml.registry import BACKBONE_REGISTRY

BACKBONE_REGISTRY.register_lazy("mace-large", "goal.ml.adapters.mace:MACEAdapter")
BACKBONE_REGISTRY.register_lazy("mace-medium", "goal.ml.adapters.mace:MACEAdapter")
BACKBONE_REGISTRY.register_lazy("mace-small", "goal.ml.adapters.mace:MACEAdapter")
BACKBONE_REGISTRY.register_lazy("uma-small", "goal.ml.adapters.fairchem:UMAAdapter")
