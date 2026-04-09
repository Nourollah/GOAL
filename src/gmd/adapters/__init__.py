"""Foundation model adapters — wrap pre-trained models as GMD backbones."""

from __future__ import annotations

from gmd.registry import BACKBONE_REGISTRY

BACKBONE_REGISTRY.register_lazy("mace-large", "gmd.adapters.mace:MACEAdapter")
BACKBONE_REGISTRY.register_lazy("mace-medium", "gmd.adapters.mace:MACEAdapter")
BACKBONE_REGISTRY.register_lazy("mace-small", "gmd.adapters.mace:MACEAdapter")
BACKBONE_REGISTRY.register_lazy("uma-small", "gmd.adapters.fairchem:UMAAdapter")
