"""Foundation model adapters — wrap pre-trained models as GOAL backbones."""

from __future__ import annotations

from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY

BACKBONE_REGISTRY.register_lazy("mace-large", "goal.ml.adapters.mace:MACEAdapter")
BACKBONE_REGISTRY.register_lazy("mace-medium", "goal.ml.adapters.mace:MACEAdapter")
BACKBONE_REGISTRY.register_lazy("mace-small", "goal.ml.adapters.mace:MACEAdapter")
BACKBONE_REGISTRY.register_lazy("uma-small", "goal.ml.adapters.fairchem:UMAAdapter")

# LoRA fine-tuning of a trained SIMURGH checkpoint (needs peft at build time).
BACKBONE_REGISTRY.register_lazy(
    "simurgh_lora", "goal.ml.adapters.simurgh_lora:SimurghLoRABackbone"
)
MODEL_REGISTRY.register_lazy("simurgh_lora", "goal.ml.adapters.simurgh_lora:SimurghLoRABackbone")

# nvalchemi-toolkit model wrapper (NVIDIA only; importable everywhere).
from goal.ml.adapters.nvalchemi_wrapper import SIMURGHNvalchemiWrapper
from goal.ml.adapters.simurgh_lora import SIMURGHLoRAAdapter, SIMURGHLoRAConfig

__all__ = ["SIMURGHLoRAAdapter", "SIMURGHLoRAConfig", "SIMURGHNvalchemiWrapper"]
