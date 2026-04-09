"""Full model architectures — registered for discovery via the model registry."""

from __future__ import annotations

from gmd.registry import MODEL_REGISTRY

MODEL_REGISTRY.register_lazy("hyperspec", "gmd.nn.models.hyperspec:HyperSpecModel")
MODEL_REGISTRY.register_lazy("invariant_gnn", "gmd.nn.models.invariant:InvariantGNN")
