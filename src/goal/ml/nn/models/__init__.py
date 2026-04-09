"""Full model architectures — registered for discovery via the model registry."""

from __future__ import annotations

from goal.ml.registry import MODEL_REGISTRY

MODEL_REGISTRY.register_lazy("hyperspec", "goal.ml.nn.models.hyperspec:HyperSpecModel")
MODEL_REGISTRY.register_lazy("invariant_gnn", "goal.ml.nn.models.invariant:InvariantGNN")
