"""Full model architectures — registered for discovery via the model registry."""

from __future__ import annotations

from goal.ml.registry import MODEL_REGISTRY

MODEL_REGISTRY.register_lazy("hyperspec", "goal.ml.nn.models.hyperspec:HyperSpecModel")
MODEL_REGISTRY.register_lazy("invariant_gnn", "goal.ml.nn.models.invariant:InvariantGNN")

# Invariant backbones — edge-based scalar feature extractors
MODEL_REGISTRY.register_lazy("deepset", "goal.ml.nn.models.deepset:DeepSet")
MODEL_REGISTRY.register_lazy("hyperset", "goal.ml.nn.models.hyperset:HyperSet")
MODEL_REGISTRY.register_lazy("lucidset", "goal.ml.nn.models.lucidset:LucidSet")

# Monolithic example — bypasses backbone→head split (head: null)
MODEL_REGISTRY.register_lazy("monolithic_example", "goal.ml.nn.models.monolithic_example:MonolithicExample")
