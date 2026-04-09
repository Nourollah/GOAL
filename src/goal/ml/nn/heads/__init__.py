"""Task-specific output heads — registered via the head registry."""

from __future__ import annotations

from goal.ml.registry import HEAD_REGISTRY

HEAD_REGISTRY.register_lazy("energy_forces", "goal.ml.nn.heads.energy_forces:EnergyForcesHead")
HEAD_REGISTRY.register_lazy("energy", "goal.ml.nn.heads.energy:EnergyHead")
HEAD_REGISTRY.register_lazy("direct_forces", "goal.ml.nn.heads.direct_forces:DirectForcesHead")
HEAD_REGISTRY.register_lazy("stress", "goal.ml.nn.heads.stress:StressHead")
HEAD_REGISTRY.register_lazy("dipole", "goal.ml.nn.heads.dipole:DipoleHead")
