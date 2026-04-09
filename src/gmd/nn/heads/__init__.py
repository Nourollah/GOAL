"""Task-specific output heads — registered via the head registry."""

from __future__ import annotations

from gmd.registry import HEAD_REGISTRY

HEAD_REGISTRY.register_lazy(
    "energy_forces", "gmd.nn.heads.energy_forces:EnergyForcesHead"
)
HEAD_REGISTRY.register_lazy(
    "energy", "gmd.nn.heads.energy:EnergyHead"
)
HEAD_REGISTRY.register_lazy(
    "direct_forces", "gmd.nn.heads.direct_forces:DirectForcesHead"
)
HEAD_REGISTRY.register_lazy(
    "stress", "gmd.nn.heads.stress:StressHead"
)
HEAD_REGISTRY.register_lazy(
    "dipole", "gmd.nn.heads.dipole:DipoleHead"
)
