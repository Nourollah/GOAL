"""Edge-energy head: bond energies, their atomic split and the total.

``ε_ij = MLP(x_ij) · u(r_ij)`` for every directed edge, from the invariant
(``m = 0``) part of its bond state; the envelope ``u`` makes ``ε`` and its
derivative vanish at the cutoff.  Atomic energies are
``e_i = ½ Σ_j (ε_ij + ε_ji)``, so ``Σ_i e_i`` equals the edge sum exactly and
energy / force heads, metrics and fragment consistency work on them
unchanged.  ``bond_energies`` holds ``ε_ij + ε_ji`` on both directed edges of
a bond.  Unlike a :class:`~goal.ml.nn.models.base.TaskHead` it consumes
edge features, so it is called directly by edge-centred models.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import scatter

__all__ = ["EdgeEnergyHead", "reverse_edges"]


def reverse_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Index of the edge ``j -> i`` for every edge ``i -> j`` (the list must be symmetric)."""
    i, j = edge_index
    key = i * num_nodes + j
    order = torch.argsort(key)
    position = torch.searchsorted(key[order], j * num_nodes + i)
    return order[position.clamp_max(key.numel() - 1)]


class EdgeEnergyHead(nn.Module):
    """Bias-free MLP on bond invariants times the envelope.

    ``zero_init`` starts the output layer at zero (an added bond line leaves
    a fresh model unchanged).
    """

    def __init__(self, in_channels: int, hidden: list[int], zero_init: bool = False) -> None:
        super().__init__()
        sizes = [in_channels, *hidden, 1]
        layers: list[nn.Module] = []
        for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(a, b, bias=False))
            if k < len(sizes) - 2:
                layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)

    def forward(
        self, invariants: torch.Tensor, env: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
    ) -> dict[str, torch.Tensor]:
        """``edge_energies`` ``(E,)``, ``atomic_energies`` ``(N,)`` and ``bond_energies`` ``(E,)``."""
        eps = self.mlp(invariants).squeeze(-1) * env
        half = 0.5 * eps
        atomic = scatter(half, edge_index[0], dim=0, dim_size=num_nodes, reduce="sum")
        atomic = atomic + scatter(half, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")
        bond = eps + eps[reverse_edges(edge_index, num_nodes)] if eps.numel() else eps
        return {"edge_energies": eps, "atomic_energies": atomic, "bond_energies": bond}
