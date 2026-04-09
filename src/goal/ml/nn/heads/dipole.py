"""Dipole moment output head.

Computes the total dipole moment of the system from per-atom partial
charges and positions:

    mu = sum_i q_i * r_i

where q_i are learned per-atom charges from a scalar readout and r_i
are atomic positions. This is an equivariant quantity (transforms as
a vector under rotations) derived from invariant per-atom charges.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.nn.blocks.readout import ScalarReadout
from goal.ml.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("dipole")
class DipoleHead(nn.Module):
    """Head that predicts dipole moment from per-atom charges.

    Charges are learned scalars; dipole = sum(q_i * r_i) per graph.
    Also outputs per-atom charges and total charge for auxiliary losses.

    Parameters
    ----------
    irreps_in : str
        Input irreps from the backbone.
    hidden_dim : int
        Width of the readout MLP for charge prediction.
    """

    def __init__(
        self,
        irreps_in: str,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.readout: ScalarReadout = ScalarReadout(irreps_in=irreps_in, hidden_dim=hidden_dim)
        self._output_keys: list[str] = ["dipole", "charges", "total_charge"]

    @property
    def output_keys(self) -> list[str]:
        return self._output_keys

    def forward(
        self,
        features: NodeFeatures,
        graph: AtomicGraph,
    ) -> dict[str, torch.Tensor]:
        # Per-atom partial charges
        charges: torch.Tensor = self.readout(features.node_feats).squeeze(-1)  # (N,)

        batch: torch.Tensor = (
            graph.batch
            if graph.batch is not None
            else torch.zeros(graph.num_atoms, dtype=torch.long, device=charges.device)  # (N,)
        )

        # Dipole moment: mu = sum_i q_i * r_i
        weighted_pos: torch.Tensor = graph.pos * charges.unsqueeze(-1)  # (N, 3)
        dipole: torch.Tensor = scatter(weighted_pos, batch, dim=0, reduce="sum")  # (B, 3)

        # Total charge per graph (useful for charge neutrality loss)
        total_charge: torch.Tensor = scatter(charges, batch, dim=0, reduce="sum")  # (B,)

        return {
            "dipole": dipole,  # (B, 3)
            "charges": charges,  # (N,)
            "total_charge": total_charge,  # (B,)
            "num_atoms": scatter(torch.ones_like(charges), batch, dim=0, reduce="sum"),  # (B,)
        }
