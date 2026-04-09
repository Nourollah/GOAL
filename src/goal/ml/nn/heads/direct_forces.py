"""Direct forces head — predicts forces without autograd.

Unlike ``EnergyForcesHead`` which derives forces as -dE/dr (energy-conserving),
this head predicts forces directly from node features via a learned linear
projection. This is the standard approach used by models like SchNet, DimeNet,
PaiNN, NequIP (in direct mode), and MACE (in direct mode).

Non-conservative but often more accurate for force-matching on
ground-truth DFT forces.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.nn.blocks.readout import ScalarReadout
from goal.ml.nn.primitives.linear import EquivariantLinear
from goal.ml.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("direct_forces")
class DirectForcesHead(nn.Module):
    """Head that predicts energy and forces independently (non-conservative).

    Energy is computed as sum of per-atom scalars.
    Forces are predicted directly from equivariant vector features (l=1),
    NOT as -dE/dr. This avoids the autograd overhead and can be more
    accurate for force-matching objectives.

    Parameters
    ----------
    irreps_in : str
        Input irreps from the backbone. Must contain l=1 (vector) channels.
    hidden_dim : int
        Width of the scalar readout MLP.
    """

    def __init__(
        self,
        irreps_in: str,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.readout = ScalarReadout(irreps_in=irreps_in, hidden_dim=hidden_dim)

        # Project equivariant features to 3D vectors for forces
        # Output: 1x1o (a single l=1 odd-parity vector = 3 components)
        vector_irreps = Irreps("1x1o")
        self.force_proj: EquivariantLinear = EquivariantLinear(self.irreps_in, vector_irreps)

        self._output_keys: typing.List[str] = ["energy", "forces"]

    @property
    def output_keys(self) -> typing.List[str]:
        return self._output_keys

    def forward(
        self,
        features: NodeFeatures,
        graph: AtomicGraph,
    ) -> typing.Dict[str, torch.Tensor]:
        node_feats: torch.Tensor = features.node_feats                      # (N, irreps_in.dim)

        # Energy: scalar readout
        node_energies: torch.Tensor = self.readout(node_feats).squeeze(-1)  # (N,)
        batch: torch.Tensor = graph.batch if graph.batch is not None else torch.zeros(  # (N,)
            graph.num_atoms, dtype=torch.long, device=node_energies.device
        )
        energy: torch.Tensor = scatter(node_energies, batch, dim=0, reduce="sum")  # (B,)

        # Forces: project equivariant features to l=1 vectors
        forces: torch.Tensor = self.force_proj(node_feats)                  # (N, 3)

        return {
            "energy": energy,                                               # (B,)
            "forces": forces,                                               # (N, 3)
            "num_atoms": scatter(                                           # (B,)
                torch.ones_like(node_energies), batch, dim=0, reduce="sum"
            ),
        }
