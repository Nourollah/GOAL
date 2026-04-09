"""Energy-only output head.

Computes energy as a sum of per-node scalar predictions.
No forces or stress — useful for datasets that only have energy labels,
or as a building block in multi-head setups.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.nn.blocks.readout import ScalarReadout
from goal.ml.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("energy")
class EnergyHead(nn.Module):
    """Head that predicts only total energy.

    Energy is computed as a sum of per-atom contributions from a scalar
    readout MLP.

    Parameters
    ----------
    irreps_in : str
        Input irreps from the backbone.
    hidden_dim : int
        Width of the readout MLP.
    """

    def __init__(
        self,
        irreps_in: str,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.readout: ScalarReadout = ScalarReadout(irreps_in=irreps_in, hidden_dim=hidden_dim)
        self._output_keys: list[str] = ["energy"]

    @property
    def output_keys(self) -> list[str]:
        return self._output_keys

    def forward(
        self,
        features: NodeFeatures,
        graph: AtomicGraph,
    ) -> dict[str, torch.Tensor]:
        node_energies: torch.Tensor = self.readout(features.node_feats).squeeze(-1)  # (N,)

        batch: torch.Tensor = (
            graph.batch
            if graph.batch is not None
            else torch.zeros(  # (N,)
                graph.num_atoms, dtype=torch.long, device=node_energies.device
            )
        )
        energy: torch.Tensor = scatter(node_energies, batch, dim=0, reduce="sum")  # (B,)

        return {
            "energy": energy,  # (B,)
            "num_atoms": scatter(  # (B,)
                torch.ones_like(node_energies), batch, dim=0, reduce="sum"
            ),
        }
