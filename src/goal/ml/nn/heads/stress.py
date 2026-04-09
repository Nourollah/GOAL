"""Stress-only output head.

Computes the stress tensor via the virial route using autograd:
stress = (1/V) * d(total_energy) / d(strain).

Useful when training on periodic systems where stress is an important
target alongside or instead of forces.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.nn.blocks.readout import ScalarReadout
from goal.ml.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("stress")
class StressHead(nn.Module):
    """Head that predicts energy and stress tensor.

    Stress is computed as the derivative of total energy with respect to
    a symmetric strain applied to the cell, following the virial formulation.

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
        self._output_keys: typing.List[str] = ["energy", "stress"]

    @property
    def output_keys(self) -> typing.List[str]:
        return self._output_keys

    def forward(
        self,
        features: NodeFeatures,
        graph: AtomicGraph,
    ) -> typing.Dict[str, torch.Tensor]:
        node_energies: torch.Tensor = self.readout(features.node_feats).squeeze(-1)  # (N,)

        batch: torch.Tensor = graph.batch if graph.batch is not None else torch.zeros(  # (N,)
            graph.num_atoms, dtype=torch.long, device=node_energies.device
        )
        energy: torch.Tensor = scatter(node_energies, batch, dim=0, reduce="sum")  # (B,)

        outputs: typing.Dict[str, torch.Tensor] = {
            "energy": energy,                                               # (B,)
            "num_atoms": scatter(                                           # (B,)
                torch.ones_like(node_energies), batch, dim=0, reduce="sum"
            ),
        }

        # Stress via virial: sigma_ij = (1/V) * sum_a r_a_i * f_a_j
        # Implemented as derivative of energy w.r.t. strain
        if graph.pos.requires_grad:
            forces_neg = torch.autograd.grad(                               # (N, 3)
                energy.sum(),
                graph.pos,
                create_graph=self.training,
                retain_graph=True,
            )[0]

            # Virial contribution from pairwise interactions
            # stress = -(1/V) * sum_edges (r_ij outer f_j)
            row, col = graph.edge_index                                     # (E,), (E,)
            virial = torch.einsum(                                          # (E, 3, 3)
                "ei,ej->eij",
                graph.edge_attr,                                            # (E, 3)
                forces_neg[col],                                            # (E, 3)
            )

            # Sum virial per graph
            num_graphs = energy.shape[0]
            stress = scatter(                                               # (B, 9)
                virial.view(-1, 9),                                         # (E, 9)
                batch[row],
                dim=0,
                reduce="sum",
                dim_size=num_graphs,
            ).view(-1, 3, 3)                                                # (B, 3, 3)

            # Normalise by cell volume
            if graph.cell is not None:
                cell = graph.cell.view(-1, 3, 3)                            # (B, 3, 3)
                volumes = torch.det(cell).abs().clamp(min=1e-10)            # (B,)
                stress = stress / volumes.unsqueeze(-1).unsqueeze(-1)       # (B, 3, 3)

            outputs["stress"] = stress                                      # (B, 3, 3)

        return outputs
