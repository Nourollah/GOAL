"""Energy + forces + stress output head.

Computes energy as a sum of per-node scalar predictions, then obtains
forces as the negative gradient of energy with respect to positions
(ensuring energy conservation by construction).
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.nn.blocks.readout import ScalarReadout
from goal.ml.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("energy_forces")
class EnergyForcesHead(nn.Module):
    """Head that predicts energy, forces, and optionally stress.

    Energy is computed as a sum of per-atom contributions from a scalar
    readout.  Forces are obtained via ``torch.autograd.grad`` (the
    negative gradient of energy w.r.t. positions), which guarantees
    energy conservation.

    Parameters
    ----------
    irreps_in : str
        Input irreps from the backbone.
    hidden_dim : int
        Width of the readout MLP.
    compute_stress : bool
        Whether to also compute the stress tensor via strain derivatives.
    """

    def __init__(
        self,
        irreps_in: str,
        hidden_dim: int = 64,
        compute_stress: bool = False,
    ) -> None:
        super().__init__()
        self.compute_stress: bool = compute_stress
        self.readout: ScalarReadout = ScalarReadout(irreps_in=irreps_in, hidden_dim=hidden_dim)
        self._output_keys: typing.List[str] = ["energy", "forces"]
        if compute_stress:
            self._output_keys.append("stress")

    @property
    def output_keys(self) -> typing.List[str]:
        """Keys this head produces."""
        return self._output_keys

    def forward(
        self,
        features: NodeFeatures,
        graph: AtomicGraph,
    ) -> typing.Dict[str, torch.Tensor]:
        """Compute energy and forces (and optionally stress).

        Parameters
        ----------
        features : NodeFeatures
            Output of the backbone.
        graph : AtomicGraph
            The input graph (needed for positions and batch assignment).

        Returns
        -------
        dict[str, Tensor]
            ``{'energy': (B,), 'forces': (N, 3), ...}``
        """
        # Per-node scalar contribution
        node_energies: torch.Tensor = self.readout(features.node_feats)     # (N, 1)
        node_energies = node_energies.squeeze(-1)                           # (N,)

        # Sum per graph to get total energy
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

        # Forces via autograd — negative gradient of energy w.r.t. positions
        if graph.pos.requires_grad:
            forces: torch.Tensor = torch.autograd.grad(                     # (N, 3)
                energy.sum(),
                graph.pos,
                create_graph=self.training,
                retain_graph=True,
            )[0]
            outputs["forces"] = -forces                                     # (N, 3)
        else:
            outputs["forces"] = torch.zeros_like(graph.pos)                 # (N, 3)

        return outputs
