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
        """Predict the total energy of each graph.

        Per-atom energies are taken from ``features.node_energies`` when
        the backbone provides them (e.g. SIMURGH); otherwise they come
        from this head's scalar readout of ``features.node_feats``.  The
        total energy is their per-graph sum.  No forces or stress.

        Parameters
        ----------
        features : NodeFeatures
            Backbone output; reads ``node_energies`` ``(N,)`` if set,
            else ``node_feats`` ``(N, irreps_in.dim)``.
        graph : AtomicGraph
            Input graph providing ``batch`` ``(N,)`` (all atoms are
            assigned to graph 0 when ``batch`` is ``None``).

        Returns
        -------
        dict[str, Tensor]
            ``'energy'`` ``(B,)`` in eV and ``'num_atoms'`` ``(B,)``
            (float atom counts per graph).
        """
        # Prefer pre-computed per-atom energies (e.g. from SIMURGH) when the
        # backbone supplies them; otherwise apply our own scalar readout.
        if features.node_energies is not None:
            node_energies: torch.Tensor = features.node_energies  # (N,)
        else:
            node_energies = self.readout(features.node_feats).squeeze(-1)  # (N,)

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
