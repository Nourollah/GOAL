"""Generic scalar property output head.

Predicts a single named scalar per structure from per-atom features.
Use this as a building block for arbitrary properties (band gap, HOMO,
LUMO, polarisability, etc.) without writing a custom head class for
each one.

The ``property_name`` parameter controls which key appears in the output
dictionary, so the loss system and logging automatically pick it up.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.nn.blocks.readout import ScalarReadout
from goal.ml.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("scalar")
class ScalarHead(nn.Module):
    """Head that predicts a single per-structure scalar property.

    The property is computed by pooling per-atom scalar contributions.
    Use ``reduction="sum"`` for extensive properties (energy, total charge)
    and ``reduction="mean"`` for intensive properties (band gap, HOMO).

    Parameters
    ----------
    irreps_in : str
        Input irreps from the backbone (must contain scalar channels).
    hidden_dim : int
        Width of the readout MLP.
    property_name : str
        Name of the predicted property — becomes the key in the output
        dict and must match the target key on ``AtomicGraph``.
    reduction : str
        Pooling strategy: ``"sum"`` (extensive) or ``"mean"`` (intensive).
    """

    def __init__(
        self,
        irreps_in: str,
        hidden_dim: int = 64,
        property_name: str = "scalar",
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("sum", "mean"):
            raise ValueError(f"reduction must be 'sum' or 'mean', got '{reduction}'")
        self.property_name: str = property_name
        self.reduction: str = reduction
        self.readout: ScalarReadout = ScalarReadout(irreps_in=irreps_in, hidden_dim=hidden_dim)
        self._output_keys: list[str] = [property_name]

    @property
    def output_keys(self) -> list[str]:
        """Keys this head produces."""
        return self._output_keys

    def forward(
        self,
        features: NodeFeatures,
        graph: AtomicGraph,
    ) -> dict[str, torch.Tensor]:
        """Compute the named scalar property.

        Returns
        -------
        dict[str, Tensor]
            ``{self.property_name: (B,)}``
        """
        node_contrib: torch.Tensor = self.readout(features.node_feats).squeeze(-1)  # (N,)

        batch: torch.Tensor = (
            graph.batch
            if graph.batch is not None
            else torch.zeros(graph.num_atoms, dtype=torch.long, device=node_contrib.device)
        )

        value: torch.Tensor = scatter(
            node_contrib, batch, dim=0, reduce=self.reduction,
        )  # (B,)

        return {self.property_name: value}
