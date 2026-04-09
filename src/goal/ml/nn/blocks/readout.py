"""Readout blocks — aggregate node features to graph-level predictions."""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import Irreps

from goal.ml.nn.primitives.linear import EquivariantLinear


class ScalarReadout(nn.Module):
    """Reduce equivariant node features to a per-node scalar.

    Extracts the scalar (l=0) channels from the irrep features,
    then applies a small MLP to produce a scalar output per node.

    Parameters
    ----------
    irreps_in : str or Irreps
        Input irreps containing at least scalar channels.
    hidden_dim : int
        Width of the readout MLP hidden layer.
    """

    def __init__(self, irreps_in: str | Irreps, hidden_dim: int = 64) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in)

        # Project to scalars only
        num_scalars = sum(mul for mul, ir in self.irreps_in if ir.l == 0)
        scalar_irreps = Irreps(f"{num_scalars}x0e")
        self.to_scalars = EquivariantLinear(self.irreps_in, scalar_irreps)

        self.mlp = nn.Sequential(
            nn.Linear(num_scalars, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-node scalar, then optionally sum over graphs.

        Parameters
        ----------
        node_feats : Tensor
            Shape ``(N, irreps_in.dim)``.
        batch : Tensor or None
            Graph membership for each node. If provided, sums per graph.

        Returns
        -------
        Tensor
            Per-graph scalar ``(B,)`` if batch is given, else per-node ``(N, 1)``.
        """
        scalars = self.to_scalars(node_feats)  # (N, num_scalars)
        node_out = self.mlp(scalars)  # (N, 1)

        if batch is not None:
            from torch_geometric.utils import scatter

            return scatter(node_out.squeeze(-1), batch, dim=0, reduce="sum")  # (B,)
        return node_out  # (N, 1)
