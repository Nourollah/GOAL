"""Equivariant message-passing interaction blocks."""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, spherical_harmonics

from gmd.nn.primitives.linear import EquivariantLinear
from gmd.nn.primitives.norm import EquivariantLayerNorm
from gmd.nn.primitives.radial import BesselBasis, PolynomialEnvelope, RadialMLP
from gmd.nn.primitives.tp import WeightedTensorProduct


class EquivariantInteractionBlock(nn.Module):
    """One layer of equivariant message passing.

    Messages are constructed via a tensor product of sender node features
    with spherical harmonics of edge vectors, weighted by a radial MLP
    that operates only on scalar edge lengths.

    Parameters
    ----------
    irreps_node : str or Irreps
        Irreps of input and output node features.
    irreps_edge : str or Irreps
        Irreps of edge spherical harmonics (e.g. ``'1x0e + 1x1o + 1x2e'``).
    num_basis : int
        Number of radial basis functions.
    cutoff : float
        Cutoff radius for the radial basis and envelope.
    hidden_dim : int
        Width of the radial MLP hidden layers.
    """

    def __init__(
        self,
        irreps_node: typing.Union[str, Irreps],
        irreps_edge: typing.Union[str, Irreps],
        num_basis: int = 8,
        cutoff: float = 5.0,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.irreps_node = Irreps(irreps_node)
        self.irreps_edge = Irreps(irreps_edge)

        # Radial basis and envelope
        self.radial_basis = BesselBasis(num_basis=num_basis, cutoff=cutoff)
        self.envelope = PolynomialEnvelope(cutoff=cutoff)

        # Tensor product: node_feats ⊗ edge_sh → message
        self.tp = WeightedTensorProduct(
            irreps_in1=self.irreps_node,
            irreps_in2=self.irreps_edge,
            irreps_out=self.irreps_node,
        )

        # Radial MLP produces weights for the tensor product
        self.radial_mlp = RadialMLP(
            num_basis=num_basis,
            hidden_dim=hidden_dim,
            num_out=self.tp.weight_numel,
        )

        # Post-message linear + normalisation
        self.linear = EquivariantLinear(self.irreps_node, self.irreps_node)
        self.norm = EquivariantLayerNorm(self.irreps_node)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one round of equivariant message passing.

        Parameters
        ----------
        node_feats : Tensor
            Node features, shape ``(N, irreps_node.dim)``.
        edge_index : Tensor
            Edge indices, shape ``(2, E)``.
        edge_vectors : Tensor
            Edge displacement vectors, shape ``(E, 3)``.
        edge_lengths : Tensor
            Edge lengths, shape ``(E,)``.

        Returns
        -------
        Tensor
            Updated node features with residual connection, shape ``(N, irreps_node.dim)``.
        """
        row, col = edge_index                                           # (E,), (E,)

        # Spherical harmonics of edge directions
        edge_sh = spherical_harmonics(                                      # (E, irreps_edge.dim)
            self.irreps_edge,
            edge_vectors,                                                   # (E, 3)
            normalize=True,
            normalization="component",
        )

        # Radial weights
        rbf = self.radial_basis(edge_lengths)                               # (E, num_basis)
        env = self.envelope(edge_lengths).unsqueeze(-1)                     # (E, 1)
        tp_weights = self.radial_mlp(rbf) * env                             # (E, weight_numel)

        # Messages via tensor product
        sender_feats = node_feats[row]                                      # (E, irreps_node.dim)
        messages = self.tp(sender_feats, edge_sh, tp_weights)               # (E, irreps_node.dim)

        # Aggregate messages (sum over neighbours)
        agg = torch.zeros_like(node_feats)                                  # (N, irreps_node.dim)
        agg.index_add_(0, col, messages)                                    # (N, irreps_node.dim)

        # Post-process with linear + norm + residual
        out = self.linear(agg)                                              # (N, irreps_node.dim)
        out = self.norm(out)                                                # (N, irreps_node.dim)
        return node_feats + out                                             # (N, irreps_node.dim)
