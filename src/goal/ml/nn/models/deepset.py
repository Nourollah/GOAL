"""DeepSet backbone — edge-based invariant feature extraction.

An invariant backbone that builds per-node features from pairwise
edge descriptors. Inspired by the DeepSet architecture from the
SCAI project, rewritten from scratch to satisfy the GOAL
``InvariantBackbone`` protocol.

Architecture
------------
1. Embed atomic numbers → node features
2. Expand edge distances with Bessel radial basis
3. Project source atoms, target atoms, and distances into a shared
   feature space
4. Concatenate ``[source, target, distance]`` per edge
5. Edge interaction MLP
6. Sum-aggregate edges to nodes → per-node invariant features

Satisfies the ``InvariantBackbone`` protocol — pair with any head
(``energy_forces``, ``direct_forces``, ``stress``, etc.).
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.nn.blocks.embedding import AtomicNumberEmbedding
from goal.ml.nn.primitives.radial import BesselBasis
from goal.ml.registry import MODEL_REGISTRY


class _MLP(nn.Module):
    """Small utility MLP with SiLU activations.

    Avoids pulling in ``torch_geometric.nn.MLP`` — keeps dependencies
    consistent with the rest of GOAL which builds MLPs from raw
    ``nn.Linear`` + ``nn.SiLU`` layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev: int = in_channels
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev, hidden_channels))
            layers.append(nn.SiLU())
            prev = hidden_channels
        layers.append(nn.Linear(prev, out_channels))
        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@MODEL_REGISTRY.register("deepset")
class DeepSet(nn.Module):
    """Edge-based invariant backbone.

    Builds per-node scalar features from edge descriptors by
    projecting source atoms, target atoms, and radial distances
    into a shared space, applying an edge interaction MLP, and
    scatter-aggregating back to nodes.

    Satisfies the ``InvariantBackbone`` protocol.

    Parameters
    ----------
    num_elements : int
        Maximum atomic number supported.
    embedding_dim : int
        Dimension of initial atomic embeddings.
    hidden_channels : int
        Width of hidden layers and output feature dimension.
    num_filters : int
        Size of the projected feature space for atoms and distances.
    num_radial_basis : int
        Number of Bessel radial basis functions.
    transform_depth : int
        Number of layers in the atom/distance projection MLPs.
    cutoff : float
        Cutoff radius for neighbour interactions (Ångström).

    Example
    -------
    >>> backbone = DeepSet(cutoff=5.0)
    >>> features = backbone(graph)  # NodeFeatures with irreps="128x0e"
    """

    def __init__(
        self,
        num_elements: int = 120,
        embedding_dim: int = 128,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_radial_basis: int = 20,
        transform_depth: int = 2,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__()
        self._hidden_dim: int = hidden_channels
        self._num_interactions: int = 1  # single edge-interaction layer

        # --- Embedding / basis expansion ---
        self.embedding: AtomicNumberEmbedding = AtomicNumberEmbedding(
            num_elements=num_elements,
            embedding_dim=embedding_dim,
        )
        self.radial_basis: BesselBasis = BesselBasis(
            num_basis=num_radial_basis,
            cutoff=cutoff,
        )

        # --- Feature projections ---
        self.distance_transform: _MLP = _MLP(
            in_channels=num_radial_basis,
            hidden_channels=hidden_channels,
            out_channels=num_filters,
            num_layers=transform_depth,
        )
        self.source_atom_transform: _MLP = _MLP(
            in_channels=embedding_dim,
            hidden_channels=hidden_channels,
            out_channels=num_filters,
            num_layers=transform_depth,
        )
        self.target_atom_transform: _MLP = _MLP(
            in_channels=embedding_dim,
            hidden_channels=hidden_channels,
            out_channels=num_filters,
            num_layers=transform_depth,
        )

        # --- Edge interaction ---
        self.edge_interaction: _MLP = _MLP(
            in_channels=num_filters * 3,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=transform_depth,
        )

    # ------------------------------------------------------------------
    # InvariantBackbone protocol
    # ------------------------------------------------------------------

    @property
    def hidden_dim(self) -> int:
        """Dimension of the invariant node embedding."""
        return self._hidden_dim

    @property
    def num_interactions(self) -> int:
        """Number of message-passing layers."""
        return self._num_interactions

    @property
    def irreps_out(self) -> typing.Any:
        """Invariant output: all scalar channels."""
        from e3nn.o3 import Irreps

        return Irreps(f"{self._hidden_dim}x0e")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        """Extract per-node invariant features from edge descriptors.

        Args:
            graph: Input ``AtomicGraph`` with topology and features.

        Returns:
            ``NodeFeatures`` with scalar irreps ``"Hx0e"`` where
            ``H = hidden_channels``.
        """
        row: torch.Tensor
        col: torch.Tensor
        row, col = graph.edge_index  # (E,), (E,)

        # Atomic embeddings
        h: torch.Tensor = self.embedding(graph.z)  # (N, embedding_dim)

        # Radial basis expansion of edge lengths
        edge_basis: torch.Tensor = self.radial_basis(
            graph.edge_weight,  # (E,)
        )  # (E, num_radial_basis)

        # Project and concatenate [distance, source, target]
        d_proj: torch.Tensor = self.distance_transform(edge_basis)  # (E, num_filters)
        src_proj: torch.Tensor = self.source_atom_transform(h[row])  # (E, num_filters)
        tgt_proj: torch.Tensor = self.target_atom_transform(h[col])  # (E, num_filters)

        edge_feats: torch.Tensor = torch.cat(
            [d_proj, src_proj, tgt_proj],
            dim=-1,
        )  # (E, num_filters * 3)

        # Edge interaction MLP
        edge_feats = self.edge_interaction(edge_feats)  # (E, hidden_channels)

        # Aggregate edge features to nodes
        node_feats: torch.Tensor = scatter(
            edge_feats,
            row,
            dim=0,
            reduce="sum",
            dim_size=graph.num_atoms,
        )  # (N, hidden_channels)

        return NodeFeatures(
            node_feats=node_feats,  # (N, hidden_channels)
            irreps=f"{self._hidden_dim}x0e",
        )
