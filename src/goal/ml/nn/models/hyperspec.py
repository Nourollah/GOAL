"""HyperSpec model — the native GOAL equivariant architecture.

This is a placeholder implementation that establishes the interface.
The full architecture will be developed iteratively.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import Irreps

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.nn.blocks.embedding import AtomicNumberEmbedding
from goal.ml.nn.blocks.interaction import EquivariantInteractionBlock
from goal.ml.nn.primitives.linear import EquivariantLinear
from goal.ml.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("hyperspec")
class HyperSpecModel(nn.Module):
    """Native GOAL equivariant graph neural network.

    Satisfies the ``EquivariantBackbone`` protocol: takes an ``AtomicGraph``
    and returns ``NodeFeatures``.

    Parameters
    ----------
    num_elements : int
        Maximum atomic number supported.
    embedding_dim : int
        Dimension of initial atomic embeddings.
    hidden_channels : int
        Number of scalar channels in hidden irreps.
    num_interactions : int
        Number of message-passing layers.
    lmax : int
        Maximum angular momentum order for spherical harmonics.
    cutoff : float
        Cutoff radius for neighbour interactions (Angstrom).
    num_radial_basis : int
        Number of radial basis functions.
    """

    def __init__(
        self,
        num_elements: int = 120,
        embedding_dim: int = 64,
        hidden_channels: int = 128,
        num_interactions: int = 3,
        lmax: int = 2,
        cutoff: float = 5.0,
        num_radial_basis: int = 8,
    ) -> None:
        super().__init__()
        self._num_interactions: int = num_interactions
        self._cutoff: float = cutoff

        # Build irreps for hidden features: hidden_channels scalars + vectors + ...
        irreps_parts: typing.List[str] = []
        for l_val in range(lmax + 1):
            parity: str = "e" if l_val % 2 == 0 else "o"
            irreps_parts.append(f"{hidden_channels}x{l_val}{parity}")
        self._irreps_hidden: Irreps = Irreps("+".join(irreps_parts))

        # Edge spherical harmonics irreps
        sh_parts: typing.List[str] = [f"1x{l_val}{'e' if l_val % 2 == 0 else 'o'}" for l_val in range(lmax + 1)]
        self._irreps_edge: Irreps = Irreps("+".join(sh_parts))

        # Atomic embedding → initial node features (scalar only → full irreps)
        self.embedding = AtomicNumberEmbedding(
            num_elements=num_elements,
            embedding_dim=embedding_dim,
        )
        scalar_irreps: Irreps = Irreps(f"{embedding_dim}x0e")
        self.input_linear: EquivariantLinear = EquivariantLinear(scalar_irreps, self._irreps_hidden)

        # Interaction layers
        self.interactions = nn.ModuleList([
            EquivariantInteractionBlock(
                irreps_node=self._irreps_hidden,
                irreps_edge=self._irreps_edge,
                num_basis=num_radial_basis,
                cutoff=cutoff,
            )
            for _ in range(num_interactions)
        ])

    @property
    def irreps_out(self) -> Irreps:
        """Output irreducible representations."""
        return self._irreps_hidden

    @property
    def num_interactions(self) -> int:
        """Number of message-passing layers."""
        return self._num_interactions

    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        """Run the full equivariant forward pass.

        Parameters
        ----------
        graph : AtomicGraph
            Input atomic graph with topology and features.

        Returns
        -------
        NodeFeatures
            Equivariant node features after all interaction layers.
        """
        # Initial embedding (scalar features only)
        h: torch.Tensor = self.embedding(graph.z)                           # (N, embedding_dim)
        h = self.input_linear(h)                                            # (N, irreps_hidden.dim)

        # Message passing
        for interaction in self.interactions:
            h = interaction(                                                 # (N, irreps_hidden.dim)
                node_feats=h,
                edge_index=graph.edge_index,                                # (2, E)
                edge_vectors=graph.edge_attr,                               # (E, 3)
                edge_lengths=graph.edge_weight,                             # (E,)
            )

        return NodeFeatures(
            node_feats=h,                                                   # (N, irreps_hidden.dim)
            irreps=str(self._irreps_hidden),
        )
