"""Graph construction transforms.

Transforms operate on ``AtomicGraph`` instances, typically building or
updating the neighbour list (edge_index, edge_vectors, edge_lengths).
"""

from __future__ import annotations

import torch
from torch_geometric.nn import radius_graph

from goal.ml.data.graph import AtomicGraph, _apply_mic
from goal.ml.registry import TRANSFORM_REGISTRY


@TRANSFORM_REGISTRY.register("radius_graph")
class RadiusGraphTransform:
    """Rebuild neighbour list with a given cutoff radius.

    Useful when you want to change the cutoff after initial graph
    construction (e.g. during multi-cutoff training).
    """

    def __init__(self, cutoff: float) -> None:
        self.cutoff: float = cutoff

    def __call__(self, graph: AtomicGraph) -> AtomicGraph:
        """Reconstruct edges for the given cutoff."""
        edge_index: torch.Tensor = radius_graph(graph.pos, r=self.cutoff, loop=False)
        row: torch.Tensor
        col: torch.Tensor
        row, col = edge_index
        edge_vectors: torch.Tensor = graph.pos[col] - graph.pos[row]

        if graph.pbc.any():
            cell: torch.Tensor = graph.cell.squeeze(0)  # (1, 3, 3) → (3, 3)
            edge_vectors = _apply_mic(edge_vectors, cell, graph.pbc)

        edge_lengths: torch.Tensor = edge_vectors.norm(dim=-1)

        # Keep everything except topology
        graph.edge_index = edge_index
        graph.edge_attr = edge_vectors
        graph.edge_weight = edge_lengths
        return graph

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cutoff={self.cutoff})"
