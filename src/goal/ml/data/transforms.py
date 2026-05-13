"""Graph construction transforms.

Transforms operate on ``AtomicGraph`` instances, typically building or
updating the neighbour list (edge_index, edge_vectors, edge_lengths).
"""

from __future__ import annotations

import torch

from goal.ml.data.graph import AtomicGraph
from goal.ml.data.neighbor_list import NeighborListResult, build_neighbor_list
from goal.ml.registry import TRANSFORM_REGISTRY


@TRANSFORM_REGISTRY.register("radius_graph")
class RadiusGraphTransform:
    """Rebuild neighbour list with a given cutoff radius.

    Useful when you want to change the cutoff after initial graph
    construction (e.g. during multi-cutoff training).

    Parameters
    ----------
    cutoff : float
        New cutoff radius in Ångströms.
    backend : str
        Neighbour-list backend.  ``"ase"`` (default, correct PBC),
        ``"matscipy"`` (faster, optional dep), or ``"radius_graph"``
        (legacy, no PBC).  See :mod:`goal.ml.data.neighbor_list`.
    """

    def __init__(self, cutoff: float, backend: str = "ase") -> None:
        self.cutoff: float = cutoff
        self.backend: str = backend

    def __call__(self, graph: AtomicGraph) -> AtomicGraph:
        """Reconstruct edges for the given cutoff."""
        from ase import Atoms

        atoms = Atoms(
            numbers=graph.z.cpu().numpy().astype(int),
            positions=graph.pos.cpu().numpy(),
            cell=graph.cell.squeeze(0).cpu().numpy(),
            pbc=graph.pbc.cpu().numpy(),
        )
        nl: NeighborListResult = build_neighbor_list(
            atoms, self.cutoff, backend=self.backend, dtype=graph.pos.dtype
        )

        graph.edge_index = nl.edge_index
        graph.edge_attr = nl.edge_vectors
        graph.edge_weight = nl.edge_lengths
        graph.unit_shifts = nl.unit_shifts
        return graph

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cutoff={self.cutoff}, backend={self.backend!r})"
