"""Differentiable edge-geometry recomputation.

Neighbour lists shipped with :class:`goal.ml.data.graph.AtomicGraph` are
materialised once (typically inside ``from_ase``).  The resulting
``edge_attr`` and ``edge_weight`` tensors are concrete data, *not*
connected to ``graph.pos`` in the autograd graph.  For force prediction
via ``torch.autograd.grad(E, positions)`` the model must therefore
recompute edge vectors and lengths *from* ``graph.pos`` inside the
forward pass, so that the dependency exists.

Helper preserves PBC correctness using the integer ``unit_shifts``
stored on the graph (when present).  The closed-form rule is::

    r_ij = positions[col] - positions[row] + S_ij @ cell_of_graph(row)

where ``S_ij`` is the integer cell-shift vector and ``cell_of_graph``
selects the unit cell of the graph that owns atom ``row``.  For
non-periodic systems ``S_ij`` is zero and the shift term vanishes.
"""

from __future__ import annotations

import torch

from goal.ml.data.graph import AtomicGraph


def differentiable_edges(
    graph: AtomicGraph,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(edge_vectors, edge_lengths)`` derived from ``positions``.

    Parameters
    ----------
    graph : AtomicGraph
        Graph providing ``edge_index``, ``unit_shifts``, ``cell``, and
        (optional) ``batch`` membership.
    positions : Tensor
        ``(N, 3)`` tensor of atomic positions, typically ``graph.pos``
        with ``requires_grad=True`` set by the caller.

    Returns
    -------
    edge_vectors : Tensor
        ``(E, 3)`` differentiable edge displacements.
    edge_lengths : Tensor
        ``(E,)`` differentiable edge lengths.
    """
    row: torch.Tensor
    col: torch.Tensor
    row, col = graph.edge_index  # (E,), (E,)

    edge_vectors: torch.Tensor = positions[col] - positions[row]  # (E, 3)

    unit_shifts: torch.Tensor | None = graph.get("unit_shifts", None)
    cell: torch.Tensor | None = graph.get("cell", None)

    if unit_shifts is not None and cell is not None and unit_shifts.abs().sum() > 0:
        cell = cell.to(positions.dtype)
        if cell.dim() == 2:
            # Single graph: (3, 3)
            shift: torch.Tensor = unit_shifts.to(positions.dtype) @ cell
        else:
            # Batched: (B, 3, 3).  Look up cell of the graph each edge
            # belongs to via ``batch[row]``.
            batch: torch.Tensor | None = graph.get("batch", None)
            if batch is None:
                cell_per_edge: torch.Tensor = cell[0]
                shift = unit_shifts.to(positions.dtype) @ cell_per_edge
            else:
                cell_per_edge = cell[batch[row]]  # (E, 3, 3)
                shift = torch.einsum(
                    "ej,ejk->ek",
                    unit_shifts.to(positions.dtype),
                    cell_per_edge,
                )
        edge_vectors = edge_vectors + shift

    edge_lengths: torch.Tensor = edge_vectors.norm(dim=-1)  # (E,)
    return edge_vectors, edge_lengths
