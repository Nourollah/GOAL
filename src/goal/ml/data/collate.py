"""Batching helpers: GOAL ``AtomicGraph`` batches and nvalchemi batches.

* :func:`goal_collate_fn` — the PyG ``Batch`` every GOAL model trains on.
  Works on every device.
* :func:`batch_from_tensors` — a GOAL batch built straight from device
  tensors (positions, atomic numbers, cells), with the neighbour list
  computed on the same device (the ``nvalchemi-toolkit-ops`` kernel on
  NVIDIA GPUs, pure PyTorch elsewhere).  Used by the nvalchemi model
  wrapper and the batched simulation backend, where atoms move every step.
* :func:`nvalchemi_collate_fn` — an ``nvalchemi.data.Batch`` (GPU-resident,
  the format nvalchemi's dynamics engine consumes) when nvalchemi-toolkit
  and an NVIDIA GPU are available; otherwise falls back to
  :func:`goal_collate_fn`.  GOAL models themselves never consume it — it
  is for feeding GOAL datasets into nvalchemi pipelines.
"""

from __future__ import annotations

import logging
import typing
from functools import cache

import torch
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph
from goal.ml.data.neighbor_list import build_neighbor_list_batch

log = logging.getLogger(__name__)


def goal_collate_fn(graphs: typing.Sequence[AtomicGraph]) -> Batch:
    """Collate ``AtomicGraph`` objects into a PyG ``Batch`` (any device)."""
    return Batch.from_data_list(list(graphs))


def batch_from_tensors(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cutoff: float,
    batch: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    neighbor_backend: str = "auto",
) -> Batch:
    """A GOAL ``Batch`` for structures given as concatenated tensors.

    ``positions`` is used as-is (no copy) inside the batch, so a caller that
    sets ``positions.requires_grad_(True)`` can differentiate the model's
    energy with respect to it.

    Parameters
    ----------
    positions : Tensor
        ``(N, 3)`` concatenated positions.
    atomic_numbers : Tensor
        ``(N,)`` atomic numbers.
    cutoff : float
        Neighbour-list cutoff (the model's).
    batch : Tensor, optional
        ``(N,)`` sorted structure index per atom; ``None`` = one structure.
    cell, pbc : Tensor, optional
        ``(B, 3, 3)`` lattice vectors (rows) and ``(B, 3)`` periodic flags.
    neighbor_backend : str
        See :func:`~goal.ml.data.neighbor_list.build_neighbor_list_batch`.
    """
    device = positions.device
    n = positions.shape[0]
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=device)
    batch = batch.to(device=device, dtype=torch.long)
    n_graphs = int(batch.max()) + 1 if n else 0
    if cell is not None and cell.dim() == 2:
        cell = cell.expand(n_graphs, 3, 3)
    if pbc is not None and pbc.dim() == 1:
        pbc = pbc.expand(n_graphs, 3)

    nl = build_neighbor_list_batch(
        positions.detach(), cutoff, batch=batch, cell=cell, pbc=pbc, backend=neighbor_backend
    )
    counts = torch.bincount(batch, minlength=n_graphs)
    atom_ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).tolist()
    # Edges are sorted by centre atom, so each structure's edges are contiguous.
    edge_ptr = torch.searchsorted(
        nl.edge_index[0].contiguous(), torch.tensor(atom_ptr, device=device)
    ).tolist()

    zeros_cell = positions.new_zeros(3, 3)
    no_pbc = torch.zeros(3, dtype=torch.bool, device=device)
    graphs: list[AtomicGraph] = []
    for g in range(n_graphs):
        lo, hi = atom_ptr[g], atom_ptr[g + 1]
        e_lo, e_hi = edge_ptr[g], edge_ptr[g + 1]
        graphs.append(
            AtomicGraph(
                positions=positions[lo:hi],
                atomic_numbers=atomic_numbers[lo:hi].to(device=device, dtype=torch.long),
                cell=cell[g].to(positions.dtype) if cell is not None else zeros_cell,
                pbc=pbc[g].to(torch.bool) if pbc is not None else no_pbc,
                edge_index=nl.edge_index[:, e_lo:e_hi] - lo,
                unit_shifts=nl.unit_shifts[e_lo:e_hi],
            )
        )
    return Batch.from_data_list(graphs)


# ---------------------------------------------------------------------------
# nvalchemi interoperability
# ---------------------------------------------------------------------------


def atomic_graph_to_atomic_data(
    graph: AtomicGraph, device: str | torch.device | None = None
) -> typing.Any:
    """One GOAL ``AtomicGraph`` → ``nvalchemi.data.AtomicData``.

    Labels are carried over in nvalchemi's shapes (``energy`` ``[1, 1]``,
    ``forces`` ``[N, 3]``, ``stress`` ``[1, 3, 3]``); ``energy`` / ``forces``
    buffers are always allocated because nvalchemi 0.2 silently drops model
    outputs that have no preallocated buffer.
    """
    from nvalchemi.data import AtomicData

    device = torch.device(device) if device is not None else graph.pos.device
    pos = graph.pos.to(device)
    n = pos.shape[0]
    fields: dict[str, typing.Any] = {
        "positions": pos,
        "atomic_numbers": graph.z.to(device=device, dtype=torch.long),
        "forces": (
            graph.get("forces") if graph.get("forces") is not None else torch.zeros(n, 3)
        ).to(device=device, dtype=pos.dtype),
        "energy": (
            graph.get("energy").reshape(1, 1)
            if graph.get("energy") is not None
            else torch.zeros(1, 1)
        ).to(device=device, dtype=pos.dtype),
    }
    pbc = graph.get("pbc")
    cell = graph.get("cell")
    if pbc is not None and bool(torch.as_tensor(pbc).any()) and cell is not None:
        fields["cell"] = cell.reshape(1, 3, 3).to(device=device, dtype=pos.dtype)
        fields["pbc"] = torch.as_tensor(pbc, device=device).reshape(1, 3)
    stress = graph.get("stress")
    if stress is not None:
        fields["stress"] = stress.reshape(1, 3, 3).to(device=device, dtype=pos.dtype)
    return AtomicData(**fields)


@cache
def _log_collate_backend(backend: str) -> None:
    log.info("GOAL collate backend: %s", backend)


def nvalchemi_collate_fn(
    graphs: typing.Sequence[AtomicGraph], device: str | torch.device = "cuda"
) -> typing.Any:
    """Collate into a GPU-resident ``nvalchemi.data.Batch``.

    Falls back to :func:`goal_collate_fn` (a PyG batch) when
    nvalchemi-toolkit or an NVIDIA GPU is unavailable.
    """
    from goal.ml.utils.hardware import nvalchemi_available

    if not nvalchemi_available():
        _log_collate_backend("pyg (nvalchemi unavailable)")
        return goal_collate_fn(graphs)
    from nvalchemi.data import Batch as NvBatch

    _log_collate_backend("nvalchemi")
    return NvBatch.from_data_list(
        [atomic_graph_to_atomic_data(g, device) for g in graphs], device=torch.device(device)
    )
