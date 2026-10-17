"""Tensor-level neighbour lists for code that already holds positions on a device.

A thin front end to :func:`goal.ml.data.neighbor_list.build_neighbor_list_batch`
— that module is the single implementation.  Backend selection:

* ``"auto"`` (default) — the batched ``nvalchemi-toolkit-ops`` Warp kernel
  when the package and an NVIDIA GPU are present and the tensors are on it;
  otherwise the pure-PyTorch implementation (CUDA, ROCm, MPS, CPU).
* ``"nvalchemi"`` / ``"nvalchemiops"`` — request the kernel; degrades to
  ``"torch"`` with a warning when it cannot run.
* ``"torch"`` / ``"pyg"`` — portable implementation.  (``"pyg"`` is accepted
  for compatibility: PyG's own ``radius_graph`` needs ``pyg-lib`` or
  ``torch-cluster`` and has no periodic boundaries, so it is not used.)

GOAL models do not call this during training: neighbour lists are built
once per structure when a dataset is processed, and the models re-derive
differentiable edge vectors from ``graph.pos`` + ``unit_shifts``.  This is
for loops that move atoms and must rebuild the list every step — batched
MD, GPU-resident batches.
"""

from __future__ import annotations

import torch

from goal.ml.data.neighbor_list import build_neighbor_list_batch

_ALIASES: dict[str, str] = {
    "auto": "auto",
    "nvalchemi": "nvalchemiops",
    "nvalchemiops": "nvalchemiops",
    "torch": "torch",
    "pyg": "torch",
}


def build_neighbor_list(
    pos: torch.Tensor,
    cutoff: float,
    batch: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(edge_index (2, E), edge_vec (E, 3))`` for a batch of structures.

    Parameters
    ----------
    pos : Tensor
        ``(N, 3)`` positions in Å.
    cutoff : float
        Cutoff radius in Å.
    batch : Tensor, optional
        ``(N,)`` structure index per atom (sorted).
    cell : Tensor, optional
        ``(B, 3, 3)`` or ``(3, 3)`` lattice vectors as rows.
    pbc : Tensor, optional
        ``(B, 3)`` or ``(3,)`` periodic flags (required with ``cell``).
    backend : str
        ``"auto"``, ``"nvalchemi"``, ``"torch"`` (see module docstring).

    Returns
    -------
    edge_index : Tensor
        Global indices, ``edge_index[0]`` the centre atom ``i``.
    edge_vec : Tensor
        ``r_j - r_i`` including the periodic shift.
    """
    try:
        resolved = _ALIASES[backend]
    except KeyError as exc:
        raise ValueError(f"backend must be one of {sorted(_ALIASES)}, got {backend!r}.") from exc
    nl = build_neighbor_list_batch(pos, cutoff, batch=batch, cell=cell, pbc=pbc, backend=resolved)
    return nl.edge_index, nl.edge_vectors


__all__ = ["build_neighbor_list", "build_neighbor_list_batch"]
