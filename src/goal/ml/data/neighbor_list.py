"""Backend-agnostic neighbour-list construction for periodic and non-periodic systems.

Provides a single entry point ``build_neighbor_list`` that dispatches to
one of several implementations:

* ``"ase"``            — ``ase.neighborlist.neighbor_list`` (default).
                         Correct PBC via minimum-image convention, zero extra
                         dependencies (ASE is already a core GOAL dependency).
* ``"matscipy"``       — ``matscipy.neighbours.neighbour_list``.
                         10–100× faster than ASE for large cells; optional dep.
                         Falls back to ``"ase"`` with a warning if not installed.
* ``"nvalchemiops"``   — ``nvalchemiops.torch.neighbors.neighbor_list``.
                         GPU-accelerated O(N) cell-list kernels via NVIDIA Warp.
                         Requires CUDA and ``nvalchemi-toolkit-ops[torch]``.
                         Falls back to ``"ase"`` if CUDA is unavailable or the
                         package is not installed.  Best suited for large periodic
                         structures (≥1 000 atoms) in on-the-fly MD/inference
                         loops where neighbour lists are rebuilt every step.
* ``"radius_graph"``   — Legacy ``torch_geometric.nn.radius_graph`` path.
                         No PBC support.  Kept for backward compatibility with
                         non-periodic structures and pre-existing datasets.

The key correctness fix over the old ``radius_graph`` approach is that the
ASE, matscipy, and nvalchemiops backends find neighbours using the
*minimum-image* distance, so cross-boundary pairs in periodic cells are never
missed.

Typical usage::

    from goal.ml.data.neighbor_list import build_neighbor_list

    nl = build_neighbor_list(atoms, cutoff=5.0, backend="ase")
    # nl.edge_index  — (2, E) int64
    # nl.edge_vectors — (E, 3) float64  (already MIC-corrected)
    # nl.edge_lengths — (E,)   float64
    # nl.unit_shifts  — (E, 3) int64   integer cell-shift vectors
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch


@dataclass
class NeighborListResult:
    """Output of a neighbour-list computation.

    Attributes
    ----------
    edge_index : Tensor
        Source/destination atom indices, shape ``(2, E)`` int64.
    edge_vectors : Tensor
        Displacement vectors ``r_j − r_i`` in Cartesian coordinates,
        shape ``(E, 3)`` float64.  Already minimum-image-corrected for
        periodic backends.
    edge_lengths : Tensor
        Euclidean lengths ``‖r_j − r_i‖``, shape ``(E,)`` float64.
    unit_shifts : Tensor
        Integer cell-shift vectors ``S`` such that
        ``r_j − r_i = D + S @ cell``, shape ``(E, 3)`` int64.
        Required by MACE and other MLIPs that use equivariant message
        passing under PBC.  Zero for non-periodic systems.
    """

    edge_index: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    unit_shifts: torch.Tensor


def build_neighbor_list(
    atoms,  # ase.Atoms — typed loosely to avoid mandatory top-level import
    cutoff: float,
    backend: str = "ase",
    dtype: torch.dtype = torch.float64,
) -> NeighborListResult:
    """Build a neighbour list for an ASE ``Atoms`` object.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure to compute neighbours for.
    cutoff : float
        Neighbour cutoff radius in Ångströms.
    backend : str
        Neighbour-list backend.  One of ``"ase"``, ``"matscipy"``, or
        ``"radius_graph"``.  See module docstring for details.
    dtype : torch.dtype
        Floating-point dtype for returned tensors.

    Returns
    -------
    NeighborListResult
        Edge index, displacement vectors, lengths, and integer cell shifts.
    """
    if backend == "matscipy":
        return _matscipy_neighbor_list(atoms, cutoff, dtype)
    elif backend == "nvalchemiops":
        return _nvalchemiops_neighbor_list(atoms, cutoff, dtype)
    elif backend == "radius_graph":
        return _radius_graph_neighbor_list(atoms, cutoff, dtype)
    else:
        # Default: "ase" — also the fallback for unknown values
        if backend != "ase":
            warnings.warn(
                f"Unknown neighbor_list backend '{backend}'; falling back to 'ase'.",
                stacklevel=2,
            )
        return _ase_neighbor_list(atoms, cutoff, dtype)


def build_neighbor_list_from_tensors(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    backend: str = "ase",
    dtype: torch.dtype = torch.float64,
) -> NeighborListResult:
    """Build a neighbour list from raw tensors (no pre-existing ``Atoms``).

    Constructs a temporary ``ase.Atoms`` object from the supplied
    tensors and delegates to ``build_neighbor_list``.

    Parameters
    ----------
    positions : Tensor
        Cartesian positions, shape ``(N, 3)``.
    atomic_numbers : Tensor
        Integer atomic numbers, shape ``(N,)``.
    cell : Tensor
        Unit-cell matrix, shape ``(3, 3)``.  Ignored for non-periodic dims.
    pbc : Tensor
        Periodic-boundary flags, shape ``(3,)`` bool.
    cutoff : float
        Neighbour cutoff radius in Ångströms.
    backend : str
        Neighbour-list backend (see ``build_neighbor_list``).
    dtype : torch.dtype
        Floating-point dtype for returned tensors.

    Returns
    -------
    NeighborListResult
    """
    import numpy as np
    from ase import Atoms

    atoms = Atoms(
        numbers=atomic_numbers.cpu().numpy().astype(int),
        positions=positions.cpu().numpy(),
        cell=cell.cpu().numpy(),
        pbc=pbc.cpu().numpy(),
    )
    return build_neighbor_list(atoms, cutoff, backend=backend, dtype=dtype)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _ase_neighbor_list(
    atoms,
    cutoff: float,
    dtype: torch.dtype,
) -> NeighborListResult:
    """ASE-based neighbour list — correct PBC, no extra dependencies."""
    from ase.neighborlist import neighbor_list as ase_nl

    i, j, d, D, S = ase_nl("ijdDS", atoms, cutoff)

    edge_index = torch.stack(
        [
            torch.tensor(i, dtype=torch.long),
            torch.tensor(j, dtype=torch.long),
        ]
    )
    edge_vectors = torch.tensor(D, dtype=dtype)
    edge_lengths = torch.tensor(d, dtype=dtype)
    unit_shifts = torch.tensor(S, dtype=torch.long)

    return NeighborListResult(
        edge_index=edge_index,
        edge_vectors=edge_vectors,
        edge_lengths=edge_lengths,
        unit_shifts=unit_shifts,
    )


def _matscipy_neighbor_list(
    atoms,
    cutoff: float,
    dtype: torch.dtype,
) -> NeighborListResult:
    """matscipy-based neighbour list — faster, optional dependency."""
    try:
        from matscipy.neighbours import neighbour_list as msc_nl
    except ImportError:
        warnings.warn(
            "matscipy is not installed; falling back to ASE neighbor list. "
            "Install it with: pip install matscipy",
            stacklevel=3,
        )
        return _ase_neighbor_list(atoms, cutoff, dtype)

    i, j, d, D, S = msc_nl("ijdDS", atoms, cutoff)

    edge_index = torch.stack(
        [
            torch.tensor(i, dtype=torch.long),
            torch.tensor(j, dtype=torch.long),
        ]
    )
    edge_vectors = torch.tensor(D, dtype=dtype)
    edge_lengths = torch.tensor(d, dtype=dtype)
    unit_shifts = torch.tensor(S, dtype=torch.long)

    return NeighborListResult(
        edge_index=edge_index,
        edge_vectors=edge_vectors,
        edge_lengths=edge_lengths,
        unit_shifts=unit_shifts,
    )


def _nvalchemiops_neighbor_list(
    atoms,
    cutoff: float,
    dtype: torch.dtype,
) -> NeighborListResult:
    """NVIDIA nvalchemi-toolkit-ops GPU backend — O(N) cell list on CUDA.

    Requires ``nvalchemi-toolkit-ops[torch]`` and a CUDA-capable GPU.
    Falls back to the ASE backend with a warning when either is absent.

    The high-level ``nvalchemiops.torch.neighbors.neighbor_list`` entry
    point is used with ``return_neighbor_list=True`` to get COO-format output
    directly compatible with GOAL's ``edge_index`` convention.
    """
    try:
        from nvalchemiops.torch.neighbors import neighbor_list as nval_nl
    except ImportError:
        warnings.warn(
            "nvalchemi-toolkit-ops is not installed; falling back to ASE neighbor "
            "list.  Install it with: pip install 'nvalchemi-toolkit-ops[torch]'",
            stacklevel=3,
        )
        return _ase_neighbor_list(atoms, cutoff, dtype)

    if not torch.cuda.is_available():
        warnings.warn(
            "nvalchemiops neighbor list requires CUDA but no GPU was found; "
            "falling back to ASE neighbor list.",
            stacklevel=3,
        )
        return _ase_neighbor_list(atoms, cutoff, dtype)

    import numpy as np

    cuda = torch.device("cuda")
    positions_cuda = torch.tensor(atoms.positions, dtype=dtype, device=cuda)
    cell_np = np.array(atoms.cell)
    pbc_np = np.array(atoms.pbc, dtype=bool)
    has_pbc = bool(pbc_np.any())

    if has_pbc:
        # nvalchemiops high-level API accepts (3, 3) cell and (3,) pbc
        cell_cuda = torch.tensor(cell_np, dtype=dtype, device=cuda)
        pbc_cuda = torch.tensor(pbc_np, dtype=torch.bool, device=cuda)
        # Returns (neighbor_list, neighbor_ptr, unit_shifts) for PBC + list fmt
        nl_out, _ptr, shifts_out = nval_nl(
            positions_cuda,
            cutoff,
            cell=cell_cuda,
            pbc=pbc_cuda,
            return_neighbor_list=True,
        )
        # nl_out: (2, E) int32 COOP  →  int64 on CPU
        edge_index = nl_out.to(torch.long).cpu()
        # shifts_out: (E, 3) int32  →  int64 on CPU
        unit_shifts = shifts_out.to(torch.long).cpu()
    else:
        # No PBC: (neighbor_list, neighbor_ptr)
        nl_out, _ptr = nval_nl(
            positions_cuda,
            cutoff,
            return_neighbor_list=True,
        )
        edge_index = nl_out.to(torch.long).cpu()
        unit_shifts = torch.zeros(edge_index.shape[1], 3, dtype=torch.long)

    # Compute edge displacement vectors on CPU.
    # D = positions[col] - positions[row] + unit_shifts @ cell
    # This matches the ASE convention exactly.
    positions_cpu = torch.tensor(atoms.positions, dtype=dtype)
    cell_cpu = torch.tensor(cell_np, dtype=dtype)
    row, col = edge_index
    edge_vectors = positions_cpu[col] - positions_cpu[row] + unit_shifts.to(dtype) @ cell_cpu
    edge_lengths = edge_vectors.norm(dim=-1)

    return NeighborListResult(
        edge_index=edge_index,
        edge_vectors=edge_vectors,
        edge_lengths=edge_lengths,
        unit_shifts=unit_shifts,
    )


def _radius_graph_neighbor_list(
    atoms,
    cutoff: float,
    dtype: torch.dtype,
) -> NeighborListResult:
    """Legacy torch_geometric ``radius_graph`` backend — no PBC support."""
    from torch_geometric.nn import radius_graph

    positions = torch.tensor(atoms.positions, dtype=dtype)
    edge_index = radius_graph(positions, r=cutoff, loop=False)
    row, col = edge_index
    edge_vectors = positions[col] - positions[row]
    edge_lengths = edge_vectors.norm(dim=-1)
    unit_shifts = torch.zeros(edge_index.shape[1], 3, dtype=torch.long)

    return NeighborListResult(
        edge_index=edge_index,
        edge_vectors=edge_vectors,
        edge_lengths=edge_lengths,
        unit_shifts=unit_shifts,
    )
