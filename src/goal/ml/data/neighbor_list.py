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
* ``"torch"``          — Pure PyTorch (dense distances, periodic images by
                         explicit cell shifts).  No extra dependency and runs on
                         every device PyTorch supports (CUDA, ROCm, MPS, CPU).
                         The portable fallback of the GPU path.
* ``"radius_graph"``   — Legacy ``torch_geometric.nn.radius_graph`` path.
                         No PBC support.  Kept for backward compatibility with
                         non-periodic structures and pre-existing datasets.
                         Needs ``pyg-lib`` or ``torch-cluster``; without them it
                         falls back to ``"torch"``.
* ``"auto"``           — ``"nvalchemiops"`` for large structures (≥
                         :data:`AUTO_GPU_MIN_ATOMS` atoms) when the package and an
                         NVIDIA GPU are present; otherwise ``"matscipy"`` when it
                         is installed, else ``"ase"``.

For many structures at once, as tensors already on a device (batched MD,
GPU-resident batches), use :func:`build_neighbor_list_batch`: one
``nvalchemiops`` kernel launch for the whole batch on NVIDIA hardware, the
``"torch"`` implementation everywhere else.

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

import logging
import math
import warnings
from dataclasses import dataclass
from functools import cache

import torch

log = logging.getLogger(__name__)

#: ``backend="auto"`` hands structures at least this large to the GPU
#: kernels; below it the host↔device round trip costs more than it saves.
AUTO_GPU_MIN_ATOMS: int = 1000


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
        Neighbour-list backend: ``"ase"``, ``"matscipy"``, ``"nvalchemiops"``,
        ``"torch"``, ``"radius_graph"`` or ``"auto"``.  See module docstring.
    dtype : torch.dtype
        Floating-point dtype for returned tensors.

    Returns
    -------
    NeighborListResult
        Edge index, displacement vectors, lengths, and integer cell shifts.
    """
    if backend == "auto":
        backend = _auto_backend(len(atoms))
    if backend == "torch":
        return _torch_neighbor_list_atoms(atoms, cutoff, dtype)
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

    from goal.ml.utils.hardware import detect_compute_backend

    if detect_compute_backend() != "nvidia":  # ROCm also reports torch.cuda as available
        warnings.warn(
            "nvalchemiops neighbor list requires an NVIDIA CUDA GPU; "
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
    try:
        edge_index = radius_graph(positions, r=cutoff, loop=False)
    except ImportError as exc:  # PyG >= 2.7 needs pyg-lib or torch-cluster for this
        warnings.warn(
            f"radius_graph is unavailable ({exc}); using the pure-PyTorch backend.",
            stacklevel=3,
        )
        return _torch_neighbor_list_atoms(atoms, cutoff, dtype)
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


# ---------------------------------------------------------------------------
# "auto" resolution
# ---------------------------------------------------------------------------


@cache
def _auto_backend_for(large: bool) -> str:
    """Backend ``"auto"`` resolves to, for small or large structures (cached)."""
    import importlib.util

    from goal.ml.utils.hardware import nvalchemiops_available

    if large and nvalchemiops_available():
        chosen = "nvalchemiops"
    elif importlib.util.find_spec("matscipy") is not None:
        chosen = "matscipy"
    else:
        chosen = "ase"
    log.info(
        "GOAL neighbour-list backend (auto, %s structures): %s",
        f">= {AUTO_GPU_MIN_ATOMS} atoms" if large else f"< {AUTO_GPU_MIN_ATOMS} atoms",
        chosen,
    )
    return chosen


def _auto_backend(n_atoms: int) -> str:
    return _auto_backend_for(n_atoms >= AUTO_GPU_MIN_ATOMS)


# ---------------------------------------------------------------------------
# Pure-PyTorch backend (portable: CUDA, ROCm, MPS, CPU)
# ---------------------------------------------------------------------------


def _image_shifts(cell: torch.Tensor, pbc: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Integer cell shifts that can hold a neighbour within ``cutoff``.

    For wrapped positions (fractional coordinates in ``[0, 1)``) along a
    periodic direction, ``ceil(cutoff / h)`` images on each side suffice,
    where ``h`` is the distance between the two cell faces spanned by the
    other lattice vectors.
    """
    ranges: list[torch.Tensor] = []
    volume = torch.det(cell).abs()
    for axis in range(3):
        if bool(pbc[axis]):
            other = [k for k in range(3) if k != axis]
            area = torch.linalg.cross(cell[other[0]], cell[other[1]]).norm()
            height = float(volume / area)
            n = math.ceil(cutoff / height)
            ranges.append(torch.arange(-n, n + 1, device=cell.device))
        else:
            ranges.append(torch.zeros(1, dtype=torch.long, device=cell.device))
    grid = torch.cartesian_prod(*ranges)
    return grid.reshape(-1, 3).long()


def _torch_neighbor_list_single(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
    shift_chunk: int = 27,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(edge_index (2, E), unit_shifts (E, 3))`` for one structure, ASE convention.

    ``r_j - r_i + S @ cell`` lies within ``cutoff`` for every edge
    ``(i, j, S)``; self-pairs are excluded only for the zero shift (a
    periodic image of the atom itself is a neighbour, as in ASE).
    """
    device = positions.device
    n = positions.shape[0]
    periodic = pbc is not None and cell is not None and bool(pbc.any())
    if not periodic:
        dist = torch.cdist(positions, positions)
        mask = dist <= cutoff
        mask.fill_diagonal_(False)
        i, j = mask.nonzero(as_tuple=True)
        return torch.stack([i, j]), torch.zeros(i.shape[0], 3, dtype=torch.long, device=device)

    cell = cell.to(positions.dtype)
    pbc = pbc.to(torch.bool)
    # Wrap periodic directions into the cell; remember the wrap so the
    # returned shifts refer to the *original* positions.
    frac = torch.linalg.solve(cell.T, positions.T).T
    wrap = torch.where(pbc, torch.floor(frac), torch.zeros_like(frac)).long()
    wrapped = positions - wrap.to(positions.dtype) @ cell

    shifts = _image_shifts(cell, pbc, cutoff)
    rows, cols, shift_ids = [], [], []
    zero_id = int(((shifts == 0).all(dim=1)).nonzero()[0, 0])
    for start in range(0, shifts.shape[0], shift_chunk):
        block = shifts[start : start + shift_chunk]
        images = wrapped[None, :, :] + (block.to(positions.dtype) @ cell)[:, None, :]  # (K, n, 3)
        dist = torch.cdist(wrapped[None].expand(block.shape[0], n, 3), images)  # (K, n, n)
        mask = dist <= cutoff
        if start <= zero_id < start + block.shape[0]:
            mask[zero_id - start].fill_diagonal_(False)
        k, i, j = mask.nonzero(as_tuple=True)
        rows.append(i)
        cols.append(j)
        shift_ids.append(k + start)
    i = torch.cat(rows)
    j = torch.cat(cols)
    s_wrapped = shifts[torch.cat(shift_ids)]
    unit_shifts = s_wrapped - wrap[j] + wrap[i]
    return torch.stack([i, j]), unit_shifts


def _torch_neighbor_list_atoms(atoms, cutoff: float, dtype: torch.dtype) -> NeighborListResult:
    """``"torch"`` backend for an ASE ``Atoms`` object."""
    import numpy as np

    positions = torch.tensor(atoms.positions, dtype=dtype)
    cell = torch.tensor(np.array(atoms.cell), dtype=dtype)
    pbc = torch.tensor(atoms.pbc, dtype=torch.bool)
    edge_index, unit_shifts = _torch_neighbor_list_single(positions, cutoff, cell, pbc)
    row, col = edge_index
    edge_vectors = positions[col] - positions[row] + unit_shifts.to(dtype) @ cell
    return NeighborListResult(
        edge_index=edge_index,
        edge_vectors=edge_vectors,
        edge_lengths=edge_vectors.norm(dim=-1),
        unit_shifts=unit_shifts,
    )


# ---------------------------------------------------------------------------
# Batched, tensor-level construction (GPU simulation / GPU-resident batches)
# ---------------------------------------------------------------------------


def _resolve_batch_backend(backend: str, device: torch.device) -> str:
    from goal.ml.utils.hardware import nvalchemiops_available, resolve_backend

    if backend not in ("auto", "nvalchemiops", "torch"):
        raise ValueError(
            f"build_neighbor_list_batch backend must be 'auto', 'nvalchemiops' or 'torch', "
            f"got {backend!r}."
        )
    resolved = resolve_backend(
        backend,
        fast="nvalchemiops",
        fallback="torch",
        available=nvalchemiops_available,
        feature="batched neighbour-list",
    )
    if resolved == "nvalchemiops" and device.type != "cuda":
        return "torch"  # the Warp kernels need the tensors on the GPU
    return resolved


def build_neighbor_list_batch(
    positions: torch.Tensor,
    cutoff: float,
    batch: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    backend: str = "auto",
    sort: bool = True,
) -> NeighborListResult:
    """Neighbour list for a batch of structures given as tensors.

    Parameters
    ----------
    positions : Tensor
        ``(N, 3)`` concatenated positions (any device).
    cutoff : float
        Cutoff radius in Ångström.
    batch : Tensor, optional
        ``(N,)`` structure index per atom, sorted (PyG convention).
        ``None`` = one structure.
    cell : Tensor, optional
        ``(B, 3, 3)`` or ``(3, 3)`` lattice vectors as rows.
    pbc : Tensor, optional
        ``(B, 3)`` or ``(3,)`` periodic flags.  Required with ``cell``.
    backend : str
        ``"auto"`` (default), ``"nvalchemiops"`` (NVIDIA only; degrades to
        ``"torch"`` with a warning when unavailable) or ``"torch"``.
    sort : bool
        Sort edges by ``(i, j, unit_shift)`` so every backend returns the same order.

    Returns
    -------
    NeighborListResult
        Global (batch-level) ``edge_index`` in the ASE convention
        (``edge_vectors = r_j - r_i + S @ cell_of(i)``), on ``positions.device``.
    """
    device = positions.device
    n = positions.shape[0]
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=device)
    batch = batch.to(device=device, dtype=torch.long)
    n_graphs = int(batch.max()) + 1 if n else 0
    if cell is not None:
        cell = cell.to(device=device, dtype=positions.dtype)
        if cell.dim() == 2:
            cell = cell.expand(n_graphs, 3, 3)
        if pbc is None:
            raise ValueError("pbc is required whenever cell is given.")
        pbc = pbc.to(device=device, dtype=torch.bool)
        if pbc.dim() == 1:
            pbc = pbc.expand(n_graphs, 3)
    periodic = pbc is not None and bool(pbc.any())

    resolved = _resolve_batch_backend(backend, device)
    if resolved == "nvalchemiops":
        edge_index, unit_shifts = _nvalchemiops_batch(
            positions, cutoff, batch, cell, pbc, n_graphs
        )
    else:
        counts = torch.bincount(batch, minlength=n_graphs)
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        rows, cols, shifts = [], [], []
        for g in range(n_graphs):
            lo, hi = int(ptr[g]), int(ptr[g + 1])
            ei, sh = _torch_neighbor_list_single(
                positions[lo:hi],
                cutoff,
                cell[g] if cell is not None else None,
                pbc[g] if pbc is not None else None,
            )
            rows.append(ei[0] + lo)
            cols.append(ei[1] + lo)
            shifts.append(sh)
        edge_index = (
            torch.stack([torch.cat(rows), torch.cat(cols)])
            if rows
            else (torch.zeros(2, 0, dtype=torch.long, device=device))
        )
        unit_shifts = (
            torch.cat(shifts) if shifts else torch.zeros(0, 3, dtype=torch.long, device=device)
        )

    if sort and edge_index.shape[1]:
        # Lexicographic (i, j, S): small periodic cells list one (i, j) pair
        # once per image, so the shift is part of the key.
        order = torch.arange(edge_index.shape[1], device=device)
        for column in (
            unit_shifts[:, 2],
            unit_shifts[:, 1],
            unit_shifts[:, 0],
            edge_index[1],
            edge_index[0],
        ):
            order = order[torch.argsort(column[order], stable=True)]
        edge_index, unit_shifts = edge_index[:, order], unit_shifts[order]

    row, col = edge_index
    edge_vectors = positions[col] - positions[row]
    if periodic and cell is not None:
        edge_vectors = edge_vectors + torch.einsum(
            "ej,ejk->ek", unit_shifts.to(positions.dtype), cell[batch[row]]
        )
    return NeighborListResult(
        edge_index=edge_index,
        edge_vectors=edge_vectors,
        edge_lengths=edge_vectors.norm(dim=-1),
        unit_shifts=unit_shifts,
    )


def _nvalchemiops_batch(
    positions: torch.Tensor,
    cutoff: float,
    batch: torch.Tensor,
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
    n_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One batched Warp kernel launch (``nvalchemiops`` >= 0.4.1)."""
    from nvalchemiops.torch.neighbors import neighbor_list as nval_nl

    counts = torch.bincount(batch, minlength=n_graphs)
    batch_ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)
    batch_idx = batch.to(torch.int32)
    if pbc is not None and bool(pbc.any()):
        out = nval_nl(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            return_neighbor_list=True,
        )
        edge_index, unit_shifts = out[0].long(), out[2].long()
    else:
        out = nval_nl(
            positions, cutoff, batch_idx=batch_idx, batch_ptr=batch_ptr, return_neighbor_list=True
        )
        edge_index = out[0].long()
        unit_shifts = torch.zeros(
            edge_index.shape[1], 3, dtype=torch.long, device=positions.device
        )
    return edge_index, unit_shifts
