"""AtomicGraph — the central data contract of GOAL.

Every model, dataset, and adapter in the framework speaks this language.
Pure tensors inside the training loop — no ASE objects, no numpy arrays,
no dicts. ASE is used only at the ``from_ase()`` boundary.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import torch
from torch_geometric.data import Data


class AtomicGraph(Data):
    """The central data contract of GOAL.

    All models, datasets, and adapters speak this language.
    Pure tensors — no ASE objects, no numpy arrays, no dicts inside
    the training loop.  ASE is used only at ``from_ase()`` boundary.

    PyG's ``Data`` base class gives us batching for free:

    - ``Batch.from_data_list([g1, g2, g3])`` stacks multiple graphs
    - ``graph.batch`` tensor tracks which atoms belong to which structure
    - All tensor fields are automatically concatenated along dim 0
    """

    def __init__(
        self,
        # Atomic structure — always required
        positions: torch.Tensor,       # (N, 3) float64
        atomic_numbers: torch.Tensor,  # (N,)   int64
        cell: torch.Tensor,            # (3, 3) float64, zeros if no PBC
        pbc: torch.Tensor,             # (3,)   bool
        # Graph topology — always required
        edge_index: torch.Tensor,      # (2, E) int64
        edge_vectors: torch.Tensor,    # (E, 3) float64, r_j - r_i
        edge_lengths: torch.Tensor,    # (E,)   float64
        # Training targets — optional, None for inference
        energy: typing.Optional[torch.Tensor] = None,    # (1,)   float64
        forces: typing.Optional[torch.Tensor] = None,    # (N, 3) float64
        stress: typing.Optional[torch.Tensor] = None,    # (3, 3) float64
        # Metadata
        weight: typing.Optional[torch.Tensor] = None,    # (1,)   float64, sample weight
        head: typing.Optional[str] = None,               # multihead training tag
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(
            pos=positions,
            z=atomic_numbers,
            cell=cell.unsqueeze(0),  # PyG expects (1, 3, 3) for cell
            pbc=pbc,
            edge_index=edge_index,
            edge_attr=edge_vectors,
            edge_weight=edge_lengths,
            energy=energy,
            forces=forces,
            stress=stress,
            weight=weight,
            head=head,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Convenient property accessors
    # ------------------------------------------------------------------

    @property
    def positions(self) -> torch.Tensor:
        """Atomic positions (N, 3)."""
        return self.pos

    @property
    def atomic_numbers(self) -> torch.Tensor:
        """Atomic numbers (N,)."""
        return self.z

    @property
    def edge_vectors(self) -> torch.Tensor:
        """Edge displacement vectors r_j − r_i, shape (E, 3)."""
        return self.edge_attr

    @property
    def edge_lengths(self) -> torch.Tensor:
        """Edge lengths ‖r_j − r_i‖, shape (E,)."""
        return self.edge_weight

    @property
    def num_atoms(self) -> int:
        """Total number of atoms in this graph (or batch of graphs)."""
        return self.pos.shape[0]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ase(
        cls,
        atoms,  # ase.Atoms — typed loosely to avoid hard ase import
        cutoff: float,
        energy: typing.Optional[float] = None,
        forces=None,
        stress=None,
        weight: float = 1.0,
        head: typing.Optional[str] = None,
        dtype: torch.dtype = torch.float64,
    ) -> AtomicGraph:
        """Convert ASE ``Atoms`` to ``AtomicGraph``.

        This is the *only* place ASE appears in the entire framework.
        Called once per structure during dataset preprocessing.
        """
        import numpy as np
        from torch_geometric.nn import radius_graph

        positions: torch.Tensor = torch.tensor(atoms.positions, dtype=dtype)
        atomic_numbers: torch.Tensor = torch.tensor(atoms.numbers, dtype=torch.long)
        cell: torch.Tensor = torch.tensor(np.array(atoms.cell), dtype=dtype)
        pbc_tensor: torch.Tensor = torch.tensor(atoms.pbc, dtype=torch.bool)

        # Build neighbour list — pure PyTorch, works on GPU
        edge_index: torch.Tensor = radius_graph(positions, r=cutoff, loop=False)
        row: torch.Tensor
        col: torch.Tensor
        row, col = edge_index
        edge_vecs: torch.Tensor = positions[col] - positions[row]

        # Apply minimum image convention for PBC
        if pbc_tensor.any():
            edge_vecs = _apply_mic(edge_vecs, cell, pbc_tensor)

        edge_lens: torch.Tensor = edge_vecs.norm(dim=-1)

        return cls(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc_tensor,
            edge_index=edge_index,
            edge_vectors=edge_vecs,
            edge_lengths=edge_lens,
            energy=(
                torch.tensor([energy], dtype=dtype) if energy is not None else None
            ),
            forces=(
                torch.tensor(forces, dtype=dtype) if forces is not None else None
            ),
            stress=(
                torch.tensor(stress, dtype=dtype) if stress is not None else None
            ),
            weight=torch.tensor([weight], dtype=dtype),
            head=head,
        )

    @classmethod
    def from_dict(cls, d: typing.Dict[str, typing.Any], cutoff: float) -> AtomicGraph:
        """Build from raw dict — used in adapters to translate
        MACE / fairchem dict conventions into ``AtomicGraph``.
        """
        positions: torch.Tensor = d["positions"]
        atomic_numbers: torch.Tensor = d["atomic_numbers"]
        cell: torch.Tensor = d.get("cell", torch.zeros(3, 3, dtype=positions.dtype))
        pbc: torch.Tensor = d.get("pbc", torch.zeros(3, dtype=torch.bool))

        from torch_geometric.nn import radius_graph

        edge_index: torch.Tensor = radius_graph(positions, r=cutoff, loop=False)
        row: torch.Tensor
        col: torch.Tensor
        row, col = edge_index
        edge_vecs: torch.Tensor = positions[col] - positions[row]

        if pbc.any():
            edge_vecs = _apply_mic(edge_vecs, cell, pbc)

        edge_lens: torch.Tensor = edge_vecs.norm(dim=-1)

        return cls(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            edge_index=edge_index,
            edge_vectors=edge_vecs,
            edge_lengths=edge_lens,
            energy=d.get("energy"),
            forces=d.get("forces"),
            stress=d.get("stress"),
            weight=d.get("weight"),
            head=d.get("head"),
        )


# ---------------------------------------------------------------------------
# Typed output container
# ---------------------------------------------------------------------------


@dataclass
class NodeFeatures:
    """Output of any ``EquivariantBackbone`` forward pass.

    Typed container — not a raw dict.
    """

    node_feats: torch.Tensor
    """(N, channels) — irrep feature vectors."""

    irreps: str
    """e3nn irreps string, e.g. ``'256x0e+256x1o'``."""

    node_energies: typing.Optional[torch.Tensor] = None
    """(N,) atomic energy contributions, if available."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_mic(
    edge_vectors: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Apply minimum image convention for periodic boundary conditions.

    Projects edge vectors into fractional coordinates, wraps them to
    the range [−0.5, 0.5), and converts back to Cartesian.
    """
    # Convert to fractional coordinates
    inv_cell: torch.Tensor = torch.linalg.inv(cell)
    frac: torch.Tensor = edge_vectors @ inv_cell.T

    # Wrap periodic dimensions to [-0.5, 0.5)
    for dim in range(3):
        if pbc[dim]:
            frac[:, dim] = frac[:, dim] - torch.round(frac[:, dim])

    # Back to Cartesian
    return frac @ cell
