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
        positions: torch.Tensor,  # (N, 3) float64
        atomic_numbers: torch.Tensor,  # (N,)   int64
        cell: torch.Tensor,  # (3, 3) float64, zeros if no PBC
        pbc: torch.Tensor,  # (3,)   bool
        # Graph topology — only the index pairs are mandatory.
        edge_index: torch.Tensor,  # (2, E) int64
        # ``edge_vectors`` and ``edge_lengths`` are kept *optional* because
        # the production force-prediction path (``SimurghBackbone`` +
        # ``DualForcesHead``) recomputes both from ``positions`` inside
        # ``goal.ml.nn.models.simurgh.geometry.differentiable_edges`` so the
        # autograd chain links forces back to ``graph.pos``.  Storing them
        # on the graph just adds ~30% per-batch VRAM with no consumer in
        # that pipeline.  Callers that *do* need them at rest (e.g. the
        # virial-style ``StressHead`` reading ``graph.edge_attr``
        # directly, or downstream transforms that mutate edge geometry)
        # may pass them explicitly — they still flow into the standard
        # PyG slots (``edge_attr`` / ``edge_weight``).
        edge_vectors: torch.Tensor | None = None,  # (E, 3) float, r_j - r_i
        edge_lengths: torch.Tensor | None = None,  # (E,)   float
        # PBC shifts — integer cell-shift vectors per edge (needed by MACE/MLIPs)
        unit_shifts: torch.Tensor | None = None,  # (E, 3) int64
        # Fragment labels — optional.  ``(N,)`` int64 connected-component
        # labels under a short covalent cutoff, consumed by
        # the fragment-interaction module.  Deliberately *not* required:
        # graphs built without it simply carry no attribute, and the ARACE
        # rounds treat a missing ``fragment_index`` as "channel disabled".
        # See ``goal.ml.data.fragments.compute_fragment_index``.
        fragment_index: torch.Tensor | None = None,  # (N,) int64
        # Training targets — optional, None for inference
        energy: torch.Tensor | None = None,  # (1,)   float64
        forces: torch.Tensor | None = None,  # (N, 3) float64
        stress: torch.Tensor | None = None,  # (3, 3) float64
        # Metadata
        weight: torch.Tensor | None = None,  # (1,)   float64, sample weight
        head: str | None = None,  # multihead training tag
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(
            pos=positions,
            z=atomic_numbers,
            cell=cell.unsqueeze(0) if cell is not None else None,  # PyG expects (1, 3, 3) for cell
            pbc=pbc,
            edge_index=edge_index,
            edge_attr=edge_vectors,
            edge_weight=edge_lengths,
            unit_shifts=unit_shifts,
            fragment_index=fragment_index,
            energy=energy,
            forces=forces,
            stress=stress,
            weight=weight,
            head=head,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Batching semantics for fragment labels
    # ------------------------------------------------------------------

    def __inc__(self, key: str, value: typing.Any, *args: typing.Any, **kwargs: typing.Any):
        """Offset ``fragment_index`` by the *fragment* count, not the atom count.

        PyG's default increments any key containing ``"index"`` by
        ``num_nodes``, which would leave the labels non-contiguous
        (``labels.max() + 1`` far larger than the true number of
        fragments).  ``EquivariantFragmentInteraction`` sizes its
        fragment graph as ``fragment_index.max() + 1``, so sparse labels
        would create phantom empty fragments.  Incrementing by
        ``max + 1`` keeps a batch's labels dense while still guaranteeing
        no collisions between graphs.
        """
        if key == "fragment_index":
            return int(value.max()) + 1 if value.numel() > 0 else 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: typing.Any, *args: typing.Any, **kwargs: typing.Any):
        """Concatenate ``fragment_index`` along the atom axis.

        PyG defaults ``"index"``-like keys to ``-1``, which happens to
        coincide with dim 0 for a 1-D tensor — pinned explicitly so the
        behaviour cannot drift with a future PyG release.
        """
        if key == "fragment_index":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

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
    def edge_vectors(self) -> torch.Tensor | None:
        """Stored edge displacement vectors r_j − r_i, shape (E, 3).

        ``None`` on graphs built via :meth:`from_ase` / :meth:`from_dict`
        (the production path) — callers should recompute from
        ``positions`` + ``edge_index`` so the autograd chain is intact.
        Only populated when the graph was constructed explicitly with
        ``edge_vectors=...``.
        """
        return self.get("edge_attr", None)

    @property
    def edge_lengths(self) -> torch.Tensor | None:
        """Stored edge lengths ‖r_j − r_i‖, shape (E,).

        ``None`` on graphs built via :meth:`from_ase` / :meth:`from_dict`
        — see :attr:`edge_vectors` for the rationale.
        """
        return self.get("edge_weight", None)

    @property
    def fragment_index(self) -> torch.Tensor | None:
        """Fragment labels ``(N,)`` int64, or ``None``.

        Populated only when the dataset was built with
        ``compute_fragment_index=True`` (config key
        ``data.compute_fragment_index``).  ``None`` on every other graph
        — the ARACE rounds read it defensively and only require it when
        the fragment-interaction module is enabled.
        """
        return self.get("fragment_index", None)

    @property
    def num_fragments(self) -> int:
        """Number of fragments in this graph (or batch), 0 when unlabelled."""
        labels: torch.Tensor | None = self.get("fragment_index", None)
        if labels is None or labels.numel() == 0:
            return 0
        return int(labels.max()) + 1

    @property
    def energy(self) -> torch.Tensor | None:
        """Potential energy (1,) float64, or None for inference structures."""
        return self.get("energy", None)

    @property
    def forces(self) -> torch.Tensor | None:
        """Atomic forces (N, 3) float64, or None."""
        return self.get("forces", None)

    @property
    def stress(self) -> torch.Tensor | None:
        """Stress tensor (3, 3) float64, or None."""
        return self.get("stress", None)

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
        energy: float | None = None,
        forces=None,
        stress=None,
        weight: float = 1.0,
        head: str | None = None,
        dtype: torch.dtype = torch.float64,
        neighbor_list_backend: str = "ase",
        compute_fragment_index: bool = False,
        fragment_covalent_cutoff: float = 1.8,
        fragment_scheme: str = "connected",
        fragment_charge: int = 0,
        fragment_on_failure: str = "fallback",
        fragment_smarts: typing.Sequence[str] | None = None,
        fragment_keep_groups: typing.Sequence[str] | None = None,
    ) -> AtomicGraph:
        """Convert ASE ``Atoms`` to ``AtomicGraph``.

        This is the *only* place ASE appears in the entire framework.
        Called once per structure during dataset preprocessing.

        Parameters
        ----------
        neighbor_list_backend : str
            Neighbour-list backend — ``"ase"`` (default, correct PBC),
            ``"matscipy"`` (faster, optional dep), or ``"radius_graph"``
            (legacy, no PBC support).  See
            ``goal.ml.data.neighbor_list`` for details.
        compute_fragment_index : bool
            Attach ``fragment_index`` — connected-component labels under
            a short covalent cutoff, required by
            the fragment-interaction module.  Computed here, at graph
            construction, so the training loop never pays for it.
            Default ``False`` (no field, no cost).
        fragment_covalent_cutoff : float
            Bond threshold for the fragment decomposition (Angstrom).
        fragment_scheme : str
            How to split the structure — see
            :data:`goal.ml.data.fragments.FRAGMENT_SCHEMES`.
        fragment_charge : int
            Total charge for the rdkit schemes' bond perception.
        fragment_on_failure : str
            ``"fallback"`` or ``"raise"`` when perception fails.
        fragment_smarts : sequence of str, optional
            Bond patterns to cut, for ``fragment_scheme="custom"``.
        fragment_keep_groups : sequence of str, optional
            SMARTS whose matched atoms must stay in one fragment.
        """
        import numpy as np

        from goal.ml.data.neighbor_list import build_neighbor_list

        positions: torch.Tensor = torch.tensor(atoms.positions, dtype=dtype)
        atomic_numbers: torch.Tensor = torch.tensor(atoms.numbers, dtype=torch.long)
        cell: torch.Tensor = torch.tensor(np.array(atoms.cell), dtype=dtype)
        pbc_tensor: torch.Tensor = torch.tensor(atoms.pbc, dtype=torch.bool)

        nl = build_neighbor_list(atoms, cutoff, backend=neighbor_list_backend, dtype=dtype)

        # ``nl.edge_vectors`` / ``nl.edge_lengths`` are deliberately *not*
        # forwarded here.  Production heads recompute both from positions
        # in the forward pass (so the autograd graph wires force gradients
        # back to ``graph.pos``); keeping them on every graph in the
        # dataloader just inflates per-batch VRAM with redundant data.
        # Callers needing the stored copy (e.g. stress training before
        # the StressHead update lands) can construct ``AtomicGraph``
        # directly and pass them through ``__init__``.
        fragment_index: torch.Tensor | None = None
        if compute_fragment_index:
            from goal.ml.data.fragments import compute_fragment_index as _fragments

            fragment_index = _fragments(
                positions=positions,
                atomic_numbers=atomic_numbers,
                covalent_cutoff=fragment_covalent_cutoff,
                cell=cell,
                pbc=pbc_tensor,
                scheme=fragment_scheme,
                charge=fragment_charge,
                on_failure=fragment_on_failure,
                smarts=fragment_smarts,
                keep_groups=fragment_keep_groups,
            )

        graph = cls(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc_tensor,
            edge_index=nl.edge_index,
            unit_shifts=nl.unit_shifts,
            fragment_index=fragment_index,
            energy=(torch.tensor([energy], dtype=dtype) if energy is not None else None),
            forces=(torch.tensor(forces, dtype=dtype) if forces is not None else None),
            stress=(torch.tensor(stress, dtype=dtype) if stress is not None else None),
            weight=torch.tensor([weight], dtype=dtype),
            head=head,
        )
        # Hessian label (eV/Å²): a per-atom array (N, 9N) of each atom's three
        # rows.  Kept whole here; HessianRows samples rows for batching.
        hessian = getattr(atoms, "arrays", {}).get("hessian")
        if hessian is not None:
            n = len(atoms)
            graph.hessian = torch.as_tensor(hessian, dtype=dtype).reshape(3 * n, 3 * n)
        return graph

    @classmethod
    def from_dict(
        cls,
        d: dict[str, typing.Any],
        cutoff: float,
        neighbor_list_backend: str = "ase",
        compute_fragment_index: bool = False,
        fragment_covalent_cutoff: float = 1.8,
        fragment_scheme: str = "connected",
        fragment_charge: int = 0,
        fragment_on_failure: str = "fallback",
        fragment_smarts: typing.Sequence[str] | None = None,
        fragment_keep_groups: typing.Sequence[str] | None = None,
    ) -> AtomicGraph:
        """Build from raw dict — used in adapters to translate
        MACE / fairchem dict conventions into ``AtomicGraph``.

        ``compute_fragment_index`` / ``fragment_covalent_cutoff`` behave
        exactly as in :meth:`from_ase`.
        """
        from goal.ml.data.neighbor_list import build_neighbor_list_from_tensors

        positions: torch.Tensor = d["positions"]
        atomic_numbers: torch.Tensor = d["atomic_numbers"]
        cell: torch.Tensor = d.get("cell", torch.zeros(3, 3, dtype=positions.dtype))
        pbc: torch.Tensor = d.get("pbc", torch.zeros(3, dtype=torch.bool))

        nl = build_neighbor_list_from_tensors(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            cutoff=cutoff,
            backend=neighbor_list_backend,
            dtype=positions.dtype,
        )

        # See ``from_ase`` for why edge_vectors / edge_lengths are *not*
        # forwarded — production heads recompute them from positions so
        # storing them per-graph is wasted memory.
        fragment_index: torch.Tensor | None = d.get("fragment_index")
        if compute_fragment_index and fragment_index is None:
            from goal.ml.data.fragments import compute_fragment_index as _fragments

            fragment_index = _fragments(
                positions=positions,
                atomic_numbers=atomic_numbers,
                covalent_cutoff=fragment_covalent_cutoff,
                cell=cell,
                pbc=pbc,
                scheme=fragment_scheme,
                charge=fragment_charge,
                on_failure=fragment_on_failure,
                smarts=fragment_smarts,
                keep_groups=fragment_keep_groups,
            )

        return cls(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            edge_index=nl.edge_index,
            unit_shifts=nl.unit_shifts,
            fragment_index=fragment_index,
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

    node_energies: torch.Tensor | None = None
    """(N,) atomic energy contributions, if available."""

    node_forces: torch.Tensor | None = None
    """(N, 3) per-atom forces produced by a pairwise-force backbone.

    When populated (e.g. by ``SimurghBackbone`` with
    ``compute_pairwise_forces=True``), force-aware heads can consume
    these directly instead of taking an additional
    ``autograd.grad(energy, positions)`` pass.  Newton's third law is
    satisfied by construction whenever a backbone fills this slot.
    """


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
