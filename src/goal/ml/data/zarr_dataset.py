"""Zarr storage for atomic structures: chunked, compressed, random access.

Layout (``goal-zarr/1``), one group per split, structures concatenated
along the first axis and delimited by per-structure counts::

    store.zarr/                    attrs: format = "goal-zarr/1"
      {split}/                     attrs: num_structures, [cutoff]
        n_atoms         (S,)       int64
        positions       (N, 3)     float64   Å
        atomic_numbers  (N,)       int64
        cell            (S, 3, 3)  float64   Å, lattice vectors as rows
        pbc             (S, 3)     bool
        energies        (S,)       float64   eV          (optional)
        forces          (N, 3)     float64   eV/Å        (optional)
        stress          (S, 3, 3)  float64   eV/Å³       (optional)
        n_edges         (S,)       int64                 (optional, with cutoff)
        edge_index      (E, 2)     int64     per-structure atom indices
        unit_shifts     (E, 3)     int64
        info/<key>      (S,)       float64 / int64  scalar ``atoms.info`` values (optional, ``info_keys``)

A label array is written only when every structure has that label.  With
``cutoff`` given at write time the neighbour lists are stored too and
reused by :class:`ZarrDataset` whenever it is opened with the same cutoff.

Write with :func:`write_zarr` or ``goal-pack-zarr``; train on it with
``data.dataset_type: zarr``.  Needs ``zarr>=3`` (``pip install
'goal-atomistic[zarr]'``).
"""

from __future__ import annotations

import os
import typing
from pathlib import Path

import numpy as np
import torch

from goal.ml.data.datasets.base import BaseAtomicDataset
from goal.ml.data.graph import AtomicGraph

ZARR_FORMAT: str = "goal-zarr/1"
_INSTALL_HINT: str = (
    "Zarr storage needs zarr>=3: pip install 'goal-atomistic[zarr]' (or 'zarr>=3')"
)


def _require_zarr() -> typing.Any:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    if int(zarr.__version__.split(".")[0]) < 3:
        raise ImportError(f"{_INSTALL_HINT} (found zarr {zarr.__version__})")
    return zarr


def _stress_3x3(stress: typing.Any) -> np.ndarray:
    arr = np.asarray(stress, dtype=float)
    if arr.size == 6:
        from ase.stress import voigt_6_to_full_3x3_stress

        return voigt_6_to_full_3x3_stress(arr)
    return arr.reshape(3, 3)


class _Buffer:
    """Accumulates structures and appends them to the store in blocks."""

    ARRAYS: typing.ClassVar[dict[str, tuple[tuple[int, ...], str]]] = {
        "n_atoms": ((), "int64"),
        "positions": ((3,), "float64"),
        "atomic_numbers": ((), "int64"),
        "cell": ((3, 3), "float64"),
        "pbc": ((3,), "bool"),
        "energies": ((), "float64"),
        "forces": ((3,), "float64"),
        "stress": ((3, 3), "float64"),
        "n_edges": ((), "int64"),
        "edge_index": ((2,), "int64"),
        "unit_shifts": ((3,), "int64"),
    }

    def __init__(self, group: typing.Any, chunk_rows: int) -> None:
        self.group = group
        self.chunk_rows = chunk_rows
        self.rows: dict[str, list[np.ndarray]] = {k: [] for k in self.ARRAYS}
        self.missing: set[str] = set()
        self.count = 0

    def add(self, record: dict[str, np.ndarray | None]) -> None:
        for key, value in record.items():
            if value is None:
                self.missing.add(key)
            else:
                self.rows[key].append(np.asarray(value))
        self.count += 1

    def flush(self) -> None:
        for key, chunks in self.rows.items():
            if not chunks or key in self.missing:
                continue
            tail, dtype = self.ARRAYS[key]
            data = np.concatenate([c.reshape(-1, *tail) for c in chunks]).astype(dtype)
            if key not in self.group:
                self.group.create_array(
                    key,
                    shape=(0, *tail),
                    dtype=dtype,
                    chunks=(self.chunk_rows, *tail) if tail else (self.chunk_rows,),
                )
            self.group[key].append(data)
            self.rows[key] = []

    def finalise(self) -> None:
        self.flush()
        for key in self.missing:  # a label some structure lacked is not stored at all
            if key in self.group:
                del self.group[key]


def write_zarr(
    frames: typing.Iterable[typing.Any],
    store: str | Path,
    split: str = "train",
    cutoff: float | None = None,
    neighbor_list_backend: str = "ase",
    overwrite: bool = False,
    chunk_rows: int = 4096,
    flush_every: int = 1000,
    info_keys: typing.Sequence[str] = (),
) -> Path:
    """Write ASE ``Atoms`` frames to a GOAL Zarr store.

    Parameters
    ----------
    frames : iterable of ase.Atoms
        Labels are read like the xyz/trajectory datasets do
        (``atoms.info`` / ``atoms.arrays`` first, then an attached
        calculator).
    store : str or Path
        Store directory (``*.zarr``); created if missing.
    split : str
        Group name (``train`` / ``val`` / ``test`` / …).
    cutoff : float, optional
        Also store neighbour lists for this cutoff.
    overwrite : bool
        Replace an existing ``split`` group.
    info_keys : sequence of str
        Scalar ``atoms.info`` entries to keep, one value per structure,
        under ``{split}/info/<key>`` (e.g. ``("step", "structure_index")``
        for trajectories).  Read back with :meth:`ZarrDataset.structure_info`.
    """
    zarr = _require_zarr()
    from goal.ml.data.keys import extract_ase_labels, resolve_label_keys

    key_map, explicit = resolve_label_keys(None, None, None, None)
    root = zarr.open_group(str(store), mode="a")
    root.attrs["format"] = ZARR_FORMAT
    if split in root:
        if not overwrite:
            raise FileExistsError(f"{store} already has a '{split}' group (pass overwrite=True).")
        del root[split]
    group = root.create_group(split)
    buffer = _Buffer(group, chunk_rows)
    info_values: dict[str, list[typing.Any]] = {key: [] for key in info_keys}

    for atoms in frames:
        for key in info_keys:
            if key not in atoms.info:
                raise KeyError(f"info_keys: a structure has no atoms.info[{key!r}]")
            info_values[key].append(atoms.info[key])
        labels = extract_ase_labels(atoms, key_map, explicit)
        n = len(atoms)
        record: dict[str, np.ndarray | None] = {
            "n_atoms": np.array([n]),
            "positions": atoms.get_positions(),
            "atomic_numbers": atoms.get_atomic_numbers(),
            "cell": np.array(atoms.cell).reshape(1, 3, 3),
            "pbc": np.asarray(atoms.pbc, dtype=bool).reshape(1, 3),
            "energies": None if labels["energy"] is None else np.array([float(labels["energy"])]),
            "forces": (
                None if labels["forces"] is None else np.asarray(labels["forces"]).reshape(n, 3)
            ),
            "stress": (
                None
                if labels["stress"] is None
                else _stress_3x3(labels["stress"]).reshape(1, 3, 3)
            ),
        }
        if cutoff is not None:
            from goal.ml.data.neighbor_list import build_neighbor_list

            nl = build_neighbor_list(atoms, cutoff, backend=neighbor_list_backend)
            record["n_edges"] = np.array([nl.edge_index.shape[1]])
            record["edge_index"] = nl.edge_index.T.numpy()
            record["unit_shifts"] = nl.unit_shifts.numpy()
        buffer.add(record)
        if buffer.count % flush_every == 0:
            buffer.flush()
    buffer.finalise()
    if info_keys:
        info_group = group.create_group("info")
        for key, values in info_values.items():
            arr = np.asarray(values)
            arr = arr.astype("int64" if np.issubdtype(arr.dtype, np.integer) else "float64")
            info_group.create_array(key, data=arr, chunks=(max(1, min(chunk_rows, len(arr))),))
    group.attrs["num_structures"] = buffer.count
    if cutoff is not None:
        group.attrs["cutoff"] = float(cutoff)
    return Path(store)


class ZarrDataset(BaseAtomicDataset):
    """Random-access dataset over a GOAL Zarr store (``dataset_type: zarr``).

    Parameters
    ----------
    root : str or Path
        The ``*.zarr`` store.
    cutoff : float
        Neighbour-list cutoff.  Stored neighbour lists are reused when the
        store was written with the same cutoff; otherwise they are built
        per structure with ``neighbor_list_backend``.
    split : str
        Group to read.
    dtype, neighbor_list_backend, fragment options
        As for every :class:`BaseAtomicDataset`.

    ``__getitem__`` returns the same ``AtomicGraph`` fields as the
    xyz / trajectory datasets (``pos``, ``z``, ``cell``, ``pbc``,
    ``edge_index``, ``unit_shifts``, ``energy`` ``(1,)``, ``forces``,
    ``stress``, ``weight``, ``fragment_index``).

    A store missing the requested split raises ``FileNotFoundError``, so
    ``GOALDataModule`` with ``data.root: store.zarr`` falls back to splitting
    the ``train`` group by ``data.split_ratio``.
    """

    #: The ``root`` directory is one store, not a folder of data files.
    ROOT_IS_DIRECTORY: typing.ClassVar[bool] = True

    def __init__(
        self,
        root: str | Path,
        cutoff: float,
        split: str = "train",
        dtype: torch.dtype = torch.float64,
        neighbor_list_backend: str = "ase",
        **fragment_kwargs: typing.Any,
    ) -> None:
        super().__init__(
            root=root,
            cutoff=cutoff,
            split=split,
            dtype=dtype,
            neighbor_list_backend=neighbor_list_backend,
            **fragment_kwargs,
        )
        _require_zarr()
        if not self.root.exists():
            raise FileNotFoundError(f"Zarr store not found: {self.root}")
        group = self._open()
        self._n_atoms: np.ndarray = group["n_atoms"][:]
        self._atom_ptr: np.ndarray = np.concatenate([[0], np.cumsum(self._n_atoms)])
        stored_cutoff = group.attrs.get("cutoff", None)
        self._use_stored_edges: bool = (
            "edge_index" in group
            and stored_cutoff is not None
            and abs(float(stored_cutoff) - float(cutoff)) < 1e-9
        )
        if self._use_stored_edges:
            n_edges = group["n_edges"][:]
            self._edge_ptr: np.ndarray = np.concatenate([[0], np.cumsum(n_edges)])
        self._labels: set[str] = {k for k in ("energies", "forces", "stress") if k in group}

    def _open(self) -> typing.Any:
        """The split group, opened once per process (safe with DataLoader workers)."""
        if getattr(self, "_group_pid", None) != os.getpid():
            zarr = _require_zarr()
            root = zarr.open_group(str(self.root), mode="r")
            if self.split not in root:
                # FileNotFoundError (not KeyError): GOALDataModule treats a
                # missing named split like a missing split file and falls back
                # to splitting the 'train' group by data.split_ratio.
                raise FileNotFoundError(
                    f"{self.root} has no '{self.split}' group; groups: {list(root.group_keys())}"
                )
            self._group = root[self.split]
            self._group_pid = os.getpid()
        return self._group

    def __getstate__(self) -> dict[str, typing.Any]:
        state = dict(self.__dict__)
        state.pop("_group", None)
        state.pop("_group_pid", None)
        return state

    def __len__(self) -> int:
        return int(self._n_atoms.shape[0])

    def structure_info(self, idx: int) -> dict[str, typing.Any]:
        """Scalar ``atoms.info`` values stored with ``info_keys`` for structure ``idx``."""
        group = self._open()
        if "info" not in group:
            return {}
        info = group["info"]
        return {key: info[key][idx].item() for key in info.array_keys()}

    def __getitem__(self, idx: int) -> AtomicGraph:
        if idx < 0:
            idx += len(self)
        group = self._open()
        lo, hi = int(self._atom_ptr[idx]), int(self._atom_ptr[idx + 1])
        positions = torch.tensor(group["positions"][lo:hi], dtype=self.dtype)
        numbers = torch.tensor(group["atomic_numbers"][lo:hi], dtype=torch.long)
        cell = torch.tensor(group["cell"][idx], dtype=self.dtype)
        pbc = torch.tensor(group["pbc"][idx], dtype=torch.bool)

        if self._use_stored_edges:
            e_lo, e_hi = int(self._edge_ptr[idx]), int(self._edge_ptr[idx + 1])
            edge_index = torch.tensor(
                group["edge_index"][e_lo:e_hi], dtype=torch.long
            ).T.contiguous()
            unit_shifts = torch.tensor(group["unit_shifts"][e_lo:e_hi], dtype=torch.long)
        else:
            from ase import Atoms

            from goal.ml.data.neighbor_list import build_neighbor_list

            atoms = Atoms(
                numbers=numbers.numpy(),
                positions=positions.numpy(),
                cell=cell.numpy(),
                pbc=pbc.numpy(),
            )
            nl = build_neighbor_list(
                atoms, self.cutoff, backend=self.neighbor_list_backend, dtype=self.dtype
            )
            edge_index, unit_shifts = nl.edge_index, nl.unit_shifts

        graph = AtomicGraph(
            positions=positions,
            atomic_numbers=numbers,
            cell=cell,
            pbc=pbc,
            edge_index=edge_index,
            unit_shifts=unit_shifts,
            energy=(
                torch.tensor([float(group["energies"][idx])], dtype=self.dtype)
                if "energies" in self._labels
                else None
            ),
            forces=(
                torch.tensor(group["forces"][lo:hi], dtype=self.dtype)
                if "forces" in self._labels
                else None
            ),
            stress=(
                torch.tensor(group["stress"][idx], dtype=self.dtype)
                if "stress" in self._labels
                else None
            ),
            weight=torch.tensor([1.0], dtype=self.dtype),
        )
        return self.attach_fragment_index(graph)
