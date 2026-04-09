"""LMDB dataset — reads pre-processed atomic graphs from LMDB databases.

LMDB (Lightning Memory-Mapped Database) provides fast random access with
memory-mapped I/O, making it the preferred format for very large datasets
(millions of structures). This is the same format used by FairChem/OCP
for their large-scale training datasets.

Data layout: each LMDB key is "idx" (bytes), each value is a serialised
dict with positions, atomic_numbers, cell, pbc, edge_index, and targets.
"""

from __future__ import annotations

import pickle
import typing
from pathlib import Path

import torch

from goal.ml.data.datasets.base import BaseAtomicDataset
from goal.ml.data.graph import AtomicGraph
from goal.ml.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("lmdb")
class LMDBDataset(BaseAtomicDataset):
    """Dataset backed by an LMDB database with pre-computed atomic graphs.

    Supports the FairChem/OCP data layout where each entry is a pickled
    dict keyed by integer index.

    Parameters
    ----------
    root : str or Path
        Directory containing LMDB files, or a single ``.lmdb`` file.
    cutoff : float
        Cutoff radius (used for metadata; graph should be pre-built).
    split : str
        Which split to load — looks for ``{split}.lmdb`` inside *root*.
    energy_key : str
        Key in the stored dict for total energy.
    forces_key : str
        Key in the stored dict for per-atom forces.
    stress_key : str
        Key in the stored dict for stress tensor.
    """

    def __init__(
        self,
        root: typing.Union[str, Path],
        cutoff: float,
        split: str = "train",
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(root=root, cutoff=cutoff, split=split, dtype=dtype)
        self.energy_key: str = energy_key
        self.forces_key: str = forces_key
        self.stress_key: str = stress_key
        self._env: typing.Any = None
        self._length: int = 0
        self._open()

    def _open(self) -> None:
        """Open the LMDB environment."""
        import lmdb

        path: Path = self.root / f"{self.split}.lmdb"
        if not path.exists():
            path = self.root
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        self._env = lmdb.open(
            str(path),
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            subdir=path.is_dir(),
        )

        with self._env.begin(write=False) as txn:
            length_bytes: typing.Optional[bytes] = txn.get(b"length")
            if length_bytes is not None:
                self._length = int(length_bytes)
            else:
                self._length = txn.stat()["entries"]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> AtomicGraph:
        with self._env.begin(write=False) as txn:
            raw: typing.Optional[bytes] = txn.get(str(idx).encode())
            if raw is None:
                raise IndexError(f"Index {idx} not found in LMDB")

        data: typing.Dict[str, typing.Any] = pickle.loads(raw)
        return self._dict_to_graph(data)

    def _dict_to_graph(self, data: typing.Dict[str, typing.Any]) -> AtomicGraph:
        """Convert a stored dict to an AtomicGraph."""

        def _to_tensor(val: typing.Any, dtype: torch.dtype = self.dtype) -> typing.Optional[torch.Tensor]:
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                return val.to(dtype)
            return torch.tensor(val, dtype=dtype)

        positions: typing.Optional[torch.Tensor] = _to_tensor(data["positions"])
        atomic_numbers: typing.Optional[torch.Tensor] = _to_tensor(data["atomic_numbers"], dtype=torch.long)
        cell: typing.Optional[torch.Tensor] = _to_tensor(data.get("cell", torch.zeros(3, 3)))
        pbc: typing.Optional[torch.Tensor] = _to_tensor(data.get("pbc", torch.zeros(3)), dtype=torch.bool)

        # Use pre-computed graph if available, else build on-the-fly
        if "edge_index" in data:
            edge_index: typing.Optional[torch.Tensor] = _to_tensor(data["edge_index"], dtype=torch.long)
            edge_vectors: typing.Optional[torch.Tensor] = _to_tensor(data.get("edge_vectors"))
            edge_lengths: typing.Optional[torch.Tensor] = _to_tensor(data.get("edge_lengths"))
        else:
            from torch_geometric.nn import radius_graph
            edge_index = radius_graph(positions, r=self.cutoff, loop=False)
            row: torch.Tensor
            col: torch.Tensor
            row, col = edge_index
            edge_vectors = positions[col] - positions[row]
            edge_lengths = edge_vectors.norm(dim=-1)

        return AtomicGraph(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            edge_index=edge_index,
            edge_vectors=edge_vectors,
            edge_lengths=edge_lengths,
            energy=_to_tensor(data.get(self.energy_key)),
            forces=_to_tensor(data.get(self.forces_key)),
            stress=_to_tensor(data.get(self.stress_key)),
        )

    def close(self) -> None:
        """Close the LMDB environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self) -> None:
        self.close()
