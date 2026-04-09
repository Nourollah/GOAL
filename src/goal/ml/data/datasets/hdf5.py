"""HDF5 dataset — reads pre-processed atomic graphs from HDF5 files.

HDF5 is preferred over ExtXYZ for large datasets because it supports
random access without loading the entire file into memory.
"""

from __future__ import annotations

import typing
from pathlib import Path

import torch

from goal.ml.data.datasets.base import BaseAtomicDataset
from goal.ml.data.graph import AtomicGraph
from goal.ml.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("hdf5")
class HDF5Dataset(BaseAtomicDataset):
    """Dataset backed by an HDF5 file with pre-computed graphs.

    Each group in the HDF5 file represents one atomic structure with
    datasets for positions, atomic_numbers, cell, pbc, edge_index,
    edge_vectors, edge_lengths, and optional targets.

    Parameters
    ----------
    root : str or Path
        Directory containing HDF5 files, or a single ``.h5`` file.
    cutoff : float
        Cutoff radius used during preprocessing (stored as metadata).
    split : str
        Which split to load — looks for ``{split}.h5`` inside *root*.
    """

    def __init__(
        self,
        root: str | Path,
        cutoff: float,
        split: str = "train",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(root=root, cutoff=cutoff, split=split, dtype=dtype)
        self._file: typing.Any = None
        self._keys: list[str] = []
        self._open()

    def _open(self) -> None:
        """Open the HDF5 file and index its groups."""
        import h5py

        path: Path = self.root / f"{self.split}.h5"
        if not path.exists():
            path = self.root
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        self._file = h5py.File(str(path), "r")
        self._keys = list(self._file.keys())

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> AtomicGraph:
        grp: typing.Any = self._file[self._keys[idx]]

        def _tensor(name: str, dtype: torch.dtype = self.dtype) -> torch.Tensor | None:
            if name in grp:
                return torch.tensor(grp[name][()], dtype=dtype)
            return None

        return AtomicGraph(
            positions=_tensor("positions"),
            atomic_numbers=_tensor("atomic_numbers", dtype=torch.long),
            cell=_tensor("cell"),
            pbc=_tensor("pbc", dtype=torch.bool),
            edge_index=_tensor("edge_index", dtype=torch.long),
            edge_vectors=_tensor("edge_vectors"),
            edge_lengths=_tensor("edge_lengths"),
            energy=_tensor("energy"),
            forces=_tensor("forces"),
            stress=_tensor("stress"),
            weight=_tensor("weight"),
        )

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()
