"""ExtXYZ dataset — reads ``.xyz`` files via ASE and converts to ``AtomicGraph``.

ASE is used at the dataset boundary only. Once structures are loaded and
converted to ``AtomicGraph``, ASE is never touched again inside the
training loop.
"""

from __future__ import annotations

import typing
from pathlib import Path

import torch

from goal.ml.data.datasets.base import BaseAtomicDataset
from goal.ml.data.graph import AtomicGraph
from goal.ml.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("xyz")
class ExtXYZDataset(BaseAtomicDataset):
    """Dataset that reads extended XYZ files using ASE.

    Structures are read once at initialisation, converted to
    ``AtomicGraph``, and cached in memory. This trades RAM for
    zero per-epoch I/O overhead.

    Parameters
    ----------
    root : str or Path
        Directory containing ``.xyz`` files, or a single ``.xyz`` file.
    cutoff : float
        Cutoff radius for neighbour list construction (Angstrom).
    split : str
        Which split to load — used to find ``{split}.xyz`` inside *root*.
    energy_key : str
        Key in the ASE ``atoms.info`` dict that holds total energy.
    forces_key : str
        Key in ``atoms.arrays`` that holds per-atom forces.
    stress_key : str
        Key in ``atoms.info`` that holds the virial stress tensor.
    head : str or None
        Multihead training tag applied to every graph in this dataset.
    """

    def __init__(
        self,
        root: typing.Union[str, Path],
        cutoff: float,
        split: str = "train",
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
        head: typing.Optional[str] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(root=root, cutoff=cutoff, split=split, dtype=dtype)
        self.energy_key: str = energy_key
        self.forces_key: str = forces_key
        self.stress_key: str = stress_key
        self.head: typing.Optional[str] = head

        self._graphs: typing.List[AtomicGraph] = []
        self._load()

    def _load(self) -> None:
        """Read the XYZ file and convert every frame to ``AtomicGraph``."""
        from ase.io import read

        path: Path = self.root / f"{self.split}.xyz"
        if not path.exists():
            # Fall back to root itself if it's a file
            path = self.root
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        frames: typing.Any = read(str(path), index=":")
        if not isinstance(frames, list):
            frames = [frames]

        for atoms in frames:
            energy: typing.Any = atoms.info.get(self.energy_key)
            forces: typing.Any = atoms.arrays.get(self.forces_key)
            stress: typing.Any = atoms.info.get(self.stress_key)

            graph: AtomicGraph = AtomicGraph.from_ase(
                atoms,
                cutoff=self.cutoff,
                energy=energy,
                forces=forces,
                stress=stress,
                head=self.head,
                dtype=self.dtype,
            )
            self._graphs.append(graph)

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> AtomicGraph:
        return self._graphs[idx]
