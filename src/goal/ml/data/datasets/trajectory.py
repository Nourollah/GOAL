"""ASE Trajectory dataset — reads ``.traj`` files via ASE.

ASE trajectory files (``.traj``) are a common output format for molecular
dynamics simulations and geometry optimisations in ASE. This loader reads
them using ASE's ``ase.io.Trajectory`` reader and converts each frame
to an ``AtomicGraph``.
"""

from __future__ import annotations

import typing
from pathlib import Path

import torch

from goal.ml.data.datasets.base import BaseAtomicDataset
from goal.ml.data.graph import AtomicGraph
from goal.ml.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("trajectory")
class TrajectoryDataset(BaseAtomicDataset):
    """Dataset that reads ASE trajectory (``.traj``) files.

    Structures are read once at initialisation, converted to
    ``AtomicGraph``, and cached in memory.

    Parameters
    ----------
    root : str or Path
        Directory containing ``.traj`` files, or a single file.
    cutoff : float
        Cutoff radius for neighbour list construction (Angstrom).
    split : str
        Which split to load — looks for ``{split}.traj`` inside *root*.
    energy_key : str
        ASE ``atoms.info`` key for total energy.
        Falls back to ``atoms.get_potential_energy()`` if not in info.
    forces_key : str
        ASE ``atoms.arrays`` key for forces.
        Falls back to ``atoms.get_forces()`` if not in arrays.
    """

    def __init__(
        self,
        root: typing.Union[str, Path],
        cutoff: float,
        split: str = "train",
        energy_key: str = "energy",
        forces_key: str = "forces",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(root=root, cutoff=cutoff, split=split, dtype=dtype)
        self.energy_key: str = energy_key
        self.forces_key: str = forces_key
        self._graphs: typing.List[AtomicGraph] = []
        self._load()

    def _load(self) -> None:
        """Read the trajectory file and convert every frame."""
        from ase.io import Trajectory

        path: Path = self.root / f"{self.split}.traj"
        if not path.exists():
            path = self.root
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        traj: typing.Any = Trajectory(str(path), mode="r")

        for atoms in traj:
            # Extract energy
            energy: typing.Any = atoms.info.get(self.energy_key)
            if energy is None:
                try:
                    energy = atoms.get_potential_energy()
                except Exception:
                    energy = None

            # Extract forces
            forces: typing.Any = atoms.arrays.get(self.forces_key)
            if forces is None:
                try:
                    forces = atoms.get_forces()
                except Exception:
                    forces = None

            # Extract stress
            stress: typing.Any = atoms.info.get("stress")
            if stress is None:
                try:
                    stress = atoms.get_stress(voigt=False)
                except Exception:
                    stress = None

            graph: AtomicGraph = AtomicGraph.from_ase(
                atoms,
                cutoff=self.cutoff,
                energy=energy,
                forces=forces,
                stress=stress,
                dtype=self.dtype,
            )
            self._graphs.append(graph)

        traj.close()

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> AtomicGraph:
        return self._graphs[idx]
