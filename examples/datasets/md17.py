"""MD17 and revised MD17 (rMD17) molecular dynamics datasets.

MD17 contains molecular dynamics trajectories for small organic
molecules.  rMD17 is the revised version with more accurate DFT
energies and forces.  Both sourced via ``torch_geometric.datasets.MD17``.

Data source: `<https://sgdml.org/>`_
"""

from __future__ import annotations

import typing
from pathlib import Path

import torch

from examples.datasets.base import BenchmarkDataset

MD17_MOLECULES: list[str] = [
    "aspirin",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "salicylic_acid",
    "toluene",
    "uracil",
]

REVISED_MD17_MOLECULES: list[str] = [
    "aspirin",
    "azobenzene",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "paracetamol",
    "salicylic_acid",
    "toluene",
    "uracil",
]

# kcal/mol → eV
KCAL_TO_EV: float = 0.043364


class MD17Dataset(BenchmarkDataset):
    """MD17 and revised MD17 (rMD17) molecular dynamics datasets.

    MD17: CCSD(T)/cc-pVTZ energies and forces for 8 small organic
    molecules from short MD trajectories at 500 K.

    rMD17: Recomputed at PBE/def2-SVP level with more consistent
    reference frame.  More suitable for benchmarking MLFFs.

    Standard splits used in the MLFF literature:
        train: 950 structures (following Schütt et al. 2017)
        val:   50 structures
        test:  remaining (~9000 structures)

    Units (after conversion):
        energy: eV
        forces: eV/Å

    Parameters
    ----------
    root : str
        Directory for downloaded and cached data.
    molecule : str
        One of :data:`MD17_MOLECULES` or :data:`REVISED_MD17_MOLECULES`.
    revised : bool
        If ``True``, use rMD17; if ``False``, use original MD17.
    cutoff : float
        Neighbour list cutoff in Ångström.
    split : str
        ``'train'``, ``'val'``, or ``'test'``.
    train_size : int
        Number of training structures.
    val_size : int
        Number of validation structures.
    seed : int
        Random seed for split reproducibility.
    """

    def __init__(
        self,
        root: str,
        molecule: str = "aspirin",
        revised: bool = True,
        cutoff: float = 5.0,
        split: str = "train",
        train_size: int = 950,
        val_size: int = 50,
        seed: int = 42,
        dtype: torch.dtype = torch.float64,
        transform: typing.Callable[..., typing.Any] | None = None,
    ) -> None:
        allowed: list[str] = REVISED_MD17_MOLECULES if revised else MD17_MOLECULES
        if molecule not in allowed:
            raise ValueError(f"Unknown molecule '{molecule}'. " f"Available: {allowed}")
        self.molecule: str = molecule
        self.revised: bool = revised
        self.train_size: int = train_size
        self.val_size: int = val_size
        self.seed: int = seed
        super().__init__(root=root, cutoff=cutoff, split=split, dtype=dtype, transform=transform)

    def _cache_name(self) -> str:
        prefix: str = "rmd17" if self.revised else "md17"
        return f"{prefix}_{self.molecule}"

    def _download_and_process(self) -> list[typing.Any]:
        """Download via PyG and convert to AtomicGraph list."""
        from torch_geometric.datasets import MD17 as PyGMD17

        from goal.ml.data.graph import AtomicGraph

        name: str = f"revised {self.molecule}" if self.revised else self.molecule
        pyg_dataset = PyGMD17(root=str(self.root / "raw"), name=name)

        graphs: list[AtomicGraph] = []
        for data in pyg_dataset:
            positions: torch.Tensor = data.pos.to(self.dtype)
            atomic_numbers: torch.Tensor = data.z.long()

            energy_kcal: float = data.energy.item()
            forces_kcal: torch.Tensor = data.force.to(self.dtype)

            energy_ev: torch.Tensor = torch.tensor([energy_kcal * KCAL_TO_EV], dtype=self.dtype)
            forces_ev: torch.Tensor = forces_kcal * KCAL_TO_EV

            from torch_geometric.nn import radius_graph

            edge_index: torch.Tensor = radius_graph(positions, r=self.cutoff, loop=False)
            row, col = edge_index
            edge_vecs: torch.Tensor = positions[col] - positions[row]
            edge_lens: torch.Tensor = edge_vecs.norm(dim=-1)

            graph: AtomicGraph = AtomicGraph(
                positions=positions,
                atomic_numbers=atomic_numbers,
                cell=torch.zeros(3, 3, dtype=self.dtype),
                pbc=torch.zeros(3, dtype=torch.bool),
                edge_index=edge_index,
                edge_vectors=edge_vecs,
                edge_lengths=edge_lens,
                energy=energy_ev,
                forces=forces_ev,
            )
            graphs.append(graph)
        return graphs

    def split_indices(self) -> dict[str, list[int]]:
        """Reproducible random split following MD17 benchmark protocol."""
        return self._random_split_indices(
            total=len(self._data),
            train_size=self.train_size,
            val_size=self.val_size,
            seed=self.seed,
        )

    def citation(self) -> str:
        return (
            "@article{chmiela2017machine,\n"
            "  title={Machine learning of accurate energy-conserving "
            "molecular force fields},\n"
            "  author={Chmiela, Stefan and Tkatchenko, Alexandre and "
            "Sauceda, Huziel E and Poltavsky, Igor and "
            'Sch{\\"u}tt, Kristof T and M{\\"u}ller, Klaus-Robert},\n'
            "  journal={Science advances},\n"
            "  year={2017}\n"
            "}"
        )
