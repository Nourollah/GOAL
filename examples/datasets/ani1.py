"""ANI-1 and ANI-1x datasets for organic molecules.

ANI-1: ~20 million conformations of drug-like organic molecules
containing H, C, N, O at ωB97x/6-31G* level of theory.

ANI-1x: Extended ANI-1 with active learning, ~5M structures,
higher quality reference data.  Preferred for benchmarking.

Data source: ``torch_geometric.datasets.ANI1``
"""

from __future__ import annotations

import typing
from pathlib import Path

import torch

from examples.datasets.base import BenchmarkDataset

# Hartree → eV
HARTREE_TO_EV: float = 27.211396
# Bohr → Ångström
BOHR_TO_ANGSTROM: float = 0.529177


class ANI1Dataset(BenchmarkDataset):
    """ANI-1 and ANI-1x datasets for organic molecules.

    ANI-1: ~20M conformations at ωB97x/6-31G* level.
    ANI-1x: ~5M structures with active learning, higher quality.

    Note on size: ANI-1 is ~30 GB, ANI-1x is ~7 GB.
    First run takes significant time to download and process.
    Processed cache is saved as ``.pt`` files.

    Units (after conversion):
        energy: eV (from Hartree)
        forces: eV/Å (from Hartree/Bohr)

    Parameters
    ----------
    root : str
        Directory for downloaded and cached data.
    version : str
        ``'1'`` for ANI-1, ``'1x'`` for ANI-1x.
    cutoff : float
        Neighbour list cutoff in Ångström.
    split : str
        ``'train'``, ``'val'``, or ``'test'``.
    train_fraction : float
        Fraction of data for training.
    val_fraction : float
        Fraction of data for validation.
    max_structures : int or None
        If set, subsample to this many total structures.
    elements : list[int] or None
        If set, filter to structures containing only these atomic numbers.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        root: str,
        version: str = "1x",
        cutoff: float = 5.0,
        split: str = "train",
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        max_structures: int | None = None,
        elements: list[int] | None = None,
        seed: int = 42,
        dtype: torch.dtype = torch.float64,
        transform: typing.Callable[..., typing.Any] | None = None,
    ) -> None:
        if version not in ("1", "1x"):
            raise ValueError(f"version must be '1' or '1x', got '{version}'")
        self.version: str = version
        self.train_fraction: float = train_fraction
        self.val_fraction: float = val_fraction
        self.max_structures: int | None = max_structures
        self.elements: list[int] | None = elements
        self.seed: int = seed
        super().__init__(root=root, cutoff=cutoff, split=split, dtype=dtype, transform=transform)

    def _cache_name(self) -> str:
        name: str = f"ani{self.version}"
        if self.max_structures is not None:
            name += f"_max{self.max_structures}"
        if self.elements is not None:
            elems: str = "_".join(str(e) for e in sorted(self.elements))
            name += f"_elems{elems}"
        return name

    def _download_and_process(self) -> list[typing.Any]:
        """Download via PyG and convert to AtomicGraph list."""
        from goal.ml.data.graph import AtomicGraph

        if self.version == "1x":
            from torch_geometric.datasets import ANI1x as PyGDataset

            pyg_dataset = PyGDataset(root=str(self.root / "raw"))
        else:
            from torch_geometric.datasets import ANI1 as PyGDataset

            pyg_dataset = PyGDataset(root=str(self.root / "raw"))

        from torch_geometric.nn import radius_graph

        graphs: list[AtomicGraph] = []
        for data in pyg_dataset:
            atomic_numbers: torch.Tensor = data.z.long()

            # Element filter
            if self.elements is not None:
                allowed = set(self.elements)
                if not all(z.item() in allowed for z in atomic_numbers):
                    continue

            positions: torch.Tensor = data.pos.to(self.dtype)

            # Energy: Hartree → eV
            energy_ev: torch.Tensor | None = None
            if hasattr(data, "energy") and data.energy is not None:
                energy_ev = torch.tensor([data.energy.item() * HARTREE_TO_EV], dtype=self.dtype)

            # Forces: Hartree/Bohr → eV/Å
            forces_ev: torch.Tensor | None = None
            if hasattr(data, "force") and data.force is not None:
                forces_ev = data.force.to(self.dtype) * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)

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

            # Subsample limit
            if self.max_structures is not None and len(graphs) >= self.max_structures:
                break

        return graphs

    def split_indices(self) -> dict[str, list[int]]:
        """Reproducible fraction-based split."""
        total: int = len(self._data)
        train_size: int = int(total * self.train_fraction)
        val_size: int = int(total * self.val_fraction)
        return self._random_split_indices(
            total=total,
            train_size=train_size,
            val_size=val_size,
            seed=self.seed,
        )

    def citation(self) -> str:
        if self.version == "1x":
            return (
                "@article{smith2020ani1x,\n"
                "  title={The ANI-1ccx and ANI-1x data sets, coupled-cluster "
                "and density functional theory properties for molecules},\n"
                "  author={Smith, Justin S and Zubatyuk, Roman and Nebgen, "
                "Benjamin and Lubbers, Nicholas and Barros, Kipton and "
                "Roitberg, Adrian E and Isayev, Olexandr and Tretiak, Sergei},\n"
                "  journal={Scientific data},\n"
                "  year={2020}\n"
                "}"
            )
        return (
            "@article{smith2017ani,\n"
            "  title={ANI-1: an extensible neural network potential with "
            "DFT accuracy at force field computational cost},\n"
            "  author={Smith, Justin S and Isayev, Olexandr and "
            "Roitberg, Adrian E},\n"
            "  journal={Chemical science},\n"
            "  year={2017}\n"
            "}"
        )
