"""SPICE dataset for drug-like molecules and protein fragments.

~1.1M conformations of small molecules and protein fragments at
ωB97M-D3BJ/def2-TZVPPD level — higher quality than ANI-1.
Contains elements H, C, N, O, F, P, S, Cl, Br, I.

SPICE 1.0: original release (~1.1M structures)
SPICE 2.0: extended with more protein fragment diversity

Data source: downloaded from Zenodo as HDF5, parsed using ``h5py``.
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

# SPICE Zenodo download URLs
_SPICE_URLS: typing.Dict[str, str] = {
    "1": "https://zenodo.org/records/8222043/files/SPICE-1.1.4.hdf5",
    "2": "https://zenodo.org/records/10835749/files/SPICE-2.0.1.hdf5",
}

_SPICE_SUBSETS: typing.Dict[str, typing.List[str]] = {
    "small_molecules": ["Solvated Amino Acids", "Dipeptides", "PubChem"],
    "amino_acids": ["Solvated Amino Acids"],
    "dipeptides": ["Dipeptides"],
}

# Element symbol → atomic number (for SPICE parsing)
_ELEMENT_TO_Z: typing.Dict[str, int] = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15,
    "S": 16, "Cl": 17, "Br": 35, "I": 53,
}


class SPICEDataset(BenchmarkDataset):
    """SPICE dataset for drug-like molecules and protein fragments.

    ~1.1M conformations at ωB97M-D3BJ/def2-TZVPPD level.
    Higher quality than ANI-1, more representative of pharmaceutical
    chemistry.

    Units (after conversion):
        energy: eV (from Hartree)
        forces: eV/Å (from Hartree/Bohr)

    Parameters
    ----------
    root : str
        Directory for downloaded and cached data.
    version : str
        ``'1'`` or ``'2'``.
    subset : str
        Which subset to use: ``'all'``, ``'small_molecules'``,
        ``'amino_acids'``, or ``'dipeptides'``.
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
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        root: str,
        version: str = "1",
        subset: str = "all",
        cutoff: float = 5.0,
        split: str = "train",
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        max_structures: typing.Optional[int] = None,
        seed: int = 42,
        dtype: torch.dtype = torch.float64,
        transform: typing.Optional[typing.Callable[..., typing.Any]] = None,
    ) -> None:
        if version not in _SPICE_URLS:
            raise ValueError(f"version must be '1' or '2', got '{version}'")
        if subset not in ("all", *_SPICE_SUBSETS):
            raise ValueError(
                f"subset must be 'all', 'small_molecules', 'amino_acids', "
                f"or 'dipeptides', got '{subset}'"
            )
        self.version: str = version
        self.subset: str = subset
        self.train_fraction: float = train_fraction
        self.val_fraction: float = val_fraction
        self.max_structures: typing.Optional[int] = max_structures
        self.seed: int = seed
        super().__init__(root=root, cutoff=cutoff, split=split, dtype=dtype, transform=transform)

    def _cache_name(self) -> str:
        name: str = f"spice{self.version}_{self.subset}"
        if self.max_structures is not None:
            name += f"_max{self.max_structures}"
        return name

    def _download_hdf5(self) -> Path:
        """Download SPICE HDF5 file if not already present."""
        import requests

        url: str = _SPICE_URLS[self.version]
        filename: str = url.rsplit("/", 1)[-1]
        raw_dir: Path = self.root / "raw"
        raw_dir.mkdir(exist_ok=True)
        filepath: Path = raw_dir / filename

        if filepath.exists():
            return filepath

        print(f"Downloading SPICE {self.version} from {url}...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return filepath

    def _download_and_process(self) -> typing.List[typing.Any]:
        """Download HDF5 and convert to AtomicGraph list."""
        import h5py
        import numpy as np

        from torch_geometric.nn import radius_graph

        from gmd.data.graph import AtomicGraph

        hdf5_path: Path = self._download_hdf5()
        graphs: typing.List[AtomicGraph] = []

        with h5py.File(hdf5_path, "r") as f:
            for group_name in f:
                # Subset filter
                if self.subset != "all":
                    allowed_groups: typing.List[str] = _SPICE_SUBSETS[self.subset]
                    if not any(ag in group_name for ag in allowed_groups):
                        continue

                group = f[group_name]
                n_conformations: int = group["conformations"].shape[0]

                for i in range(n_conformations):
                    positions_np = group["conformations"][i]
                    atomic_nums_np = group["atomic_numbers"][:]

                    positions: torch.Tensor = torch.tensor(
                        positions_np, dtype=self.dtype
                    )
                    atomic_numbers: torch.Tensor = torch.tensor(
                        atomic_nums_np, dtype=torch.long
                    )

                    # Energy: Hartree → eV
                    energy_ev: typing.Optional[torch.Tensor] = None
                    if "dft_total_energy" in group:
                        e_hartree: float = float(group["dft_total_energy"][i])
                        energy_ev = torch.tensor(
                            [e_hartree * HARTREE_TO_EV], dtype=self.dtype
                        )

                    # Forces: Hartree/Bohr → eV/Å
                    forces_ev: typing.Optional[torch.Tensor] = None
                    if "dft_total_gradient" in group:
                        grad = group["dft_total_gradient"][i]
                        forces_ev = torch.tensor(
                            -grad * (HARTREE_TO_EV / BOHR_TO_ANGSTROM),
                            dtype=self.dtype,
                        )

                    edge_index: torch.Tensor = radius_graph(
                        positions, r=self.cutoff, loop=False
                    )
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

                    if (
                        self.max_structures is not None
                        and len(graphs) >= self.max_structures
                    ):
                        return graphs

        return graphs

    def split_indices(self) -> typing.Dict[str, typing.List[int]]:
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
        return (
            "@article{eastman2023spice,\n"
            "  title={SPICE, A Dataset of Drug-like Molecules and Peptides "
            "for Training Machine Learning Potentials},\n"
            "  author={Eastman, Peter and Behara, Pavan Kumar and "
            "Dotson, David L and Galvelis, Raimondas and Herr, "
            "John E and Horton, Josh T and Mao, Yuezhi and "
            "Chodera, John D and Pritchard, Benjamin P and "
            "Wang, Yuanqing and others},\n"
            "  journal={Scientific Data},\n"
            "  year={2023}\n"
            "}"
        )
