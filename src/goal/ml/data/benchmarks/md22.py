"""MD22 — molecular dynamics trajectories for large molecules.

MD22 is the successor to MD17 from the same group: seven molecules from
42 to 370 atoms, an order of magnitude larger than MD17's, sampled from
AIMD at PBE+MBD level.  It exists because MD17's molecules are small
enough that a force field can memorise them; MD22 asks whether a model
scales to peptides, nucleic acids and supramolecular complexes.

File format is sGDML, identical to *original* MD17 (``z``, ``R``, ``E``,
``F`` in one ``.npz``), so the reader is shared with
:mod:`goal.ml.data.benchmarks.md17`.  The files also declare their own units —
``e_unit='kcal/mol'``, ``r_unit='Ang'`` — which this module checks rather
than assumes.

Data source: https://www.quantum-machine.org/gdml/repo/datasets/

Reference: Chmiela et al., "Accurate global machine learning force
fields for molecules with hundreds of atoms", Science Advances 9, 2023.
"""

from __future__ import annotations

import typing
from pathlib import Path

import numpy as np
import torch

from goal.ml.data.benchmarks.base import BenchmarkDataset
from goal.ml.data.benchmarks.md17 import KCAL_TO_EV, _download_file, _load_npz

#: Molecule → (file name, atom count, conformations in the file).
#: The counts are what the distributed files contain, verified on read.
MD22_MOLECULES: dict[str, tuple[str, int, int]] = {
    "Ac-Ala3-NHMe": ("md22_Ac-Ala3-NHMe.npz", 42, 85_109),
    "DHA": ("md22_DHA.npz", 56, 69_753),
    "stachyose": ("md22_stachyose.npz", 87, 27_272),
    "AT-AT": ("md22_AT-AT.npz", 60, 20_001),
    "AT-AT-CG-CG": ("md22_AT-AT-CG-CG.npz", 118, 10_153),
    "buckyball-catcher": ("md22_buckyball-catcher.npz", 148, 6_102),
    "double-walled_nanotube": ("md22_double-walled_nanotube.npz", 370, 5_032),
}

#: Training-set sizes used for the sGDML models in the MD22 paper.  They
#: are the closest thing this dataset has to a standard protocol, so a
#: number published against MD22 is almost always at one of these — hence
#: the per-molecule default rather than one size for all.  Override with
#: ``train_size``; nothing in the loader depends on these values.
MD22_TRAIN_SIZES: dict[str, int] = {
    "Ac-Ala3-NHMe": 6_000,
    "DHA": 8_000,
    "stachyose": 8_000,
    "AT-AT": 3_000,
    "AT-AT-CG-CG": 2_000,
    "buckyball-catcher": 600,
    "double-walled_nanotube": 800,
}

_MD22_BASE_URL: str = "https://www.quantum-machine.org/gdml/repo/datasets/"


class MD22Dataset(BenchmarkDataset):
    """MD22 — seven large-molecule AIMD trajectories at PBE+MBD.

    Units (after conversion):
        energy: eV
        forces: eV/Å

    Parameters
    ----------
    root : str
        Directory for downloaded and cached data.
    molecule : str
        One of :data:`MD22_MOLECULES`.
    cutoff : float
        Neighbour list cutoff in Ångström.
    split : str
        ``'train'``, ``'val'``, or ``'test'``.
    train_size : int, optional
        Number of training structures.  Default is the molecule's entry
        in :data:`MD22_TRAIN_SIZES`.
    val_size : int
        Number of validation structures.
    test_size : int, optional
        Cap on the test split.  ``None`` gives it every frame train
        and val did not take (Ac-Ala3-NHMe leaves ~78,600 frames).
    seed : int
        Random seed for split reproducibility.
    **fragment_kwargs
        Fragment-label options forwarded to :class:`BenchmarkDataset`
        (``compute_fragment_index``, ``fragment_scheme``, ...).  An MD22
        trajectory is one molecule, so ``fragment_scheme: connected``
        returns a single fragment per frame; ``rotatable`` cuts it into
        chemical groups, and because perception is cached on the bond
        graph — which an MD trajectory does not change — the whole
        trajectory costs one decomposition.

    Notes
    -----
    These molecules are large.  ``double-walled_nanotube`` is 370 atoms,
    which at a 5 Å cutoff is tens of thousands of edges per structure —
    size the batch by edge count, not by structure count.
    """

    def __init__(
        self,
        root: str,
        molecule: str = "Ac-Ala3-NHMe",
        cutoff: float = 5.0,
        split: str = "train",
        train_size: int | None = None,
        val_size: int = 500,
        test_size: int | None = None,
        seed: int = 42,
        dtype: torch.dtype = torch.float64,
        transform: typing.Callable[..., typing.Any] | None = None,
        **fragment_kwargs: typing.Any,
    ) -> None:
        if molecule not in MD22_MOLECULES:
            raise ValueError(
                f"Unknown MD22 molecule '{molecule}'. Available: {sorted(MD22_MOLECULES)}"
            )
        self.molecule: str = molecule
        self.train_size: int = (
            int(train_size) if train_size is not None else MD22_TRAIN_SIZES[molecule]
        )
        self.val_size: int = int(val_size)
        self.test_size: int | None = int(test_size) if test_size is not None else None
        self.seed: int = seed
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        super().__init__(
            root=root,
            cutoff=cutoff,
            split=split,
            dtype=dtype,
            transform=transform,
            **fragment_kwargs,
        )

    def _cache_name(self) -> str:
        return f"md22_{self.molecule}"

    def _get_npz(self) -> Path:
        """Download this molecule's ``.npz`` if not already cached."""
        raw_dir: Path = self.root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        filename: str = MD22_MOLECULES[self.molecule][0]
        npz_path: Path = raw_dir / filename
        if npz_path.exists():
            return npz_path
        url: str = _MD22_BASE_URL + filename
        print(f"Downloading MD22 {self.molecule} from {url}")
        _download_file(url, npz_path)
        return npz_path

    def _download_and_process(self) -> list[typing.Any]:
        from goal.ml.data.graph import AtomicGraph
        from goal.ml.data.neighbor_list import build_neighbor_list_from_tensors

        npz_path: Path = self._get_npz()
        self._check_units(npz_path)
        # MD22 ships in the same sGDML layout as original (non-revised)
        # MD17 — ``E`` is (n_frames, 1) here, which ``_load_npz`` flattens.
        z_np, R_np, E_np, F_np = _load_npz(npz_path, revised=False)

        expected_atoms: int = MD22_MOLECULES[self.molecule][1]
        if z_np.shape[0] != expected_atoms:
            raise ValueError(
                f"{npz_path.name} has {z_np.shape[0]} atoms, expected "
                f"{expected_atoms} for '{self.molecule}' — wrong or corrupt file."
            )

        z: torch.Tensor = torch.from_numpy(z_np)
        cell: torch.Tensor = torch.zeros(3, 3, dtype=self.dtype)
        pbc: torch.Tensor = torch.zeros(3, dtype=torch.bool)

        graphs: list[AtomicGraph] = []
        for i in range(len(E_np)):
            positions: torch.Tensor = torch.from_numpy(R_np[i]).to(self.dtype)
            nl = build_neighbor_list_from_tensors(
                positions=positions,
                atomic_numbers=z,
                cell=cell,
                pbc=pbc,
                cutoff=self.cutoff,
                backend="ase",
                dtype=self.dtype,
            )
            graphs.append(
                AtomicGraph(
                    positions=positions,
                    atomic_numbers=z,
                    cell=cell,
                    pbc=pbc,
                    edge_index=nl.edge_index,
                    energy=torch.tensor([E_np[i] * KCAL_TO_EV], dtype=self.dtype),
                    forces=torch.from_numpy(F_np[i]).to(self.dtype) * KCAL_TO_EV,
                )
            )
        return graphs

    @staticmethod
    def _check_units(npz_path: Path) -> None:
        """Verify the file's declared units are the ones we convert from.

        MD22 records ``e_unit`` / ``r_unit`` in the archive, so the
        kcal/mol → eV conversion can be checked instead of assumed.  A
        silently mis-scaled energy is the kind of error that trains
        happily and reports plausible numbers.
        """
        data = np.load(npz_path, allow_pickle=True)
        energy_unit: str = str(data["e_unit"]) if "e_unit" in data else "kcal/mol"
        length_unit: str = str(data["r_unit"]) if "r_unit" in data else "Ang"
        if energy_unit.lower() != "kcal/mol" or not length_unit.lower().startswith("ang"):
            raise ValueError(
                f"{npz_path.name} declares units ({energy_unit}, {length_unit}); "
                f"this loader converts kcal/mol and Angstrom. Check the file."
            )

    def split_indices(self) -> dict[str, list[int]]:
        """Reproducible random split, MD17-style (train / val / rest)."""
        total: int = len(self._data)
        if self.train_size + self.val_size > total:
            raise ValueError(
                f"train_size ({self.train_size}) + val_size ({self.val_size}) "
                f"exceeds the {total} conformations in "
                f"'{self.molecule}'. Lower them or pick a larger molecule."
            )
        return self._random_split_indices(
            total=total,
            train_size=self.train_size,
            val_size=self.val_size,
            seed=self.seed,
            test_size=self.test_size,
        )

    def citation(self) -> str:
        """Return BibTeX for the MD22 paper (Chmiela et al., *Science Advances*, 2023)."""
        return (
            "@article{chmiela2023accurate,\n"
            "  title={Accurate global machine learning force fields for "
            "molecules with hundreds of atoms},\n"
            "  author={Chmiela, Stefan and Vassilev-Galindo, Valentin and "
            "Unke, Oliver T and Kabylda, Adil and Sauceda, Huziel E and "
            'Tkatchenko, Alexandre and M{\\"u}ller, Klaus-Robert},\n'
            "  journal={Science Advances},\n"
            "  volume={9},\n"
            "  number={2},\n"
            "  year={2023}\n"
            "}"
        )
