"""MD17 and revised MD17 (rMD17) molecular dynamics datasets.

MD17 contains molecular dynamics trajectories for small organic
molecules.  rMD17 is the revised version with more accurate DFT
energies and forces.

Data sources:
  MD17:  https://www.quantum-machine.org/gdml/data/npz/
  rMD17: https://figshare.com/ndownloader/files/23950376 (primary)
         https://archive.materialscloud.org/records/pfffs-fff86/files/rmd17.tar.bz2 (mirror)
"""

from __future__ import annotations

import shutil
import tarfile
import typing
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

from goal.ml.data.benchmarks.base import BenchmarkDataset

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

# Original MD17: individual .npz files per molecule
_MD17_FILENAMES: dict[str, str] = {
    "aspirin": "md17_aspirin.npz",
    "benzene": "md17_benzene2017.npz",
    "ethanol": "md17_ethanol.npz",
    "malonaldehyde": "md17_malonaldehyde.npz",
    "naphthalene": "md17_naphthalene.npz",
    "salicylic_acid": "md17_salicylic.npz",
    "toluene": "md17_toluene.npz",
    "uracil": "md17_uracil.npz",
}
_MD17_BASE_URL: str = "https://www.quantum-machine.org/gdml/data/npz/"

# rMD17: single tar.bz2 archive; primary + mirror URLs
_RMD17_URLS: list[str] = [
    "https://figshare.com/ndownloader/files/23950376",
    "https://archive.materialscloud.org/records/pfffs-fff86/files/rmd17.tar.bz2?download=1",
]
_RMD17_FILENAMES: dict[str, str] = {
    "aspirin": "rmd17_aspirin.npz",
    "azobenzene": "rmd17_azobenzene.npz",
    "benzene": "rmd17_benzene.npz",
    "ethanol": "rmd17_ethanol.npz",
    "malonaldehyde": "rmd17_malonaldehyde.npz",
    "naphthalene": "rmd17_naphthalene.npz",
    "paracetamol": "rmd17_paracetamol.npz",
    "salicylic_acid": "rmd17_salicylic.npz",
    "toluene": "rmd17_toluene.npz",
    "uracil": "rmd17_uracil.npz",
}

# kcal/mol → eV
KCAL_TO_EV: float = 0.043364


def _download_file(url: str, dest: Path, timeout: int = 600) -> None:
    """Download ``url`` (HTTPS only) to ``dest``, rejecting HTML error pages."""
    if not url.startswith("https://"):
        raise ValueError(f"Refusing to download from a non-HTTPS URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    # nosec B310: the scheme is restricted to https:// just above.
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:  # nosec B310
        content_type: str = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise RuntimeError(f"Got HTML response from {url} — URL may have changed")
        shutil.copyfileobj(resp, f)
    if dest.stat().st_size == 0:
        dest.unlink()
        raise RuntimeError(f"Downloaded empty file from {url}")


def _load_npz(path: Path, revised: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (atomic_numbers, positions, energies, forces) arrays from .npz."""
    data = np.load(path)
    if revised:
        z = data["nuclear_charges"].astype(np.int64)  # (n_atoms,)
        R = data["coords"].astype(np.float64)  # (n_frames, n_atoms, 3)
        E = data["energies"].astype(np.float64)  # (n_frames,)
        F = data["forces"].astype(np.float64)  # (n_frames, n_atoms, 3)
    else:
        z = data["z"].astype(np.int64)  # (n_atoms,)
        R = data["R"].astype(np.float64)  # (n_frames, n_atoms, 3)
        E = data["E"].astype(np.float64).reshape(-1)  # (n_frames,)
        F = data["F"].astype(np.float64)  # (n_frames, n_atoms, 3)
    return z, R, E, F


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
        One of ``MD17_MOLECULES`` or ``REVISED_MD17_MOLECULES``.
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
    test_size : int, optional
        Cap on the test split.  ``None`` gives it every frame train
        and val did not take (rMD17 aspirin leaves ~99,000 frames).
    seed : int
        Random seed for split reproducibility.
    **fragment_kwargs
        Fragment-label options forwarded to
        :class:`~goal.ml.data.benchmarks.base.BenchmarkDataset`
        (``compute_fragment_index``, ``fragment_scheme``, ...).  An MD17
        trajectory is one molecule, so ``fragment_scheme: connected``
        returns a single fragment for every frame; ``rotatable`` cuts it
        into chemical groups, and because perception is cached on the
        bond graph — which an MD trajectory does not change — the whole
        trajectory costs one decomposition.
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
        test_size: int | None = None,
        seed: int = 42,
        dtype: torch.dtype = torch.float64,
        transform: typing.Callable[..., typing.Any] | None = None,
        **fragment_kwargs: typing.Any,
    ) -> None:
        allowed: list[str] = REVISED_MD17_MOLECULES if revised else MD17_MOLECULES
        if molecule not in allowed:
            raise ValueError(f"Unknown molecule '{molecule}'. Available: {allowed}")
        self.molecule: str = molecule
        self.revised: bool = revised
        self.train_size: int = train_size
        self.val_size: int = val_size
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
        prefix: str = "rmd17" if self.revised else "md17"
        return f"{prefix}_{self.molecule}"

    def _get_npz(self) -> Path:
        """Download the .npz for this molecule if not already cached, return its path."""
        raw_dir: Path = self.root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        if self.revised:
            npz_name = _RMD17_FILENAMES[self.molecule]
            npz_path = raw_dir / npz_name
            if npz_path.exists():
                return npz_path

            # Download full rMD17 archive (try primary then mirror)
            tar_path = raw_dir / "rmd17.tar.bz2"
            if not tar_path.exists():
                last_err: Exception = RuntimeError("No URLs tried")
                for url in _RMD17_URLS:
                    try:
                        print(f"Downloading rMD17 from {url}")
                        _download_file(url, tar_path)
                        break
                    except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as e:
                        last_err = e
                        print(f"  Failed ({e}), trying next mirror...")
                        if tar_path.exists():
                            tar_path.unlink()
                else:
                    raise RuntimeError(
                        f"All rMD17 URLs failed. Last error: {last_err}"
                    ) from last_err

            # Extract all .npz files from the archive
            print("Extracting rMD17 archive...")
            with tarfile.open(tar_path, "r:bz2") as tf:
                for member in tf.getmembers():
                    if member.name.endswith(".npz"):
                        fname = Path(member.name).name
                        out = raw_dir / fname
                        if not out.exists():
                            src = tf.extractfile(member)
                            if src is not None:
                                with open(out, "wb") as f:
                                    shutil.copyfileobj(src, f)

            if not npz_path.exists():
                raise FileNotFoundError(
                    f"{npz_name} not found in rMD17 archive. "
                    f"Available files in archive may differ from expected names."
                )
            return npz_path

        else:
            npz_name = _MD17_FILENAMES[self.molecule]
            npz_path = raw_dir / npz_name
            if npz_path.exists():
                return npz_path
            url = _MD17_BASE_URL + npz_name
            print(f"Downloading MD17 {self.molecule} from {url}")
            _download_file(url, npz_path)
            return npz_path

    def _download_and_process(self) -> list[typing.Any]:
        from goal.ml.data.graph import AtomicGraph
        from goal.ml.data.neighbor_list import build_neighbor_list_from_tensors

        npz_path = self._get_npz()
        z_np, R_np, E_np, F_np = _load_npz(npz_path, self.revised)

        z = torch.from_numpy(z_np)
        cell = torch.zeros(3, 3, dtype=self.dtype)
        pbc = torch.zeros(3, dtype=torch.bool)

        graphs: list[AtomicGraph] = []
        for i in range(len(E_np)):
            positions = torch.from_numpy(R_np[i]).to(self.dtype)
            energy_ev = torch.tensor([E_np[i] * KCAL_TO_EV], dtype=self.dtype)
            forces_ev = torch.from_numpy(F_np[i]).to(self.dtype) * KCAL_TO_EV

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
                    edge_vectors=nl.edge_vectors,
                    edge_lengths=nl.edge_lengths,
                    energy=energy_ev,
                    forces=forces_ev,
                )
            )
        return graphs

    def split_indices(self) -> dict[str, list[int]]:
        """Reproducible random split following MD17 benchmark protocol."""
        return self._random_split_indices(
            total=len(self._data),
            train_size=self.train_size,
            val_size=self.val_size,
            seed=self.seed,
            test_size=self.test_size,
        )

    def citation(self) -> str:
        """Return BibTeX for the MD17 paper (Chmiela et al., *Science Advances*, 2017).

        The same entry is returned for ``revised=True`` (rMD17).
        """
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
