"""GMD, the compositional-generalisation benchmark (arXiv:2605.08988), from the Hugging Face Hub.

Selected with ``data.dataset_type: gmd``.  The files a run needs are downloaded on first
use from https://huggingface.co/datasets/AmirMasoud/GMD into the Hugging Face cache, and
the graphs are cached under ``data.root``; no file is ever added by hand.

    data:
      dataset_type: gmd
      task: fragment_duplication   # a GMD task (snake_case or CamelCase), all, or custom
      format: traj                 # traj | lmdb | zarr | npz: identical labels
      seed: null                   # null: the paper's frame windows; int: random disjoint frames
      n_train: 1000                # frames per training molecule
      n_val: 200                   # validation frames per training molecule
      n_test: 200                  # test frames per molecule
      val_split: test_id           # GOAL's validation set: test_id (frames 1400-1599), val (1000-1199) or test_ood
      test_split: test_ood         # GOAL's test set: test_ood (the unseen molecules) or test_id
      train_molecules: null        # replace the task's molecules: "Family/Molecule" or a whole "Family"
      test_molecules: null
      extra_train_molecules: []    # or add to them, e.g. [CarboxylicAcids/O2C4H8]
      extra_test_molecules: []
      fractions: [0.8, 0.1, 0.1]   # task: all only
      revision: main               # Hub revision; a commit hash pins the data
      local_root: null             # a local copy of the repository: nothing is downloaded

Splits: GOAL ``train`` = GMD train, ``val`` = ``val_split``, ``test`` = ``test_split`` (by
default the unseen molecules); with ``task: all``, the shuffled frames split by ``fractions``.

Frame selection follows the dataset's own loader (``gmd.py`` in the repository):
``seed: null`` takes, from each training molecule, frames [0, n_train) for train,
[1000, 1000 + n_val) for val and [1400, 1400 + n_test) for test_id, and [1400, 1400 + n_test)
from each OOD molecule; ``seed: <int>`` draws ``n_train + n_val + n_test`` disjoint random
frames per training molecule and ``n_test`` per OOD molecule, with one generator over the
molecules in order, so a seed gives the same frames as the dataset's loader.  Frames are
non-periodic: the stored cell is the padding box of the sampling MD.
"""

from __future__ import annotations

import hashlib
import json
import re
import typing
import warnings
from pathlib import Path

import numpy as np
import torch

from goal.ml.data.benchmarks.base import BenchmarkDataset

REPO_ID = "AmirMasoud/GMD"
EXT = {"traj": ".traj", "lmdb": ".aselmdb", "zarr": ".zarr", "npz": ".npz"}
#: First frame of each window of the paper protocol.
STARTS = {"train": 0, "val": 1000, "test_id": 1400, "test_ood": 1400}
#: Isolated-atom energies (eV) at the labels' level of theory, by atomic number.
ATOM_REFS = {1: -13.595298, 6: -1028.481145, 7: -1483.895483, 8: -2041.117735}


def _series(family: str, formula: str, dh: int, lo: int, hi: int) -> list[str]:
    """Homologous series C_lo..C_hi with 2n + dh hydrogens, as ``Family/Molecule``."""
    return [f"{family}/" + formula.format(n=n, h=2 * n + dh) for n in range(lo, hi + 1)]


def alkanes(lo: int, hi: int) -> list[str]:
    return _series("Alkanes", "C{n}H{h}", 2, lo, hi)


def alcohols(lo: int, hi: int) -> list[str]:
    return _series("Alcohols", "OC{n}H{h}", 2, lo, hi)


def aldehydes(lo: int, hi: int) -> list[str]:
    return _series("Aldehydes", "OC{n}H{h}", 0, lo, hi)


def acids(lo: int, hi: int) -> list[str]:
    return _series("CarboxylicAcids", "O2C{n}H{h}", 0, lo, hi)


def diacids(lo: int, hi: int) -> list[str]:
    return _series("DiCarboxylicAcids", "O4C{n}H{h}", -2, lo, hi)


def diamines(lo: int, hi: int) -> list[str]:
    return _series("DiAmines", "N2C{n}H{h}", 4, lo, hi)


def amino_acids(lo: int, hi: int) -> list[str]:
    return _series("AminoAcids", "NC{n}O2H{h}", 1, lo, hi)


def amines(lo: int, hi: int) -> list[str]:
    return _series("Amines", "C{n}H{h}N", 3, lo, hi)


def amides(lo: int, hi: int) -> list[str]:
    return _series("Amides", "C{n}H{h}NO", 1, lo, hi)


#: Fragment Fusion's complex polyols and polycarbonyls (C7-C11).
COMPLEX = [
    f"Polyols/{m}"
    for m in (
        "C3OC3OC4H22",
        "C3OC6OCH22",
        "OC3OC3OC3OCH22",
        "OC3OC6OCOH22",
        "OC6OC3OCH22",
        "OC6OC4OH22",
        "OC7OH16",
        "OC8OH18",
        "OC9OH20",
        "OC10OH22",
        "OC11OH24",
    )
] + [
    f"KetoAldehydes/{m}"
    for m in (
        "C4OC5OC2H20",
        "COC10OH20",
        "COC2OC5OC2OH14",
        "COC2OC6OC2OH16",
        "COC2OCOC3OC2OH10",
        "COC2OCOC5OC2OH14",
        "COC2OCOC7H18",
        "COC8OC2OH18",
    )
]

#: task -> (training molecules, OOD test molecules): Table 1 of the paper, augmented
#: variants from its Appendix A.3.
TASKS: dict[str, tuple[list[str], list[str]]] = {
    "fragment_chain_extension": (alkanes(2, 6), alkanes(7, 13)),
    "fragment_fusion": (alcohols(4, 10) + aldehydes(4, 10) + COMPLEX, acids(4, 10)),
    "fragment_duplication": (acids(5, 10), diacids(5, 10)),
    "fragment_combination": (diamines(2, 9) + diacids(2, 9), amino_acids(2, 9)),
    "fragment_fission": (acids(5, 10), alcohols(5, 10) + aldehydes(5, 10)),
    "fragment_chain_extension_augmented": (
        alcohols(2, 3) + alcohols(9, 15) + acids(4, 8),
        alcohols(4, 8) + acids(2, 3) + acids(9, 15),
    ),
    "fragment_fusion_augmented": (
        alcohols(4, 10) + aldehydes(4, 10) + COMPLEX + amines(4, 8) + amides(4, 8),
        acids(4, 10),
    ),
}


def task_key(task: str) -> str:
    """``"FragmentFusion"`` or ``"fragment_fusion"`` -> ``"fragment_fusion"``; also all, custom."""
    key = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", str(task)).lower()
    if key not in (*TASKS, "all", "custom"):
        raise ValueError(f"unknown GMD task {task!r}; choose all, custom or one of {list(TASKS)}")
    return key


def family_molecules(family: str, fmt: str, root: Path | None, repo_id: str, revision: str):
    """Every molecule of a family, as ``Family/Molecule``, from a local copy or the Hub."""
    ext = EXT[fmt]
    if root is not None:
        names = sorted(p.name[: -len(ext)] for p in (root / fmt / family).glob(f"*{ext}"))
    else:
        from huggingface_hub import HfApi

        prefix = f"{fmt}/{family}/"
        files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
        heads = {f[len(prefix) :].split("/")[0] for f in files if f.startswith(prefix)}
        names = sorted(h[: -len(ext)] for h in heads if h.endswith(ext))
    if not names:
        raise ValueError(f"GMD has no molecules in family {family!r} ({fmt})")
    return [f"{family}/{n}" for n in names]


def download(molecules: list[str] | None, fmt: str, repo_id: str, revision: str) -> Path:
    """Fetch the ``fmt`` files of ``molecules`` (``None``: all) from the Hub; the local root."""
    from huggingface_hub import snapshot_download

    if molecules is None:
        patterns = [f"{fmt}/*"]
    else:
        patterns = [f"{fmt}/{m}{EXT[fmt]}*" for m in molecules]
    kw = {"repo_type": "dataset", "revision": revision, "allow_patterns": patterns}
    try:
        return Path(snapshot_download(repo_id, **kw))
    except Exception as online:  # noqa: BLE001 - no network (a compute node): use the cache
        try:
            return Path(snapshot_download(repo_id, local_files_only=True, **kw))
        except Exception:  # noqa: BLE001 - not cached either: report the network error
            raise RuntimeError(
                "GMD: cannot reach the Hugging Face Hub and the files are not cached; run "
                "once where there is network (e.g. a login node) or set data.local_root"
            ) from online


def read_frames(path: str | Path) -> list:
    """One molecule (.traj, .aselmdb, .zarr or .npz) as non-periodic labelled ``ase.Atoms``,
    sorted by trajectory frame, with ``info = {molecule, family, frame}``."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    path = Path(path)
    meta = {"molecule": path.name[: -len(path.suffix)], "family": path.parent.name}

    def labelled(numbers, positions, energy, forces, frame) -> Atoms:
        atoms = Atoms(numbers=numbers, positions=positions, pbc=False)
        atoms.info = {**meta, "frame": int(frame)}
        atoms.calc = SinglePointCalculator(atoms, energy=float(energy), forces=np.asarray(forces))
        return atoms

    frames: list[Atoms] = []
    if path.suffix == ".traj":
        from ase.io import read

        for a in read(path, ":"):
            frame = a.info.get("frame", len(frames))
            frames.append(
                labelled(a.numbers, a.positions, a.get_potential_energy(), a.get_forces(), frame)
            )
    elif path.suffix == ".aselmdb":
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        reader = ASELMDBReader(path)
        for i in range(len(reader)):
            row = reader.read(i)
            kv = row.get("key_value_pairs", {}) or {}
            frames.append(
                labelled(row["numbers"], row["positions"], row["energy"], row["forces"],
                         kv.get("frame", i))
            )  # fmt: skip
        reader.close()
    elif path.suffix == ".npz":
        d = np.load(path)
        for p, e, f, k in zip(d["pos"], d["energy"], d["forces"], d["frame"], strict=True):
            frames.append(labelled(d["z"], p, e, f, k))
    elif path.suffix == ".zarr":
        import zarr

        g = zarr.open_group(str(path), mode="r")
        numbers, frame = g["numbers"][:], g["frame"][:]
        positions, energy, forces = g["positions"][:], g["energy"][:], g["forces"][:]
        for i in range(len(frame)):
            frames.append(labelled(numbers, positions[i], energy[i], forces[i], frame[i]))
    else:
        raise ValueError(f"not a GMD file: {path}")
    return sorted(frames, key=lambda a: a.info["frame"])


class GMDDataset(BenchmarkDataset):
    """One split of a GMD task, downloaded from the Hugging Face Hub (see the module docstring)."""

    def __init__(
        self,
        root: str,
        cutoff: float,
        split: str = "train",
        task: str = "fragment_duplication",
        format: str = "traj",  # the dataset card's name for it
        seed: int | None = None,
        n_train: int = 1000,
        n_val: int = 200,
        n_test: int = 200,
        val_split: str = "test_id",
        test_split: str = "test_ood",
        train_molecules: typing.Sequence[str] | None = None,
        test_molecules: typing.Sequence[str] | None = None,
        extra_train_molecules: typing.Sequence[str] = (),
        extra_test_molecules: typing.Sequence[str] = (),
        fractions: typing.Sequence[float] = (0.8, 0.1, 0.1),
        repo_id: str = REPO_ID,
        revision: str = "main",
        local_root: str | None = None,
        dtype: torch.dtype = torch.float64,
        transform: typing.Callable[..., typing.Any] | None = None,
        compute_fragment_index: bool = False,
        fragment_covalent_cutoff: float = 1.8,
        fragment_scheme: str = "connected",
        fragment_charge: int = 0,
        fragment_on_failure: str = "fallback",
        fragment_smarts: typing.Sequence[str] | None = None,
        fragment_keep_groups: typing.Sequence[str] | None = None,
    ) -> None:
        if format not in EXT:
            raise ValueError(f"unknown GMD format {format!r}; choose one of {list(EXT)}")
        if val_split not in ("test_id", "val", "test_ood"):
            raise ValueError(f"val_split must be test_id, val or test_ood, got {val_split!r}")
        if test_split not in ("test_ood", "test_id"):
            raise ValueError(f"test_split must be test_ood or test_id, got {test_split!r}")
        self.task = task_key(task)
        if self.task == "custom" and not (train_molecules and test_molecules):
            raise ValueError("task: custom needs train_molecules and test_molecules")
        if self.task == "all" and not np.isclose(sum(fractions), 1.0):
            raise ValueError(f"fractions must sum to 1, got {list(fractions)}")
        if seed is None and (n_train > STARTS["val"] or STARTS["val"] + n_val > STARTS["test_id"]):
            raise ValueError(
                "the paper windows overlap at these sizes; set a seed for random splits"
            )
        self.format, self.seed = format, seed
        self.val_split, self.test_split = val_split, test_split
        self.sizes = {"train": int(n_train), "val": int(n_val), "test": int(n_test)}
        self.fractions = [float(f) for f in fractions]
        self.repo_id, self.revision = repo_id, revision
        self.local_root = None if local_root is None else Path(local_root)
        base_train, base_test = TASKS.get(self.task, ([], []))
        self.molecule_entries = {
            "train": list(train_molecules or base_train) + list(extra_train_molecules or ()),
            "test": list(test_molecules or base_test) + list(extra_test_molecules or ()),
        }
        super().__init__(
            root=root,
            cutoff=cutoff,
            split=split,
            dtype=dtype,
            transform=transform,
            compute_fragment_index=compute_fragment_index,
            fragment_covalent_cutoff=fragment_covalent_cutoff,
            fragment_scheme=fragment_scheme,
            fragment_charge=fragment_charge,
            fragment_on_failure=fragment_on_failure,
            fragment_smarts=fragment_smarts,
            fragment_keep_groups=fragment_keep_groups,
        )

    # -- what this split holds ------------------------------------------------

    def _gmd_split(self) -> str:
        """The GMD split behind this GOAL split."""
        if self.task == "all":
            return self.split
        return {"train": "train", "val": self.val_split, "test": self.test_split}[self.split]

    def _cache_name(self) -> str:
        spec = {
            "task": self.task,
            "format": self.format,
            "seed": self.seed,
            "sizes": self.sizes,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "molecules": self.molecule_entries,
            "fractions": self.fractions,
            "repo": self.repo_id,
            "revision": self.revision,
        }
        digest = hashlib.sha1(
            json.dumps(spec, sort_keys=True).encode(), usedforsecurity=False
        ).hexdigest()[:12]
        return f"gmd_{self.task}_{self._gmd_split()}_{digest}"

    def split_indices(self) -> dict[str, list[int]]:
        out = {s: [] for s in ("train", "val", "test")}
        out[self.split] = list(range(len(self._data)))
        return out

    def citation(self) -> str:
        return (
            "@inproceedings{nourollah2026gmd,\n"
            "  title     = {Benchmarking Compositional Generalisation for Machine Learning "
            "Interatomic Potentials},\n"
            "  author    = {Nourollah, Amir Masoud and Khalid, Irtaza and Leoni, Stefano and "
            "Schockaert, Steven},\n"
            "  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), "
            "Evaluations and Datasets Track},\n"
            "  year      = {2026},\n"
            "  eprint    = {2605.08988}\n"
            "}"
        )

    # -- download and selection -----------------------------------------------

    def _molecules(self, which: str) -> list[str]:
        out: list[str] = []
        for entry in self.molecule_entries[which]:
            if "/" in entry:
                out.append(entry)
            else:
                out += family_molecules(
                    entry, self.format, self.local_root, self.repo_id, self.revision
                )
        return list(dict.fromkeys(out))

    def _root(self, molecules: list[str] | None) -> Path:
        if self.local_root is not None:
            return self.local_root
        return download(molecules, self.format, self.repo_id, self.revision)

    def _file(self, root: Path, molecule: str) -> Path:
        return root / self.format / f"{molecule}{EXT[self.format]}"

    def selected_frames(self) -> list:
        """The labelled ``ase.Atoms`` of this split, in molecule order."""
        want = self._gmd_split()
        if self.task == "all":
            root = self._root(None)
            files = sorted((root / self.format).glob(f"*/*{EXT[self.format]}"))
            frames = [a for f in files for a in read_frames(f)]
            order = np.random.default_rng(0 if self.seed is None else self.seed).permutation(
                len(frames)
            )
            cuts = (np.cumsum(self.fractions)[:2] * len(frames)).astype(int)
            part = dict(zip(("train", "val", "test"), np.split(order, cuts), strict=True))
            return [frames[i] for i in part[want]]
        train, ood = self._molecules("train"), self._molecules("test")
        root = self._root(train + ood)
        n = self.sizes
        sizes_train = {"train": n["train"], "val": n["val"], "test_id": n["test"]}
        picked: list = []
        rng = np.random.default_rng(self.seed)
        for molecule in train + ood:
            sizes = sizes_train if molecule in train else {"test_ood": n["test"]}
            if self.seed is not None:  # replay the generator over every molecule, in order
                count = len(frames := read_frames(self._file(root, molecule)))
                perm = rng.permutation(count)
                cuts = np.cumsum([0, *sizes.values()])
                if want not in sizes:
                    continue
                lo, hi = (cuts[list(sizes).index(want)], cuts[list(sizes).index(want) + 1])
                chosen = [frames[i] for i in perm[lo:hi]]
            else:
                if want not in sizes:
                    continue
                frames = read_frames(self._file(root, molecule))
                start = STARTS[want]
                chosen = [a for a in frames if start <= a.info["frame"] < start + sizes[want]]
            if len(chosen) < sizes[want]:
                warnings.warn(
                    f"GMD {molecule}: only {len(chosen)} of {sizes[want]} frames for {want}",
                    stacklevel=2,
                )
            picked += chosen
        return picked

    def _download_and_process(self) -> list[typing.Any]:
        from goal.ml.data.graph import AtomicGraph

        return [
            AtomicGraph.from_ase(
                atoms,
                cutoff=self.cutoff,
                energy=atoms.get_potential_energy(),
                forces=atoms.get_forces(),
                dtype=self.dtype,
            )
            for atoms in self.selected_frames()
        ]
