"""``dataset_type: gmd``: GMD from the Hugging Face Hub, on a local copy of the repository."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from goal.ml.data.benchmarks.gmd import STARTS, TASKS, GMDDataset, read_frames, task_key

N_FRAMES = 1450  # enough for every paper window at the small sizes below
SIZES = {"n_train": 5, "n_val": 3, "n_test": 4}


def _write_repo(root, fmt: str) -> None:
    """A tiny repository in the published layout: two acids, one diacid, frames shuffled."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    rng = np.random.default_rng(0)
    for k, mol in enumerate(
        ["CarboxylicAcids/O2C5H10", "CarboxylicAcids/O2C6H12", "DiCarboxylicAcids/O4C5H8"]
    ):
        family, name = mol.split("/")
        numbers = np.array([8, 6, 1][: 2 + k % 2])
        order = rng.permutation(N_FRAMES)  # stored out of order: readers sort by frame
        pos = rng.normal(size=(N_FRAMES, len(numbers), 3)) * 2.0
        energy = 100.0 * k + np.arange(N_FRAMES) * 1e-3  # encodes molecule and frame
        forces = rng.normal(size=(N_FRAMES, len(numbers), 3))
        d = root / fmt / family
        d.mkdir(parents=True, exist_ok=True)
        if fmt == "traj":
            frames = []
            for t in order:
                a = Atoms(numbers=numbers, positions=pos[t], cell=[17.0] * 3, pbc=True)
                a.info = {"molecule": name, "family": family, "frame": int(t)}
                a.calc = SinglePointCalculator(a, energy=energy[t], forces=forces[t])
                frames.append(a)
            write(d / f"{name}.traj", frames)
        elif fmt == "npz":
            np.savez(d / f"{name}.npz", z=numbers, pos=pos[order], energy=energy[order],
                     forces=forces[order], frame=order)  # fmt: skip
        else:
            import zarr

            g = zarr.open_group(str(d / f"{name}.zarr"), mode="w")
            for key, value in (("numbers", numbers), ("frame", order), ("positions", pos[order]),
                               ("energy", energy[order]), ("forces", forces[order])):  # fmt: skip
                g.create_array(key, data=value)


@pytest.fixture(params=["traj", "npz", "zarr"])
def repo(request, tmp_path):
    _write_repo(tmp_path / "repo", request.param)
    return tmp_path, request.param


def _dataset(tmp_path, fmt, split, **kw) -> GMDDataset:
    args = {"task": "custom", "train_molecules": ["CarboxylicAcids"],
            "test_molecules": ["DiCarboxylicAcids/O4C5H8"], **SIZES, **kw}  # fmt: skip
    return GMDDataset(
        root=str(tmp_path / "cache"), cutoff=5.0, split=split, format=fmt,
        local_root=str(tmp_path / "repo"), **args,
    )  # fmt: skip


def _reference(root, fmt, train, ood, seed):
    """The dataset's own loader (gmd.load) on the same files: {split: [(energy)]}."""
    n_train, n_val, n_test = SIZES["n_train"], SIZES["n_val"], SIZES["n_test"]
    rng = np.random.default_rng(seed)
    out = {s: [] for s in STARTS}
    for m in train + ood:
        frames = read_frames(root / fmt / f"{m}.{fmt}")
        sizes = (
            {"train": n_train, "val": n_val, "test_id": n_test}
            if m in train
            else {"test_ood": n_test}
        )
        if seed is None:
            picked = {
                s: [a for a in frames if STARTS[s] <= a.info["frame"] < STARTS[s] + n]
                for s, n in sizes.items()
            }
        else:
            perm, cuts = rng.permutation(len(frames)), np.cumsum([0, *sizes.values()])
            picked = {
                s: [frames[i] for i in perm[lo:hi]] for s, lo, hi in zip(sizes, cuts, cuts[1:])
            }
        for s in sizes:
            out[s] += [a.get_potential_energy() for a in picked[s]]
    return out


@pytest.mark.parametrize("seed", [None, 7])
def test_splits_match_the_datasets_own_loader(repo, seed) -> None:
    tmp_path, fmt = repo
    train = ["CarboxylicAcids/O2C5H10", "CarboxylicAcids/O2C6H12"]
    ref = _reference(tmp_path / "repo", fmt, train, ["DiCarboxylicAcids/O4C5H8"], seed)
    for split, gmd in (("train", "train"), ("val", "test_id"), ("test", "test_ood")):
        ds = _dataset(tmp_path, fmt, split, seed=seed)
        got = [float(g.energy) for g in ds]
        assert got == pytest.approx(ref[gmd], abs=1e-12), (split, seed)
    val = _dataset(tmp_path, fmt, "val", seed=seed, val_split="val")
    assert [float(g.energy) for g in val] == pytest.approx(ref["val"], abs=1e-12)


def test_paper_windows_frames_and_non_periodic_graphs(repo) -> None:
    tmp_path, fmt = repo
    ds = _dataset(tmp_path, fmt, "train")
    assert len(ds) == 2 * SIZES["n_train"]
    frames = sorted(round(float(g.energy) * 1000) % 100000 for g in ds)  # energy encodes the frame
    assert frames == sorted([*range(5)] * 2)
    assert all(not bool(g.pbc.any()) for g in ds)  # the stored box is MD padding only
    test = _dataset(tmp_path, fmt, "test")
    assert len(test) == SIZES["n_test"] and all(
        1400 <= round(float(g.energy) * 1000) % 100000 < 1404 for g in test
    )


def test_extra_molecules_are_added_and_the_cache_is_reused(repo) -> None:
    tmp_path, fmt = repo
    ds = _dataset(tmp_path, fmt, "train", train_molecules=["CarboxylicAcids/O2C5H10"],
                  extra_train_molecules=["CarboxylicAcids/O2C6H12"])  # fmt: skip
    assert len(ds) == 2 * SIZES["n_train"]
    cached = list((tmp_path / "cache" / "processed").glob("gmd_custom_train_*.pt"))
    assert len(cached) == 1
    again = _dataset(tmp_path, fmt, "train", train_molecules=["CarboxylicAcids/O2C5H10"],
                     extra_train_molecules=["CarboxylicAcids/O2C6H12"])  # fmt: skip
    assert [float(g.energy) for g in again] == [float(g.energy) for g in ds]


def test_task_all_splits_by_fraction(repo) -> None:
    tmp_path, fmt = repo
    sizes = [len(GMDDataset(root=str(tmp_path / "cache"), cutoff=5.0, split=s, task="all",
                            format=fmt, local_root=str(tmp_path / "repo"), seed=1))
             for s in ("train", "val", "test")]  # fmt: skip
    assert sum(sizes) == 3 * N_FRAMES and sizes[0] == int(0.8 * 3 * N_FRAMES)


def test_task_names_and_bad_options() -> None:
    assert task_key("FragmentFusion") == "fragment_fusion" and "fragment_fission" in TASKS
    assert len(TASKS["fragment_duplication"][0]) == 6  # C5-C10, Table 1 of the paper
    with pytest.raises(ValueError, match="unknown GMD task"):
        task_key("fragment_fusions")
    with pytest.raises(ValueError, match="overlap"):
        GMDDataset(root="/nonexistent", cutoff=5.0, n_train=1001)
    with pytest.raises(ValueError, match="format"):
        GMDDataset(root="/nonexistent", cutoff=5.0, format="hdf5")


def test_dtype_follows_the_config(repo) -> None:
    tmp_path, fmt = repo
    ds = _dataset(tmp_path, fmt, "test", dtype=torch.float32)
    assert ds[0].pos.dtype == torch.float32
