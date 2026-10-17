"""GOAL Zarr stores: write, read back identical graphs, train from them."""

from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import bulk, molecule
from ase.calculators.emt import EMT
from ase.io import read, write

from goal.ml.data.graph import AtomicGraph

zarr = pytest.importorskip("zarr", minversion="3")

from goal.ml.data.zarr_dataset import ZARR_FORMAT, ZarrDataset, write_zarr

CUTOFF = 4.0


def _frames(with_stress: bool = False):
    out = []
    structures = [molecule("CH4"), bulk("Cu", "fcc", a=3.6) * (2, 1, 1), molecule("C2H6")]
    if with_stress:
        structures = [bulk("Cu", "fcc", a=3.6), bulk("Cu", "fcc", a=3.7) * (2, 1, 1)]
    for atoms in structures:
        atoms.calc = EMT()
        atoms.info["energy"] = atoms.get_potential_energy()
        atoms.arrays["forces"] = atoms.get_forces()
        if with_stress:
            atoms.info["stress"] = atoms.get_stress()
        atoms.calc = None
        out.append(atoms)
    return out


@pytest.fixture()
def xyz_file(tmp_path: Path) -> Path:
    path = tmp_path / "frames.xyz"
    write(path, _frames(), format="extxyz")
    return path


@pytest.fixture()
def store(tmp_path: Path, xyz_file: Path) -> Path:
    """A synthetic store with stored neighbour lists (train split)."""
    return write_zarr(
        read(xyz_file, index=":"), tmp_path / "data.zarr", split="train", cutoff=CUTOFF
    )


def _reference(xyz_file: Path, i: int) -> AtomicGraph:
    atoms = read(xyz_file, index=i)
    return AtomicGraph.from_ase(
        atoms, cutoff=CUTOFF, energy=atoms.get_potential_energy(), forces=atoms.get_forces()
    )


FIELDS = ("pos", "z", "cell", "pbc", "edge_index", "unit_shifts", "energy", "forces", "weight")


def test_layout(store: Path) -> None:
    root = zarr.open_group(str(store), mode="r")
    assert root.attrs["format"] == ZARR_FORMAT
    group = root["train"]
    assert group.attrs["num_structures"] == 3 and group.attrs["cutoff"] == CUTOFF
    assert {"positions", "atomic_numbers", "energies", "forces", "cell", "pbc", "n_atoms"} <= set(
        group.array_keys()
    )
    assert group["positions"].shape == (5 + 2 + 8, 3)


def test_len_and_fields_match_from_ase(store: Path, xyz_file: Path) -> None:
    ds = ZarrDataset(root=store, cutoff=CUTOFF, split="train")
    assert len(ds) == 3
    for i in range(3):
        got, ref = ds[i], _reference(xyz_file, i)
        assert set(got.keys()) == set(ref.keys())
        for field in FIELDS:
            assert torch.equal(getattr(got, field), getattr(ref, field)), field
    assert torch.equal(ds[-1].pos, ds[2].pos)


def test_other_cutoff_recomputes_the_neighbour_list(store: Path, xyz_file: Path) -> None:
    ds = ZarrDataset(root=store, cutoff=3.0)
    atoms = read(xyz_file, index=1)
    assert ds[1].edge_index.shape == AtomicGraph.from_ase(atoms, cutoff=3.0).edge_index.shape


def test_labels_missing_on_some_frames_are_dropped(tmp_path: Path) -> None:
    frames = _frames()
    del frames[1].info["energy"]
    store = write_zarr(frames, tmp_path / "s.zarr")
    ds = ZarrDataset(root=store, cutoff=CUTOFF)
    assert ds[0].get("energy") is None and ds[0].get("forces") is not None


def test_stress_round_trip(tmp_path: Path) -> None:
    frames = _frames(with_stress=True)
    ds = ZarrDataset(root=write_zarr(frames, tmp_path / "s.zarr"), cutoff=CUTOFF)
    from ase.stress import voigt_6_to_full_3x3_stress

    expected = voigt_6_to_full_3x3_stress(frames[1].info["stress"])
    np.testing.assert_allclose(ds[1].stress.numpy(), expected)


def test_existing_split_is_protected(store: Path) -> None:
    with pytest.raises(FileExistsError, match="overwrite"):
        write_zarr(_frames(), store, split="train")
    write_zarr(_frames()[:1], store, split="train", overwrite=True)
    assert len(ZarrDataset(root=store, cutoff=CUTOFF)) == 1


def test_multiple_splits_in_one_store(store: Path) -> None:
    write_zarr(_frames()[:2], store, split="val")
    assert len(ZarrDataset(root=store, cutoff=CUTOFF, split="val")) == 2
    with pytest.raises(FileNotFoundError, match="no 'test' group"):
        ZarrDataset(root=store, cutoff=CUTOFF, split="test")


def test_dataloader_workers_reopen_the_store(store: Path) -> None:
    import pickle

    ds = ZarrDataset(root=store, cutoff=CUTOFF)
    clone = pickle.loads(pickle.dumps(ds))
    assert torch.equal(clone[0].pos, ds[0].pos)


def test_registered_for_the_datamodule(store: Path) -> None:
    from omegaconf import OmegaConf

    import goal.ml.data.datasets  # noqa: F401
    from goal.ml.data.datamodule import GOALDataModule

    write_zarr(_frames()[:2], store, split="val")
    cfg = OmegaConf.create(
        {
            "data": {
                "dataset_type": "zarr",
                "root": str(store),
                "cutoff": CUTOFF,
                "batch_size": 2,
                "train_paths": [str(store)],
                "val_paths": [str(store)],
            }
        }
    )
    dm = GOALDataModule(cfg)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch.num_graphs == 2


def test_fragment_labels_are_attached(store: Path) -> None:
    ds = ZarrDataset(root=store, cutoff=CUTOFF, compute_fragment_index=True)
    assert ds[0].fragment_index.shape == (5,)


def test_pack_cli(tmp_path: Path, xyz_file: Path) -> None:
    from goal.ml.cli.pack_zarr import main

    out = tmp_path / "cli.zarr"
    assert main(["--input", str(xyz_file), "--output", str(out), "--split", "test"]) == 0
    assert len(ZarrDataset(root=out, cutoff=CUTOFF, split="test")) == 3
    assert (
        main(["--input", str(xyz_file), "--output", str(out), "--compute-edges"]) == 1
    )  # no --cutoff


def test_missing_zarr_has_an_install_hint(monkeypatch, tmp_path: Path) -> None:
    from goal.ml.data import zarr_dataset

    real_import = builtins.__import__

    def _no_zarr(name, *args, **kwargs):
        if name == "zarr":
            raise ImportError("No module named 'zarr'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_zarr)
    with pytest.raises(ImportError, match=r"goal-atomistic\[zarr\]"):
        zarr_dataset.write_zarr(_frames(), tmp_path / "x.zarr")


def test_info_keys_round_trip(tmp_path: Path) -> None:
    frames = _frames()
    for i, atoms in enumerate(frames):
        atoms.info["step"] = 10 * i
        atoms.info["temperature"] = 300.0 + i
    store = write_zarr(frames, tmp_path / "t.zarr", info_keys=("step", "temperature"))
    ds = ZarrDataset(root=store, cutoff=CUTOFF)
    assert ds.structure_info(2) == {"step": 20, "temperature": 302.0}
    with pytest.raises(KeyError, match="atoms.info"):
        write_zarr(_frames(), tmp_path / "u.zarr", info_keys=("step",))


def test_train_only_store_falls_back_to_split_ratio(tmp_path: Path) -> None:
    """A store with only a 'train' group, given as data.root, is split by
    data.split_ratio — a missing group behaves like a missing split file."""
    from omegaconf import OmegaConf

    import goal.ml.data.datasets  # noqa: F401
    from goal.ml.data.datamodule import GOALDataModule

    frames = _frames() * 4  # 12 structures
    store = write_zarr(frames, tmp_path / "train_only.zarr")
    cfg = OmegaConf.create(
        {
            "data": {
                "dataset_type": "zarr",
                "root": str(store),
                "cutoff": CUTOFF,
                "batch_size": 2,
                "split_ratio": [0.5, 0.25, 0.25],
            }
        }
    )
    dm = GOALDataModule(cfg)
    dm.setup("fit")
    assert len(dm.data_train) + len(dm.data_val) <= 12
    assert len(dm.data_train) >= 4 and len(dm.data_val) >= 2
