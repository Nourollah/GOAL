"""Tests for the ASE-LMDB dataset (OMol25 / OMat24 / ODAC23 shards).

Shards are written here rather than read from ``data/``: the format is what
is under test, the real corpus is 20 GB, and a synthetic shard can carry
exactly the chemistry, charges and energies a case needs.
"""

from __future__ import annotations

import typing
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers — write a shard the way ASE's LMDB backend does
# ---------------------------------------------------------------------------


def _atoms(symbols: str, n_atoms: int | None = None) -> typing.Any:
    """A small non-periodic molecule with well-separated atoms."""
    from ase import Atoms

    count: int = n_atoms if n_atoms is not None else len(Atoms(symbols))
    positions: np.ndarray = np.zeros((count, 3), dtype=float)
    positions[:, 0] = np.arange(count) * 1.2
    return Atoms(symbols, positions=positions)


def write_shard(
    path: Path,
    rows: typing.Sequence[tuple[typing.Any, dict[str, typing.Any], float]],
    with_nextid: bool = True,
) -> Path:
    """Write *rows* as ``(atoms, data, energy)`` into an ``.aselmdb`` shard.

    Mirrors the on-disk contract: integer-string keys from 1, values
    ``zlib(json(ase.db row))``, plus the ``nextid`` counter that lets a
    reader enumerate rows without scanning every key.
    """
    import lmdb
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.db.row import atoms2dict
    from ase.io.jsonio import encode

    env: typing.Any = lmdb.open(str(path), subdir=False, map_size=1 << 26)
    with env.begin(write=True) as txn:
        for index, (atoms, data, energy) in enumerate(rows, start=1):
            forces: np.ndarray = np.tile([0.1, -0.2, 0.3], (len(atoms), 1))
            atoms = atoms.copy()
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
            record: dict[str, typing.Any] = atoms2dict(atoms)
            record["data"] = data
            record["ctime"] = record["mtime"] = 0.0
            record["user"] = "test"
            txn.put(str(index).encode(), zlib.compress(encode(record).encode()))
        if with_nextid:
            txn.put(b"nextid", zlib.compress(encode(len(rows) + 1).encode()))
    env.close()
    return path


#: Reference rows: three chemistries, two charges, two spins, two corpora.
#: Energies are exactly ``Σ_Z n_Z e_Z`` for the coefficients in
#: :data:`REFERENCE_ENERGIES`, so the regression has an exact answer.
REFERENCE_ENERGIES: dict[int, float] = {1: -13.5, 6: -1028.0, 8: -2041.0}


def _energy(atoms: typing.Any) -> float:
    return float(sum(REFERENCE_ENERGIES[int(z)] for z in atoms.numbers))


def _reference_rows() -> list[tuple[typing.Any, dict[str, typing.Any], float]]:
    specs: list[tuple[str, int, int, str]] = [
        ("CH4", 0, 1, "biomolecules"),
        ("C2H6", 0, 1, "biomolecules"),
        ("H2O", 0, 1, "elytes"),
        ("CO2", 0, 1, "elytes"),
        ("CH4", 1, 2, "reactivity"),  # a cation, open shell
        ("H2O", -1, 1, "reactivity"),  # an anion
    ]
    rows: list[tuple[typing.Any, dict[str, typing.Any], float]] = []
    for symbols, charge, spin, corpus in specs:
        atoms: typing.Any = _atoms(symbols)
        data: dict[str, typing.Any] = {
            "charge": charge,
            "spin": spin,
            "data_id": corpus,
            "source": f"{corpus}/{symbols}_q{charge}/orca.tar.zst",
            "num_atoms": len(atoms),
            "nl_energy": 1.5 * len(atoms),
        }
        rows.append((atoms, data, _energy(atoms)))
    return rows


@pytest.fixture
def shard(tmp_path: Path) -> Path:
    """One shard holding the reference rows."""
    return write_shard(tmp_path / "data0000.aselmdb", _reference_rows())


@pytest.fixture
def split(tmp_path: Path) -> Path:
    """A directory of two shards, the OMol25 layout in miniature."""
    directory: Path = tmp_path / "train"
    directory.mkdir()
    rows = _reference_rows()
    write_shard(directory / "data0000.aselmdb", rows[:3])
    write_shard(directory / "data0001.aselmdb", rows[3:])
    return directory


# ---------------------------------------------------------------------------
# Row decoding
# ---------------------------------------------------------------------------


class TestRowDecoding:
    """The zlib+JSON ase.db row contract."""

    def test_decode_row_gives_arrays(self, shard: Path):
        import lmdb

        from goal.ml.data.datasets.aselmdb import decode_row

        env = lmdb.open(str(shard), subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            row = decode_row(txn.get(b"1"))
        env.close()

        assert isinstance(row["numbers"], np.ndarray)
        assert row["positions"].shape == (5, 3)  # CH4
        assert row["data"]["data_id"] == "biomolecules"

    def test_row_to_atoms_carries_labels(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        reader = ASELMDBReader(shard)
        atoms = reader.atoms(0)
        assert atoms.get_chemical_formula(mode="hill") == "CH4"
        assert atoms.get_potential_energy() == pytest.approx(_energy(_atoms("CH4")))
        assert atoms.get_forces().shape == (5, 3)
        reader.close()

    def test_row_to_atoms_exposes_scalar_extras(self, shard: Path):
        """Corpus scalars reach ``atoms.info`` so ``key_mapping`` can see them."""
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        reader = ASELMDBReader(shard)
        atoms = reader.atoms(0)
        assert atoms.info["charge"] == 0
        assert atoms.info["nl_energy"] == pytest.approx(7.5)
        assert atoms.info["data"]["source"].endswith("orca.tar.zst")
        reader.close()

    def test_parse_formula_round_trip(self):
        from goal.ml.data.datasets.aselmdb import parse_formula

        assert parse_formula("C19H18N4O5S") == {6: 19, 1: 18, 7: 4, 8: 5, 16: 1}
        # OMol25's own style writes an explicit 1
        assert parse_formula("C2H4OS2") == parse_formula("C2H4O1S2")

    def test_parse_formula_rejects_nonsense(self):
        from goal.ml.data.datasets.aselmdb import parse_formula

        with pytest.raises(ValueError, match="Unknown element"):
            parse_formula("Xx2")


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class TestReader:
    """Flat addressing over one or many shards."""

    def test_length_and_addresses(self, split: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        reader = ASELMDBReader(split)
        assert len(reader.files) == 2
        assert len(reader) == 6
        assert reader.address(0) == (split / "data0000.aselmdb", 1)
        assert reader.address(5) == (split / "data0001.aselmdb", 3)
        reader.close()

    def test_position_round_trip(self, split: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        reader = ASELMDBReader(split)
        for position in range(len(reader)):
            path, row_id = reader.address(position)
            assert reader.position_of(path, row_id) == position
        reader.close()

    def test_addresses_match_across_path_spellings(self, split: Path):
        """A cached catalogue records the path spelling used when it was scanned.

        A reader opened with a different spelling of the same file — relative
        vs absolute, or through a symlink — must still be able to address its
        rows, or a perfectly good catalogue becomes unusable.
        """
        import os

        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        reader = ASELMDBReader(split)
        absolute = (split / "data0001.aselmdb").resolve()
        relative = Path(os.path.relpath(absolute, Path.cwd()))

        assert reader.position_of(relative, 1) == reader.position_of(absolute, 1)
        assert reader.read_row(relative, 1)["numbers"].shape == (3,)

        # ... and a catalogue whose stored paths resolve to nothing from here
        # (an older cache written relative to a different working directory)
        # is still addressable by shard name.
        catalogue = reader.catalogue(cache=False)
        object.__setattr__(
            catalogue,
            "files",
            np.asarray([f"somewhere/else/{f.name}" for f in reader.files], dtype=str),
        )
        assert reader.positions_of(catalogue).tolist() == list(range(6))
        reader.close()

    def test_key_scan_fallback_without_nextid(self, tmp_path: Path):
        """A shard with no ``nextid`` counter still enumerates correctly."""
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        path = write_shard(tmp_path / "bare.aselmdb", _reference_rows()[:3], with_nextid=False)
        reader = ASELMDBReader(path)
        assert len(reader) == 3
        assert reader.read(2)["data"]["data_id"] == "elytes"
        reader.close()

    def test_read_many_visits_every_row(self, split: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        reader = ASELMDBReader(split)
        seen = dict(reader.read_many(np.arange(len(reader))))
        assert sorted(seen) == list(range(6))
        assert seen[0]["data"]["data_id"] == "biomolecules"
        reader.close()

    def test_two_readers_share_one_shard(self, shard: Path):
        """LMDB refuses a second open of one file; the shared cache must not.

        Two readers over the same shard is the normal case — a notebook
        browsing what a dataset is training on — and used to raise
        ``lmdb.Error: already open in this process``.
        """
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        first = ASELMDBReader(shard)
        second = ASELMDBReader(shard)
        assert first.read(0)["energy"] == second.read(0)["energy"]

        first.close()
        # Closing one holder must leave the other's handle usable.
        assert second.read(1)["data"]["data_id"] == "biomolecules"
        second.close()

    def test_reopen_after_close(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        reader = ASELMDBReader(shard)
        reader.read(0)
        reader.close()
        assert reader.read(0)["numbers"].shape == (5,)
        reader.close()

    def test_out_of_range_raises(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        reader = ASELMDBReader(shard)
        with pytest.raises(IndexError):
            reader.read(999)
        reader.close()

    def test_missing_root_raises(self, tmp_path: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        with pytest.raises(FileNotFoundError):
            ASELMDBReader(tmp_path / "nope.aselmdb")

    def test_empty_directory_raises(self, tmp_path: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            ASELMDBReader(tmp_path / "empty")


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


class TestCatalogue:
    """The per-row summary that makes filtering and searching cheap."""

    def test_scan_fields(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        reader = ASELMDBReader(shard)
        cat = reader.catalogue(cache=False)

        assert len(cat) == 6
        assert cat.num_atoms.tolist() == [5, 8, 3, 3, 5, 3]
        assert cat.formula[0] == "CH4"
        assert cat.charge.tolist() == [0, 0, 0, 0, 1, -1]
        assert cat.spin.tolist() == [1, 1, 1, 1, 2, 1]
        assert sorted(set(cat.data_id.tolist())) == ["biomolecules", "elytes", "reactivity"]
        assert cat.energy[0] == pytest.approx(_energy(_atoms("CH4")))
        assert cat.source is None  # off by default
        reader.close()

    def test_scan_can_include_source(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        reader = ASELMDBReader(shard)
        cat = reader.catalogue(cache=False, include_source=True)
        assert cat.source is not None
        assert "CH4_q0" in cat.source[0]
        reader.close()

    def test_elements_and_masks(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        reader = ASELMDBReader(shard)
        cat = reader.catalogue(cache=False)

        assert cat.unique_elements() == [1, 6, 8]
        assert cat.elements(0) == [1, 6]  # CH4
        assert cat.elements(3) == [6, 8]  # CO2

        # allowed: keep only rows made entirely of these
        hydrocarbons = cat.element_mask(allowed=[1, 6])
        assert cat.formula[hydrocarbons].tolist() == ["CH4", "C2H6", "CH4"]

        # excluded: drop rows containing any of these
        no_oxygen = cat.element_mask(excluded=[8])
        assert set(cat.formula[no_oxygen].tolist()) == {"CH4", "C2H6"}
        reader.close()

    def test_select_keeps_rows_aligned(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        reader = ASELMDBReader(shard)
        cat = reader.catalogue(cache=False)
        picked = cat.select(np.array([1, 4]))

        assert len(picked) == 2
        assert picked.formula.tolist() == ["C2H6", "CH4"]
        assert picked.charge.tolist() == [0, 1]
        assert reader.positions_of(picked).tolist() == [1, 4]
        reader.close()

    def test_cache_round_trip(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        reader = ASELMDBReader(shard)
        first = reader.catalogue(cache=True)
        cache_path = reader.catalogue_path(0)
        assert cache_path.is_file()

        second = reader.catalogue(cache=True)  # read back, not re-scanned
        assert second.num_atoms.tolist() == first.num_atoms.tolist()
        assert second.formula.tolist() == first.formula.tolist()
        assert second.energy.tolist() == first.energy.tolist()
        reader.close()

    def test_cache_dir_override(self, shard: Path, tmp_path: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBReader

        elsewhere = tmp_path / "cache"
        reader = ASELMDBReader(shard, cache_dir=elsewhere)
        reader.catalogue(cache=True)
        assert list(elsewhere.glob("*.npz"))
        assert not list(shard.parent.glob("*.npz"))
        reader.close()

    def test_merge_is_order_preserving(self, split: Path):
        """Merging shard summaries must keep rows addressed correctly."""
        from goal.ml.data.datasets.aselmdb import ASELMDBReader, concat_catalogues

        reader = ASELMDBReader(split)
        merged = reader.catalogue(cache=False)
        assert len(merged) == 6
        assert merged.files.shape == (2,)
        assert merged.file_index.tolist() == [0, 0, 0, 1, 1, 1]
        assert reader.positions_of(merged).tolist() == [0, 1, 2, 3, 4, 5]

        # A pairwise concat of the same pieces agrees with the batched one.
        left = reader.catalogue(cache=False).select(np.arange(3))
        pairwise = concat_catalogues([left, left])
        assert pairwise.formula.tolist() == left.formula.tolist() * 2
        reader.close()

    def test_merge_drops_partial_source(self, split: Path):
        """A merged catalogue must not claim source it has for only some rows."""
        from goal.ml.data.datasets.aselmdb import ASELMDBReader, concat_catalogues

        reader = ASELMDBReader(split)
        with_source = reader.catalogue(cache=False, include_source=True)
        without = reader.catalogue(cache=False)
        assert concat_catalogues([with_source, without]).source is None
        reader.close()


# ---------------------------------------------------------------------------
# Statistics from a catalogue
# ---------------------------------------------------------------------------


class TestCatalogueStatistics:
    """Energy baseline and scale, computed without decoding geometry."""

    def test_reference_energies_are_exact(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import (
            ASELMDBReader,
            linear_reference_energies,
        )

        reader = ASELMDBReader(shard)
        cat = reader.catalogue(cache=False)
        recovered = linear_reference_energies(cat)

        assert set(recovered) == set(REFERENCE_ENERGIES)
        for z, expected in REFERENCE_ENERGIES.items():
            assert recovered[z] == pytest.approx(expected, abs=1e-6)
        reader.close()

    def test_reference_energies_ignore_absent_elements(self, shard: Path):
        """A filtered catalogue shares its parent's category table.

        Elements only its *siblings* contain must not come back as a
        meaningless 0.0 reference energy.
        """
        from goal.ml.data.datasets.aselmdb import (
            ASELMDBReader,
            linear_reference_energies,
        )

        reader = ASELMDBReader(shard)
        cat = reader.catalogue(cache=False)
        hydrocarbons = cat.select(cat.element_mask(allowed=[1, 6]))
        recovered = linear_reference_energies(hydrocarbons)

        assert set(recovered) == {1, 6}
        reader.close()

    def test_energy_scale_matches_definition(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBReader, residual_energy_scale

        reader = ASELMDBReader(shard)
        cat = reader.catalogue(cache=False)
        # Energies are exactly the baseline here, so the residual spread —
        # and therefore the scale — is zero, which reports as 1.0 (a no-op
        # gain) rather than 0.
        assert residual_energy_scale(cat, REFERENCE_ENERGIES) == pytest.approx(1.0)

        # Perturb one energy and the scale becomes std/mean_atoms.
        shifted = cat.energy.copy()
        shifted[0] += 10.0
        perturbed = ASELMDBReader(shard).catalogue(cache=False)
        object.__setattr__(perturbed, "energy", shifted)
        expected = float(np.std(shifted - cat.energy, ddof=1) / cat.num_atoms.mean())
        assert residual_energy_scale(perturbed, REFERENCE_ENERGIES) == pytest.approx(expected)
        reader.close()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TestDataset:
    """``dataset_type: aselmdb`` — filtering and graph construction."""

    def test_registered(self):
        from goal.ml.data.datasets.aselmdb import ASELMDBDataset
        from goal.ml.registry import DATASET_REGISTRY

        assert DATASET_REGISTRY.get("aselmdb") is ASELMDBDataset

    def test_yields_atomic_graphs(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBDataset
        from goal.ml.data.graph import AtomicGraph

        dataset = ASELMDBDataset(root=shard, cutoff=5.0, dtype=torch.float32)
        assert len(dataset) == 6

        graph = dataset[0]
        assert isinstance(graph, AtomicGraph)
        assert graph.z.tolist() == [6, 1, 1, 1, 1]
        assert graph.pos.shape == (5, 3)
        assert graph.energy.item() == pytest.approx(_energy(_atoms("CH4")), abs=1e-3)
        assert graph.forces.shape == (5, 3)
        dataset.close()

    def test_element_filter(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBDataset

        dataset = ASELMDBDataset(root=shard, cutoff=5.0, allowed_elements=[1, 6])
        assert dataset.catalogue.formula.tolist() == ["CH4", "C2H6", "CH4"]
        dataset.close()

    def test_charge_and_spin_filters(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBDataset

        neutral = ASELMDBDataset(root=shard, cutoff=5.0, charges=[0], spins=[1])
        assert len(neutral) == 4
        assert set(neutral.catalogue.charge.tolist()) == {0}
        neutral.close()

        cations = ASELMDBDataset(root=shard, cutoff=5.0, charges=[1])
        assert len(cations) == 1
        assert cations.catalogue.spin.tolist() == [2]
        cations.close()

    def test_corpus_and_size_filters(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBDataset

        dataset = ASELMDBDataset(
            root=shard, cutoff=5.0, data_ids=["elytes"], min_atoms=3, max_atoms=3
        )
        assert sorted(dataset.catalogue.formula.tolist()) == ["CO2", "H2O"]
        dataset.close()

    def test_sample_cap_and_stride(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBDataset

        head = ASELMDBDataset(root=shard, cutoff=5.0, max_samples=2)
        assert head.catalogue.formula.tolist() == ["CH4", "C2H6"]
        head.close()

        strided = ASELMDBDataset(root=shard, cutoff=5.0, sample_stride=3)
        assert strided.catalogue.formula.tolist() == ["CH4", "CO2"]
        strided.close()

        seeded = ASELMDBDataset(root=shard, cutoff=5.0, max_samples=3, sample_seed=7)
        again = ASELMDBDataset(root=shard, cutoff=5.0, max_samples=3, sample_seed=7)
        assert len(seeded) == 3
        assert seeded.catalogue.row_id.tolist() == again.catalogue.row_id.tolist()
        seeded.close()
        again.close()

    def test_catalogue_is_aligned_with_items(self, shard: Path):
        """``dataset.catalogue`` row *i* must describe ``dataset[i]``."""
        from goal.ml.data.datasets.aselmdb import ASELMDBDataset

        dataset = ASELMDBDataset(
            root=shard, cutoff=5.0, allowed_elements=[1, 8], dtype=torch.float32
        )
        for index in range(len(dataset)):
            graph = dataset[index]
            assert graph.z.shape[0] == int(dataset.catalogue.num_atoms[index])
            assert graph.energy.item() == pytest.approx(
                float(dataset.catalogue.energy[index]), abs=1e-3
            )
            assert dataset.atoms(index).get_chemical_formula(mode="hill") == (
                dataset.catalogue.formula[index]
            )
        dataset.close()

    def test_fragment_labels_attached(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBDataset

        dataset = ASELMDBDataset(
            root=shard,
            cutoff=5.0,
            compute_fragment_index=True,
            fragment_scheme="connected",
            fragment_covalent_cutoff=1.8,
        )
        graph = dataset[0]
        assert graph.fragment_index is not None
        assert graph.fragment_index.shape[0] == graph.z.shape[0]
        dataset.close()

    def test_no_fragment_labels_by_default(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBDataset

        dataset = ASELMDBDataset(root=shard, cutoff=5.0)
        assert dataset[0].get("fragment_index", None) is None
        dataset.close()

    def test_charge_spin_attachment_is_opt_in(self, shard: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBDataset

        plain = ASELMDBDataset(root=shard, cutoff=5.0)
        assert plain[4].get("total_charge", None) is None
        plain.close()

        annotated = ASELMDBDataset(root=shard, cutoff=5.0, attach_charge_spin=True)
        graph = annotated[4]
        assert graph.total_charge.item() == pytest.approx(1.0)
        assert graph.spin_multiplicity.item() == 2
        annotated.close()

    def test_key_mapping_reaches_corpus_scalars(self, shard: Path):
        """A ``data`` scalar can be trained on via the usual key mapping."""
        from goal.ml.data.datasets.aselmdb import ASELMDBDataset

        dataset = ASELMDBDataset(root=shard, cutoff=5.0, key_mapping={"energy": "nl_energy"})
        assert dataset[0].energy.item() == pytest.approx(7.5)  # 1.5 * 5 atoms
        dataset.close()

    def test_wrong_key_fails_at_construction(self, shard: Path):
        """A typo must not fall through to the row's calculator.

        An ase.db row always carries a SinglePointCalculator, so the
        generic key resolution would happily return the total energy for
        a misspelled ``energy_key`` and train on the wrong target.
        """
        from goal.ml.data.datasets.aselmdb import ASELMDBDataset

        with pytest.raises(ValueError, match="not found in this corpus"):
            ASELMDBDataset(root=shard, cutoff=5.0, energy_key="not_a_key")

    def test_split_file_resolution(self, tmp_path: Path):
        from goal.ml.data.datasets.aselmdb import ASELMDBDataset

        write_shard(tmp_path / "val.aselmdb", _reference_rows()[:2])
        dataset = ASELMDBDataset(root=tmp_path, cutoff=5.0, split="val")
        assert len(dataset) == 2
        dataset.close()

    def test_directory_without_split_file_raises(self, split: Path):
        """The signal that makes ``GOALDataModule`` fall back to a ratio split.

        Loading the whole directory here instead would give train, val and
        test the same rows.
        """
        from goal.ml.data.datasets.aselmdb import ASELMDBDataset

        with pytest.raises(FileNotFoundError, match="train/val/test would"):
            ASELMDBDataset(root=split, cutoff=5.0, split="train")


# ---------------------------------------------------------------------------
# DataModule integration
# ---------------------------------------------------------------------------


class TestDataModuleIntegration:
    """A directory of shards loaded the way a config declares it."""

    def test_extension_is_discoverable(self):
        from goal.ml.data.datamodule import _DATA_EXTENSIONS

        assert ".aselmdb" in _DATA_EXTENSIONS

    def test_directory_root_numeric_split(self, split: Path):
        from omegaconf import OmegaConf

        from goal.ml.data.datamodule import GOALDataModule

        cfg = OmegaConf.create(
            {
                "data": {
                    "dataset_type": "aselmdb",
                    "root": str(split),
                    "cutoff": 5.0,
                    "batch_size": 2,
                    "num_workers": 0,
                    "split_ratio": [0.5, 0.25, 0.25],
                    "split_seed": 3,
                    "allowed_elements": [1, 6, 8],
                    "compute_fragment_index": True,
                    "fragment_scheme": "connected",
                }
            }
        )
        datamodule = GOALDataModule(cfg)
        datamodule.setup(None)

        assert len(datamodule.data_train) == 4
        assert len(datamodule.data_val) == 1
        assert len(datamodule.data_test) == 1

        batch = next(iter(datamodule.train_dataloader()))
        assert batch.z.shape[0] == batch.pos.shape[0]
        assert batch.energy.shape[0] == 2
        assert batch.fragment_index.shape[0] == batch.z.shape[0]

    def test_per_split_directories(self, tmp_path: Path):
        from omegaconf import OmegaConf

        from goal.ml.data.datamodule import GOALDataModule

        for name, rows in (("train", _reference_rows()[:4]), ("val", _reference_rows()[4:])):
            directory = tmp_path / name
            directory.mkdir()
            write_shard(directory / "data0000.aselmdb", rows)

        cfg = OmegaConf.create(
            {
                "data": {
                    "dataset_type": "aselmdb",
                    "train_dir": str(tmp_path / "train"),
                    "val_dir": str(tmp_path / "val"),
                    "cutoff": 5.0,
                    "batch_size": 2,
                    "num_workers": 0,
                }
            }
        )
        datamodule = GOALDataModule(cfg)
        datamodule.setup("fit")
        assert len(datamodule.data_train) == 4
        assert len(datamodule.data_val) == 2
