"""The five ``data.fragment_scheme`` options.

``connected`` is pure torch; the other four go through rdkit bond
perception.  What is checked here:

* each scheme returns a valid partition — contiguous labels, every atom
  covered, deterministic ordering;
* they differ in the way their definitions say they should (a saturated
  acid is one connected fragment but several rotatable-bond groups);
* the rdkit path caches on the bond graph, so a trajectory pays for
  perception once;
* failures degrade to ``connected`` with one warning, or raise on demand.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from goal.ml.data.fragments import FRAGMENT_SCHEMES, compute_fragment_index

pytest.importorskip("rdkit")

RDKIT_SCHEMES = [s for s in FRAGMENT_SCHEMES if s != "connected"]


def _butanoic_acid() -> tuple[torch.Tensor, torch.Tensor]:
    """CCCC(=O)O with explicit hydrogens, from a 3D embedding."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC(=O)O"))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    positions = torch.tensor(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=torch.float64,
    )
    numbers = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    return positions, numbers


def _two_waters(separation: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.tensor(
        [[0.0, 0.0, 0.0], [0.957, 0.0, 0.0], [-0.240, 0.927, 0.0]], dtype=torch.float64
    )
    shift = torch.tensor([separation, 0.0, 0.0], dtype=torch.float64)
    return torch.cat([base, base + shift]), torch.tensor([8, 1, 1, 8, 1, 1])


def _labels(positions, numbers, **kwargs) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_fragment_index(positions, numbers, **kwargs)


def _any_scheme(positions, numbers, scheme: str, **kwargs) -> torch.Tensor:
    """Labels for *scheme*, supplying patterns for ``custom``.

    ``custom`` has no meaning without ``fragment_smarts`` — it raises
    rather than guess — so the generic "every scheme" tests hand it one.
    """
    if scheme == "custom":
        kwargs.setdefault("smarts", ["[CX4]-[CX4]"])
    return _labels(positions, numbers, scheme=scheme, **kwargs)


class TestEverySchemeReturnsAValidPartition:
    @pytest.mark.parametrize("scheme", FRAGMENT_SCHEMES)
    def test_partition_is_well_formed(self, scheme: str) -> None:
        positions, numbers = _butanoic_acid()
        labels = _any_scheme(positions, numbers, scheme)

        assert labels.dtype == torch.long
        assert labels.shape == (positions.shape[0],)
        # Contiguous 0..K-1, every label present, atom 0 in fragment 0.
        assert {int(v) for v in labels.unique()} == set(range(int(labels.max()) + 1))
        assert int(labels[0]) == 0

    @pytest.mark.parametrize("scheme", FRAGMENT_SCHEMES)
    def test_deterministic(self, scheme: str) -> None:
        positions, numbers = _butanoic_acid()
        assert torch.equal(
            _any_scheme(positions, numbers, scheme),
            _any_scheme(positions, numbers, scheme),
        )

    @pytest.mark.parametrize("scheme", FRAGMENT_SCHEMES)
    def test_rotation_and_translation_invariant(self, scheme: str) -> None:
        from tests.ml.simurgh.conftest import random_so3

        positions, numbers = _butanoic_acid()
        rotation = random_so3()
        shift = torch.tensor([2.5, -1.0, 0.75], dtype=torch.float64)
        assert torch.equal(
            _any_scheme(positions, numbers, scheme),
            _any_scheme(positions @ rotation.T + shift, numbers, scheme),
        )

    def test_rejects_unknown_scheme(self) -> None:
        positions, numbers = _two_waters()
        with pytest.raises(ValueError, match="Unknown fragment scheme"):
            compute_fragment_index(positions, numbers, scheme="nonsense")


class TestSchemesDifferAsTheirDefinitionsSay:
    def test_saturated_acid_is_one_connected_fragment(self) -> None:
        """The molecule is covalently continuous, so distance-based and
        perception-based components both see exactly one fragment."""
        positions, numbers = _butanoic_acid()
        assert int(_labels(positions, numbers, scheme="connected").max()) + 1 == 1
        assert int(_labels(positions, numbers, scheme="rdkit_components").max()) + 1 == 1

    def test_rotatable_splits_the_chain_into_groups(self) -> None:
        positions, numbers = _butanoic_acid()
        labels = _labels(positions, numbers, scheme="rotatable")
        sizes = sorted(torch.bincount(labels).tolist(), reverse=True)

        assert int(labels.max()) + 1 == 5, "butanoic acid has 4 rotatable bonds"
        assert sizes == [4, 3, 3, 2, 2]
        assert sum(sizes) == positions.shape[0], "every atom must land in a group"

    def test_brics_only_snips_the_acid_group(self) -> None:
        """BRICS encodes retrosynthetic disconnections, and a saturated
        chain has almost none — this is the scheme behaving correctly, not
        failing."""
        positions, numbers = _butanoic_acid()
        labels = _labels(positions, numbers, scheme="brics")
        assert int(labels.max()) + 1 == 2
        assert sorted(torch.bincount(labels).tolist(), reverse=True) == [12, 2]

    def test_recap_isolates_the_bridging_oxygen(self) -> None:
        """Documented consequence: cutting both bonds around a bridging
        heteroatom leaves it as a one-atom fragment."""
        positions, numbers = _butanoic_acid()
        labels = _labels(positions, numbers, scheme="recap")
        sizes = sorted(torch.bincount(labels).tolist(), reverse=True)
        assert sizes == [11, 2, 1]

    def test_separate_molecules_split_under_every_scheme(self) -> None:
        """Whatever the scheme, two molecules 4 Å apart are never merged."""
        positions, numbers = _two_waters()
        for scheme in FRAGMENT_SCHEMES:
            labels = _any_scheme(positions, numbers, scheme)
            assert int(labels.max()) + 1 >= 2, f"{scheme} merged two molecules"
            assert set(labels[:3].tolist()).isdisjoint(
                set(labels[3:].tolist())
            ), f"{scheme} put atoms of different molecules in one fragment"


class TestPerceptionCaching:
    def test_same_topology_reuses_one_perception(self) -> None:
        """Perception depends only on connectivity and charge, so jiggled
        copies of one molecule must hit the cache — this is what keeps a
        7000-frame trajectory affordable."""
        from goal.ml.data import fragments

        fragments._LABEL_CACHE.clear()
        positions, numbers = _butanoic_acid()

        first = _labels(positions, numbers, scheme="rotatable")
        assert len(fragments._LABEL_CACHE) == 1

        torch.manual_seed(0)
        for _ in range(5):
            jiggled = positions + 0.01 * torch.randn_like(positions)
            assert torch.equal(_labels(jiggled, numbers, scheme="rotatable"), first)

        assert len(fragments._LABEL_CACHE) == 1, "cache missed on unchanged topology"

    def test_different_schemes_do_not_share_cache_entries(self) -> None:
        from goal.ml.data import fragments

        fragments._LABEL_CACHE.clear()
        positions, numbers = _butanoic_acid()
        rot = _labels(positions, numbers, scheme="rotatable")
        bri = _labels(positions, numbers, scheme="brics")

        assert len(fragments._LABEL_CACHE) == 2
        assert not torch.equal(rot, bri)


class TestFailureHandling:
    def _impossible(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Two atoms at an absurd distance with a charge rdkit cannot fit —
        perception is expected to fail."""
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float64)
        return positions, torch.tensor([6, 6], dtype=torch.long)

    def test_fallback_warns_once_and_uses_connected(self) -> None:
        from goal.ml.data import fragments

        fragments._WARNED.clear()
        fragments._LABEL_CACHE.clear()
        positions, numbers = self._impossible()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            labels = compute_fragment_index(
                positions, numbers, scheme="rotatable", charge=3, on_failure="fallback"
            )
            second = compute_fragment_index(
                positions, numbers, scheme="rotatable", charge=3, on_failure="fallback"
            )

        # Fell back to the distance graph: the two carbons are 1.2 Å apart,
        # inside the covalent cutoff, so they are one fragment.
        assert int(labels.max()) + 1 == 1
        assert torch.equal(labels, second)
        failures = [w for w in caught if "falling back" in str(w.message)]
        assert len(failures) == 1, "the warning should fire once, not once per frame"

    def test_raise_mode_propagates(self) -> None:
        from goal.ml.data import fragments

        fragments._LABEL_CACHE.clear()
        positions, numbers = self._impossible()
        with pytest.raises(Exception):  # noqa: B017 — rdkit raises its own types
            compute_fragment_index(
                positions, numbers, scheme="rotatable", charge=3, on_failure="raise"
            )

    def test_rejects_unknown_failure_mode(self) -> None:
        positions, numbers = _two_waters()
        with pytest.raises(ValueError, match="on_failure"):
            compute_fragment_index(positions, numbers, on_failure="explode")


class TestDatasetPlumbing:
    """``data.fragment_scheme`` must actually reach the dataset."""

    def test_scheme_reaches_the_graphs(self, tmp_path) -> None:  # noqa: ANN001
        from ase import Atoms
        from ase.io import write

        from goal.ml.data.datasets.xyz import ExtXYZDataset

        positions, numbers = _butanoic_acid()
        atoms = Atoms(numbers=numbers.tolist(), positions=positions.numpy())
        atoms.info["energy"] = -1.0
        atoms.arrays["forces"] = torch.zeros(len(numbers), 3).numpy()
        write(str(tmp_path / "train.xyz"), atoms, format="extxyz")

        def build(scheme: str):
            return ExtXYZDataset(
                root=tmp_path,
                cutoff=5.0,
                split="train",
                compute_fragment_index=True,
                fragment_scheme=scheme,
            )[0]

        assert build("connected").num_fragments == 1
        assert build("rotatable").num_fragments == 5
        assert build("brics").num_fragments == 2

    def test_datamodule_forwards_the_new_keys(self) -> None:
        from goal.ml.data.datamodule import _resolve_fragment_keys
        from goal.ml.data.datasets.trajectory import TrajectoryDataset

        extra = {
            "compute_fragment_index": True,
            "fragment_covalent_cutoff": 1.8,
            "fragment_scheme": "rotatable",
            "fragment_charge": 0,
            "fragment_on_failure": "fallback",
        }
        assert _resolve_fragment_keys(TrajectoryDataset, extra) == extra


# ----------------------------------------------------------------------
# Hinting: keep_groups (any scheme) and the custom SMARTS scheme
# ----------------------------------------------------------------------


class TestKeepGroups:
    """``fragment_keep_groups`` guarantees matched atoms share a fragment."""

    def test_protects_a_group_the_scheme_would_have_split(self) -> None:
        positions, numbers = _butanoic_acid()

        plain = _labels(positions, numbers, scheme="rotatable")
        kept = _labels(positions, numbers, scheme="rotatable", keep_groups=["C(=O)O"])

        # The carboxyl carbon, both oxygens and the acid H must coincide.
        from rdkit import Chem

        assert int(plain.max()) + 1 == 5
        assert int(kept.max()) + 1 == 4, "protecting the carboxyl should merge two groups"

        # Every atom of every C(=O)O match now carries one label.
        mol = Chem.AddHs(Chem.MolFromSmiles("CCCC(=O)O"))
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)O")):
            assert len({int(kept[i]) for i in match}) == 1

    def test_brics_only_cut_was_inside_the_protected_group(self) -> None:
        """A documented interaction worth knowing: BRICS' single cut on a
        saturated acid is the carboxyl C-OH bond, so protecting the group
        turns BRICS into a no-op on this molecule."""
        positions, numbers = _butanoic_acid()
        assert int(_labels(positions, numbers, scheme="brics").max()) + 1 == 2
        assert (
            int(_labels(positions, numbers, scheme="brics", keep_groups=["C(=O)O"]).max()) + 1 == 1
        )

    def test_merges_even_when_the_scheme_never_cuts(self) -> None:
        """``connected`` cuts nothing, so the guarantee can only come from
        the label merge — two waters 4 Å apart, forced together."""
        positions, numbers = _two_waters()
        assert int(_labels(positions, numbers, scheme="connected").max()) + 1 == 2
        merged = _labels(positions, numbers, scheme="connected", keep_groups=["[OX2]"])
        # [OX2] matches each water's O separately, so they stay apart …
        assert int(merged.max()) + 1 == 2

    def test_overlapping_groups_merge_transitively(self) -> None:
        positions, numbers = _butanoic_acid()
        chained = _labels(
            positions,
            numbers,
            scheme="rotatable",
            keep_groups=["C(=O)O", "[CX4]-[CX3]"],
        )
        # Two overlapping protections collapse their fragments into one.
        assert int(chained.max()) + 1 < 4

    def test_partition_stays_valid(self) -> None:
        positions, numbers = _butanoic_acid()
        labels = _labels(positions, numbers, scheme="rotatable", keep_groups=["C(=O)O"])
        assert {int(v) for v in labels.unique()} == set(range(int(labels.max()) + 1))
        assert labels.shape == (positions.shape[0],)

    def test_rejects_invalid_smarts_at_config_time(self) -> None:
        from goal.ml.data.fragments import validate_fragment_config

        with pytest.raises(ValueError, match="not a valid SMARTS"):
            validate_fragment_config("rotatable", keep_groups=["C(=O"])


class TestCustomScheme:
    def test_two_atom_pattern_cuts_that_bond(self) -> None:
        positions, numbers = _butanoic_acid()
        labels = _labels(positions, numbers, scheme="custom", smarts=["[CX4]-[CX4]"])
        # Butanoic acid has 2 C(sp3)-C(sp3) bonds → 3 fragments.
        assert int(labels.max()) + 1 == 3

    def test_atom_mapped_pattern_selects_the_bond(self) -> None:
        positions, numbers = _butanoic_acid()
        labels = _labels(positions, numbers, scheme="custom", smarts=["[C:1](=O)-[O:2]"])
        assert int(labels.max()) + 1 == 2

    def test_multiple_patterns_compose(self) -> None:
        positions, numbers = _butanoic_acid()
        both = _labels(
            positions,
            numbers,
            scheme="custom",
            smarts=["[CX4]-[CX4]", "[C:1](=O)-[O:2]"],
        )
        assert int(both.max()) + 1 == 4

    def test_custom_respects_keep_groups(self) -> None:
        positions, numbers = _butanoic_acid()
        cut = _labels(positions, numbers, scheme="custom", smarts=["[C:1](=O)-[O:2]"])
        kept = _labels(
            positions,
            numbers,
            scheme="custom",
            smarts=["[C:1](=O)-[O:2]"],
            keep_groups=["C(=O)O"],
        )
        assert int(cut.max()) + 1 == 2
        assert int(kept.max()) + 1 == 1

    def test_custom_without_patterns_is_rejected(self) -> None:
        positions, numbers = _butanoic_acid()
        with pytest.raises(ValueError, match="requires data.fragment_smarts"):
            compute_fragment_index(positions, numbers, scheme="custom")

    def test_ambiguous_pattern_is_rejected(self) -> None:
        from goal.ml.data.fragments import validate_fragment_config

        with pytest.raises(ValueError, match="ambiguous"):
            validate_fragment_config("custom", smarts=["CCC"])

    def test_hints_reach_the_dataset(self, tmp_path) -> None:  # noqa: ANN001
        from ase import Atoms
        from ase.io import write

        from goal.ml.data.datasets.xyz import ExtXYZDataset

        positions, numbers = _butanoic_acid()
        atoms = Atoms(numbers=numbers.tolist(), positions=positions.numpy())
        atoms.info["energy"] = -1.0
        atoms.arrays["forces"] = torch.zeros(len(numbers), 3).numpy()
        write(str(tmp_path / "train.xyz"), atoms, format="extxyz")

        graph = ExtXYZDataset(
            root=tmp_path,
            cutoff=5.0,
            split="train",
            compute_fragment_index=True,
            fragment_scheme="rotatable",
            fragment_keep_groups=["C(=O)O"],
        )[0]
        assert graph.num_fragments == 4

        custom = ExtXYZDataset(
            root=tmp_path,
            cutoff=5.0,
            split="train",
            compute_fragment_index=True,
            fragment_scheme="custom",
            fragment_smarts=["[CX4]-[CX4]"],
        )[0]
        assert custom.num_fragments == 3

    def test_bad_config_fails_at_dataset_construction(self, tmp_path) -> None:  # noqa: ANN001
        from ase import Atoms
        from ase.io import write

        from goal.ml.data.datasets.xyz import ExtXYZDataset

        positions, numbers = _butanoic_acid()
        atoms = Atoms(numbers=numbers.tolist(), positions=positions.numpy())
        write(str(tmp_path / "train.xyz"), atoms, format="extxyz")

        with pytest.raises(ValueError, match="requires data.fragment_smarts"):
            ExtXYZDataset(
                root=tmp_path,
                cutoff=5.0,
                split="train",
                compute_fragment_index=True,
                fragment_scheme="custom",
            )


class TestConfigMistakes:
    """The two ways a hand-written config silently does the wrong thing."""

    def test_smarts_with_a_non_custom_scheme_warns(self) -> None:
        from goal.ml.data import fragments

        fragments._WARNED.clear()
        positions, numbers = _butanoic_acid()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            labels = compute_fragment_index(
                positions, numbers, scheme="brics", smarts=["[CX4]-[CX4]"]
            )

        ignored = [w for w in caught if "IGNORES the" in str(w.message)]
        assert ignored, "setting fragment_smarts on a rule-based scheme must warn"
        # …and the patterns really are ignored: plain BRICS output.
        assert torch.equal(labels, _labels(positions, numbers, scheme="brics"))

    def test_bare_string_is_taken_as_one_pattern(self) -> None:
        """`fragment_smarts: "[CX4]-[CX4]"` without list brackets used to be
        iterated character by character."""
        positions, numbers = _butanoic_acid()
        assert torch.equal(
            _labels(positions, numbers, scheme="custom", smarts="[CX4]-[CX4]"),
            _labels(positions, numbers, scheme="custom", smarts=["[CX4]-[CX4]"]),
        )
        assert torch.equal(
            _labels(positions, numbers, scheme="rotatable", keep_groups="C(=O)O"),
            _labels(positions, numbers, scheme="rotatable", keep_groups=["C(=O)O"]),
        )


class TestPositiveGroupSpecification:
    """Defining fragments by naming the bond BETWEEN groups.

    The intent "carboxyl is one fragment, the alkyl chain is another" is
    expressed by cutting the single bond that joins them — one pattern,
    not one pattern per group.
    """

    def test_carboxyl_versus_alkyl_chain(self) -> None:
        positions, numbers = _butanoic_acid()
        labels = _labels(positions, numbers, scheme="custom", smarts=["[CX4]-[CX3]"])

        assert int(labels.max()) + 1 == 2
        from rdkit import Chem

        mol = Chem.AddHs(Chem.MolFromSmiles("CCCC(=O)O"))
        (carboxyl,) = mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)O"))
        # The whole carboxyl sits in one fragment …
        assert len({int(labels[i]) for i in carboxyl}) == 1
        # … and no chain carbon shares it.
        chain = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetSymbol() == "C" and a.GetIdx() not in carboxyl
        ]
        assert {int(labels[i]) for i in chain}.isdisjoint({int(labels[carboxyl[0]])})

    def test_also_splitting_the_chain_into_units(self) -> None:
        positions, numbers = _butanoic_acid()
        labels = _labels(
            positions, numbers, scheme="custom", smarts=["[CX4]-[CX4]", "[CX4]-[CX3]"]
        )
        sizes = sorted(torch.bincount(labels).tolist(), reverse=True)
        # carboxyl (4) + CH2 (3) + CH2 (3) + CH3 (4) for butanoic acid.
        assert int(labels.max()) + 1 == 4
        assert sizes == [4, 4, 3, 3]
