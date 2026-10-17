"""Dataset-level plumbing for fragment labels (Part 0 of the add-on work).

Complements ``tests/ml/simurgh/test_fragment_ca.py`` (which covers the
label algorithm and batching) by checking the path the labels actually
travel in a run: ``data.compute_fragment_index`` →
``BaseAtomicDataset`` → ``AtomicGraph.from_ase`` → ``graph.fragment_index``.

Real ASE files written to ``tmp_path`` — the point is the loader wiring, so
mocking the loader would test nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("ase")


def _write_two_waters(path: Path) -> Path:
    """An extXYZ frame holding two water molecules 4 Å apart."""
    from ase import Atoms
    from ase.io import write

    positions = [
        (0.000, 0.000, 0.000),
        (0.957, 0.000, 0.000),
        (-0.240, 0.927, 0.000),
        (4.000, 0.000, 0.000),
        (4.957, 0.000, 0.000),
        (3.760, 0.927, 0.000),
    ]
    atoms = Atoms("OHHOHH", positions=positions)
    atoms.info["energy"] = -152.0
    atoms.arrays["forces"] = torch.zeros(6, 3).numpy()
    write(str(path), atoms, format="extxyz")
    return path


class TestDatasetAttachesFragmentLabels:
    def test_enabled_labels_two_molecules(self, tmp_path: Path) -> None:
        from goal.ml.data.datasets.xyz import ExtXYZDataset

        _write_two_waters(tmp_path / "train.xyz")
        ds = ExtXYZDataset(
            root=tmp_path,
            cutoff=5.0,
            split="train",
            compute_fragment_index=True,
            fragment_covalent_cutoff=1.8,
        )
        graph = ds[0]

        assert graph.fragment_index is not None
        assert graph.fragment_index.dtype == torch.long
        assert graph.num_fragments == 2
        assert torch.equal(graph.fragment_index, torch.tensor([0, 0, 0, 1, 1, 1]))

    def test_disabled_by_default(self, tmp_path: Path) -> None:
        from goal.ml.data.datasets.xyz import ExtXYZDataset

        _write_two_waters(tmp_path / "train.xyz")
        ds = ExtXYZDataset(root=tmp_path, cutoff=5.0, split="train")
        assert ds[0].fragment_index is None

    def test_cutoff_controls_the_decomposition(self, tmp_path: Path) -> None:
        from goal.ml.data.datasets.xyz import ExtXYZDataset

        _write_two_waters(tmp_path / "train.xyz")
        # 5 Å joins the two molecules into one fragment; 0.5 Å splits every
        # atom apart.  Both are wrong for chemistry — the point is that the
        # knob reaches the algorithm.
        joined = ExtXYZDataset(
            root=tmp_path,
            cutoff=5.0,
            split="train",
            compute_fragment_index=True,
            fragment_covalent_cutoff=5.0,
        )[0]
        shattered = ExtXYZDataset(
            root=tmp_path,
            cutoff=5.0,
            split="train",
            compute_fragment_index=True,
            fragment_covalent_cutoff=0.5,
        )[0]

        assert joined.num_fragments == 1
        assert shattered.num_fragments == 6


class TestUnsupportedDatasetHandling:
    """``goal.ml.data.benchmarks`` loaders are a separate hierarchy that never
    learned about fragment labels — the keys must not break them when the
    feature is off, and must fail loudly when it is on."""

    def _fake_dataset_cls(self) -> type:
        class LegacyDataset:  # noqa: D401 — stand-in for a benchmark loader
            def __init__(self, root, cutoff, split="train", dtype=torch.float64):
                self.root, self.cutoff, self.split = root, cutoff, split

        return LegacyDataset

    def test_disabled_keys_are_dropped(self) -> None:
        from goal.ml.data.datamodule import _resolve_fragment_keys

        extra = {
            "energy_key": "energy",
            "compute_fragment_index": False,
            "fragment_covalent_cutoff": 1.8,
        }
        resolved = _resolve_fragment_keys(self._fake_dataset_cls(), extra)
        assert resolved == {"energy_key": "energy"}

    def test_enabled_keys_raise_a_pointed_error(self) -> None:
        from goal.ml.data.datamodule import _resolve_fragment_keys

        with pytest.raises(ValueError, match="does not support fragment labels"):
            _resolve_fragment_keys(self._fake_dataset_cls(), {"compute_fragment_index": True})

    def test_supporting_datasets_keep_the_keys(self) -> None:
        from goal.ml.data.datamodule import _resolve_fragment_keys
        from goal.ml.data.datasets.trajectory import TrajectoryDataset

        extra = {"compute_fragment_index": True, "fragment_covalent_cutoff": 1.8}
        assert _resolve_fragment_keys(TrajectoryDataset, extra) == extra

    def test_absent_keys_are_a_no_op(self) -> None:
        from goal.ml.data.datamodule import _resolve_fragment_keys

        extra = {"energy_key": "energy"}
        assert _resolve_fragment_keys(self._fake_dataset_cls(), extra) is extra
