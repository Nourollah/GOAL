"""Tests for benchmark example datasets — all mocked, no real downloads."""

from __future__ import annotations

import typing
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Fixtures: fake PyG datasets
# ---------------------------------------------------------------------------


def _make_fake_md17_data(n: int = 100) -> list[Data]:
    """Create fake MD17-like Data objects (legacy, kept for reference)."""
    items: list[Data] = []
    for _ in range(n):
        num_atoms = 21  # aspirin
        data = Data(
            pos=torch.randn(num_atoms, 3, dtype=torch.float64),
            z=torch.randint(1, 9, (num_atoms,)),
            energy=torch.tensor([-300.0 + torch.randn(1).item()]),
            force=torch.randn(num_atoms, 3, dtype=torch.float64),
        )
        items.append(data)
    return items


def _make_fake_atomic_graphs(n: int = 100) -> list:
    """Create fake AtomicGraph objects with energies already in eV range."""
    from goal.ml.data.graph import AtomicGraph

    num_atoms = 21  # aspirin
    graphs = []
    for _ in range(n):
        pos = torch.randn(num_atoms, 3, dtype=torch.float64)
        graphs.append(
            AtomicGraph(
                positions=pos,
                atomic_numbers=torch.randint(1, 9, (num_atoms,)),
                cell=torch.zeros(3, 3, dtype=torch.float64),
                pbc=torch.zeros(3, dtype=torch.bool),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                energy=torch.tensor([-13.0 + 0.1 * torch.randn(1).item()], dtype=torch.float64),
                forces=torch.randn(num_atoms, 3, dtype=torch.float64),
            )
        )
    return graphs


def _make_fake_ani1_data(n: int = 50) -> list[Data]:
    """Create fake ANI-1-like Data objects."""
    items: list[Data] = []
    for _ in range(n):
        num_atoms = torch.randint(3, 15, (1,)).item()
        data = Data(
            pos=torch.randn(num_atoms, 3, dtype=torch.float64),
            z=torch.randint(1, 9, (num_atoms,)),
            energy=torch.tensor([-76.0 + 0.1 * torch.randn(1).item()]),
            force=torch.randn(num_atoms, 3, dtype=torch.float64),
        )
        items.append(data)
    return items


def _make_fake_qm9_data(n: int = 200) -> list[Data]:
    """Create fake QM9-like Data objects with 19 properties."""
    items: list[Data] = []
    for _ in range(n):
        num_atoms = torch.randint(3, 9, (1,)).item()
        data = Data(
            pos=torch.randn(num_atoms, 3, dtype=torch.float64),
            z=torch.randint(1, 9, (num_atoms,)),
            y=torch.randn(1, 19, dtype=torch.float64),
        )
        items.append(data)
    return items


# ---------------------------------------------------------------------------
# Test: BenchmarkDataset base class
# ---------------------------------------------------------------------------


class TestBenchmarkDatasetBase:
    """Test shared base class functionality."""

    def test_invalid_split_raises(self, tmp_path: Path):
        """Invalid split name should raise ValueError."""
        from goal.ml.data.benchmarks.base import BenchmarkDataset

        class _DummyDataset(BenchmarkDataset):
            def _download_and_process(self):
                return [Data(pos=torch.randn(3, 3), z=torch.ones(3, dtype=torch.long))]

            def split_indices(self):
                return {"train": [0], "val": [], "test": []}

            def citation(self):
                return "@article{dummy}"

        with pytest.raises(ValueError, match="split must be"):
            _DummyDataset(root=str(tmp_path), cutoff=5.0, split="invalid")

    def test_random_split_indices_deterministic(self):
        """Same seed should produce identical splits."""
        from goal.ml.data.benchmarks.base import BenchmarkDataset

        s1 = BenchmarkDataset._random_split_indices(1000, 800, 100, seed=42)
        s2 = BenchmarkDataset._random_split_indices(1000, 800, 100, seed=42)
        assert s1["train"] == s2["train"]
        assert s1["val"] == s2["val"]
        assert s1["test"] == s2["test"]

    def test_random_split_indices_sizes(self):
        """Split sizes should sum to total."""
        from goal.ml.data.benchmarks.base import BenchmarkDataset

        s = BenchmarkDataset._random_split_indices(1000, 800, 100, seed=42)
        assert len(s["train"]) == 800
        assert len(s["val"]) == 100
        assert len(s["test"]) == 100
        all_indices = set(s["train"] + s["val"] + s["test"])
        assert len(all_indices) == 1000

    def test_random_split_different_seeds(self):
        """Different seeds should produce different splits."""
        from goal.ml.data.benchmarks.base import BenchmarkDataset

        s1 = BenchmarkDataset._random_split_indices(1000, 800, 100, seed=42)
        s2 = BenchmarkDataset._random_split_indices(1000, 800, 100, seed=99)
        assert s1["train"] != s2["train"]


# ---------------------------------------------------------------------------
# Test: MD17Dataset
# ---------------------------------------------------------------------------


class TestMD17Dataset:
    """Test MD17 dataset adapter with mocked PyG download."""

    @pytest.fixture()
    def mock_pyg_md17(self):
        """Patch MD17Dataset._download_and_process to avoid network calls."""
        from goal.ml.data.benchmarks.md17 import MD17Dataset

        fake_graphs = _make_fake_atomic_graphs(100)
        with patch.object(MD17Dataset, "_download_and_process", return_value=fake_graphs) as mock:
            yield mock

    def test_creates_cache_on_first_call(self, tmp_path: Path, mock_pyg_md17):
        """First call should create a .pt cache file."""
        from goal.ml.data.benchmarks.md17 import MD17Dataset

        ds = MD17Dataset(root=str(tmp_path), molecule="aspirin", split="train")
        cache_files = list((tmp_path / "processed").glob("*.pt"))
        assert len(cache_files) == 1
        assert "rmd17_aspirin" in cache_files[0].name

    def test_loads_from_cache_on_second_call(self, tmp_path: Path, mock_pyg_md17):
        """Second call should load from cache, not re-download."""
        from goal.ml.data.benchmarks.md17 import MD17Dataset

        MD17Dataset(root=str(tmp_path), molecule="aspirin", split="train")
        call_count_first = mock_pyg_md17.call_count

        MD17Dataset(root=str(tmp_path), molecule="aspirin", split="train")
        assert mock_pyg_md17.call_count == call_count_first  # no new PyG call

    def test_split_sizes_sum_to_total(self, tmp_path: Path, mock_pyg_md17):
        """Train + val + test should equal total dataset size."""
        from goal.ml.data.benchmarks.md17 import MD17Dataset

        train = MD17Dataset(root=str(tmp_path), molecule="aspirin", split="train")
        val = MD17Dataset(root=str(tmp_path), molecule="aspirin", split="val")
        test = MD17Dataset(root=str(tmp_path), molecule="aspirin", split="test")
        assert len(train) + len(val) + len(test) == 100

    def test_default_split_sizes(self, tmp_path: Path, mock_pyg_md17):
        """Default: 950 train, 50 val (but we only have 100 fake items)."""
        from goal.ml.data.benchmarks.md17 import MD17Dataset

        # With 100 items, train_size=950 exceeds total, so test would be empty
        ds = MD17Dataset(
            root=str(tmp_path),
            molecule="aspirin",
            split="train",
            train_size=80,
            val_size=10,
        )
        assert len(ds) == 80

    def test_unit_conversion(self, tmp_path: Path, mock_pyg_md17):
        """Energies should be converted from kcal/mol to eV."""
        from goal.ml.data.benchmarks.md17 import KCAL_TO_EV, MD17Dataset

        ds = MD17Dataset(
            root=str(tmp_path),
            molecule="aspirin",
            split="train",
            train_size=80,
            val_size=10,
        )
        graph = ds[0]
        # Energy should be in eV (original is ~-300 kcal/mol × 0.043364)
        assert graph.energy is not None
        assert abs(graph.energy.item()) < 100  # eV range, not kcal range

    def test_returns_atomic_graph(self, tmp_path: Path, mock_pyg_md17):
        """Items should have required AtomicGraph fields."""
        from goal.ml.data.benchmarks.md17 import MD17Dataset

        ds = MD17Dataset(
            root=str(tmp_path),
            molecule="aspirin",
            split="train",
            train_size=80,
            val_size=10,
        )
        graph = ds[0]
        assert hasattr(graph, "pos")
        assert hasattr(graph, "z")
        assert hasattr(graph, "edge_index")
        assert hasattr(graph, "energy")
        assert hasattr(graph, "forces")

    def test_invalid_molecule_raises(self, tmp_path: Path):
        """Unknown molecule name should raise ValueError."""
        from goal.ml.data.benchmarks.md17 import MD17Dataset

        with pytest.raises(ValueError, match="Unknown molecule"):
            MD17Dataset(root=str(tmp_path), molecule="nonexistent", split="train")

    def test_citation_nonempty(self, tmp_path: Path, mock_pyg_md17):
        """Citation should be non-empty BibTeX."""
        from goal.ml.data.benchmarks.md17 import MD17Dataset

        ds = MD17Dataset(
            root=str(tmp_path),
            molecule="aspirin",
            split="train",
            train_size=80,
            val_size=10,
        )
        cit = ds.citation()
        assert "@article" in cit
        assert "chmiela" in cit


# ---------------------------------------------------------------------------
# Test: ANI1Dataset
# ---------------------------------------------------------------------------


class TestANI1Dataset:
    """Test ANI-1/1x adapter with mocked PyG download."""

    @pytest.fixture()
    def mock_pyg_ani1(self):
        """Patch PyG ANI1 and ANI1x."""
        fake_data = _make_fake_ani1_data(50)
        with (
            patch("goal.ml.data.benchmarks.ani1.ANI1x") as mock_1x,
            patch("goal.ml.data.benchmarks.ani1.ANI1") as mock_1,
        ):
            mock_1x.return_value = fake_data
            mock_1.return_value = fake_data
            yield mock_1x, mock_1

    def test_ani1x_loads(self, tmp_path: Path, mock_pyg_ani1):
        """ANI-1x should load without error."""
        from goal.ml.data.benchmarks.ani1 import ANI1Dataset

        ds = ANI1Dataset(root=str(tmp_path), version="1x", split="train")
        assert len(ds) > 0

    def test_max_structures(self, tmp_path: Path, mock_pyg_ani1):
        """max_structures should limit dataset size."""
        from goal.ml.data.benchmarks.ani1 import ANI1Dataset

        ds = ANI1Dataset(
            root=str(tmp_path),
            version="1x",
            split="train",
            max_structures=10,
        )
        # Total is capped at 10, train fraction 0.8 → 8
        assert len(ds._data) <= 10

    def test_invalid_version_raises(self, tmp_path: Path):
        """Invalid version should raise ValueError."""
        from goal.ml.data.benchmarks.ani1 import ANI1Dataset

        with pytest.raises(ValueError, match="version must be"):
            ANI1Dataset(root=str(tmp_path), version="2", split="train")


# ---------------------------------------------------------------------------
# Test: QM9Dataset
# ---------------------------------------------------------------------------


class TestQM9Dataset:
    """Test QM9 adapter with mocked PyG download."""

    @pytest.fixture()
    def mock_pyg_qm9(self):
        """Patch PyG's QM9 dataset."""
        fake_data = _make_fake_qm9_data(200)
        with patch("goal.ml.data.benchmarks.qm9.PyGQM9") as mock_cls:
            mock_cls.return_value = fake_data
            yield mock_cls

    def test_qm9_loads_with_target_name(self, tmp_path: Path, mock_pyg_qm9):
        """QM9 should accept target name like 'homo'."""
        from goal.ml.data.benchmarks.qm9 import QM9Dataset

        ds = QM9Dataset(root=str(tmp_path), target="homo", split="train")
        assert len(ds) > 0

    def test_qm9_loads_with_target_index(self, tmp_path: Path, mock_pyg_qm9):
        """QM9 should accept target index like 4 (gap)."""
        from goal.ml.data.benchmarks.qm9 import QM9Dataset

        ds = QM9Dataset(root=str(tmp_path), target=4, split="train")
        assert ds.target_name == "gap"

    def test_qm9_all_targets(self, tmp_path: Path, mock_pyg_qm9):
        """target=None should store all 19 properties."""
        from goal.ml.data.benchmarks.qm9 import QM9Dataset

        ds = QM9Dataset(root=str(tmp_path), target=None, split="train")
        graph = ds[0]
        assert graph.energy is not None

    def test_invalid_target_raises(self, tmp_path: Path):
        """Unknown target name should raise ValueError."""
        from goal.ml.data.benchmarks.qm9 import QM9Dataset

        with pytest.raises(ValueError, match="Unknown QM9 target"):
            QM9Dataset(root=str(tmp_path), target="nonexistent", split="train")

    def test_invalid_target_index_raises(self, tmp_path: Path):
        """Out-of-range target index should raise ValueError."""
        from goal.ml.data.benchmarks.qm9 import QM9Dataset

        with pytest.raises(ValueError, match="target index must be"):
            QM9Dataset(root=str(tmp_path), target=99, split="train")


# ---------------------------------------------------------------------------
# Test: MD22
# ---------------------------------------------------------------------------


def _write_md22_npz(
    root: Path,
    molecule: str = "AT-AT",
    n_frames: int = 40,
    n_atoms: int | None = None,
    energy_unit: str = "kcal/mol",
    length_unit: str = "Ang",
) -> Path:
    """Write a miniature sGDML-format .npz where the loader expects it.

    Real files rather than mocks: the layout (``E`` shaped ``(N, 1)``,
    the declared unit strings, the kcal/mol scale) is exactly what the
    loader has to cope with, and a mock would assert nothing about it.
    """
    import numpy as np

    from goal.ml.data.benchmarks.md22 import MD22_MOLECULES

    filename, default_atoms, _ = MD22_MOLECULES[molecule]
    count: int = default_atoms if n_atoms is None else n_atoms
    raw: Path = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path: Path = raw / filename
    rng = np.random.default_rng(0)
    np.savez(
        path,
        z=np.resize(np.array([1, 6, 7, 8], dtype=np.int64), count),
        R=rng.normal(scale=3.0, size=(n_frames, count, 3)),
        E=rng.normal(loc=-1_000.0, size=(n_frames, 1)),  # (N, 1), as MD22 ships
        F=rng.normal(size=(n_frames, count, 3)),
        e_unit=np.array(energy_unit),
        r_unit=np.array(length_unit),
    )
    return path


class TestMD22Dataset:
    """MD22 adapter — real miniature .npz files, no downloads."""

    def test_registered(self):
        import goal.ml.data.benchmarks  # noqa: F401
        from goal.ml.registry import DATASET_REGISTRY

        assert DATASET_REGISTRY.get("md22").__name__ == "MD22Dataset"

    def test_loads_and_yields_atomic_graphs(self, tmp_path: Path):
        from goal.ml.data.benchmarks.md22 import MD22Dataset
        from goal.ml.data.graph import AtomicGraph

        _write_md22_npz(tmp_path, n_frames=40)
        dataset = MD22Dataset(
            root=str(tmp_path), molecule="AT-AT", train_size=20, val_size=5, cutoff=5.0
        )
        assert len(dataset) == 20

        graph = dataset[0]
        assert isinstance(graph, AtomicGraph)
        assert graph.z.shape[0] == 60
        assert graph.energy.shape == (1,)
        assert graph.forces.shape == (60, 3)

    def test_units_are_converted_to_ev(self, tmp_path: Path):
        """kcal/mol → eV, using the same constant as the MD17 adapter."""
        import numpy as np

        from goal.ml.data.benchmarks.md17 import KCAL_TO_EV
        from goal.ml.data.benchmarks.md22 import MD22Dataset

        path = _write_md22_npz(tmp_path, n_frames=10)
        raw = np.load(path)
        dataset = MD22Dataset(
            root=str(tmp_path), molecule="AT-AT", train_size=10, val_size=0, cutoff=5.0
        )
        # split_indices shuffles, so compare the set of energies.
        produced = sorted(float(dataset[i].energy) for i in range(len(dataset)))
        expected = sorted(float(e) * KCAL_TO_EV for e in raw["E"].reshape(-1))
        assert produced == pytest.approx(expected, rel=1e-9)

    def test_default_train_size_is_per_molecule(self, tmp_path: Path):
        """Published MD22 numbers sit at these sizes, so they are defaults."""
        from goal.ml.data.benchmarks.md22 import MD22_TRAIN_SIZES, MD22Dataset

        _write_md22_npz(tmp_path, n_frames=4_000)
        dataset = MD22Dataset(root=str(tmp_path), molecule="AT-AT", val_size=10, cutoff=5.0)
        assert dataset.train_size == MD22_TRAIN_SIZES["AT-AT"] == 3_000
        assert len(dataset) == 3_000

    def test_train_size_override(self, tmp_path: Path):
        from goal.ml.data.benchmarks.md22 import MD22Dataset

        _write_md22_npz(tmp_path, n_frames=40)
        dataset = MD22Dataset(
            root=str(tmp_path), molecule="AT-AT", train_size=7, val_size=3, cutoff=5.0
        )
        assert len(dataset) == 7

    def test_split_is_disjoint_and_covers_everything(self, tmp_path: Path):
        from goal.ml.data.benchmarks.md22 import MD22Dataset

        _write_md22_npz(tmp_path, n_frames=40)
        common = dict(root=str(tmp_path), molecule="AT-AT", train_size=20, val_size=5, cutoff=5.0)
        splits = {
            name: set(MD22Dataset(split=name, **common)._indices)
            for name in ("train", "val", "test")
        }
        assert len(splits["train"]) == 20
        assert len(splits["val"]) == 5
        assert len(splits["test"]) == 15
        assert not splits["train"] & splits["val"]
        assert set().union(*splits.values()) == set(range(40))

    def test_oversized_split_is_refused(self, tmp_path: Path):
        """A clear error, not a silently short training set."""
        from goal.ml.data.benchmarks.md22 import MD22Dataset

        _write_md22_npz(tmp_path, n_frames=40)
        with pytest.raises(ValueError, match="exceeds the 40 conformations"):
            MD22Dataset(
                root=str(tmp_path),
                molecule="AT-AT",
                train_size=38,
                val_size=10,
                cutoff=5.0,
            )

    def test_unknown_molecule_is_refused(self, tmp_path: Path):
        from goal.ml.data.benchmarks.md22 import MD22Dataset

        with pytest.raises(ValueError, match="Unknown MD22 molecule"):
            MD22Dataset(root=str(tmp_path), molecule="aspirin")

    def test_wrong_units_are_refused(self, tmp_path: Path):
        """A silently mis-scaled energy trains happily and reports nonsense."""
        from goal.ml.data.benchmarks.md22 import MD22Dataset

        _write_md22_npz(tmp_path, n_frames=10, energy_unit="eV")
        with pytest.raises(ValueError, match="declares units"):
            MD22Dataset(
                root=str(tmp_path),
                molecule="AT-AT",
                train_size=5,
                val_size=1,
                cutoff=5.0,
            )

    def test_wrong_atom_count_is_refused(self, tmp_path: Path):
        """Guards against a truncated download or a mislabelled file."""
        from goal.ml.data.benchmarks.md22 import MD22Dataset

        _write_md22_npz(tmp_path, n_frames=10, n_atoms=59)
        with pytest.raises(ValueError, match="expected 60"):
            MD22Dataset(
                root=str(tmp_path),
                molecule="AT-AT",
                train_size=5,
                val_size=1,
                cutoff=5.0,
            )

    def test_fragment_options_are_forwarded(self, tmp_path: Path):
        from goal.ml.data.benchmarks.md22 import MD22Dataset

        _write_md22_npz(tmp_path, n_frames=10)
        dataset = MD22Dataset(
            root=str(tmp_path),
            molecule="AT-AT",
            train_size=5,
            val_size=1,
            cutoff=5.0,
            compute_fragment_index=True,
            fragment_scheme="connected",
        )
        graph = dataset[0]
        assert graph.fragment_index is not None
        assert graph.fragment_index.shape[0] == graph.z.shape[0]

    def test_citation_nonempty(self, tmp_path: Path):
        from goal.ml.data.benchmarks.md22 import MD22Dataset

        _write_md22_npz(tmp_path, n_frames=10)
        dataset = MD22Dataset(
            root=str(tmp_path), molecule="AT-AT", train_size=5, val_size=1, cutoff=5.0
        )
        assert "chmiela2023accurate" in dataset.citation()


# ---------------------------------------------------------------------------
# Test: fragment labels on benchmark datasets
# ---------------------------------------------------------------------------


def _two_molecules(separation: float) -> typing.Any:
    """Two H2 molecules *separation* Angstrom apart, as an AtomicGraph.

    At 1.0 Angstrom apart within a molecule and ``separation`` between
    them, ``fragment_scheme: connected`` with the default 1.8 Angstrom
    bond threshold sees one fragment when they overlap and two when they
    do not — a decomposition with an answer known by construction.
    """
    from goal.ml.data.graph import AtomicGraph

    positions: torch.Tensor = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [separation, 0.0, 0.0],
            [separation + 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    return AtomicGraph(
        positions=positions,
        atomic_numbers=torch.ones(4, dtype=torch.long),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.zeros(3, dtype=torch.bool),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        energy=torch.tensor([-1.0], dtype=torch.float64),
    )


def _fragment_dataset(root: Path, separation: float = 10.0, **kwargs: typing.Any):
    """A minimal BenchmarkDataset over one two-molecule structure."""
    from goal.ml.data.benchmarks.base import BenchmarkDataset

    class _Dummy(BenchmarkDataset):
        def _download_and_process(self):
            return [_two_molecules(separation)]

        def split_indices(self):
            return {"train": [0], "val": [], "test": []}

        def citation(self):
            return "@article{dummy}"

    return _Dummy(root=str(root), cutoff=5.0, split="train", **kwargs)


class TestBenchmarkFragmentLabels:
    """``compute_fragment_index`` and friends on the benchmark adapters.

    NF_SIMURGH pools per-atom energies onto fragment virtual nodes and
    raises without ``fragment_index``, so a benchmark adapter that
    ignored these options could not train it at all — and the DataModule
    forwards them to whatever dataset is configured.
    """

    def test_off_by_default(self, tmp_path: Path):
        dataset = _fragment_dataset(tmp_path)
        assert dataset[0].get("fragment_index", None) is None

    def test_labels_attached_when_enabled(self, tmp_path: Path):
        dataset = _fragment_dataset(tmp_path, compute_fragment_index=True)
        graph = dataset[0]
        assert graph.fragment_index is not None
        assert graph.fragment_index.shape == (4,)

    def test_connected_separates_distant_molecules(self, tmp_path: Path):
        far = _fragment_dataset(tmp_path / "far", 10.0, compute_fragment_index=True)
        assert int(far[0].fragment_index.max()) + 1 == 2

        near = _fragment_dataset(tmp_path / "near", 1.0, compute_fragment_index=True)
        assert int(near[0].fragment_index.max()) + 1 == 1

    def test_covalent_cutoff_is_honoured(self, tmp_path: Path):
        """A threshold above the separation merges the two molecules."""
        dataset = _fragment_dataset(
            tmp_path,
            separation=2.5,
            compute_fragment_index=True,
            fragment_covalent_cutoff=3.0,
        )
        assert int(dataset[0].fragment_index.max()) + 1 == 1

    def test_labels_are_memoised_in_place(self, tmp_path: Path):
        """Second access must reuse the labels, not recompute them.

        This is what keeps the rdkit schemes affordable on a corpus of
        134k distinct molecules: the cost lands in epoch 1 only.
        """
        dataset = _fragment_dataset(tmp_path, compute_fragment_index=True)
        first = dataset[0].fragment_index
        assert dataset[0].fragment_index is first

    def test_bad_scheme_fails_at_construction(self, tmp_path: Path):
        """Not thousands of frames into the first epoch."""
        with pytest.raises(ValueError, match="fragment_scheme"):
            _fragment_dataset(
                tmp_path, compute_fragment_index=True, fragment_scheme="not_a_scheme"
            )

    def test_scheme_not_validated_when_disabled(self, tmp_path: Path):
        """A stale scheme name is harmless while the feature is off."""
        dataset = _fragment_dataset(tmp_path, fragment_scheme="not_a_scheme")
        assert dataset[0].get("fragment_index", None) is None

    def test_bare_smarts_string_is_wrapped(self, tmp_path: Path):
        """A scalar in YAML means one pattern, not a list of characters."""
        dataset = _fragment_dataset(
            tmp_path,
            compute_fragment_index=True,
            fragment_scheme="custom",
            fragment_smarts="[CX4]-[CX4]",
        )
        assert dataset.fragment_smarts == ["[CX4]-[CX4]"]

    def test_md17_forwards_fragment_options(self, tmp_path: Path):
        """The adapters pass ``**fragment_kwargs`` through to the base."""
        from goal.ml.data.benchmarks.md17 import MD17Dataset

        with patch.object(
            MD17Dataset, "_download_and_process", return_value=[_two_molecules(10.0)]
        ):
            dataset = MD17Dataset(
                root=str(tmp_path),
                molecule="aspirin",
                cutoff=5.0,
                split="train",
                train_size=1,
                val_size=0,
                compute_fragment_index=True,
                fragment_scheme="connected",
            )
            assert int(dataset[0].fragment_index.max()) + 1 == 2

    def test_qm9_forwards_fragment_options(self, tmp_path: Path, mock_pyg_qm9_module):
        from goal.ml.data.benchmarks.qm9 import QM9Dataset

        dataset = QM9Dataset(
            root=str(tmp_path),
            target="u0",
            split="train",
            compute_fragment_index=True,
            fragment_scheme="connected",
        )
        assert dataset[0].fragment_index.shape[0] == dataset[0].z.shape[0]

    def test_unknown_option_still_raises(self, tmp_path: Path):
        """``**fragment_kwargs`` must not swallow typos silently."""
        from goal.ml.data.benchmarks.md17 import MD17Dataset

        with patch.object(
            MD17Dataset, "_download_and_process", return_value=[_two_molecules(10.0)]
        ):
            with pytest.raises(TypeError):
                MD17Dataset(
                    root=str(tmp_path),
                    molecule="aspirin",
                    split="train",
                    train_size=1,
                    val_size=0,
                    energy_key="not_a_dataset_option",
                )


@pytest.fixture()
def mock_pyg_qm9_module():
    """Patch PyG's QM9 dataset (module-level twin of the class fixture)."""
    fake_data = _make_fake_qm9_data(20)
    with patch("goal.ml.data.benchmarks.qm9.PyGQM9") as mock_cls:
        mock_cls.return_value = fake_data
        yield mock_cls


# ---------------------------------------------------------------------------
# Test: QM9 label shape
# ---------------------------------------------------------------------------


class TestQM9EnergyShape:
    """``graph.energy`` must be ``(1,)``, the shape every other dataset emits.

    PyG's ``data.y`` is ``(1, 19)``, so a naive column slice is
    ``(1, 1)``.  Left that way it batches to ``(G, 1)`` and ``EnergyLoss``
    divides it by the ``(G,)`` atom counts — which broadcasts to
    ``(G, G)`` and optimises a loss over every (structure, other
    structure) pair.  Nothing raises; the number is simply wrong.
    """

    def test_single_target_energy_is_1d(self, tmp_path: Path, mock_pyg_qm9_module):
        from goal.ml.data.benchmarks.qm9 import QM9Dataset

        dataset = QM9Dataset(root=str(tmp_path), target="u0", split="train")
        assert dataset[0].energy.shape == (1,)

    def test_all_targets_energy_is_1d(self, tmp_path: Path, mock_pyg_qm9_module):
        from goal.ml.data.benchmarks.qm9 import QM9Dataset

        dataset = QM9Dataset(root=str(tmp_path), target=None, split="train")
        assert dataset[0].energy.shape == (1,)

    def test_batched_energy_matches_graph_count(self, tmp_path: Path, mock_pyg_qm9_module):
        from torch_geometric.data import Batch

        from goal.ml.data.benchmarks.qm9 import QM9Dataset

        dataset = QM9Dataset(root=str(tmp_path), target="u0", split="train")
        batch = Batch.from_data_list([dataset[i] for i in range(3)])
        assert batch.energy.shape == (3,)

    def test_energy_loss_is_not_broadcast(self, tmp_path: Path, mock_pyg_qm9_module):
        """The loss must compare graph i with graph i, and nothing else."""
        from torch_geometric.data import Batch

        from goal.ml.data.benchmarks.qm9 import QM9Dataset
        from goal.ml.training.loss import EnergyLoss

        dataset = QM9Dataset(root=str(tmp_path), target="u0", split="train")
        batch = Batch.from_data_list([dataset[i] for i in range(3)])
        num_atoms: torch.Tensor = torch.tensor(
            [dataset[i].z.shape[0] for i in range(3)], dtype=torch.float64
        )
        predictions: dict[str, torch.Tensor] = {
            "energy": torch.zeros(3, dtype=torch.float64),
            "num_atoms": num_atoms,
        }
        expected: torch.Tensor = (batch.energy / num_atoms).abs().mean()
        assert EnergyLoss("mae")(predictions, batch) == pytest.approx(float(expected), rel=1e-12)


# ---------------------------------------------------------------------------
# Test: Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Test that example datasets register correctly."""

    def test_md17_registered(self):
        """'md17' should be in DATASET_REGISTRY after import."""
        import goal.ml.data.benchmarks  # noqa: F401
        from goal.ml.registry import DATASET_REGISTRY

        assert "md17" in DATASET_REGISTRY

    def test_rmd17_registered(self):
        """'rmd17' should be in DATASET_REGISTRY."""
        import goal.ml.data.benchmarks  # noqa: F401
        from goal.ml.registry import DATASET_REGISTRY

        assert "rmd17" in DATASET_REGISTRY

    def test_ani1_registered(self):
        """'ani1' and 'ani1x' should be in DATASET_REGISTRY."""
        import goal.ml.data.benchmarks  # noqa: F401
        from goal.ml.registry import DATASET_REGISTRY

        assert "ani1" in DATASET_REGISTRY
        assert "ani1x" in DATASET_REGISTRY

    def test_qm9_registered(self):
        """'qm9' should be in DATASET_REGISTRY."""
        import goal.ml.data.benchmarks  # noqa: F401
        from goal.ml.registry import DATASET_REGISTRY

        assert "qm9" in DATASET_REGISTRY

    def test_spice_registered(self):
        """'spice' should be in DATASET_REGISTRY."""
        import goal.ml.data.benchmarks  # noqa: F401
        from goal.ml.registry import DATASET_REGISTRY

        assert "spice" in DATASET_REGISTRY

    def test_registry_resolves_md17_class(self):
        """DATASET_REGISTRY.get('md17') should resolve to MD17Dataset."""
        import goal.ml.data.benchmarks  # noqa: F401
        from goal.ml.registry import DATASET_REGISTRY

        cls = DATASET_REGISTRY.get("md17")
        assert cls.__name__ == "MD17Dataset"

    def test_importable_from_any_working_directory(self, tmp_path: Path):
        """The CLI must register benchmark datasets however it was invoked.

        Under pytest the repo root is on ``sys.path``, so every test above
        would pass even if the console script could not find
        ``examples/`` — and it could not: ``goal-train`` runs with
        ``sys.path[0]`` set to the environment's ``bin/``, only ``src/``
        is installed, and ``train.py`` swallows the resulting
        ``ImportError``.  The run then died at "'md17' not found in
        datasets registry" listing only the core loaders.

        So this runs the import the way the entry point does: a fresh
        interpreter, in a directory outside the repo, with no
        ``PYTHONPATH``.
        """
        import os
        import subprocess
        import sys

        script: str = (
            "import goal.ml.cli;"
            "import goal.ml.data.benchmarks;"
            "from goal.ml.registry import DATASET_REGISTRY;"
            "print(sorted(k for k in ('md17', 'qm9', 'spice') if k in DATASET_REGISTRY))"
        )
        environment: dict[str, str] = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(tmp_path),
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "['md17', 'qm9', 'spice']", result.stdout


# ---------------------------------------------------------------------------
# Test: Composite loss system
# ---------------------------------------------------------------------------


class TestCompositeLossFunctions:
    """Test the new composite loss per-property feature."""

    def test_rmse_loss_function(self):
        """RMSE loss should equal sqrt(MSE)."""
        from goal.ml.training.loss import _rmse_loss

        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])
        result = _rmse_loss(pred, target)
        expected = torch.sqrt(torch.nn.functional.mse_loss(pred, target))
        assert torch.allclose(result, expected)

    def test_resolve_loss_fn_rmse(self):
        """resolve_loss_fn('rmse') should work."""
        from goal.ml.training.loss import _rmse_loss, resolve_loss_fn

        assert resolve_loss_fn("rmse") is _rmse_loss

    def test_resolve_loss_fn_dotted_path(self):
        """Dotted import path should resolve to callable."""
        from goal.ml.training.loss import resolve_loss_fn

        fn = resolve_loss_fn("torch.nn.functional.mse_loss")
        assert fn is torch.nn.functional.mse_loss

    def test_resolve_loss_fn_invalid_dotted_path(self):
        """Invalid dotted path should raise ValueError."""
        from goal.ml.training.loss import resolve_loss_fn

        with pytest.raises(ValueError, match="Cannot resolve"):
            resolve_loss_fn("nonexistent.module.function")

    def test_weighted_loss_with_label(self):
        """WeightedLoss should store custom label."""
        from goal.ml.training.loss import EnergyLoss, WeightedLoss

        wl = WeightedLoss(EnergyLoss(), weight=4.0, label="energy_mse")
        assert wl.label == "energy_mse"

    def test_weighted_loss_with_group(self):
        """WeightedLoss should store group."""
        from goal.ml.training.loss import ForcesLoss, WeightedLoss

        wl = WeightedLoss(ForcesLoss(), weight=4.0, label="forces_mse", group="forces")
        assert wl.group == "forces"

    def test_weighted_loss_default_label(self):
        """Default label should be class name."""
        from goal.ml.training.loss import EnergyLoss, WeightedLoss

        wl = WeightedLoss(EnergyLoss(), weight=1.0)
        assert wl.label == "EnergyLoss"

    def test_composite_loss_group_totals(self):
        """CompositeLoss should emit group totals for multi-fn properties."""
        from goal.ml.training.loss import CompositeLoss, ForcesLoss, WeightedLoss

        composite = CompositeLoss(
            [
                WeightedLoss(
                    ForcesLoss(loss_fn="mse"), weight=4.0, label="forces_mse", group="forces"
                ),
                WeightedLoss(
                    ForcesLoss(loss_fn="mae"), weight=8.0, label="forces_mae", group="forces"
                ),
            ]
        )

        pred = {"forces": torch.randn(5, 3)}
        target = {"forces": torch.randn(5, 3)}
        result = composite(pred, target)

        assert "forces_mse" in result
        assert "forces_mae" in result
        assert "forces" in result  # group total
        assert "total" in result
        assert torch.allclose(
            result["forces"],
            result["forces_mse"] + result["forces_mae"],
        )

    def test_composite_loss_single_fn_no_group_total(self):
        """Single-fn property should NOT get a separate group total."""
        from goal.ml.training.loss import CompositeLoss, EnergyLoss, WeightedLoss

        composite = CompositeLoss(
            [
                WeightedLoss(EnergyLoss(), weight=4.0, label="energy"),
            ]
        )

        pred = {"energy": torch.tensor([1.0]), "num_atoms": torch.tensor([1.0])}
        target = {"energy": torch.tensor([2.0]), "num_atoms": torch.tensor([1.0])}
        result = composite(pred, target)

        assert "energy" in result
        assert "total" in result

    def test_composite_loss_mixed(self):
        """Mix of single-fn and multi-fn properties."""
        from goal.ml.training.loss import (
            CompositeLoss,
            EnergyLoss,
            ForcesLoss,
            WeightedLoss,
        )

        composite = CompositeLoss(
            [
                WeightedLoss(EnergyLoss(), weight=4.0, label="energy"),
                WeightedLoss(
                    ForcesLoss(loss_fn="mse"), weight=4.0, label="forces_mse", group="forces"
                ),
                WeightedLoss(
                    ForcesLoss(loss_fn="rmse"), weight=8.0, label="forces_rmse", group="forces"
                ),
            ]
        )

        pred = {
            "energy": torch.tensor([1.0]),
            "num_atoms": torch.tensor([1.0]),
            "forces": torch.randn(5, 3),
        }
        target = {
            "energy": torch.tensor([2.0]),
            "num_atoms": torch.tensor([1.0]),
            "forces": torch.randn(5, 3),
        }
        result = composite(pred, target)

        assert set(result.keys()) == {"total", "energy", "forces_mse", "forces_rmse", "forces"}
        assert torch.allclose(
            result["total"],
            result["energy"] + result["forces_mse"] + result["forces_rmse"],
        )

    def test_composite_loss_logging_keys_for_wandb(self):
        """Logged keys should be clean for W&B panels."""
        from goal.ml.training.loss import (
            CompositeLoss,
            EnergyLoss,
            ForcesLoss,
            WeightedLoss,
        )

        composite = CompositeLoss(
            [
                WeightedLoss(EnergyLoss(), weight=4.0, label="energy"),
                WeightedLoss(
                    ForcesLoss(loss_fn="mse"), weight=4.0, label="forces_mse", group="forces"
                ),
                WeightedLoss(
                    ForcesLoss(loss_fn="rmse"), weight=8.0, label="forces_rmse", group="forces"
                ),
            ]
        )

        pred = {
            "energy": torch.tensor([1.0]),
            "num_atoms": torch.tensor([1.0]),
            "forces": torch.randn(5, 3),
        }
        target = {
            "energy": torch.tensor([2.0]),
            "num_atoms": torch.tensor([1.0]),
            "forces": torch.randn(5, 3),
        }
        result = composite(pred, target)

        # Simulate what module.training_step does
        logged = {f"train/{k}": v for k, v in result.items()}
        assert "train/total" in logged
        assert "train/energy" in logged
        assert "train/forces_mse" in logged
        assert "train/forces_rmse" in logged
        assert "train/forces" in logged
