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


def _make_fake_md17_data(n: int = 100) -> typing.List[Data]:
    """Create fake MD17-like Data objects."""
    items: typing.List[Data] = []
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


def _make_fake_ani1_data(n: int = 50) -> typing.List[Data]:
    """Create fake ANI-1-like Data objects."""
    items: typing.List[Data] = []
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


def _make_fake_qm9_data(n: int = 200) -> typing.List[Data]:
    """Create fake QM9-like Data objects with 19 properties."""
    items: typing.List[Data] = []
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
        from examples.datasets.base import BenchmarkDataset

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
        from examples.datasets.base import BenchmarkDataset

        s1 = BenchmarkDataset._random_split_indices(1000, 800, 100, seed=42)
        s2 = BenchmarkDataset._random_split_indices(1000, 800, 100, seed=42)
        assert s1["train"] == s2["train"]
        assert s1["val"] == s2["val"]
        assert s1["test"] == s2["test"]

    def test_random_split_indices_sizes(self):
        """Split sizes should sum to total."""
        from examples.datasets.base import BenchmarkDataset

        s = BenchmarkDataset._random_split_indices(1000, 800, 100, seed=42)
        assert len(s["train"]) == 800
        assert len(s["val"]) == 100
        assert len(s["test"]) == 100
        all_indices = set(s["train"] + s["val"] + s["test"])
        assert len(all_indices) == 1000

    def test_random_split_different_seeds(self):
        """Different seeds should produce different splits."""
        from examples.datasets.base import BenchmarkDataset

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
        """Patch PyG's MD17 dataset to return fake data."""
        fake_data = _make_fake_md17_data(100)
        with patch("examples.datasets.md17.PyGMD17") as mock_cls:
            mock_cls.return_value = fake_data
            yield mock_cls

    def test_creates_cache_on_first_call(self, tmp_path: Path, mock_pyg_md17):
        """First call should create a .pt cache file."""
        from examples.datasets.md17 import MD17Dataset

        ds = MD17Dataset(root=str(tmp_path), molecule="aspirin", split="train")
        cache_files = list((tmp_path / "processed").glob("*.pt"))
        assert len(cache_files) == 1
        assert "rmd17_aspirin" in cache_files[0].name

    def test_loads_from_cache_on_second_call(self, tmp_path: Path, mock_pyg_md17):
        """Second call should load from cache, not re-download."""
        from examples.datasets.md17 import MD17Dataset

        MD17Dataset(root=str(tmp_path), molecule="aspirin", split="train")
        call_count_first = mock_pyg_md17.call_count

        MD17Dataset(root=str(tmp_path), molecule="aspirin", split="train")
        assert mock_pyg_md17.call_count == call_count_first  # no new PyG call

    def test_split_sizes_sum_to_total(self, tmp_path: Path, mock_pyg_md17):
        """Train + val + test should equal total dataset size."""
        from examples.datasets.md17 import MD17Dataset

        train = MD17Dataset(root=str(tmp_path), molecule="aspirin", split="train")
        val = MD17Dataset(root=str(tmp_path), molecule="aspirin", split="val")
        test = MD17Dataset(root=str(tmp_path), molecule="aspirin", split="test")
        assert len(train) + len(val) + len(test) == 100

    def test_default_split_sizes(self, tmp_path: Path, mock_pyg_md17):
        """Default: 950 train, 50 val (but we only have 100 fake items)."""
        from examples.datasets.md17 import MD17Dataset

        # With 100 items, train_size=950 exceeds total, so test would be empty
        ds = MD17Dataset(
            root=str(tmp_path), molecule="aspirin", split="train",
            train_size=80, val_size=10,
        )
        assert len(ds) == 80

    def test_unit_conversion(self, tmp_path: Path, mock_pyg_md17):
        """Energies should be converted from kcal/mol to eV."""
        from examples.datasets.md17 import MD17Dataset, KCAL_TO_EV

        ds = MD17Dataset(
            root=str(tmp_path), molecule="aspirin", split="train",
            train_size=80, val_size=10,
        )
        graph = ds[0]
        # Energy should be in eV (original is ~-300 kcal/mol × 0.043364)
        assert graph.energy is not None
        assert abs(graph.energy.item()) < 100  # eV range, not kcal range

    def test_returns_atomic_graph(self, tmp_path: Path, mock_pyg_md17):
        """Items should have required AtomicGraph fields."""
        from examples.datasets.md17 import MD17Dataset

        ds = MD17Dataset(
            root=str(tmp_path), molecule="aspirin", split="train",
            train_size=80, val_size=10,
        )
        graph = ds[0]
        assert hasattr(graph, "pos")
        assert hasattr(graph, "z")
        assert hasattr(graph, "edge_index")
        assert hasattr(graph, "energy")
        assert hasattr(graph, "forces")

    def test_invalid_molecule_raises(self, tmp_path: Path):
        """Unknown molecule name should raise ValueError."""
        from examples.datasets.md17 import MD17Dataset

        with pytest.raises(ValueError, match="Unknown molecule"):
            MD17Dataset(root=str(tmp_path), molecule="nonexistent", split="train")

    def test_citation_nonempty(self, tmp_path: Path, mock_pyg_md17):
        """Citation should be non-empty BibTeX."""
        from examples.datasets.md17 import MD17Dataset

        ds = MD17Dataset(
            root=str(tmp_path), molecule="aspirin", split="train",
            train_size=80, val_size=10,
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
        with patch("examples.datasets.ani1.ANI1x") as mock_1x, \
             patch("examples.datasets.ani1.ANI1") as mock_1:
            mock_1x.return_value = fake_data
            mock_1.return_value = fake_data
            yield mock_1x, mock_1

    def test_ani1x_loads(self, tmp_path: Path, mock_pyg_ani1):
        """ANI-1x should load without error."""
        from examples.datasets.ani1 import ANI1Dataset

        ds = ANI1Dataset(root=str(tmp_path), version="1x", split="train")
        assert len(ds) > 0

    def test_max_structures(self, tmp_path: Path, mock_pyg_ani1):
        """max_structures should limit dataset size."""
        from examples.datasets.ani1 import ANI1Dataset

        ds = ANI1Dataset(
            root=str(tmp_path), version="1x", split="train",
            max_structures=10,
        )
        # Total is capped at 10, train fraction 0.8 → 8
        assert len(ds._data) <= 10

    def test_invalid_version_raises(self, tmp_path: Path):
        """Invalid version should raise ValueError."""
        from examples.datasets.ani1 import ANI1Dataset

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
        with patch("examples.datasets.qm9.PyGQM9") as mock_cls:
            mock_cls.return_value = fake_data
            yield mock_cls

    def test_qm9_loads_with_target_name(self, tmp_path: Path, mock_pyg_qm9):
        """QM9 should accept target name like 'homo'."""
        from examples.datasets.qm9 import QM9Dataset

        ds = QM9Dataset(root=str(tmp_path), target="homo", split="train")
        assert len(ds) > 0

    def test_qm9_loads_with_target_index(self, tmp_path: Path, mock_pyg_qm9):
        """QM9 should accept target index like 4 (gap)."""
        from examples.datasets.qm9 import QM9Dataset

        ds = QM9Dataset(root=str(tmp_path), target=4, split="train")
        assert ds.target_name == "gap"

    def test_qm9_all_targets(self, tmp_path: Path, mock_pyg_qm9):
        """target=None should store all 19 properties."""
        from examples.datasets.qm9 import QM9Dataset

        ds = QM9Dataset(root=str(tmp_path), target=None, split="train")
        graph = ds[0]
        assert graph.energy is not None

    def test_invalid_target_raises(self, tmp_path: Path):
        """Unknown target name should raise ValueError."""
        from examples.datasets.qm9 import QM9Dataset

        with pytest.raises(ValueError, match="Unknown QM9 target"):
            QM9Dataset(root=str(tmp_path), target="nonexistent", split="train")

    def test_invalid_target_index_raises(self, tmp_path: Path):
        """Out-of-range target index should raise ValueError."""
        from examples.datasets.qm9 import QM9Dataset

        with pytest.raises(ValueError, match="target index must be"):
            QM9Dataset(root=str(tmp_path), target=99, split="train")


# ---------------------------------------------------------------------------
# Test: Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Test that example datasets register correctly."""

    def test_md17_registered(self):
        """'md17' should be in DATASET_REGISTRY after import."""
        import examples.datasets  # noqa: F401
        from gmd.registry import DATASET_REGISTRY

        assert "md17" in DATASET_REGISTRY

    def test_rmd17_registered(self):
        """'rmd17' should be in DATASET_REGISTRY."""
        import examples.datasets  # noqa: F401
        from gmd.registry import DATASET_REGISTRY

        assert "rmd17" in DATASET_REGISTRY

    def test_ani1_registered(self):
        """'ani1' and 'ani1x' should be in DATASET_REGISTRY."""
        import examples.datasets  # noqa: F401
        from gmd.registry import DATASET_REGISTRY

        assert "ani1" in DATASET_REGISTRY
        assert "ani1x" in DATASET_REGISTRY

    def test_qm9_registered(self):
        """'qm9' should be in DATASET_REGISTRY."""
        import examples.datasets  # noqa: F401
        from gmd.registry import DATASET_REGISTRY

        assert "qm9" in DATASET_REGISTRY

    def test_spice_registered(self):
        """'spice' should be in DATASET_REGISTRY."""
        import examples.datasets  # noqa: F401
        from gmd.registry import DATASET_REGISTRY

        assert "spice" in DATASET_REGISTRY

    def test_registry_resolves_md17_class(self):
        """DATASET_REGISTRY.get('md17') should resolve to MD17Dataset."""
        import examples.datasets  # noqa: F401
        from gmd.registry import DATASET_REGISTRY

        cls = DATASET_REGISTRY.get("md17")
        assert cls.__name__ == "MD17Dataset"


# ---------------------------------------------------------------------------
# Test: Composite loss system
# ---------------------------------------------------------------------------


class TestCompositeLossFunctions:
    """Test the new composite loss per-property feature."""

    def test_rmse_loss_function(self):
        """RMSE loss should equal sqrt(MSE)."""
        from gmd.training.loss import _rmse_loss

        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])
        result = _rmse_loss(pred, target)
        expected = torch.sqrt(torch.nn.functional.mse_loss(pred, target))
        assert torch.allclose(result, expected)

    def test_resolve_loss_fn_rmse(self):
        """resolve_loss_fn('rmse') should work."""
        from gmd.training.loss import resolve_loss_fn, _rmse_loss

        assert resolve_loss_fn("rmse") is _rmse_loss

    def test_resolve_loss_fn_dotted_path(self):
        """Dotted import path should resolve to callable."""
        from gmd.training.loss import resolve_loss_fn

        fn = resolve_loss_fn("torch.nn.functional.mse_loss")
        assert fn is torch.nn.functional.mse_loss

    def test_resolve_loss_fn_invalid_dotted_path(self):
        """Invalid dotted path should raise ValueError."""
        from gmd.training.loss import resolve_loss_fn

        with pytest.raises(ValueError, match="Cannot resolve"):
            resolve_loss_fn("nonexistent.module.function")

    def test_weighted_loss_with_label(self):
        """WeightedLoss should store custom label."""
        from gmd.training.loss import WeightedLoss, EnergyLoss

        wl = WeightedLoss(EnergyLoss(), weight=4.0, label="energy_mse")
        assert wl.label == "energy_mse"

    def test_weighted_loss_with_group(self):
        """WeightedLoss should store group."""
        from gmd.training.loss import WeightedLoss, ForcesLoss

        wl = WeightedLoss(ForcesLoss(), weight=4.0, label="forces_mse", group="forces")
        assert wl.group == "forces"

    def test_weighted_loss_default_label(self):
        """Default label should be class name."""
        from gmd.training.loss import WeightedLoss, EnergyLoss

        wl = WeightedLoss(EnergyLoss(), weight=1.0)
        assert wl.label == "EnergyLoss"

    def test_composite_loss_group_totals(self):
        """CompositeLoss should emit group totals for multi-fn properties."""
        from gmd.training.loss import CompositeLoss, WeightedLoss, ForcesLoss

        composite = CompositeLoss([
            WeightedLoss(ForcesLoss(loss_fn="mse"), weight=4.0, label="forces_mse", group="forces"),
            WeightedLoss(ForcesLoss(loss_fn="mae"), weight=8.0, label="forces_mae", group="forces"),
        ])

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
        from gmd.training.loss import CompositeLoss, WeightedLoss, EnergyLoss

        composite = CompositeLoss([
            WeightedLoss(EnergyLoss(), weight=4.0, label="energy"),
        ])

        pred = {"energy": torch.tensor([1.0]), "num_atoms": torch.tensor([1.0])}
        target = {"energy": torch.tensor([2.0]), "num_atoms": torch.tensor([1.0])}
        result = composite(pred, target)

        assert "energy" in result
        assert "total" in result

    def test_composite_loss_mixed(self):
        """Mix of single-fn and multi-fn properties."""
        from gmd.training.loss import (
            CompositeLoss, WeightedLoss, EnergyLoss, ForcesLoss,
        )

        composite = CompositeLoss([
            WeightedLoss(EnergyLoss(), weight=4.0, label="energy"),
            WeightedLoss(ForcesLoss(loss_fn="mse"), weight=4.0, label="forces_mse", group="forces"),
            WeightedLoss(ForcesLoss(loss_fn="rmse"), weight=8.0, label="forces_rmse", group="forces"),
        ])

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
        from gmd.training.loss import (
            CompositeLoss, WeightedLoss, EnergyLoss, ForcesLoss,
        )

        composite = CompositeLoss([
            WeightedLoss(EnergyLoss(), weight=4.0, label="energy"),
            WeightedLoss(ForcesLoss(loss_fn="mse"), weight=4.0, label="forces_mse", group="forces"),
            WeightedLoss(ForcesLoss(loss_fn="rmse"), weight=8.0, label="forces_rmse", group="forces"),
        ])

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
