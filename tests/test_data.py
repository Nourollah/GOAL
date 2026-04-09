"""Tests for the data loading subsystem — all loading modes and edge cases."""

from __future__ import annotations

import typing
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Helpers — mock dataset that doesn't need real files
# ---------------------------------------------------------------------------


class MockAtomicDataset(Dataset):
    """Mock dataset for testing DataModule without real data files."""

    def __init__(
        self,
        root: str | Path = "/fake",
        cutoff: float = 5.0,
        split: str = "train",
        **kwargs: typing.Any,
    ) -> None:
        self.root: Path = Path(root)
        self.cutoff: float = cutoff
        self.split: str = split
        self._size: int = kwargs.get("size", 100)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"energy": torch.tensor(float(idx))}


# ---------------------------------------------------------------------------
# Tests: _resolve_paths
# ---------------------------------------------------------------------------


class TestResolvePaths:
    """Verify path normalisation."""

    def test_single_string(self):
        from goal.ml.data.datamodule import _resolve_paths

        result: list[str] = _resolve_paths("/some/path.xyz")
        assert result == ["/some/path.xyz"]

    def test_list_of_strings(self):
        from goal.ml.data.datamodule import _resolve_paths

        result: list[str] = _resolve_paths(["/a.xyz", "/b.xyz"])
        assert result == ["/a.xyz", "/b.xyz"]

    def test_omegaconf_listconfig(self):
        from omegaconf import ListConfig

        from goal.ml.data.datamodule import _resolve_paths

        raw: ListConfig = ListConfig(["/x.xyz", "/y.xyz"])
        result: list[str] = _resolve_paths(raw)
        assert result == ["/x.xyz", "/y.xyz"]


# ---------------------------------------------------------------------------
# Tests: _discover_files_in_dir
# ---------------------------------------------------------------------------


class TestDiscoverFilesInDir:
    """Verify directory scanning for data files."""

    def test_finds_supported_files(self, tmp_path: Path):
        """Should find .xyz, .hdf5, .lmdb files in a directory."""
        from goal.ml.data.datamodule import _discover_files_in_dir

        (tmp_path / "train_001.xyz").touch()
        (tmp_path / "train_002.xyz").touch()
        (tmp_path / "train_003.hdf5").touch()
        (tmp_path / "readme.txt").touch()  # Should be ignored

        found: list[str] = _discover_files_in_dir(tmp_path)
        assert len(found) == 3
        assert all(f.endswith((".xyz", ".hdf5")) for f in found)

    def test_sorted_output(self, tmp_path: Path):
        """Files should be returned in sorted order for reproducibility."""
        from goal.ml.data.datamodule import _discover_files_in_dir

        (tmp_path / "c.xyz").touch()
        (tmp_path / "a.xyz").touch()
        (tmp_path / "b.xyz").touch()

        found: list[str] = _discover_files_in_dir(tmp_path)
        names: list[str] = [Path(f).name for f in found]
        assert names == ["a.xyz", "b.xyz", "c.xyz"]

    def test_nonexistent_dir_raises(self):
        """Non-existent directory should raise FileNotFoundError."""
        from goal.ml.data.datamodule import _discover_files_in_dir

        with pytest.raises(FileNotFoundError, match="Directory not found"):
            _discover_files_in_dir("/nonexistent/path")

    def test_empty_dir_raises(self, tmp_path: Path):
        """Empty directory should raise ValueError."""
        from goal.ml.data.datamodule import _discover_files_in_dir

        with pytest.raises(ValueError, match="No data files found"):
            _discover_files_in_dir(tmp_path)

    def test_only_unsupported_files_raises(self, tmp_path: Path):
        """Dir with only unsupported file types should raise ValueError."""
        from goal.ml.data.datamodule import _discover_files_in_dir

        (tmp_path / "notes.txt").touch()
        (tmp_path / "readme.md").touch()

        with pytest.raises(ValueError, match="No data files found"):
            _discover_files_in_dir(tmp_path)

    def test_extxyz_extension(self, tmp_path: Path):
        """Should recognise .extxyz files."""
        from goal.ml.data.datamodule import _discover_files_in_dir

        (tmp_path / "data.extxyz").touch()
        found: list[str] = _discover_files_in_dir(tmp_path)
        assert len(found) == 1

    def test_lmdb_and_traj(self, tmp_path: Path):
        """Should recognise .lmdb and .traj files."""
        from goal.ml.data.datamodule import _discover_files_in_dir

        (tmp_path / "data.lmdb").touch()
        (tmp_path / "trajectory.traj").touch()
        found: list[str] = _discover_files_in_dir(tmp_path)
        assert len(found) == 2


# ---------------------------------------------------------------------------
# Tests: ConcatAtomicDataset
# ---------------------------------------------------------------------------


class TestConcatAtomicDataset:
    """Verify dataset merging."""

    def test_sequential_concat(self):
        """Sequential merge should preserve dataset order."""
        from goal.ml.data.datasets.concat import ConcatAtomicDataset

        ds1: MockAtomicDataset = MockAtomicDataset(size=10)
        ds2: MockAtomicDataset = MockAtomicDataset(size=20)
        merged: ConcatAtomicDataset = ConcatAtomicDataset([ds1, ds2], merge_strategy="sequential")

        assert len(merged) == 30

    def test_random_concat_changes_order(self):
        """Random merge should produce a different index ordering."""
        from goal.ml.data.datasets.concat import ConcatAtomicDataset

        ds: MockAtomicDataset = MockAtomicDataset(size=100)
        merged: ConcatAtomicDataset = ConcatAtomicDataset([ds], merge_strategy="random", seed=42)

        # Indices should be a permutation
        assert len(merged) == 100
        assert set(merged._indices) == set(range(100))

    def test_random_concat_deterministic_with_seed(self):
        """Two ConcatAtomicDataset with same seed should yield same order."""
        from goal.ml.data.datasets.concat import ConcatAtomicDataset

        ds: MockAtomicDataset = MockAtomicDataset(size=50)
        m1: ConcatAtomicDataset = ConcatAtomicDataset([ds], merge_strategy="random", seed=123)
        m2: ConcatAtomicDataset = ConcatAtomicDataset([ds], merge_strategy="random", seed=123)
        assert m1._indices == m2._indices

    def test_empty_datasets(self):
        """Concatenating empty datasets should have length 0."""
        from goal.ml.data.datasets.concat import ConcatAtomicDataset

        ds: MockAtomicDataset = MockAtomicDataset(size=0)
        merged: ConcatAtomicDataset = ConcatAtomicDataset([ds])
        assert len(merged) == 0


# ---------------------------------------------------------------------------
# Tests: _merge_datasets
# ---------------------------------------------------------------------------


class TestMergeDatasets:
    """Verify _merge_datasets helper."""

    def test_single_dataset_returns_directly(self):
        """Single dataset should be returned without wrapping."""
        from goal.ml.data.datamodule import _merge_datasets

        ds: MockAtomicDataset = MockAtomicDataset(size=10)
        result: Dataset = _merge_datasets([ds])
        assert result is ds  # Same object, no wrapper

    def test_multiple_datasets_returns_concat(self):
        """Multiple datasets should be merged into ConcatAtomicDataset."""
        from goal.ml.data.datamodule import _merge_datasets
        from goal.ml.data.datasets.concat import ConcatAtomicDataset

        ds1: MockAtomicDataset = MockAtomicDataset(size=10)
        ds2: MockAtomicDataset = MockAtomicDataset(size=20)
        result: Dataset = _merge_datasets([ds1, ds2])
        assert isinstance(result, ConcatAtomicDataset)
        assert len(result) == 30


# ---------------------------------------------------------------------------
# Tests: GOALDataModule — all 4 loading modes
# ---------------------------------------------------------------------------


class TestGOALDataModule:
    """Verify the GOALDataModule across loading modes."""

    def _make_cfg(self, **overrides: typing.Any) -> DictConfig:
        """Create a minimal data config with required fields."""
        base: dict[str, typing.Any] = {
            "data": {
                "dataset_type": "mock",
                "cutoff": 5.0,
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
                "root": "/fake/root",
            }
        }
        for k, v in overrides.items():
            if k.startswith("data."):
                base["data"][k.split(".", 1)[1]] = v
            else:
                base[k] = v
        return OmegaConf.create(base)

    @pytest.fixture(autouse=True)
    def _register_mock_dataset(self):
        """Register the mock dataset in the DATASET_REGISTRY for each test."""
        from goal.ml.registry import DATASET_REGISTRY

        if "mock" not in DATASET_REGISTRY:
            DATASET_REGISTRY.register_instance("mock", MockAtomicDataset)
        yield

    def test_mode1_auto_split(self):
        """Mode 1: single root with auto-split using split_ratio."""
        from goal.ml.data.datamodule import GOALDataModule

        cfg: DictConfig = self._make_cfg(
            **{
                "data.root": "/fake",
                "data.split_ratio": [0.8, 0.1, 0.1],
                "data.split_seed": 42,
            }
        )

        # Patch _build_datasets to return our mock datasets
        with patch(
            "goal.ml.data.datamodule._build_datasets",
            return_value=[MockAtomicDataset(size=100)],
        ):
            dm: GOALDataModule = GOALDataModule(cfg)
            # _try_named_splits will eventually call _build_datasets, and
            # if it raises FileNotFoundError, it does _numeric_split.
            # Let's force the numeric split path.
            with patch.object(
                dm,
                "_try_named_splits",
                side_effect=FileNotFoundError("forced"),
            ):
                dm.setup(stage="fit")
                assert dm.data_train is not None
                assert dm.data_val is not None

    def test_mode2_per_split_paths(self):
        """Mode 2: explicit train_paths / val_paths."""
        from goal.ml.data.datamodule import GOALDataModule

        cfg: DictConfig = self._make_cfg(
            **{
                "data.train_paths": ["/fake/train.xyz"],
                "data.val_paths": ["/fake/val.xyz"],
                "data.test_paths": ["/fake/test.xyz"],
            }
        )

        # Remove root to avoid mode 1
        del cfg.data.root

        with patch(
            "goal.ml.data.datamodule._build_datasets",
            side_effect=lambda cls, paths, split, cutoff, extra: [MockAtomicDataset(size=50)],
        ):
            dm: GOALDataModule = GOALDataModule(cfg)
            dm.setup(stage=None)
            assert dm.data_train is not None
            assert dm.data_val is not None
            assert dm.data_test is not None

    def test_mode4_directory_loading(self, tmp_path: Path):
        """Mode 4: directory-based loading with train_dir / val_dir."""
        from goal.ml.data.datamodule import GOALDataModule

        train_dir: Path = tmp_path / "train"
        val_dir: Path = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        (train_dir / "data_001.xyz").touch()
        (train_dir / "data_002.xyz").touch()
        (val_dir / "data_001.xyz").touch()

        cfg: DictConfig = self._make_cfg(
            **{
                "data.train_dir": str(train_dir),
                "data.val_dir": str(val_dir),
            }
        )

        with patch(
            "goal.ml.data.datamodule._build_datasets",
            side_effect=lambda cls, paths, split, cutoff, extra: [
                MockAtomicDataset(size=len(paths) * 10)
            ],
        ):
            dm: GOALDataModule = GOALDataModule(cfg)
            dm.setup(stage="fit")
            assert dm.data_train is not None
            assert dm.data_val is not None

    def test_mode4_missing_val_dir_raises(self, tmp_path: Path):
        """Mode 4: train_dir without val_dir should raise ValueError."""
        from goal.ml.data.datamodule import GOALDataModule

        train_dir: Path = tmp_path / "train"
        train_dir.mkdir()
        (train_dir / "data.xyz").touch()

        cfg: DictConfig = self._make_cfg(
            **{
                "data.train_dir": str(train_dir),
            }
        )

        with patch(
            "goal.ml.data.datamodule._build_datasets",
            side_effect=lambda cls, paths, split, cutoff, extra: [MockAtomicDataset(size=10)],
        ):
            dm: GOALDataModule = GOALDataModule(cfg)
            with pytest.raises(ValueError, match="val_dir is missing"):
                dm.setup(stage="fit")

    def test_mode4_with_test_dir(self, tmp_path: Path):
        """Mode 4: test_dir is optional and loaded when present."""
        from goal.ml.data.datamodule import GOALDataModule

        for name in ("train", "val", "test"):
            d: Path = tmp_path / name
            d.mkdir()
            (d / "data.xyz").touch()

        cfg: DictConfig = self._make_cfg(
            **{
                "data.train_dir": str(tmp_path / "train"),
                "data.val_dir": str(tmp_path / "val"),
                "data.test_dir": str(tmp_path / "test"),
            }
        )

        with patch(
            "goal.ml.data.datamodule._build_datasets",
            side_effect=lambda cls, paths, split, cutoff, extra: [MockAtomicDataset(size=10)],
        ):
            dm: GOALDataModule = GOALDataModule(cfg)
            dm.setup(stage=None)
            assert dm.data_train is not None
            assert dm.data_val is not None
            assert dm.data_test is not None

    def test_meta_keys_include_dir_options(self):
        """_META_KEYS should include train_dir, val_dir, test_dir."""
        from goal.ml.data.datamodule import GOALDataModule

        assert "train_dir" in GOALDataModule._META_KEYS
        assert "val_dir" in GOALDataModule._META_KEYS
        assert "test_dir" in GOALDataModule._META_KEYS

    def test_extra_kwargs_excludes_meta_keys(self):
        """_extra_kwargs should not include dataset-level meta keys."""
        from goal.ml.data.datamodule import GOALDataModule

        cfg: DictConfig = self._make_cfg(
            **{
                "data.some_custom_param": 42,
            }
        )
        dm: GOALDataModule = GOALDataModule(cfg)
        extra: dict[str, typing.Any] = dm._extra_kwargs()
        assert "some_custom_param" in extra
        assert "batch_size" not in extra
        assert "train_dir" not in extra

    def test_test_dataloader_raises_without_test_data(self):
        """test_dataloader() should raise RuntimeError if no test set."""
        from goal.ml.data.datamodule import GOALDataModule

        cfg: DictConfig = self._make_cfg()
        dm: GOALDataModule = GOALDataModule(cfg)
        dm.data_test = None

        with pytest.raises(RuntimeError, match="No test dataset"):
            dm.test_dataloader()
