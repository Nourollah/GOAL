"""Lightning DataModule for atomistic datasets.

Supports:
- Single-file loading with automatic train/val/test splitting
- Multi-file loading from lists of paths (per-split or merged)
- Directory-based loading: all matching files in a directory per split
- Merge strategies: ``sequential`` (in-order) or ``random`` (shuffled)
- Separate sources for train / val / test splits
- Compatible with Lightning 2.6+ and DDP/FSDP strategies
"""

from __future__ import annotations

import typing
from pathlib import Path

import lightning as L
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader

from goal.ml.data.datasets.base import BaseAtomicDataset
from goal.ml.data.datasets.concat import ConcatAtomicDataset
from goal.ml.registry import DATASET_REGISTRY

# File extensions recognised when scanning directories
_DATA_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".xyz",
        ".extxyz",
        ".h5",
        ".hdf5",
        ".lmdb",
        ".traj",
        ".db",
    }
)


def _resolve_paths(raw: str | list[str] | ListConfig) -> list[str]:
    """Normalise a single path or list of paths into a list of strings."""
    if isinstance(raw, (list, ListConfig)):
        return [str(p) for p in raw]
    return [str(raw)]


def _discover_files_in_dir(
    directory: str | Path,
    extensions: frozenset[str] = _DATA_EXTENSIONS,
) -> list[str]:
    """Discover all data files in a directory (sorted for reproducibility).

    Parameters
    ----------
    directory : str or Path
        Directory to scan.
    extensions : frozenset of str
        Accepted file suffixes (e.g. ``".xyz"``, ``".hdf5"``).

    Returns
    -------
    list of str
        Sorted list of absolute paths to matching files.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    ValueError
        If no matching files are found.
    """
    dirpath: Path = Path(directory)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"Directory not found: {dirpath}")
    files: list[str] = sorted(
        str(f) for f in dirpath.iterdir() if f.is_file() and f.suffix.lower() in extensions
    )
    if not files:
        raise ValueError(
            f"No data files found in {dirpath}. "
            f"Supported extensions: {', '.join(sorted(extensions))}"
        )
    return files


def _build_datasets(
    ds_cls: type,
    paths: list[str],
    split: str,
    cutoff: float,
    extra: dict[str, typing.Any],
) -> list[BaseAtomicDataset]:
    """Instantiate one dataset per path."""
    datasets: list[BaseAtomicDataset] = []
    for p in paths:
        root: Path = Path(p)
        ds: BaseAtomicDataset = ds_cls(root=root, cutoff=cutoff, split=split, **extra)
        datasets.append(ds)
    return datasets


def _merge_datasets(
    datasets: list[BaseAtomicDataset],
    merge_strategy: str = "sequential",
    seed: int | None = None,
) -> Dataset:
    """Merge a list of datasets using the given strategy.

    If there's only one dataset, return it directly (no wrapper overhead).
    """
    if len(datasets) == 1:
        return datasets[0]
    return ConcatAtomicDataset(
        datasets,
        merge_strategy=merge_strategy,
        seed=seed,
    )


class GOALDataModule(L.LightningDataModule):
    """Universal ``LightningDataModule`` for any registered dataset.

    Supports four loading modes controlled by `data` config:

    **Mode 1 — Single source, auto-split** (default):
        ``data.root`` is a single path.  The dataset is loaded (with
        ``split="train"``), then split into train/val (and optionally
        test) using ``data.split_ratio``.

    **Mode 2 — Per-split paths**:
        ``data.train_paths``, ``data.val_paths``, and optionally
        ``data.test_paths`` are lists of file/dir paths.  Each split
        is loaded independently.  Multiple paths per split are merged
        using ``data.merge_strategy``.

    **Mode 3 — Merged multi-source, auto-split**:
        ``data.root`` is a list of paths.  All are loaded and merged,
        then split by ``data.split_ratio``.

    **Mode 4 — Directory-based per-split**:
        ``data.train_dir``, ``data.val_dir``, and optionally
        ``data.test_dir`` are directories.  All matching data files
        inside each directory are automatically discovered and loaded.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config — expected keys live under ``cfg.data``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    # ------------------------------------------------------------------
    # Keys that are DataModule-level, not dataset-level
    # ------------------------------------------------------------------
    _META_KEYS = {
        "dataset_type",
        "root",
        "cutoff",
        "batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "train_paths",
        "val_paths",
        "test_paths",
        "train_dir",
        "val_dir",
        "test_dir",
        "merge_strategy",
        "split_ratio",
        "split_seed",
    }

    def _extra_kwargs(self) -> dict[str, typing.Any]:
        """Dataset-specific kwargs (everything that isn't meta)."""
        return {k: v for k, v in self.cfg.data.items() if k not in self._META_KEYS}

    def setup(self, stage: str | None = None) -> None:
        """Instantiate datasets.

        Detects the loading mode automatically:
        - If ``train_dir`` exists → Mode 4 (directory-based per-split)
        - If ``train_paths`` exists → Mode 2 (per-split paths)
        - If ``root`` is a list → Mode 3 (merged multi-source, auto-split)
        - Otherwise → Mode 1 (single source, auto-split)
        """
        data_cfg: typing.Any = self.cfg.data
        ds_cls: type = DATASET_REGISTRY.get(data_cfg.dataset_type)
        cutoff: float = data_cfg.cutoff
        extra: dict[str, typing.Any] = self._extra_kwargs()
        merge: str = data_cfg.get("merge_strategy", "sequential")
        seed: int = data_cfg.get("split_seed", 42)

        has_dir: bool = "train_dir" in data_cfg and data_cfg.train_dir is not None
        has_per_split: bool = "train_paths" in data_cfg and data_cfg.train_paths is not None

        if has_dir:
            # ---- Mode 4: directory-based per-split ----
            self._setup_from_dirs(ds_cls, data_cfg, cutoff, extra, merge, seed, stage)
        elif has_per_split:
            # ---- Mode 2: per-split paths ----
            self._setup_per_split(ds_cls, data_cfg, cutoff, extra, merge, seed, stage)
        else:
            # ---- Mode 1 or 3: auto-split from root ----
            self._setup_auto_split(ds_cls, data_cfg, cutoff, extra, merge, seed, stage)

    def _setup_from_dirs(
        self,
        ds_cls,
        data_cfg,
        cutoff,
        extra,
        merge,
        seed,
        stage,
    ) -> None:
        """Mode 4: discover data files from directories for each split."""
        if stage in ("fit", None):
            train_paths: list[str] = _discover_files_in_dir(data_cfg.train_dir)
            train_datasets: list[BaseAtomicDataset] = _build_datasets(
                ds_cls,
                train_paths,
                "train",
                cutoff,
                extra,
            )
            self.data_train = _merge_datasets(train_datasets, merge, seed)

            val_dir: str | None = data_cfg.get("val_dir")
            if val_dir is not None:
                val_paths: list[str] = _discover_files_in_dir(val_dir)
                val_datasets: list[BaseAtomicDataset] = _build_datasets(
                    ds_cls,
                    val_paths,
                    "val",
                    cutoff,
                    extra,
                )
                self.data_val = _merge_datasets(val_datasets, merge, seed)
            else:
                raise ValueError(
                    "train_dir is set but val_dir is missing. "
                    "Provide val_dir for directory-based loading."
                )

        if stage in ("test", None):
            test_dir: str | None = data_cfg.get("test_dir")
            if test_dir is not None:
                test_paths: list[str] = _discover_files_in_dir(test_dir)
                test_datasets: list[BaseAtomicDataset] = _build_datasets(
                    ds_cls,
                    test_paths,
                    "test",
                    cutoff,
                    extra,
                )
                self.data_test = _merge_datasets(test_datasets, merge, seed)

    def _setup_per_split(
        self,
        ds_cls,
        data_cfg,
        cutoff,
        extra,
        merge,
        seed,
        stage,
    ) -> None:
        """Mode 2: separate file lists for train / val / test."""
        if stage in ("fit", None):
            train_paths = _resolve_paths(data_cfg.train_paths)
            train_datasets = _build_datasets(ds_cls, train_paths, "train", cutoff, extra)
            self.data_train = _merge_datasets(train_datasets, merge, seed)

            val_paths = _resolve_paths(data_cfg.val_paths)
            val_datasets = _build_datasets(ds_cls, val_paths, "val", cutoff, extra)
            self.data_val = _merge_datasets(val_datasets, merge, seed)

        if stage in ("test", None):
            test_paths = data_cfg.get("test_paths")
            if test_paths is not None:
                test_paths = _resolve_paths(test_paths)
                test_datasets = _build_datasets(ds_cls, test_paths, "test", cutoff, extra)
                self.data_test = _merge_datasets(test_datasets, merge, seed)

    def _setup_auto_split(
        self,
        ds_cls,
        data_cfg,
        cutoff,
        extra,
        merge,
        seed,
        stage,
    ) -> None:
        """Mode 1 / 3: load from root (single or list), then split."""
        roots = _resolve_paths(data_cfg.root)
        split_ratio = data_cfg.get("split_ratio", [0.8, 0.1, 0.1])

        # Try per-split files first (e.g. train.xyz, val.xyz exist)
        # If that fails, load the whole thing and split numerically
        try:
            self._try_named_splits(ds_cls, roots, cutoff, extra, merge, seed, stage)
        except FileNotFoundError:
            self._numeric_split(ds_cls, roots, cutoff, extra, merge, seed, split_ratio, stage)

    def _try_named_splits(
        self,
        ds_cls,
        roots,
        cutoff,
        extra,
        merge,
        seed,
        stage,
    ) -> None:
        """Try loading named split files ({split}.xyz etc.)."""
        if stage in ("fit", None):
            train_ds = _build_datasets(ds_cls, roots, "train", cutoff, extra)
            self.data_train = _merge_datasets(train_ds, merge, seed)

            val_ds = _build_datasets(ds_cls, roots, "val", cutoff, extra)
            self.data_val = _merge_datasets(val_ds, merge, seed)

        if stage in ("test", None):
            try:
                test_ds = _build_datasets(ds_cls, roots, "test", cutoff, extra)
                self.data_test = _merge_datasets(test_ds, merge, seed)
            except FileNotFoundError:
                self.data_test = None

    def _numeric_split(
        self,
        ds_cls,
        roots,
        cutoff,
        extra,
        merge,
        seed,
        split_ratio,
        stage,
    ) -> None:
        """Load all data at once and split by ratio."""
        all_datasets = _build_datasets(ds_cls, roots, "train", cutoff, extra)
        full = _merge_datasets(all_datasets, merge, seed)
        total = len(full)

        # Normalise split_ratio
        if isinstance(split_ratio, (list, ListConfig)):
            ratios = [float(r) for r in split_ratio]
        else:
            ratios = [float(split_ratio), 1.0 - float(split_ratio)]

        # Compute lengths
        if len(ratios) == 2:
            n_train = int(total * ratios[0])
            n_val = total - n_train
            lengths = [n_train, n_val]
        else:
            n_train = int(total * ratios[0])
            n_val = int(total * ratios[1])
            n_test = total - n_train - n_val
            lengths = [n_train, n_val, n_test]

        gen = torch.Generator().manual_seed(seed)
        splits = random_split(full, lengths, generator=gen)

        if stage in ("fit", None):
            self.data_train = splits[0]
            self.data_val = splits[1]

        if stage in ("test", None) and len(splits) == 3:
            self.data_test = splits[2]

    # ------------------------------------------------------------------
    # DataLoader construction
    # ------------------------------------------------------------------

    def _loader_kwargs(self) -> dict[str, typing.Any]:
        """Common DataLoader keyword arguments."""
        data_cfg = self.cfg.data
        num_workers = data_cfg.get("num_workers", 0)
        kwargs = dict(
            num_workers=num_workers,
            pin_memory=data_cfg.get("pin_memory", False),
        )
        # persistent_workers only valid when num_workers > 0
        if num_workers > 0:
            kwargs["persistent_workers"] = data_cfg.get("persistent_workers", True)
            pf = data_cfg.get("prefetch_factor")
            if pf is not None:
                kwargs["prefetch_factor"] = pf
        return kwargs

    def train_dataloader(self) -> DataLoader:
        return PyGDataLoader(
            self.data_train,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        return PyGDataLoader(
            self.data_val,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        if self.data_test is None:
            raise RuntimeError(
                "No test dataset available. Provide test_paths or a "
                "split_ratio with 3 values (e.g. [0.8, 0.1, 0.1])."
            )
        return PyGDataLoader(
            self.data_test,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            **self._loader_kwargs(),
        )
