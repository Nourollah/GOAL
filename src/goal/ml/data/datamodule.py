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

import sys
import typing
import warnings
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info
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
        # FairChem ASE-LMDB shards (OMol25 / OMat24 / ODAC23) — a split is
        # a directory of ``data0000.aselmdb`` … files, so discovery has to
        # know the suffix for Mode 1/4 to see them at all.
        ".aselmdb",
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


# Fragment-label options.  Understood by every ``BaseAtomicDataset``, but
# not by the benchmark loaders in ``goal.ml.data.benchmarks`` (md17 / qm9 / ani1),
# which are a separate hierarchy.
_FRAGMENT_KEYS: tuple[str, ...] = (
    "compute_fragment_index",
    "fragment_covalent_cutoff",
    "fragment_scheme",
    "fragment_charge",
    "fragment_on_failure",
    "fragment_smarts",
    "fragment_keep_groups",
)


def _resolve_fragment_keys(
    ds_cls: type,
    extra: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    """Drop the fragment-label options for datasets that cannot use them.

    ``data.compute_fragment_index`` / ``data.fragment_covalent_cutoff`` are
    forwarded to the dataset like every other non-meta ``data`` key, so a
    dataset class that predates them (the ``goal.ml.data.benchmarks`` benchmark
    loaders) would raise an opaque ``TypeError: unexpected keyword
    argument``.  Silently dropping them is right when the feature is *off*
    — the keys then say nothing about intent — but wrong when it is on,
    because the run would train with the fragment channel enabled and no labels to
    feed it.  So: drop when disabled, raise a pointed error when enabled.
    """
    present: list[str] = [k for k in _FRAGMENT_KEYS if k in extra]
    if not present:
        return extra

    import inspect

    params = inspect.signature(ds_cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return extra
    unsupported: list[str] = [k for k in present if k not in params]
    if not unsupported:
        return extra

    if bool(extra.get("compute_fragment_index", False)):
        raise ValueError(
            f"data.compute_fragment_index is enabled but dataset "
            f"{ds_cls.__name__} does not support fragment labels.  Fragment "
            f"cross-attention (model.backbone.fragment_ca) needs them, so "
            f"either switch to a dataset that provides them (xyz, trajectory, "
            f"lmdb, hdf5) or set data.compute_fragment_index: false and remove "
            f"the model.backbone.fragment_ca section."
        )
    return {k: v for k, v in extra.items() if k not in unsupported}


def _build_datasets(
    ds_cls: type,
    paths: list[str],
    split: str,
    cutoff: float,
    extra: dict[str, typing.Any],
) -> list[BaseAtomicDataset]:
    """Instantiate one dataset per path."""
    extra = _resolve_fragment_keys(ds_cls, extra)
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

    In every mode, ``data.monitor_test_during_training: true`` makes
    :meth:`val_dataloader` return the val **and** test loaders, so the
    held-out set is reported every validation epoch in its own logger
    sections (``test/``, ``test_energy/``, ``test_forces/``) instead of
    only at the end of a run.  See
    :attr:`monitor_test_during_training`.

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
        "multiprocessing_context",
        "monitor_test_during_training",
    }

    def _extra_kwargs(self) -> dict[str, typing.Any]:
        """Dataset-specific kwargs (everything that isn't meta)."""
        return {k: v for k, v in self.cfg.data.items() if k not in self._META_KEYS}

    # ------------------------------------------------------------------
    # Monitoring the held-out set during training
    # ------------------------------------------------------------------

    @property
    def monitor_test_during_training(self) -> bool:
        """Whether the test split also runs through the validation loop.

        ``data.monitor_test_during_training: true`` makes
        :meth:`val_dataloader` return **two** loaders — the val split and
        the test split — so a run reports both every epoch, separately.
        The intended use is an in-distribution val set plus an
        out-of-distribution test set (the GMD ``TestID`` / ``TestOOD``
        layout): the generalisation gap becomes a curve you watch instead
        of a number you only learn after training.

        Costs one extra forward pass over the test split per validation
        epoch, and the OOD numbers must *not* be used for model
        selection (``checkpoint_manager.top_k.metric``) or early
        stopping — monitoring a set you then select on turns it into a
        second validation set.
        """
        return bool(self.cfg.data.get("monitor_test_during_training", False))

    @property
    def val_split_names(self) -> list[str]:
        """Section name per dataloader returned by :meth:`val_dataloader`.

        ``["val"]`` normally, ``["val", "test"]`` when the test split is
        monitored during training.  ``GOALModule.validation_step`` reads
        this to pick the logger prefix per ``dataloader_idx``, which is
        what keeps the two sets in separate W&B sections.
        """
        names: list[str] = ["val"]
        if self.monitor_test_during_training and self.data_test is not None:
            names.append("test")
        return names

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

        rank_zero_info(
            f"[DataModule] Setting up dataset  type={data_cfg.dataset_type!r}  "
            f"cutoff={cutoff} Å  stage={stage!r}"
        )

        has_dir: bool = "train_dir" in data_cfg and data_cfg.train_dir is not None
        has_per_split: bool = "train_paths" in data_cfg and data_cfg.train_paths is not None

        # Every loading mode builds the test split only for
        # ``stage in ("test", None)``.  Monitoring the test set during
        # training needs it at ``fit`` time too, so widen the stage the
        # mode helpers see (``None`` = build every split).
        mode_stage: str | None = stage
        if stage == "fit" and self.monitor_test_during_training:
            mode_stage = None

        if has_dir:
            rank_zero_info("[DataModule] Mode 4 — directory-based per-split loading")
            self._setup_from_dirs(ds_cls, data_cfg, cutoff, extra, merge, seed, mode_stage)
        elif has_per_split:
            rank_zero_info("[DataModule] Mode 2 — per-split path loading")
            self._setup_per_split(ds_cls, data_cfg, cutoff, extra, merge, seed, mode_stage)
        else:
            rank_zero_info("[DataModule] Mode 1/3 — auto-split from root")
            self._setup_auto_split(ds_cls, data_cfg, cutoff, extra, merge, seed, mode_stage)

        # Report dataset sizes after setup
        if self.data_train is not None:
            rank_zero_info(f"[DataModule] Train  : {len(self.data_train):>8,} structures")
        if self.data_val is not None:
            rank_zero_info(f"[DataModule] Val    : {len(self.data_val):>8,} structures")
        if self.data_test is not None:
            rank_zero_info(f"[DataModule] Test   : {len(self.data_test):>8,} structures")

        if self.monitor_test_during_training and stage in ("fit", None):
            if self.data_test is None:
                warnings.warn(
                    "data.monitor_test_during_training is enabled but no test "
                    "split is configured (set test_dir / test_paths, or a "
                    "3-value split_ratio).  Training will report the val "
                    "split only — nothing will be logged under 'test/'.",
                    stacklevel=2,
                )
            else:
                rank_zero_info(
                    "[DataModule] Monitoring the test split during training "
                    "→ logged separately under 'test/', 'test_energy/', "
                    "'test_forces/' every validation epoch"
                )

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
        """Mode 1 / 3: load from root (single or list), then split.

        A **single file** root (the common ``root: /path/to/file.traj`` case)
        has no sibling ``{split}.<ext>`` files to discover — the datasets
        would each fall back to loading the *whole* file, so every split would
        be identical (train == val == test, full data leakage) and
        ``split_ratio`` would be silently ignored.  We therefore route file
        roots straight to the numeric split.  Named per-split files only make
        sense inside a **directory** root, so ``_try_named_splits`` is
        attempted only then, falling back to a numeric split over the
        directory's contents if no ``{split}.<ext>`` files exist.
        """
        roots = _resolve_paths(data_cfg.root)
        split_ratio = data_cfg.get("split_ratio", [0.8, 0.1, 0.1])

        all_files: bool = all(Path(r).is_file() for r in roots)
        if all_files:
            # Single file (or explicit list of files) → load once, split by ratio.
            self._numeric_split(ds_cls, roots, cutoff, extra, merge, seed, split_ratio, stage)
            return

        # Directory root(s): prefer real {split}.<ext> files; if none exist the
        # datasets raise FileNotFoundError and we numeric-split the directory.
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
        """Load all data at once and split by ratio.

        Directory roots are expanded into their contained data files so a
        single dataset can be built and split.  Split lengths always sum to
        the total (no frames lost); the remainder after flooring val/test is
        assigned to *train*, so a tiny dataset (e.g. 1 frame) puts everything
        in train rather than starving it.  A positive ratio that rounds to 0
        frames emits a warning instead of failing silently.
        """
        # Expand any directory roots into their contained data files —
        # unless the dataset's root *is* a directory (a Zarr store), which
        # declares ``ROOT_IS_DIRECTORY``.
        expanded: list[str] = []
        root_is_directory: bool = bool(getattr(ds_cls, "ROOT_IS_DIRECTORY", False))
        for r in roots:
            p = Path(r)
            if p.is_dir() and not root_is_directory:
                expanded.extend(_discover_files_in_dir(p))
            else:
                expanded.append(str(r))

        all_datasets = _build_datasets(ds_cls, expanded, "train", cutoff, extra)
        full = _merge_datasets(all_datasets, merge, seed)
        total = len(full)

        # Normalise split_ratio
        if isinstance(split_ratio, (list, ListConfig)):
            ratios = [float(r) for r in split_ratio]
        else:
            ratios = [float(split_ratio), 1.0 - float(split_ratio)]

        # Compute lengths.  Floor val/test, give the remainder to train so
        # train is never starved and small datasets don't lose their frames.
        if len(ratios) == 2:
            n_val = min(int(total * ratios[1]), total)
            n_train = total - n_val
            lengths = [n_train, n_val]
            names = ("val",)
            sizes = (n_val,)
        else:
            n_val = int(total * ratios[1])
            n_test = int(total * ratios[2])
            n_train = total - n_val - n_test
            lengths = [n_train, n_val, n_test]
            names = ("val", "test")
            sizes = (n_val, n_test)

        # Warn if a requested (positive-ratio) split rounded down to 0 frames.
        for name, size, ratio in zip(names, sizes, ratios[1:]):
            if ratio > 0.0 and size == 0:
                warnings.warn(
                    f"split_ratio requests a '{name}' split but only {total} "
                    f"frame(s) are available, so it rounded to 0. Provide more "
                    f"data or explicit per-split files/paths.",
                    stacklevel=2,
                )

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
            # Python 3.14 changed the default multiprocessing start method on
            # Linux to 'forkserver', which trips torch DataLoader workers with
            # "ValueError: too many fds". Use a 'fork' context (the historical
            # default, safe for CPU-only data loading) unless the user overrides
            # via data.multiprocessing_context. Skipped on Windows (no fork).
            mp_ctx: str | None = data_cfg.get("multiprocessing_context", None)
            if mp_ctx is None and sys.platform != "win32":
                mp_ctx = "fork"
            if mp_ctx is not None:
                kwargs["multiprocessing_context"] = mp_ctx
        return kwargs

    def train_dataloader(self) -> DataLoader:
        """The shuffled training loader over ``data_train``.

        Uses ``cfg.data.batch_size`` and the shared worker settings from
        :meth:`_loader_kwargs`.
        """
        return PyGDataLoader(
            self.data_train,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            **self._loader_kwargs(),
        )

    def _eval_loader(self, dataset: Dataset) -> DataLoader:
        """An unshuffled loader over *dataset* (val / test)."""
        return PyGDataLoader(
            dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        """The validation loader, or ``[val, test]`` when the test split
        is monitored during training.

        A plain loader is returned in the single-set case so Lightning
        does not append a ``/dataloader_idx_0`` suffix to anything and
        existing runs log exactly the keys they always did.
        """
        loaders: list[DataLoader] = [self._eval_loader(self.data_val)]
        if self.monitor_test_during_training and self.data_test is not None:
            loaders.append(self._eval_loader(self.data_test))
        return loaders[0] if len(loaders) == 1 else loaders

    def test_dataloader(self) -> DataLoader:
        """The unshuffled test loader over ``data_test``.

        Raises
        ------
        RuntimeError
            If ``data_test`` is ``None``: neither ``test_paths`` nor a
            three-value ``split_ratio`` provided a test split.
        """
        if self.data_test is None:
            raise RuntimeError(
                "No test dataset available. Provide test_paths or a "
                "split_ratio with 3 values (e.g. [0.8, 0.1, 0.1])."
            )
        return self._eval_loader(self.data_test)
