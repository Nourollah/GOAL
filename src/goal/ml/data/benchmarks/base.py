"""Shared base class and utilities for benchmark datasets.

Provides caching, automatic download, deterministic splits, and
statistics computation for all benchmark dataset adapters.
"""

from __future__ import annotations

import hashlib
import typing
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import Dataset


class BenchmarkDataset(Dataset, ABC):
    """Abstract base for benchmark datasets.

    Extends ``torch.utils.data.Dataset`` with benchmark-specific
    functionality: automatic download, split management, deterministic
    reproducible splits, and standardised statistics.

    Subclasses must implement:
        ``_download_and_process()`` — fetch raw data and convert to list of dicts
        ``split_indices()`` — return train/val/test index splits
        ``citation()`` — return BibTeX citation string for the dataset

    All benchmark datasets:
        - Cache processed data to disk as ``.pt`` files
        - Support deterministic train/val/test splits via seed
        - Report dataset statistics (num structures, avg atoms, etc.)
        - Never require manual file downloads from the user
        - Accept the ``fragment_*`` options (see ``__init__``) so models
          that need fragment labels — NF_SIMURGH pools its per-atom
          energies onto fragment virtual nodes and raises without them —
          can train on a benchmark set

    Parameters
    ----------
    root, cutoff, split, dtype, transform
        Standard dataset settings — see the concrete subclasses.
    compute_fragment_index : bool
        Attach ``fragment_index`` (per-atom labels from the chosen
        decomposition) to every graph this dataset yields.  Config key
        ``data.compute_fragment_index``.  Default ``False``, so nothing
        pays for it unless a model asks.
    fragment_covalent_cutoff : float
        Bond threshold for the distance-graph part of the decomposition
        (Angstrom).
    fragment_scheme : str
        How to split a structure — ``connected`` (default, no rdkit),
        ``rdkit_components``, ``rotatable``, ``brics``, ``recap`` or
        ``custom``.  See :mod:`goal.ml.data.fragments`.  Note that these
        are all single-molecule datasets, where ``connected`` returns one
        fragment per structure and a fragment-interaction channel then
        has no pairs to work with.
    fragment_charge : int
        Total charge used by the rdkit schemes' bond perception.
    fragment_on_failure : str
        ``"fallback"`` (warn once, use ``connected``) or ``"raise"``.
    fragment_smarts : sequence of str, optional
        Bond patterns to cut, required by ``fragment_scheme="custom"``.
    fragment_keep_groups : sequence of str, optional
        SMARTS whose matched atoms must land in the same fragment.
    """

    def __init__(
        self,
        root: str,
        cutoff: float,
        split: str = "train",
        dtype: torch.dtype = torch.float64,
        transform: typing.Callable[..., typing.Any] | None = None,
        compute_fragment_index: bool = False,
        fragment_covalent_cutoff: float = 1.8,
        fragment_scheme: str = "connected",
        fragment_charge: int = 0,
        fragment_on_failure: str = "fallback",
        fragment_smarts: typing.Sequence[str] | None = None,
        fragment_keep_groups: typing.Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        self.root: Path = Path(root)
        self.cutoff: float = cutoff
        self.split: str = split
        self.dtype: torch.dtype = dtype
        self.transform: typing.Callable[..., typing.Any] | None = transform

        # Fragment labels — same semantics, names and defaults as
        # ``goal.ml.data.datasets.base.BaseAtomicDataset``, because the
        # DataModule forwards ``data.fragment_*`` to whichever dataset is
        # configured and a benchmark adapter that silently ignored them
        # would train a fragment model with no fragments.
        from goal.ml.data.fragments import as_pattern_list

        self.compute_fragment_index: bool = bool(compute_fragment_index)
        self.fragment_covalent_cutoff: float = float(fragment_covalent_cutoff)
        self.fragment_scheme: str = str(fragment_scheme)
        self.fragment_charge: int = int(fragment_charge)
        self.fragment_on_failure: str = str(fragment_on_failure)
        # A bare string in YAML means one pattern, not a list of characters.
        self.fragment_smarts: list[str] | None = as_pattern_list(fragment_smarts) or None
        self.fragment_keep_groups: list[str] | None = as_pattern_list(fragment_keep_groups) or None
        if self.compute_fragment_index:
            # Fail on a bad scheme or a typo'd SMARTS here, at setup,
            # rather than thousands of frames into the first epoch.
            from goal.ml.data.fragments import validate_fragment_config

            validate_fragment_config(
                self.fragment_scheme, self.fragment_smarts, self.fragment_keep_groups
            )

        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "processed").mkdir(exist_ok=True)

        self._data: list[typing.Any] = self._load_or_process()
        indices: dict[str, list[int]] = self.split_indices()
        self._indices: list[int] = indices[split]

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _download_and_process(self) -> list[typing.Any]:
        """Download raw data and convert to a list of graph-like objects."""
        ...

    @abstractmethod
    def split_indices(self) -> dict[str, list[int]]:
        """Return ``{'train': [...], 'val': [...], 'test': [...]}``."""
        ...

    @abstractmethod
    def citation(self) -> str:
        """Return BibTeX citation for this dataset."""
        ...

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_path(self) -> Path:
        """Path to processed cache file, encoding cutoff and dtype."""
        dtype_str: str = str(self.dtype).replace("torch.", "")
        name: str = self._cache_name()
        return self.root / "processed" / f"{name}_cutoff{self.cutoff}_{dtype_str}.pt"

    def _cache_name(self) -> str:
        """Subclass-specific cache filename prefix. Override as needed."""
        return self.__class__.__name__.lower()

    def _load_or_process(self) -> list[typing.Any]:
        """Load from cache if exists, otherwise download and process."""
        cache: Path = self._cache_path()
        if cache.exists():
            return torch.load(cache, weights_only=False)
        data: list[typing.Any] = self._download_and_process()
        torch.save(data, cache)
        return data

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> typing.Any:
        item: typing.Any = self._data[self._indices[idx]]
        item = self._attach_fragment_index(item)
        if self.transform is not None:
            item = self.transform(item)
        return item

    def _attach_fragment_index(self, graph: typing.Any) -> typing.Any:
        """Label *graph* with ``fragment_index`` when the feature is on.

        Attached **lazily, in place**, rather than during
        ``_download_and_process``.  Two reasons:

        * the processed ``.pt`` cache is keyed by cutoff and dtype only,
          so baking labels into it would let a cache built under one
          ``fragment_scheme`` be silently reused under another;
        * the graphs live in memory for the life of the dataset, so
          writing the labels onto them means each structure is
          decomposed exactly once per process no matter how many epochs
          read it — the rdkit schemes' real cost is paid in epoch 1 and
          never again.
        """
        if not self.compute_fragment_index:
            return graph

        from goal.ml.data.fragments import attach_fragment_index

        return attach_fragment_index(
            graph,
            covalent_cutoff=self.fragment_covalent_cutoff,
            scheme=self.fragment_scheme,
            charge=self.fragment_charge,
            on_failure=self.fragment_on_failure,
            smarts=self.fragment_smarts,
            keep_groups=self.fragment_keep_groups,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self) -> dict[str, float]:
        """Compute dataset statistics for the current split.

        Returns
        -------
        dict
            Keys: ``num_structures``, ``avg_num_atoms``,
            ``energy_mean``, ``energy_std``, ``forces_rms``.
        """
        n: int = len(self)
        if n == 0:
            return {"num_structures": 0}

        num_atoms_list: list[int] = []
        energies: list[float] = []
        forces_sq_sum: float = 0.0
        forces_count: int = 0

        for i in range(n):
            item = self[i]
            if hasattr(item, "pos"):
                num_atoms_list.append(item.pos.shape[0])
            if hasattr(item, "energy") and item.energy is not None:
                e = item.energy
                energies.append(e.item() if hasattr(e, "item") else float(e))
            if hasattr(item, "forces") and item.forces is not None:
                f = item.forces
                forces_sq_sum += (f**2).sum().item()
                forces_count += f.numel()

        stats: dict[str, float] = {"num_structures": float(n)}
        if num_atoms_list:
            stats["avg_num_atoms"] = sum(num_atoms_list) / n
        if energies:
            e_t = torch.tensor(energies)
            stats["energy_mean"] = e_t.mean().item()
            stats["energy_std"] = e_t.std().item()
        if forces_count > 0:
            stats["forces_rms"] = (forces_sq_sum / forces_count) ** 0.5
        return stats

    # ------------------------------------------------------------------
    # Reproducible split helper
    # ------------------------------------------------------------------

    @staticmethod
    def _random_split_indices(
        total: int,
        train_size: int,
        val_size: int,
        seed: int = 42,
        test_size: int | None = None,
    ) -> dict[str, list[int]]:
        """Deterministic random split into train / val / test.

        Parameters
        ----------
        total : int
            Total number of structures.
        train_size : int
            Number of training structures.
        val_size : int
            Number of validation structures.
        seed : int
            Random seed for reproducibility.
        test_size : int, optional
            Cap on the test split.  ``None`` (default) gives it
            everything train and val did not take — which for an MD
            trajectory is most of the file (rMD17 aspirin: ~99,000
            frames).  A cap is worth setting when the test split is
            only there to be evaluated occasionally, and essential if
            you turn on ``monitor_test_during_training``.

        Returns
        -------
        dict
            ``{'train': [...], 'val': [...], 'test': [...]}``
        """
        gen: torch.Generator = torch.Generator().manual_seed(seed)
        perm: torch.Tensor = torch.randperm(total, generator=gen)
        indices: list[int] = perm.tolist()
        test: list[int] = indices[train_size + val_size :]
        if test_size is not None:
            test = test[: int(test_size)]
        return {
            "train": indices[:train_size],
            "val": indices[train_size : train_size + val_size],
            "test": test,
        }
