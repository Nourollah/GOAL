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
    """

    def __init__(
        self,
        root: str,
        cutoff: float,
        split: str = "train",
        dtype: torch.dtype = torch.float64,
        transform: typing.Optional[typing.Callable[..., typing.Any]] = None,
    ) -> None:
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        self.root: Path = Path(root)
        self.cutoff: float = cutoff
        self.split: str = split
        self.dtype: torch.dtype = dtype
        self.transform: typing.Optional[typing.Callable[..., typing.Any]] = transform

        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "processed").mkdir(exist_ok=True)

        self._data: typing.List[typing.Any] = self._load_or_process()
        indices: typing.Dict[str, typing.List[int]] = self.split_indices()
        self._indices: typing.List[int] = indices[split]

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _download_and_process(self) -> typing.List[typing.Any]:
        """Download raw data and convert to a list of graph-like objects."""
        ...

    @abstractmethod
    def split_indices(self) -> typing.Dict[str, typing.List[int]]:
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

    def _load_or_process(self) -> typing.List[typing.Any]:
        """Load from cache if exists, otherwise download and process."""
        cache: Path = self._cache_path()
        if cache.exists():
            return torch.load(cache, weights_only=False)
        data: typing.List[typing.Any] = self._download_and_process()
        torch.save(data, cache)
        return data

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> typing.Any:
        item: typing.Any = self._data[self._indices[idx]]
        if self.transform is not None:
            item = self.transform(item)
        return item

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self) -> typing.Dict[str, float]:
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

        num_atoms_list: typing.List[int] = []
        energies: typing.List[float] = []
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
                forces_sq_sum += (f ** 2).sum().item()
                forces_count += f.numel()

        stats: typing.Dict[str, float] = {"num_structures": float(n)}
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
    ) -> typing.Dict[str, typing.List[int]]:
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

        Returns
        -------
        dict
            ``{'train': [...], 'val': [...], 'test': [...]}``
        """
        gen: torch.Generator = torch.Generator().manual_seed(seed)
        perm: torch.Tensor = torch.randperm(total, generator=gen)
        indices: typing.List[int] = perm.tolist()
        return {
            "train": indices[:train_size],
            "val": indices[train_size : train_size + val_size],
            "test": indices[train_size + val_size :],
        }
