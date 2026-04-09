"""Concatenated dataset that merges multiple atomic datasets into one.

Supports two merge strategies:
- ``sequential``: datasets are concatenated in order
- ``random``: indices are randomly shuffled after concatenation
"""

from __future__ import annotations

import typing

import torch
from torch.utils.data import Dataset

from gmd.data.graph import AtomicGraph


class ConcatAtomicDataset(Dataset):
    """Concatenate multiple ``BaseAtomicDataset`` instances into one.

    Parameters
    ----------
    datasets : list[Dataset]
        Individual datasets to merge.
    merge_strategy : str
        ``"sequential"`` (default) or ``"random"``.
    seed : int | None
        Random seed for the ``"random"`` strategy. ``None`` uses a
        non-deterministic seed.
    """

    def __init__(
        self,
        datasets: typing.List[Dataset],
        merge_strategy: str = "sequential",
        seed: typing.Optional[int] = None,
    ) -> None:
        super().__init__()
        self.datasets: typing.List[Dataset] = datasets
        self.merge_strategy: str = merge_strategy

        # Build cumulative lengths for O(log n) index lookup
        self._cumulative_lengths: typing.List[int] = []
        total: int = 0
        for ds in datasets:
            total += len(ds)
            self._cumulative_lengths.append(total)

        # Build index mapping
        self._indices: typing.List[int] = list(range(total))
        if merge_strategy == "random":
            gen: torch.Generator = torch.Generator()
            if seed is not None:
                gen.manual_seed(seed)
            perm: typing.List[int] = torch.randperm(total, generator=gen).tolist()
            self._indices = perm

    def __len__(self) -> int:
        if not self._cumulative_lengths:
            return 0
        return self._cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> AtomicGraph:
        mapped: int = self._indices[idx]
        # Binary search for which dataset this index belongs to
        ds_idx: int = 0
        for i, cum in enumerate(self._cumulative_lengths):
            if mapped < cum:
                ds_idx = i
                break
        offset: int = self._cumulative_lengths[ds_idx - 1] if ds_idx > 0 else 0
        return self.datasets[ds_idx][mapped - offset]
