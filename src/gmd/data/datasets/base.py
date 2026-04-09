"""Abstract base class for all GMD datasets."""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import Dataset

from gmd.data.graph import AtomicGraph


class BaseAtomicDataset(Dataset, ABC):
    """Abstract base dataset that yields ``AtomicGraph`` instances.

    All concrete datasets must implement ``__len__`` and ``__getitem__``.
    The ``__getitem__`` method must return an ``AtomicGraph``.
    """

    def __init__(
        self,
        root: typing.Union[str, Path],
        cutoff: float,
        split: str = "train",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.root: Path = Path(root)
        self.cutoff: float = cutoff
        self.split: str = split
        self.dtype: torch.dtype = dtype

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> AtomicGraph: ...
