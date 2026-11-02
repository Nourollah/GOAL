"""Row sampling of per-structure Hessian labels for batching (Hessian distillation).

A graph may carry its full Hessian label ``hessian`` ``(3N, 3N)`` (eV/Å²),
which PyG cannot batch.  :class:`HessianRows` wraps a dataset: every time
an item is fetched it draws ``rows`` distinct rows ``j`` uniformly, stores
them as ``hessian_row`` ``(N, rows, 3)`` (row ``j`` reshaped like the
forces) and ``hessian_row_number`` ``(1, rows)``, and drops the full matrix.
(Not ``..._index``: PyG would offset an ``index`` attribute by the atom count.)
Row ``j`` belongs to atom ``j // 3``, Cartesian axis ``j % 3``.  Graphs
without a ``hessian`` pass through unchanged.
"""

from __future__ import annotations

import typing

import torch
from torch.utils.data import Dataset


class HessianRows(Dataset):
    """``dataset`` with one fresh sample of ``rows`` Hessian rows per access."""

    def __init__(self, dataset: typing.Any, rows: int = 1) -> None:
        self.dataset, self.rows = dataset, rows

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> typing.Any:
        graph = self.dataset[idx]
        hessian = getattr(graph, "hessian", None)
        if hessian is None:
            return graph
        graph = graph.clone()
        del graph.hessian
        n3 = hessian.shape[0]
        index = torch.randperm(n3)[: self.rows]
        graph.hessian_row = hessian[index].reshape(self.rows, n3 // 3, 3).transpose(0, 1)
        graph.hessian_row_number = index.reshape(1, -1)
        return graph

    def __getattr__(self, name: str) -> typing.Any:
        """Everything else (``graphs``, statistics helpers ...) comes from the dataset."""
        if name in ("dataset", "rows"):
            raise AttributeError(name)
        return getattr(self.dataset, name)


def with_hessian_rows(dataset: typing.Any, rows: int = 1) -> typing.Any:
    """``dataset`` itself, or wrapped in :class:`HessianRows` when its graphs carry Hessians."""
    if dataset is None or len(dataset) == 0:
        return dataset
    return HessianRows(dataset, rows) if hasattr(dataset[0], "hessian") else dataset
