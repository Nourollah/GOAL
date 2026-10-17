"""Abstract base class for all GOAL datasets."""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import Dataset

from goal.ml.data.graph import AtomicGraph


class BaseAtomicDataset(Dataset, ABC):
    """Abstract base dataset that yields ``AtomicGraph`` instances.

    All concrete datasets must implement ``__len__`` and ``__getitem__``.
    The ``__getitem__`` method must return an ``AtomicGraph``.

    Parameters
    ----------
    root, cutoff, split, dtype, neighbor_list_backend
        Standard dataset settings — see the concrete subclasses.
    compute_fragment_index : bool
        Attach ``fragment_index`` (connected-component labels under a
        short covalent cutoff) to every graph this dataset produces.
        Required by the fragment-interaction module; config key
        ``data.compute_fragment_index``.  Default ``False`` so datasets
        pay nothing unless the model asks for it.
    fragment_covalent_cutoff : float
        Bond threshold for that decomposition (Angstrom); config key
        ``data.fragment_covalent_cutoff``.
    fragment_scheme : str
        How to split a structure into fragments — ``connected`` (default,
        no rdkit), ``rdkit_components``, ``rotatable``, ``brics`` or
        ``recap``.  See :mod:`goal.ml.data.fragments`.
    fragment_charge : int
        Total charge used by the rdkit schemes' bond perception.
    fragment_on_failure : str
        ``"fallback"`` (warn once, use ``connected``) or ``"raise"``.
    fragment_smarts : sequence of str, optional
        Bond patterns to cut, required by ``fragment_scheme="custom"``.
    fragment_keep_groups : sequence of str, optional
        SMARTS whose matched atoms must land in the same fragment,
        applied on top of any scheme.
    """

    def __init__(
        self,
        root: str | Path,
        cutoff: float,
        split: str = "train",
        dtype: torch.dtype = torch.float64,
        neighbor_list_backend: str = "ase",
        compute_fragment_index: bool = False,
        fragment_covalent_cutoff: float = 1.8,
        fragment_scheme: str = "connected",
        fragment_charge: int = 0,
        fragment_on_failure: str = "fallback",
        fragment_smarts: typing.Sequence[str] | None = None,
        fragment_keep_groups: typing.Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.root: Path = Path(root)
        self.cutoff: float = cutoff
        self.split: str = split
        self.dtype: torch.dtype = dtype
        self.neighbor_list_backend: str = neighbor_list_backend
        self.compute_fragment_index: bool = bool(compute_fragment_index)
        self.fragment_covalent_cutoff: float = float(fragment_covalent_cutoff)
        self.fragment_scheme: str = str(fragment_scheme)
        self.fragment_charge: int = int(fragment_charge)
        self.fragment_on_failure: str = str(fragment_on_failure)
        from goal.ml.data.fragments import as_pattern_list

        # A bare string in YAML means one pattern, not a list of characters.
        self.fragment_smarts: list[str] | None = as_pattern_list(fragment_smarts) or None
        self.fragment_keep_groups: list[str] | None = as_pattern_list(fragment_keep_groups) or None
        if self.compute_fragment_index:
            # Fail on a bad scheme or a typo'd SMARTS here, at setup, rather
            # than thousands of frames into the first epoch.
            from goal.ml.data.fragments import validate_fragment_config

            validate_fragment_config(
                self.fragment_scheme, self.fragment_smarts, self.fragment_keep_groups
            )

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> AtomicGraph: ...

    # ------------------------------------------------------------------
    # Fragment labels
    # ------------------------------------------------------------------

    def attach_fragment_index(self, graph: AtomicGraph) -> AtomicGraph:
        """Add ``fragment_index`` to *graph* in place when enabled.

        For datasets that build their graphs through
        :meth:`AtomicGraph.from_ase` / :meth:`~AtomicGraph.from_dict` the
        labels come straight from those constructors; this helper covers
        the datasets that instantiate ``AtomicGraph`` directly (HDF5).
        A no-op when ``compute_fragment_index`` is off or the graph is
        already labelled.
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
