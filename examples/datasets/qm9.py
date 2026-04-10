"""QM9 quantum chemistry dataset.

134k drug-like organic molecules with up to 9 heavy atoms (C, H, O, N, F),
with 19 quantum chemical properties computed at B3LYP/6-31G(2df,p) level.

Unlike MD17/ANI-1, QM9 is a property prediction dataset — no forces.
Use for benchmarking molecular property prediction heads.

Data source: ``torch_geometric.datasets.QM9``
"""

from __future__ import annotations

import typing
from pathlib import Path

import torch

from examples.datasets.base import BenchmarkDataset

# QM9 target properties — all 19 properties in the dataset
QM9_TARGETS: dict[int, str] = {
    0: "mu",  # Dipole moment [D]
    1: "alpha",  # Isotropic polarisability [a₀³]
    2: "homo",  # HOMO energy [eV]
    3: "lumo",  # LUMO energy [eV]
    4: "gap",  # HOMO-LUMO gap [eV]
    5: "r2",  # Electronic spatial extent [a₀²]
    6: "zpve",  # Zero point vibrational energy [eV]
    7: "u0",  # Internal energy at 0 K [eV]
    8: "u298",  # Internal energy at 298.15 K [eV]
    9: "h298",  # Enthalpy at 298.15 K [eV]
    10: "g298",  # Free energy at 298.15 K [eV]
    11: "cv",  # Heat capacity at 298.15 K [cal/mol/K]
    12: "u0_atom",  # Atomisation energy at 0 K [eV]
    13: "u298_atom",  # Atomisation energy at 298.15 K [eV]
    14: "h298_atom",  # Atomisation enthalpy at 298.15 K [eV]
    15: "g298_atom",  # Atomisation free energy at 298.15 K [eV]
    16: "A",  # Rotational constant A [GHz]
    17: "B",  # Rotational constant B [GHz]
    18: "C",  # Rotational constant C [GHz]
}

# Reverse: name → index
_QM9_TARGETS_BY_NAME: dict[str, int] = {v: k for k, v in QM9_TARGETS.items()}

# Standard DimeNet++ split sizes
_QM9_TRAIN_SIZE: int = 110_000
_QM9_VAL_SIZE: int = 10_000


class QM9Dataset(BenchmarkDataset):
    """QM9 quantum chemistry dataset.

    134k drug-like organic molecules with 19 quantum chemical properties.
    Standard split: 110k train, 10k val, ~14k test (DimeNet++ protocol).

    Parameters
    ----------
    root : str
        Directory for downloaded and cached data.
    target : str or int or None
        Which property to predict.  Can be an integer (0–18) or a string
        from ``QM9_TARGETS`` values (e.g. ``'homo'``, ``'gap'``).
        If ``None``, all 19 properties are stored.
    cutoff : float
        Neighbour list cutoff in Ångström.
    split : str
        ``'train'``, ``'val'``, or ``'test'``.
    seed : int
        Random seed for split reproducibility.
    """

    def __init__(
        self,
        root: str,
        target: str | int | None = "homo",
        cutoff: float = 5.0,
        split: str = "train",
        seed: int = 42,
        dtype: torch.dtype = torch.float64,
        transform: typing.Callable[..., typing.Any] | None = None,
    ) -> None:
        self.target_idx: int | None = self._resolve_target(target)
        self.target_name: str | None = (
            QM9_TARGETS.get(self.target_idx) if self.target_idx is not None else None
        )
        self.seed: int = seed
        super().__init__(root=root, cutoff=cutoff, split=split, dtype=dtype, transform=transform)

    @staticmethod
    def _resolve_target(
        target: str | int | None,
    ) -> int | None:
        """Convert target name/index to integer index."""
        if target is None:
            return None
        if isinstance(target, int):
            if target not in QM9_TARGETS:
                raise ValueError(f"QM9 target index must be 0–18, got {target}")
            return target
        if target in _QM9_TARGETS_BY_NAME:
            return _QM9_TARGETS_BY_NAME[target]
        raise ValueError(
            f"Unknown QM9 target '{target}'. " f"Available: {list(_QM9_TARGETS_BY_NAME.keys())}"
        )

    def _cache_name(self) -> str:
        suffix: str = f"_target{self.target_idx}" if self.target_idx is not None else "_all"
        return f"qm9{suffix}"

    def _download_and_process(self) -> list[typing.Any]:
        """Download via PyG and convert to AtomicGraph list."""
        from torch_geometric.datasets import QM9 as PyGQM9
        from torch_geometric.nn import radius_graph

        from goal.ml.data.graph import AtomicGraph

        pyg_dataset = PyGQM9(root=str(self.root / "raw"))

        graphs: list[AtomicGraph] = []
        for data in pyg_dataset:
            positions: torch.Tensor = data.pos.to(self.dtype)
            atomic_numbers: torch.Tensor = data.z.long()

            # Build graph
            edge_index: torch.Tensor = radius_graph(positions, r=self.cutoff, loop=False)
            row, col = edge_index
            edge_vecs: torch.Tensor = positions[col] - positions[row]
            edge_lens: torch.Tensor = edge_vecs.norm(dim=-1)

            # Target property/properties
            energy: torch.Tensor | None = None
            extra_props: dict[str, torch.Tensor] = {}

            if self.target_idx is not None:
                # Single-property mode — store as energy for backward compat
                energy = data.y[:, self.target_idx : self.target_idx + 1].to(self.dtype)
            else:
                # Multi-property mode — store each property under its own key
                # so MultiHead + ScalarPropertyLoss can consume them by name
                for idx, prop_name in QM9_TARGETS.items():
                    extra_props[prop_name] = data.y[:, idx : idx + 1].to(self.dtype).squeeze(0)

            graph: AtomicGraph = AtomicGraph(
                positions=positions,
                atomic_numbers=atomic_numbers,
                cell=torch.zeros(3, 3, dtype=self.dtype),
                pbc=torch.zeros(3, dtype=torch.bool),
                edge_index=edge_index,
                edge_vectors=edge_vecs,
                edge_lengths=edge_lens,
                energy=energy,
                **extra_props,
            )
            graphs.append(graph)
        return graphs

    def split_indices(self) -> dict[str, list[int]]:
        """Reproducible split following DimeNet++ protocol."""
        return self._random_split_indices(
            total=len(self._data),
            train_size=_QM9_TRAIN_SIZE,
            val_size=_QM9_VAL_SIZE,
            seed=self.seed,
        )

    def citation(self) -> str:
        return (
            "@article{ramakrishnan2014quantum,\n"
            "  title={Quantum chemistry structures and properties of "
            "134 kilo molecules},\n"
            "  author={Ramakrishnan, Raghunathan and Dral, Pavlo O and "
            "Rupp, Matthias and von Lilienfeld, O Anatole},\n"
            "  journal={Scientific data},\n"
            "  year={2014}\n"
            "}"
        )
