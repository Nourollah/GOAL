"""SIMURGH Simplified — Monolithic (the fully simplified version).

Three parallel lines (atom, pair, fragment) over one shared input,
combined only at the readout — see
:mod:`~goal.ml.nn.models.simurgh.simurgh_simplified_layers` for the
skeleton and its design invariants.

**Monolithic** means every line's processor slot holds a *single* stack
of simple equivariant layers that processes all of that line's inputs:
no artisans, no routing, no experts.  It is the ablation baseline for
``simurgh_simplified_routed``: both models run the identical skeleton
(shared input, line wiring, residuals, norms, readouts, combination,
E0/scale, forces), and differ only in the three processors below.

Processors
----------
* atom line — one :class:`NodewiseUnit` (layer 1) and one
  :class:`EquivariantMessageUnit` per message-passing layer;
* pair line — one :class:`PairEdgeUnit` per edge layer;
* fragment line — one :class:`EquivariantMessageUnit` per virtual-node
  layer.

Registered as ``"simurgh_simplified_monolithic"`` in ``MODEL_REGISTRY``
and ``BACKBONE_REGISTRY`` (auto-discovered; configure with
``head: null``).  Config: ``src/goal/configs/ml/simurgh_simplified_monolithic_gmd26.yaml``.

Isolation: a new file that imports stable primitives, blocks and the
shared skeleton; no existing model is modified.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import Irreps

from goal.ml.nn.models.simurgh.simurgh_simplified_layers import (
    AtomLineProcessor,
    EquivariantMessageUnit,
    FragmentLineProcessor,
    NodewiseUnit,
    PairEdgeUnit,
    PairLineProcessor,
    SharedInputs,
    SimplifiedSettings,
    SimurghSimplifiedModel,
)
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY

__all__ = [
    "MonolithicAtomProcessor",
    "MonolithicFragmentProcessor",
    "MonolithicPairProcessor",
    "MonolithicSimurghSimplified",
]


# ---------------------------------------------------------------------------
# Processors — one stack per line
# ---------------------------------------------------------------------------


class MonolithicAtomProcessor(AtomLineProcessor):
    """Atom line: one node-wise unit, then one message unit per layer."""

    def __init__(
        self,
        irreps: Irreps,
        irreps_sh: Irreps,
        num_layers: int,
        num_rbf: int,
        radial_hidden: int,
    ) -> None:
        super().__init__()
        self.nodewise_unit: NodewiseUnit = NodewiseUnit(irreps)
        self.message_units: nn.ModuleList = nn.ModuleList(
            EquivariantMessageUnit(irreps, irreps_sh, num_rbf, radial_hidden)
            for _ in range(int(num_layers) - 1)
        )

    def nodewise(self, h: torch.Tensor, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Apply the single node-wise unit to every atom; returns ``(N, irreps.dim)``."""
        return self.nodewise_unit(h, atomic_numbers)

    def message(
        self,
        layer: int,
        h: torch.Tensor,
        atomic_numbers: torch.Tensor,
        inputs: SharedInputs,
        agg_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Run this layer's message unit over every edge.

        Messages flow source → receiver along ``inputs.edge_index``; the
        result is the ``(N, irreps.dim)`` equivariant update of every atom.
        """
        row, col = inputs.edge_index
        unit = typing.cast(EquivariantMessageUnit, self.message_units[layer])
        return unit(
            h[row], inputs.edge_sh, inputs.edge_rbf, inputs.edge_env, col, h, agg_scale=agg_scale
        )


class MonolithicPairProcessor(PairLineProcessor):
    """Pair line: one edge unit per layer, applied to every edge."""

    def __init__(
        self,
        irreps: Irreps,
        irreps_sh_even: Irreps,
        num_layers: int,
        num_rbf: int,
        radial_hidden: int,
    ) -> None:
        super().__init__()
        self.edge_units: nn.ModuleList = nn.ModuleList(
            PairEdgeUnit(irreps, irreps_sh_even, num_rbf, radial_hidden)
            for _ in range(int(num_layers))
        )

    def edge(
        self,
        layer: int,
        s: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh_even: torch.Tensor,
        edge_rbf: torch.Tensor,
        edge_env: torch.Tensor,
        pair_types: torch.Tensor,
    ) -> torch.Tensor:
        """Run this layer's edge unit on every edge; returns ``(E, irreps.dim)``.

        *pair_types* is ignored: the monolithic unit is not type-modulated.
        """
        row, col = edge_index
        unit = typing.cast(PairEdgeUnit, self.edge_units[layer])
        return unit(s[row], s[col], edge_sh_even, edge_rbf, edge_env)


class MonolithicFragmentProcessor(FragmentLineProcessor):
    """Fragment line: one virtual-node message unit per layer."""

    def __init__(
        self,
        irreps: Irreps,
        irreps_sh: Irreps,
        num_layers: int,
        num_rbf: int,
        radial_hidden: int,
    ) -> None:
        super().__init__()
        self.units: nn.ModuleList = nn.ModuleList(
            EquivariantMessageUnit(irreps, irreps_sh, num_rbf, radial_hidden)
            for _ in range(int(num_layers))
        )

    def forward(
        self,
        layer: int,
        v: torch.Tensor,
        state: typing.Any,
        pair_index: torch.Tensor,
        pair_sh: torch.Tensor,
        pair_rbf: torch.Tensor,
        pair_env: torch.Tensor,
        agg_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Run this layer's message unit over every fragment pair.

        *state* is unused (no routing); returns the ``(K, irreps.dim)``
        equivariant virtual-node update.
        """
        source, receiver = pair_index
        unit = typing.cast(EquivariantMessageUnit, self.units[layer])
        return unit(v[source], pair_sh, pair_rbf, pair_env, receiver, v, agg_scale=agg_scale)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@MODEL_REGISTRY.register("simurgh_simplified_monolithic")
@BACKBONE_REGISTRY.register("simurgh_simplified_monolithic")
class MonolithicSimurghSimplified(SimurghSimplifiedModel):
    """SIMURGH Simplified with a single monolithic stack in every line.

    ``E = Σ_i E0(Z_i) + scale · (E_atom + E_pair + E_frag)`` (``additive``)
    or the per-atom ``learned`` combination; ``F = -∂E/∂r``.

    Parameters
    ----------
    cutoff : float
        Atom-level neighbour cutoff (Å); must match ``data.cutoff``.
    embedding_dim, num_elements :
        Shared element-embedding width and table size.
    num_rbf, radial_hidden :
        Bessel basis size and radial-MLP width, used by every unit.
    sh_lmax : int
        Spherical-harmonic order of the shared input (>= 2).
    avg_num_neighbors : float, optional
        Atom/pair aggregation normaliser.  **Not auto-injected** by
        train.py for this model — set it in the config.
    combination : {"additive", "learned"}
        How the line energies are combined.
    atom_line, pair_line, fragment_line : dict, optional
        Per-line hyperparameters (``hidden_irreps``, ``num_layers``; the
        fragment line also ``vnode_cutoff``, ``fragment_radius``,
        ``avg_num_fragment_neighbors``, ``max_fragments``).
    readout : dict, optional
        ``n_scalar`` / ``hidden`` of every line readout.
    atomic_energies, scale, num_elements_table :
        Baseline and gain, three-mode contract as ``MonolithicArace``.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        embedding_dim: int = 32,
        num_elements: int = 120,
        num_rbf: int = 8,
        radial_hidden: int = 32,
        sh_lmax: int = 2,
        avg_num_neighbors: float | None = None,
        combination: str = "additive",
        atom_line: dict[str, typing.Any] | None = None,
        pair_line: dict[str, typing.Any] | None = None,
        fragment_line: dict[str, typing.Any] | None = None,
        readout: dict[str, typing.Any] | None = None,
        atomic_energies: dict[str, typing.Any] | None = None,
        scale: float | None = None,
        num_elements_table: int = 120,
    ) -> None:
        settings: SimplifiedSettings = SimplifiedSettings.build(
            cutoff=cutoff,
            embedding_dim=embedding_dim,
            num_elements=num_elements,
            num_rbf=num_rbf,
            radial_hidden=radial_hidden,
            sh_lmax=sh_lmax,
            avg_num_neighbors=avg_num_neighbors,
            combination=combination,
            atom_line=atom_line,
            pair_line=pair_line,
            fragment_line=fragment_line,
            readout=readout,
        )
        super().__init__(
            settings=settings,
            atom_processor=MonolithicAtomProcessor(
                settings.atom_irreps,
                settings.irreps_sh,
                settings.atom_layers,
                settings.num_rbf,
                settings.radial_hidden,
            ),
            pair_processor=MonolithicPairProcessor(
                settings.pair_irreps,
                settings.irreps_sh_even,
                settings.pair_layers,
                settings.num_rbf,
                settings.radial_hidden,
            ),
            fragment_processor=MonolithicFragmentProcessor(
                settings.fragment_irreps,
                settings.irreps_sh,
                settings.fragment_layers,
                settings.num_rbf,
                settings.radial_hidden,
            ),
            atomic_energies=atomic_energies,
            scale=scale,
            num_elements_table=num_elements_table,
        )
