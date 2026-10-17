"""SIMURGH Simplified — Routed (the less simplified version).

Three parallel lines (atom, pair, fragment) over one shared input,
combined only at the readout — see
:mod:`~goal.ml.nn.models.simurgh.simurgh_simplified_layers` for the
skeleton and its design invariants.  This model and
``simurgh_simplified_monolithic`` run the identical skeleton and differ
**only** in the processors plugged into the three line slots:

* **atom line — frequency-routed artisans.**  Each frequent element owns
  a dedicated artisan (one node-wise unit + one message unit per layer);
  every other element shares one **rare artisan**.  Routing is by the
  element of the atom being updated, so it is static for a structure.
* **pair line — frequency-routed pair artisans.**  Each frequent
  unordered element pair ``(Z_lo, Z_hi)`` owns a dedicated pair artisan;
  every other pair shares one rare pair artisan.  The edge type is the
  element pair only — no geometric attribute — so routing never depends
  on positions.
* **fragment line — mixture of experts.**  A gate scores the experts for
  every virtual node from a ``0e`` **composition code** (mean element
  embedding ‖ ``log1p(n_atoms)``) and activates the top-k.  Expert outputs
  are equivariant and are combined with the scalar gate weights.

Routing decisions
-----------------
* **Training split only.**  Frequencies come from ``artisan.pair_counts``,
  which ``train.py`` counts over ``datamodule.data_train`` — and only
  there — when ``artisan.rare_artisan.enabled`` is true, then writes into
  the run's saved ``config.yaml``.  Validation and test data never touch
  routing (the model is evaluated on compositional OOD splits).
* **Element frequency** is derived *exactly* from the same counts: the
  number of directed edges whose receiver has element ``Z``,
  ``n_Z = c(Z,Z) + ½ Σ_{Y≠Z} c(Z,Y)`` — the share of messages ``Z``
  receives, which is what a message-passing artisan processes.
* Thresholds ``min_element_frequency`` / ``min_pair_frequency`` are
  fractions; a type at or above its threshold is dedicated.
* The type → artisan maps are **buffers** (``element_route``,
  ``pair_route``) and so travel in the checkpoint; a loaded state dict
  restores the trained routing.
* **Any unseen type routes to the rare artisan** and never raises —
  including atomic numbers beyond the table.  The rare artisans are
  therefore *always* built (unlike the existing ARACE bank, which builds
  its rare artisan only when the threshold routes something).  Their
  type code is zero-initialised, so an unseen type gets the plain shared
  rare artisan rather than a random embedding row.

Execution
---------
Routed units run on an **index gather** of their own members (approved
Phase 0 item 12), so the cost is proportional to the items routed, not
``num_units ×`` the batch as with zero-masking.  A unit with no members
in a batch still runs on an empty gather, which keeps every parameter in
the autograd graph (DDP static graph).  Experts run densely over the
(few) fragments and are weighted by the gate — non-selected experts get
weight exactly zero.

MoE continuity
--------------
The gate input is a function of element labels and the fragment
partition only, never of positions.  Hard top-k therefore cannot switch
along a trajectory, and the energy stays as smooth as the experts.  The
cost: the gate specialises by fragment chemistry, not conformation.
For ``top_k == 1`` the weight is the raw gate probability (Switch-style)
so the gate still receives gradient; for ``top_k >= 2`` the selected
probabilities are renormalised.  An optional Switch load-balancing loss
``n_e · Σ_e f_e P_e`` (weight ``fragment_experts.balance_loss_weight``,
default ``0``) is returned as ``aux_loss``, which ``GOALModule`` adds to
the training total.

Registered as ``"simurgh_simplified_routed"`` (auto-discovered;
``head: null``).  Config: ``src/goal/configs/ml/simurgh_simplified_routed_gmd26.yaml``.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import torch
import torch.nn as nn
from e3nn.o3 import Irreps

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.blocks.artisans import parse_pair_counts, route_element_pairs
from goal.ml.nn.models.simurgh.simurgh_simplified_layers import (
    AtomLineProcessor,
    EquivariantMessageUnit,
    FragmentLine,
    FragmentLineProcessor,
    NodewiseUnit,
    PairEdgeUnit,
    PairLine,
    PairLineProcessor,
    SharedInputs,
    SimplifiedSettings,
    SimurghSimplifiedModel,
)
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY

__all__ = [
    "FragmentMixtureOfExperts",
    "FrequencyRoutedAtomProcessor",
    "FrequencyRoutedPairProcessor",
    "RoutedSimurghSimplified",
    "element_receiver_counts",
]

_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Frequency statistics
# ---------------------------------------------------------------------------


def element_receiver_counts(
    pair_counts: typing.Mapping[tuple[int, int], int],
) -> dict[int, float]:
    """Directed edges received by each element, derived exactly from pair counts.

    ``pair_counts[(Z_lo, Z_hi)]`` counts *directed* edges, i.e. both
    directions of every bond.  A same-element pair puts an element ``Z``
    on the receiving end of both, a mixed pair on exactly one — hence
    ``n_Z = c(Z,Z) + ½ Σ_{Y≠Z} c(Z,Y)``.
    """
    counts: dict[int, float] = {}
    for (a, b), count in pair_counts.items():
        lo, hi = sorted((int(a), int(b)))
        if lo == hi:
            counts[lo] = counts.get(lo, 0.0) + float(count)
        else:
            counts[lo] = counts.get(lo, 0.0) + 0.5 * float(count)
            counts[hi] = counts.get(hi, 0.0) + 0.5 * float(count)
    return counts


def frequent_elements(
    element_counts: typing.Mapping[int, float], min_frequency: float
) -> tuple[int, ...]:
    """Elements whose share of the counts reaches *min_frequency*, ascending Z."""
    total: float = max(1.0, sum(element_counts.values()))
    return tuple(sorted(z for z, c in element_counts.items() if c / total >= float(min_frequency)))


def _route_lookup(table: torch.Tensor, keys: torch.Tensor, rare_index: int) -> torch.Tensor:
    """Unit index per key; anything off the table goes to *rare_index*."""
    size: int = table.shape[0]
    inside: torch.Tensor = keys < size
    if keys.dim() == 2:
        inside = inside.all(dim=-1)
    safe: torch.Tensor = keys.clamp(0, size - 1)
    routed: torch.Tensor = table[safe] if keys.dim() == 1 else table[safe[:, 0], safe[:, 1]]
    return torch.where(inside, routed, torch.full_like(routed, rare_index))


def _members(assignment: torch.Tensor, unit: int) -> torch.Tensor:
    return (assignment == unit).nonzero(as_tuple=False).squeeze(-1)


# ---------------------------------------------------------------------------
# Atom line — frequency-routed artisans
# ---------------------------------------------------------------------------


class FrequencyRoutedAtomProcessor(AtomLineProcessor):
    """Dedicated artisan per frequent element + one rare artisan (index ``D``).

    An artisan = one :class:`NodewiseUnit` (layer 1) and one
    :class:`EquivariantMessageUnit` per message-passing layer; the unit of
    the *receiving* atom's artisan processes all messages that atom gets.
    """

    element_route: torch.Tensor

    def __init__(
        self,
        irreps: Irreps,
        irreps_sh: Irreps,
        num_layers: int,
        num_rbf: int,
        radial_hidden: int,
        num_elements: int,
        dedicated_elements: typing.Sequence[int],
        type_embed_dim: int,
    ) -> None:
        super().__init__()
        self.dedicated_elements: tuple[int, ...] = tuple(int(z) for z in dedicated_elements)
        self.rare_index: int = len(self.dedicated_elements)
        table: torch.Tensor = torch.full((int(num_elements),), self.rare_index, dtype=torch.long)
        for index, z in enumerate(self.dedicated_elements):
            table[z] = index
        self.register_buffer("element_route", table)

        def artisan_units(build: typing.Callable[[int | None], nn.Module]) -> nn.ModuleList:
            return nn.ModuleList(
                [build(None) for _ in self.dedicated_elements] + [build(type_embed_dim)]
            )

        self.nodewise_units: nn.ModuleList = artisan_units(
            lambda dim: NodewiseUnit(irreps, num_elements, dim)
        )
        self.message_units: nn.ModuleList = nn.ModuleList(
            artisan_units(
                lambda dim: EquivariantMessageUnit(
                    irreps, irreps_sh, num_rbf, radial_hidden, num_elements, dim
                )
            )
            for _ in range(int(num_layers) - 1)
        )

    def route(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """``(N,)`` artisan index of every atom."""
        return _route_lookup(self.element_route, atomic_numbers, self.rare_index)

    def nodewise(self, h: torch.Tensor, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Run each artisan's node-wise unit on the atoms routed to it.

        Every unit runs on an index gather of its members (possibly empty)
        and the updates are scattered back into an ``(N, irreps.dim)``
        tensor.
        """
        assignment: torch.Tensor = self.route(atomic_numbers)
        out: torch.Tensor = torch.zeros_like(h)
        for index, unit in enumerate(self.nodewise_units):
            atoms: torch.Tensor = _members(assignment, index)
            out = out.index_add(0, atoms, unit(h[atoms], atomic_numbers[atoms]))
        return out

    def message(
        self,
        layer: int,
        h: torch.Tensor,
        atomic_numbers: torch.Tensor,
        inputs: SharedInputs,
        agg_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Run each artisan's message unit on the atoms routed to it.

        An edge belongs to the artisan of its *receiving* atom, so each unit
        sees its member atoms plus every message they receive, re-indexed
        locally; the rare unit is type-modulated by the receiver's element.
        Returns the ``(N, irreps.dim)`` update of every atom.
        """
        row, col = inputs.edge_index
        assignment: torch.Tensor = self.route(atomic_numbers)
        edge_assignment: torch.Tensor = assignment[col]
        local: torch.Tensor = torch.zeros_like(assignment)
        out: torch.Tensor = torch.zeros_like(h)
        for index, unit in enumerate(typing.cast(nn.ModuleList, self.message_units[layer])):
            atoms: torch.Tensor = _members(assignment, index)
            edges: torch.Tensor = _members(edge_assignment, index)
            local[atoms] = torch.arange(atoms.shape[0], device=h.device)
            receivers: torch.Tensor = col[edges]
            update: torch.Tensor = unit(
                h[row[edges]],
                inputs.edge_sh[edges],
                inputs.edge_rbf[edges],
                inputs.edge_env[edges],
                local[receivers],
                h[atoms],
                atomic_numbers[receivers],
                agg_scale,
            )
            out = out.index_add(0, atoms, update)
        return out


# ---------------------------------------------------------------------------
# Pair line — frequency-routed pair artisans
# ---------------------------------------------------------------------------


class FrequencyRoutedPairProcessor(PairLineProcessor):
    """Dedicated pair artisan per frequent unordered element pair + one rare (index ``D``)."""

    pair_route: torch.Tensor

    def __init__(
        self,
        irreps: Irreps,
        irreps_sh_even: Irreps,
        num_layers: int,
        num_rbf: int,
        radial_hidden: int,
        num_elements: int,
        dedicated_pairs: typing.Sequence[tuple[int, int]],
        type_embed_dim: int,
    ) -> None:
        super().__init__()
        self.dedicated_pairs: tuple[tuple[int, int], ...] = tuple(
            (int(a), int(b)) for a, b in dedicated_pairs
        )
        self.rare_index: int = len(self.dedicated_pairs)
        size: int = int(num_elements)
        table: torch.Tensor = torch.full((size, size), self.rare_index, dtype=torch.long)
        for index, (a, b) in enumerate(self.dedicated_pairs):
            table[a, b] = index
            table[b, a] = index
        self.register_buffer("pair_route", table)
        self.edge_units: nn.ModuleList = nn.ModuleList(
            nn.ModuleList(
                [
                    PairEdgeUnit(irreps, irreps_sh_even, num_rbf, radial_hidden, num_elements)
                    for _ in self.dedicated_pairs
                ]
                + [
                    PairEdgeUnit(
                        irreps,
                        irreps_sh_even,
                        num_rbf,
                        radial_hidden,
                        num_elements,
                        type_embed_dim,
                    )
                ]
            )
            for _ in range(int(num_layers))
        )

    def route(self, pair_types: torch.Tensor) -> torch.Tensor:
        """``(E,)`` pair-artisan index of every edge."""
        return _route_lookup(self.pair_route, pair_types, self.rare_index)

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
        """Run each pair artisan of this layer on the edges routed to it.

        Routing is by the unordered element pair only, so both directions
        of an edge go to the same unit and stay identical.  Returns the
        ``(E, irreps.dim)`` update of every edge.
        """
        row, col = edge_index
        assignment: torch.Tensor = self.route(pair_types)
        out: torch.Tensor = torch.zeros(row.shape[0], s.shape[-1], device=s.device, dtype=s.dtype)
        for index, unit in enumerate(typing.cast(nn.ModuleList, self.edge_units[layer])):
            edges: torch.Tensor = _members(assignment, index)
            update: torch.Tensor = unit(
                s[row[edges]],
                s[col[edges]],
                edge_sh_even[edges],
                edge_rbf[edges],
                edge_env[edges],
                pair_types[edges],
            )
            out = out.index_add(0, edges, update)
        return out


# ---------------------------------------------------------------------------
# Fragment line — mixture of experts
# ---------------------------------------------------------------------------


@dataclass
class MoEState:
    """Per-forward routing of the fragments, from the composition gate."""

    probabilities: torch.Tensor
    """(K, n_experts) softmax of the gate logits."""
    selected: torch.Tensor
    """(K, n_experts) 1.0 for the top-k experts of each fragment."""
    weights: torch.Tensor
    """(K, n_experts) scalar weight applied to each expert's update."""


class FragmentMixtureOfExperts(FragmentLineProcessor):
    """Composition-gated top-k mixture of equivariant virtual-node experts.

    One gate (shared by all layers, so a fragment keeps its experts
    through the line) and ``num_experts`` :class:`EquivariantMessageUnit`
    experts per layer.  ``u_k = Σ_e w_ke · expert_e(V)_k``.
    """

    def __init__(
        self,
        irreps: Irreps,
        irreps_sh: Irreps,
        num_layers: int,
        num_rbf: int,
        radial_hidden: int,
        composition_dim: int,
        num_experts: int,
        top_k: int,
        balance_loss_weight: float,
    ) -> None:
        super().__init__()
        if int(num_experts) < 1:
            raise ValueError(f"fragment_experts.num_experts must be >= 1, got {num_experts}.")
        if not 1 <= int(top_k) <= int(num_experts):
            raise ValueError(
                f"fragment_experts.top_k must be in [1, num_experts={num_experts}], got {top_k}."
            )
        self.num_experts: int = int(num_experts)
        self.top_k: int = int(top_k)
        self.balance_loss_weight: float = float(balance_loss_weight)
        self.gate: nn.Linear = nn.Linear(int(composition_dim), self.num_experts)
        self.experts: nn.ModuleList = nn.ModuleList(
            nn.ModuleList(
                EquivariantMessageUnit(irreps, irreps_sh, num_rbf, radial_hidden)
                for _ in range(self.num_experts)
            )
            for _ in range(int(num_layers))
        )

    def prepare(self, composition: torch.Tensor) -> MoEState:
        """Score the experts from the ``(K, composition_dim)`` codes and pick the top-k.

        For ``top_k == 1`` the weight is the raw gate probability; for
        ``top_k >= 2`` the selected probabilities are renormalised to sum
        to one.  Positions never enter, so the selection cannot switch
        along a trajectory.  Returns the :class:`MoEState` (all
        ``(K, num_experts)``).
        """
        probabilities: torch.Tensor = torch.softmax(self.gate(composition), dim=-1)
        top: torch.Tensor = probabilities.topk(self.top_k, dim=-1).indices
        selected: torch.Tensor = torch.zeros_like(probabilities).scatter(1, top, 1.0)
        weights: torch.Tensor = probabilities * selected
        if self.top_k > 1:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        return MoEState(probabilities=probabilities, selected=selected, weights=weights)

    def forward(
        self,
        layer: int,
        v: torch.Tensor,
        state: MoEState,
        pair_index: torch.Tensor,
        pair_sh: torch.Tensor,
        pair_rbf: torch.Tensor,
        pair_env: torch.Tensor,
        agg_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Gate-weighted sum of this layer's expert updates.

        Every expert runs densely over all fragments; non-selected experts
        get weight exactly zero.  The ``0e`` gate weights multiply
        equivariant updates, so the ``(K, irreps.dim)`` result stays
        equivariant.
        """
        source, receiver = pair_index
        out: torch.Tensor = torch.zeros_like(v)
        for index, expert in enumerate(typing.cast(nn.ModuleList, self.experts[layer])):
            update: torch.Tensor = expert(
                v[source], pair_sh, pair_rbf, pair_env, receiver, v, agg_scale=agg_scale
            )
            out = out + state.weights[:, index : index + 1].to(v.dtype) * update
        return out

    def aux_loss(self, state: MoEState) -> torch.Tensor | None:
        """Switch load-balancing loss, or ``None`` when its weight is zero."""
        if self.balance_loss_weight <= 0.0 or state.probabilities.shape[0] == 0:
            return None
        fraction: torch.Tensor = state.selected.mean(dim=0) / self.top_k  # f_e, sums to 1
        importance: torch.Tensor = state.probabilities.mean(dim=0)  # P_e
        return self.balance_loss_weight * self.num_experts * (fraction * importance).sum()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@MODEL_REGISTRY.register("simurgh_simplified_routed")
@BACKBONE_REGISTRY.register("simurgh_simplified_routed")
class RoutedSimurghSimplified(SimurghSimplifiedModel):
    """SIMURGH Simplified with frequency-routed artisans and a fragment MoE.

    Parameters
    ----------
    cutoff, embedding_dim, num_elements, num_rbf, radial_hidden, sh_lmax,
    avg_num_neighbors, combination, atom_line, pair_line, fragment_line,
    readout, atomic_energies, scale, num_elements_table :
        Exactly as :class:`~goal.ml.nn.models.simurgh.simurgh_simplified_monolithic.MonolithicSimurghSimplified`.
    artisan : dict
        ``rare_artisan: {enabled, min_element_frequency, min_pair_frequency,
        type_embed_dim}`` and ``pair_counts``.  ``enabled`` must be
        ``true``: it is the key ``train.py`` reads to count the training
        split's pairs and inject ``pair_counts``.  The section is spelled
        like the ARACE family's so that existing injection works unchanged.
    fragment_experts : dict, optional
        ``num_experts`` (4), ``top_k`` (1), ``balance_loss_weight`` (0.0).
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
        artisan: dict[str, typing.Any] | None = None,
        fragment_experts: dict[str, typing.Any] | None = None,
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

        # ----- Routing configuration and training-split statistics.
        artisan_cfg: dict[str, typing.Any] = {str(k): v for k, v in dict(artisan or {}).items()}
        unknown: set[str] = set(artisan_cfg) - {"rare_artisan", "pair_counts"}
        if unknown:
            raise ValueError(f"Unknown key(s) {sorted(unknown)} in artisan.")
        rare_cfg: dict[str, typing.Any] = {
            str(k): v for k, v in dict(artisan_cfg.get("rare_artisan") or {}).items()
        }
        unknown = set(rare_cfg) - {
            "enabled",
            "min_element_frequency",
            "min_pair_frequency",
            "type_embed_dim",
        }
        if unknown:
            raise ValueError(f"Unknown key(s) {sorted(unknown)} in artisan.rare_artisan.")
        if not bool(rare_cfg.get("enabled", False)):
            raise ValueError(
                "simurgh_simplified_routed needs artisan.rare_artisan.enabled: true.  The "
                "rare artisans always exist in this model (unseen types route to them); "
                "the flag is what makes train.py count the training split's element "
                "pairs and inject artisan.pair_counts."
            )
        pair_counts: dict[tuple[int, int], int] | None = parse_pair_counts(
            artisan_cfg.get("pair_counts")
        )
        if pair_counts is None:
            raise ValueError(
                "simurgh_simplified_routed needs artisan.pair_counts (training-split "
                "statistics).  train.py injects them; outside it, pass "
                "compute_pair_counts(train_graphs) explicitly."
            )
        min_element_frequency: float = float(rare_cfg.get("min_element_frequency", 0.05))
        min_pair_frequency: float = float(rare_cfg.get("min_pair_frequency", 0.02))
        type_embed_dim: int = int(rare_cfg.get("type_embed_dim", 16))
        if type_embed_dim < 1:
            raise ValueError(
                f"artisan.rare_artisan.type_embed_dim must be >= 1, got {type_embed_dim}."
            )

        canonical: dict[tuple[int, int], int] = {}
        for (a, b), count in pair_counts.items():
            key: tuple[int, int] = (min(a, b), max(a, b))
            canonical[key] = canonical.get(key, 0) + int(count)
        in_table: dict[tuple[int, int], int] = {
            p: c for p, c in canonical.items() if p[1] < settings.num_elements
        }
        element_counts: dict[int, float] = element_receiver_counts(in_table)
        dedicated_elements: tuple[int, ...] = frequent_elements(
            element_counts, min_element_frequency
        )
        dedicated_pairs, _ = route_element_pairs(sorted(in_table), in_table, min_pair_frequency)

        experts_cfg: dict[str, typing.Any] = {
            str(k): v for k, v in dict(fragment_experts or {}).items()
        }
        unknown = set(experts_cfg) - {"num_experts", "top_k", "balance_loss_weight"}
        if unknown:
            raise ValueError(f"Unknown key(s) {sorted(unknown)} in fragment_experts.")

        super().__init__(
            settings=settings,
            atom_processor=FrequencyRoutedAtomProcessor(
                settings.atom_irreps,
                settings.irreps_sh,
                settings.atom_layers,
                settings.num_rbf,
                settings.radial_hidden,
                settings.num_elements,
                dedicated_elements,
                type_embed_dim,
            ),
            pair_processor=FrequencyRoutedPairProcessor(
                settings.pair_irreps,
                settings.irreps_sh_even,
                settings.pair_layers,
                settings.num_rbf,
                settings.radial_hidden,
                settings.num_elements,
                dedicated_pairs,
                type_embed_dim,
            ),
            fragment_processor=FragmentMixtureOfExperts(
                settings.fragment_irreps,
                settings.irreps_sh,
                settings.fragment_layers,
                settings.num_rbf,
                settings.radial_hidden,
                settings.composition_dim,
                int(experts_cfg.get("num_experts", 4)),
                int(experts_cfg.get("top_k", 1)),
                float(experts_cfg.get("balance_loss_weight", 0.0)),
            ),
            atomic_energies=atomic_energies,
            scale=scale,
            num_elements_table=num_elements_table,
        )
        self.min_element_frequency: float = min_element_frequency
        self.min_pair_frequency: float = min_pair_frequency
        self._log_routing(element_counts, in_table)

    # ------------------------------------------------------------------
    # Routing introspection
    # ------------------------------------------------------------------

    @property
    def atom_processor(self) -> FrequencyRoutedAtomProcessor:
        return typing.cast(FrequencyRoutedAtomProcessor, self.atom_line.processor)

    @property
    def pair_processor(self) -> FrequencyRoutedPairProcessor:
        return typing.cast(FrequencyRoutedPairProcessor, self.pair_line.processor)

    @property
    def fragment_processor(self) -> FragmentMixtureOfExperts:
        return typing.cast(FragmentMixtureOfExperts, self.fragment_line.processor)

    def _log_routing(
        self,
        element_counts: typing.Mapping[int, float],
        pair_counts: typing.Mapping[tuple[int, int], int],
    ) -> None:
        from ase.data import chemical_symbols

        def sym(z: int) -> str:
            return chemical_symbols[z] if 0 <= z < len(chemical_symbols) else str(z)

        atoms = self.atom_processor
        pairs = self.pair_processor
        e_total: float = max(1.0, sum(element_counts.values()))
        p_total: float = float(max(1, sum(pair_counts.values())))
        rare_e: float = sum(
            c for z, c in element_counts.items() if z not in atoms.dedicated_elements
        )
        rare_p: float = sum(c for p, c in pair_counts.items() if p not in pairs.dedicated_pairs)
        print(
            f"[SimurghSimplifiedRouted] atom line: {len(atoms.dedicated_elements)} dedicated "
            f"artisans ({' '.join(sym(z) for z in atoms.dedicated_elements)}) + 1 rare "
            f"({rare_e / e_total:.1%} of training messages, min_element_frequency="
            f"{self.min_element_frequency:g})"
        )
        print(
            f"[SimurghSimplifiedRouted] pair line: {len(pairs.dedicated_pairs)} dedicated "
            f"pair artisans ({' '.join(f'{sym(a)}-{sym(b)}' for a, b in pairs.dedicated_pairs)})"
            f" + 1 rare ({rare_p / p_total:.1%} of training edges, min_pair_frequency="
            f"{self.min_pair_frequency:g})"
        )
        moe = self.fragment_processor
        print(
            f"[SimurghSimplifiedRouted] fragment line: {moe.num_experts} experts, "
            f"top_k={moe.top_k}, balance_loss_weight={moe.balance_loss_weight:g}"
        )

    @torch.no_grad()
    def utilisation(self, graph: AtomicGraph) -> dict[str, dict[str, int]]:
        """Items routed to each artisan / expert for *graph* (no forward pass).

        Returns ``{"atom": {...}, "pair": {...}, "fragment": {...}}`` with
        counts keyed by unit name (``"Z6"``, ``"Z1_Z6"``, ``"rare"``,
        ``"expert0"``...).  Fragment counts are top-k memberships, so they
        sum to ``top_k × K``.
        """
        z: torch.Tensor = graph.atomic_numbers
        atoms = self.atom_processor
        pairs = self.pair_processor
        moe = self.fragment_processor

        atom_assign: torch.Tensor = atoms.route(z)
        atom_names: list[str] = [f"Z{e}" for e in atoms.dedicated_elements] + ["rare"]
        edge_index: torch.Tensor = typing.cast(torch.Tensor, graph.edge_index)
        pair_assign: torch.Tensor = pairs.route(PairLine.pair_types(z, edge_index))
        pair_names: list[str] = [f"Z{a}_Z{b}" for a, b in pairs.dedicated_pairs] + ["rare"]

        fragment_index: torch.Tensor | None = graph.get("fragment_index", None)
        fragment_counts: dict[str, int] = {}
        if fragment_index is not None and fragment_index.numel():
            n_fragments: int = int(fragment_index.max().item()) + 1
            composition, _ = FragmentLine.composition(
                self.shared_input.embedding(z).to(self.scale.dtype),
                fragment_index.to(torch.long),
                n_fragments,
            )
            selected: torch.Tensor = moe.prepare(composition).selected
            fragment_counts = {
                f"expert{e}": int(selected[:, e].sum().item()) for e in range(moe.num_experts)
            }
        return {
            "atom": {n: int((atom_assign == i).sum().item()) for i, n in enumerate(atom_names)},
            "pair": {n: int((pair_assign == i).sum().item()) for i, n in enumerate(pair_names)},
            "fragment": fragment_counts,
        }
