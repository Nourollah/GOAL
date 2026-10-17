"""Modular ARACE SIMURGH backbone.

**ARACE = ARtisan + Atomic Cluster Expansion** — the default SIMURGH
architecture: potential artisans and ACE message passing alternate as
equals, instead of the legacy pipeline of ACE dressing first and
artisans as an auxiliary readout last (kept as
:class:`~goal.ml.nn.models.simurgh.backbone.SimurghBackbone`,
registered ``"simurgh"`` / ``"simurgh_ace_first"``).

One full round (stacked ``num_rounds`` times)::

    ┌───────────────────────────────────────────────────┐
    │ ARTISAN LAYER L                                   │
    │   h_AB   = L(h[row]) + L(h[col])      (symmetric) │
    │   h_pair = TP(h_AB, Y^l(r̂); w(d))    (E, irreps) │
    │   E_L    = MLP(scalars(h_pair))       → (E,)      │
    │   E_atom += index_add(row, 0.5 × E_L × cutoff)    │
    └───────────────────────────────────────────────────┘
              ↓ h_pair (edge-level, equivariant)
    ┌───────────────────────────────────────────────────┐
    │ ACE MESSAGE PASSING BLOCK L                       │
    │   agg_i  = Σ_j h_pair_{j→i} / N̄                  │
    │   h_new  = RMSNorm(EquivLinear(agg) + h)          │
    └───────────────────────────────────────────────────┘

    E_total = E_atomic + scale × Σ_L E_artisan_L

This class implements the **modular** ``EquivariantBackbone`` contract:
``forward(graph)`` returns :class:`~goal.ml.data.graph.NodeFeatures`
with ``node_energies`` already populated, so the standard task heads
(``energy``, ``energy_forces``, ``dual_forces``) consume it without
modification and derive forces via ``-∂E/∂r`` autograd through
``graph.pos``.  The monolithic counterpart lives in
:mod:`goal.ml.nn.models.simurgh.arace` (``"monolithic_arace"``).
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, spherical_harmonics
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.nn.blocks.ace_block import AraceRound, normalise_addon_config
from goal.ml.nn.blocks.artisans import (
    _AracePairArtisan,
    _AraceRarePairArtisan,
    enumerate_element_pairs,
    log_pair_routing,
    pair_label,
    parse_pair_counts,
    resolve_n_elements,
    route_element_pairs,
)
from goal.ml.nn.blocks.embedding import AtomicNumberEmbedding
from goal.ml.nn.blocks.fragment_interaction import FragmentGeometry
from goal.ml.nn.models.simurgh.geometry import differentiable_edges
from goal.ml.nn.primitives.linear import EquivariantLinear
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY


@MODEL_REGISTRY.register("simurgh_arace")
@BACKBONE_REGISTRY.register("simurgh_arace")
class SimurghAraceBackbone(nn.Module):
    """ARACE backbone for modular SIMURGH.

    Implements the backbone protocol expected by :class:`GOALModule`:

    * ``forward(graph)`` returns ``NodeFeatures`` whose
      ``node_energies`` field already holds the per-atom total
      ``scale × Σ_L E_artisan_L + e(Z)`` — the standard energy /
      forces heads pick it up unmodified;
    * exposes ``.elements``, ``.cutoff``, ``.irreps_out`` and
      ``.num_interactions`` properties;
    * exposes ``.avg_num_neighbors`` as a settable property backed by
      the per-round ``agg_norm_scale`` buffers (``train.py`` injects
      the dataset value);
    * exposes ``.atomic_energies`` and ``.scale`` buffers/params in the
      same 3-mode contract as the legacy backbone
      (``learned`` / ``dataset`` / ``provided``);
    * exposes ``.gates()`` → ``{"round{L}/{pair}": Tensor}`` and
      ``.last_round_energies`` (detached ``(L, B)`` per-round totals of
      the most recent forward) for ARACE-specific logging;
    * when the optional add-ons are enabled, exposes
      ``.last_aux_loss`` (0-d, grad-enabled — ``GOALModule`` adds it to
      the total loss) and ``.last_gate_scores`` (per-round detached
      ``(N,)`` gate values, logged as ``gate_mean_round_{L}``).

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model (injected by ``train.py``
        from the training set).
    num_rounds : int
        Number of artisan + ACE rounds.
    share_artisan_weights : bool
        ``True`` → one artisan bank shared by every round;
        ``False`` (default) → an independent bank per round.
    artisan : dict, optional
        Artisan sub-config.  ``architecture`` must be ``"equivariant"``
        (ARACE is equivariant throughout); the remaining keys are
        forwarded to :class:`_AracePairArtisan` (``hidden_irreps``,
        ``num_layers``, ``num_rbf``, ``radial_hidden``,
        ``n_scalar_out``, ``final_hidden``, ``element_conditioned``).
    cutoff : float
        Neighbour cutoff radius (Angstrom); must match ``data.cutoff``.
    embedding_dim : int
        Width of the initial scalar embedding.
    num_elements : int
        Size of the embedding table (must exceed the largest Z).
    avg_num_neighbors : float, optional
        Edge → node aggregation normaliser for every ACE block.
        ``None`` at construction can be filled in later via the
        ``avg_num_neighbors`` property setter.
    atomic_energies : dict, optional
        Per-element baseline with the three standard modes:
        ``learned`` (default, ``nn.Parameter``), ``dataset`` /
        ``provided`` (fixed buffer seeded from ``values``).
    scale : float, optional
        Multiplicative gain on the artisan interaction energy.
    num_elements_table : int
        Size of the Z-indexed atomic-energy parameter/buffer.
    fragment_interaction : dict, optional
        Fragment-channel sub-config.  ``None`` (default) → disabled, zero
        parameters and a forward path identical to plain ARACE.  A dict
        enables
        :class:`~goal.ml.nn.blocks.fragment_interaction.EquivariantFragmentInteraction`
        in every round with those hyperparameters (``irreps_hidden``,
        ``num_rbf``, ``radial_hidden``, ``num_layers``, ``init_zero``,
        ``max_fragments``, ``cutoff``).  Requires the dataset to supply
        ``fragment_index`` (``data.compute_fragment_index: true``).
    adaptive_gate : dict, optional
        Adaptive depth-gate sub-config.  ``None`` (default) → disabled.
        A dict enables
        :class:`~goal.ml.nn.blocks.adaptive_gate.AdaptiveDepthGate` in
        every round with those hyperparameters (``n_scalar``,
        ``hard_threshold``, ``aux_loss_weight``, ``init_bias``).  The
        summed sparsity penalty of all rounds is exposed as
        ``last_aux_loss`` and added to the training loss by
        ``GOALModule``.
    """

    atomic_energies: torch.Tensor
    scale: torch.Tensor

    def __init__(
        self,
        elements: typing.Sequence[int] = (1, 6, 7, 8),
        num_rounds: int = 2,
        share_artisan_weights: bool = False,
        artisan: dict[str, typing.Any] | None = None,
        cutoff: float = 5.0,
        embedding_dim: int = 32,
        num_elements: int = 120,
        avg_num_neighbors: float | None = None,
        atomic_energies: dict[str, typing.Any] | None = None,
        scale: float | None = None,
        num_elements_table: int = 120,
        fragment_interaction: dict[str, typing.Any] | None = None,
        adaptive_gate: dict[str, typing.Any] | None = None,
    ) -> None:
        super().__init__()
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {num_rounds}.")
        self._elements: tuple[int, ...] = tuple(sorted({int(z) for z in elements}))
        self._num_rounds: int = int(num_rounds)
        self._share_artisan_weights: bool = bool(share_artisan_weights)
        self._cutoff: float = float(cutoff)
        self._avg_num_neighbors: float | None = (
            float(avg_num_neighbors) if avg_num_neighbors is not None else None
        )

        # Normalise the artisan sub-config (may arrive as an OmegaConf
        # DictConfig) to a plain dict.
        artisan_cfg: dict[str, typing.Any] = {str(k): v for k, v in dict(artisan or {}).items()}
        # ``n_elements: auto`` → max(Z)+1 rows in the element-conditioning
        # tables; absent → the legacy 120 (see resolve_n_elements).
        if "n_elements" in artisan_cfg:
            artisan_cfg["n_elements"] = resolve_n_elements(
                artisan_cfg["n_elements"], self._elements
            )
        # Frequency routing — identical layout and semantics to
        # AraceBackbone; see goal.ml.nn.blocks.artisans.route_element_pairs.
        rare_cfg: dict[str, typing.Any] = dict(artisan_cfg.pop("rare_artisan", {}) or {})
        rare_pair_enabled: bool = bool(rare_cfg.get("enabled", False))
        min_pair_frequency: float = float(rare_cfg.get("min_pair_frequency", 0.01))
        rare_pair_embed_dim: int = int(rare_cfg.get("pair_embed_dim", 16))
        pair_counts: dict[tuple[int, int], int] | None = parse_pair_counts(
            artisan_cfg.pop("pair_counts", None)
        )
        if rare_pair_enabled and pair_counts is None:
            raise ValueError(
                "artisan.rare_artisan.enabled is true but no artisan.pair_counts "
                "were provided.  train.py computes them from the training set; "
                "outside it, pass them explicitly or disable routing — routing "
                "without counts would dedicate every pair and silently do nothing."
            )

        architecture: str = str(artisan_cfg.pop("architecture", "equivariant"))
        if architecture != "equivariant":
            raise ValueError(
                "The ARACE variant is equivariant throughout; "
                f"artisan.architecture must be 'equivariant', got '{architecture}'."
            )
        self._artisan_kwargs: dict[str, typing.Any] = artisan_cfg

        hidden_irreps: Irreps = Irreps(artisan_cfg.get("hidden_irreps", "16x0e + 16x1o + 16x2e"))
        self._irreps_hidden: Irreps = hidden_irreps

        # Optional add-on sub-configs.  Both may arrive as OmegaConf
        # ``DictConfig`` — normalise to plain dicts, and treat an *empty*
        # section the same as an absent one so commenting out every key
        # under ``fragment_interaction:`` disables the module rather than
        # silently building it with defaults.
        self._fragment_cfg: dict[str, typing.Any] | None = normalise_addon_config(
            fragment_interaction
        )
        self._adaptive_gate_cfg: dict[str, typing.Any] | None = normalise_addon_config(
            adaptive_gate
        )

        # Initial node features: embedding lookup → equivariant space.
        self.embedding: AtomicNumberEmbedding = AtomicNumberEmbedding(
            num_elements=num_elements,
            embedding_dim=embedding_dim,
        )
        self.input_linear: EquivariantLinear = EquivariantLinear(
            Irreps(f"{embedding_dim}x0e"), hidden_irreps
        )

        # Shared bank (or None → each round builds its own).
        all_pairs: tuple[tuple[int, int], ...] = tuple(enumerate_element_pairs(self._elements))
        self._dedicated_pairs, self._rare_pairs = route_element_pairs(
            all_pairs, pair_counts if rare_pair_enabled else None, min_pair_frequency
        )
        log_pair_routing(
            "SimurghAraceBackbone",
            self._elements,
            all_pairs,
            self._dedicated_pairs,
            rare_pair_enabled,
            min_pair_frequency,
        )

        shared_bank: nn.ModuleDict | None = None
        shared_rare: _AraceRarePairArtisan | None = None
        if self._share_artisan_weights:
            shared_bank = nn.ModuleDict(
                {
                    f"z{a}_z{b}": _AracePairArtisan(
                        irreps_node=hidden_irreps, cutoff=self._cutoff, **artisan_cfg
                    )
                    for a, b in self._dedicated_pairs
                }
            )
            if rare_pair_enabled and self._rare_pairs:
                shared_rare = _AraceRarePairArtisan(
                    irreps_node=hidden_irreps,
                    cutoff=self._cutoff,
                    pair_embed_dim=rare_pair_embed_dim,
                    **artisan_cfg,
                )

        self.rounds: nn.ModuleList = nn.ModuleList(
            AraceRound(
                elements=self._elements,
                irreps_node=hidden_irreps,
                cutoff=self._cutoff,
                artisan_kwargs=artisan_cfg,
                artisans=shared_bank,
                avg_num_neighbors=self._avg_num_neighbors,
                fragment_interaction_config=self._fragment_cfg,
                adaptive_gate_config=self._adaptive_gate_cfg,
                pair_counts=pair_counts,
                rare_pair_enabled=rare_pair_enabled,
                min_pair_frequency=min_pair_frequency,
                rare_pair_embed_dim=rare_pair_embed_dim,
                rare_artisan=shared_rare,
            )
            for _ in range(self._num_rounds)
        )
        self.irreps_edge: Irreps = typing.cast(AraceRound, self.rounds[0]).irreps_edge

        # ----- Per-element atomic-energy baseline (3-mode contract) -----
        self._atomic_energies_mode: str = self._init_atomic_energies(
            atomic_energies, num_elements_table
        )

        scale_val: float = float(scale) if scale is not None else 1.0
        self.register_buffer(
            "scale",
            torch.tensor(scale_val, dtype=torch.get_default_dtype()),
        )

        # Per-round per-graph energy totals of the most recent forward,
        # detached — consumed by GOALModule for "energy_round_{L}" logging.
        self.last_round_energies: torch.Tensor | None = None

        # Add-on side channels of the most recent forward.
        # ``last_aux_loss`` is grad-enabled (GOALModule adds it to the
        # training loss); ``last_gate_scores`` is detached (logging only).
        self.last_aux_loss: torch.Tensor | None = None
        self.last_gate_scores: list[torch.Tensor | None] = []

        n_pairs: int = len(self._dedicated_pairs) + (1 if self._rare_pairs else 0)
        addons: list[str] = []
        if self._fragment_cfg is not None:
            addons.append(f"fragment_interaction({self._fragment_cfg})")
        if self._adaptive_gate_cfg is not None:
            addons.append(f"adaptive_gate({self._adaptive_gate_cfg})")
        rank_zero_info(
            f"[SIMURGH/ARACE] backbone initialised\n"
            f"  elements       : {list(self._elements)}\n"
            f"  irreps_hidden  : {hidden_irreps}\n"
            f"  num_rounds     : {self._num_rounds}  "
            f"share_artisan_weights={self._share_artisan_weights}\n"
            f"  artisans       : {n_pairs} per round\n"
            f"  cutoff         : {self._cutoff}  "
            f"avg_num_neigh={self._avg_num_neighbors}\n"
            f"  atomic_E mode  : {self._atomic_energies_mode}  "
            f"scale={scale_val:.4g}\n"
            f"  add-ons        : {'  '.join(addons) if addons else 'none (plain ARACE)'}"
        )

    def _init_atomic_energies(
        self,
        atomic_energies: dict[str, typing.Any] | None,
        num_elements_table: int,
    ) -> str:
        """Register ``self.atomic_energies`` per the three-mode contract.

        Same contract as :class:`MonolithicArace` /
        :class:`SimurghBackbone`: ``learned`` → zero-init
        ``nn.Parameter``; ``dataset`` / ``provided`` → fixed buffer
        seeded from ``values`` (``train.py`` injects the LSQ values for
        ``dataset`` mode).
        """
        ae_cfg: dict[str, typing.Any] = dict(atomic_energies or {})
        mode: str = str(ae_cfg.get("mode", "learned"))
        if ae_cfg.get("compute_from_dataset", False) and mode == "learned":
            mode = "dataset"
        if mode not in ("learned", "dataset", "provided"):
            raise ValueError(
                f"atomic_energies.mode must be 'learned', 'dataset' or 'provided', got {mode!r}."
            )
        if mode == "learned":
            self.atomic_energies = nn.Parameter(
                torch.zeros(num_elements_table, dtype=torch.get_default_dtype())
            )
            return mode

        values: typing.Any = ae_cfg.get("values")
        if values is None:
            raise ValueError(
                f"atomic_energies.mode={mode!r} requires 'values' to be a "
                "{Z: e_Z} dict (train.py injects it for mode='dataset')."
            )
        buf: torch.Tensor = torch.zeros(num_elements_table, dtype=torch.get_default_dtype())
        for z_key, e_val in dict(values).items():
            z: int = int(z_key)
            if not 0 <= z < num_elements_table:
                raise ValueError(
                    f"atomic_energies.values: Z={z} is outside "
                    f"[0, {num_elements_table}).  Increase "
                    "num_elements_table or fix the key."
                )
            buf[z] = float(e_val)
        self.register_buffer("atomic_energies", buf)
        return mode

    # ------------------------------------------------------------------
    # EquivariantBackbone protocol / introspection
    # ------------------------------------------------------------------

    @property
    def irreps_out(self) -> Irreps:
        return self._irreps_hidden

    @property
    def num_interactions(self) -> int:
        """Number of message-passing rounds (backbone protocol)."""
        return self._num_rounds

    @property
    def elements(self) -> tuple[int, ...]:
        return self._elements

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def num_rounds(self) -> int:
        return self._num_rounds

    @property
    def share_artisan_weights(self) -> bool:
        return self._share_artisan_weights

    @property
    def atomic_energies_mode(self) -> str:
        return self._atomic_energies_mode

    @property
    def fragment_interaction_enabled(self) -> bool:
        """Whether the equivariant fragment channel is active in every round."""
        return self._fragment_cfg is not None

    @property
    def adaptive_gate_enabled(self) -> bool:
        """Whether the adaptive depth gate is active in every round."""
        return self._adaptive_gate_cfg is not None

    @property
    def avg_num_neighbors(self) -> float | None:
        """Mean neighbour count normalising the ACE aggregation."""
        return self._avg_num_neighbors

    @avg_num_neighbors.setter
    def avg_num_neighbors(self, value: float | None) -> None:
        """Update the aggregation normaliser on every round in place."""
        self._avg_num_neighbors = float(value) if value is not None else None
        for rnd in self.rounds:
            typing.cast(AraceRound, rnd).ace_block.set_avg_num_neighbors(self._avg_num_neighbors)

    @property
    def dedicated_pairs(self) -> tuple[tuple[int, int], ...]:
        """Pairs that own a dedicated artisan."""
        return self._dedicated_pairs

    @property
    def rare_pairs(self) -> tuple[tuple[int, int], ...]:
        """Pairs routed to the shared rare artisan (empty when routing is off)."""
        return self._rare_pairs

    def gates(self) -> dict[str, torch.Tensor]:
        """Snapshot of every artisan gate, keyed ``round{L}/{pair}``.

        With routing on, the shared rare artisan reports under
        ``round{L}/rare`` — one curve for every pair below the threshold.
        """
        result: dict[str, torch.Tensor] = {}
        for round_idx, rnd in enumerate(self.rounds):
            layer = typing.cast(AraceRound, rnd).artisan_layer
            for (a, b), key in zip(layer._pairs, layer._pair_keys):
                artisan = typing.cast(_AracePairArtisan, layer.artisans[key])
                result[f"round{round_idx}/{pair_label(a, b)}"] = artisan.gate.detach()
            if layer.rare_artisan is not None:
                result[f"round{round_idx}/rare"] = layer.rare_artisan.gate.detach()
        return result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _build_fragment_geometry(
        self,
        graph: AtomicGraph,
        positions: torch.Tensor,
    ) -> FragmentGeometry | None:
        """Fragment centroids, pairs and SH for this batch, or ``None``.

        Shared by every round: the fragment decomposition depends only on
        geometry, so recomputing it per round would be pure waste.  The
        pair cutoff and SH order come from the first round's module, so
        they follow the config without a second source of truth.
        """
        if not self.fragment_interaction_enabled:
            return None
        fragment_index: torch.Tensor | None = graph.get("fragment_index", None)
        if fragment_index is None:
            raise ValueError(
                "model.backbone.fragment_interaction is enabled but the graphs "
                "carry no 'fragment_index'.  Set data.compute_fragment_index: "
                "true in the config (and rebuild any cached dataset) so the "
                "fragment labels are attached at load time."
            )
        module = typing.cast(AraceRound, self.rounds[0]).fragment_interaction
        assert module is not None  # guarded by fragment_interaction_enabled
        return module.build_geometry(
            positions=positions,
            fragment_index=fragment_index,
            batch=graph.get("batch", None),
            cell=graph.get("cell", None),
            pbc=graph.get("pbc", None),
        )

    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        """Run every ARACE round and sum the per-round energies onto the atoms.

        Side channels for ``GOALModule`` are refreshed on every call:
        ``last_aux_loss`` (adaptive-gate loss, or ``None``),
        ``last_gate_scores`` (detached, one entry per round) and
        ``last_round_energies`` (detached ``(L, B)`` per-round totals).

        Parameters
        ----------
        graph : AtomicGraph
            Batched graph with ``pos`` (Å), ``atomic_numbers``,
            ``edge_index`` and, when the fragment channel is enabled,
            ``fragment_index``.  If ``graph.pos`` does not require grad it
            is replaced by a detached, grad-enabled copy.

        Returns
        -------
        NodeFeatures
            ``node_feats`` — ``(N, irreps_hidden.dim)`` final equivariant
            features; ``node_energies`` — ``(N,)`` invariant
            ``scale · Σ_rounds e_round + E0(Z)``.
        """
        # Re-derive edge geometry from positions so the autograd graph
        # links energy back to ``graph.pos`` — the energy/forces heads
        # call ``torch.autograd.grad(energy, graph.pos)`` themselves.
        # Same pattern (and rationale) as ``SimurghBackbone.forward``.
        positions: torch.Tensor = typing.cast(torch.Tensor, graph.pos)
        if not positions.requires_grad:
            positions = positions.detach().requires_grad_(True)
            graph.pos = positions
        edge_vectors, edge_lengths = differentiable_edges(graph, positions)
        edge_index: torch.Tensor = typing.cast(torch.Tensor, graph.edge_index)

        h: torch.Tensor = self.input_linear(self.embedding(graph.atomic_numbers))

        # Edge SH — geometry only, computed ONCE and shared by every round.
        edge_sh: torch.Tensor = spherical_harmonics(
            self.irreps_edge,
            edge_vectors.to(h.dtype),
            normalize=True,
            normalization="component",
        )

        # Fragment graph — geometry only, so like ``edge_sh`` it is built
        # ONCE and shared by every round.  Built from ``positions`` (not a
        # detached copy) so the fragment channel contributes to the forces.
        fragment_geometry: FragmentGeometry | None = self._build_fragment_geometry(
            graph, positions
        )

        per_round: list[torch.Tensor] = []
        gate_scores_per_round: list[torch.Tensor | None] = []
        total_aux_loss: torch.Tensor = torch.zeros((), device=h.device, dtype=h.dtype)
        for rnd in self.rounds:
            h, e_round, gate_scores, aux_loss = rnd(
                h,
                graph.atomic_numbers,
                edge_index,
                edge_sh,
                edge_lengths,
                fragment_geometry=fragment_geometry,
            )
            per_round.append(e_round)
            gate_scores_per_round.append(gate_scores)
            total_aux_loss = total_aux_loss + aux_loss

        # Side channels for GOALModule: the aux loss keeps its graph (it
        # joins the training loss), the gate scores are detached (logging).
        self.last_aux_loss = total_aux_loss if self.adaptive_gate_enabled else None
        self.last_gate_scores = [
            g.detach() if g is not None else None for g in gate_scores_per_round
        ]

        layer_energies: torch.Tensor = torch.stack(per_round, dim=0)  # (L, N)
        node_interaction: torch.Tensor = layer_energies.sum(dim=0)  # (N,)

        scale_v: torch.Tensor = self.scale.to(node_interaction.dtype)
        baseline: torch.Tensor = self.atomic_energies[graph.atomic_numbers].to(
            node_interaction.dtype
        )
        node_energies: torch.Tensor = scale_v * node_interaction + baseline  # (N,)

        # Per-round energy totals (L, B) — diagnostic snapshot for the
        # training loop's "energy_round_{L}" logging; detached so it
        # never interferes with training.
        batch: torch.Tensor = (
            graph.batch
            if graph.batch is not None
            else torch.zeros(graph.num_atoms, dtype=torch.long, device=node_energies.device)
        )
        self.last_round_energies = scatter(
            scale_v * layer_energies, batch, dim=1, reduce="sum"
        ).detach()  # (L, B)

        return NodeFeatures(
            node_feats=h,
            irreps=str(self._irreps_hidden),
            node_energies=node_energies,
        )
