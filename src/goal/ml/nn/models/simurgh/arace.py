"""SIMURGH ARACE model variant.

**ARACE = ARtisan + Atomic Cluster Expansion** — a monolithic SIMURGH
variant where potential **AR**\\ tisans and **ACE** message passing
alternate as equals, rather than the standard pipeline of ACE dressing
first and artisans as an auxiliary readout last.

The standard SIMURGH design runs ACE-style environment dressing first
and reads out pair energies through the artisan bank *last* (an
auxiliary readout).  This variant reverses the philosophy:

* **Potential artisans are the primary learning component at every
  stage** — each round starts with an equivariant element-pair artisan
  layer that builds per-edge representations and contributes a per-layer
  pair energy.
* **ACE message passing updates nodes between artisan stages** — the
  per-edge artisan features are aggregated back onto nodes so artisan
  layer ``L + 1`` sees richer context than layer ``L``.
* **Everything stays equivariant throughout**; the only transition to
  invariants is the scalar readout inside each artisan layer.

The core architectural rhythm is the node/edge alternation::

    nodes → edges (artisan) → nodes (ACE) → edges (artisan) → ...

One full round (stacked ``num_rounds`` times)::

    ┌───────────────────────────────────────────────────┐
    │ ARTISAN LAYER L                                   │
    │   h_AB   = L(h[row]) + L(h[col])      (symmetric) │
    │   h_pair = TP(h_AB, Y^l(r̂); w(d))    (E, irreps) │
    │   E_L    = MLP(scalars(h_pair))       → (E,)      │
    │   E_atom += index_add(row, 0.5 × E_L)             │
    └───────────────────────────────────────────────────┘
              ↓ h_pair (edge-level, equivariant)
    ┌───────────────────────────────────────────────────┐
    │ ACE MESSAGE PASSING BLOCK L                       │
    │   agg_i  = Σ_j h_pair_{j→i} / N̄                  │
    │   h_new  = RMSNorm(EquivLinear(agg) + h)          │
    └───────────────────────────────────────────────────┘
              ↓ repeat, then
    E_total = E_atomic + Σ_L E_artisan_L,  F = -∂E/∂r (autograd)

This module is completely isolated from the standard SIMURGH classes —
it reuses only the shared primitives and helpers, and registers the
monolithic model as ``"monolithic_arace"``.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, spherical_harmonics
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.blocks.ace_block import normalise_addon_config
from goal.ml.nn.blocks.adaptive_gate import AdaptiveDepthGate
from goal.ml.nn.blocks.artisans import (
    _AracePairArtisan,
    _AraceRarePairArtisan,
    _PairRMSNorm,
    cosine_cutoff,
    enumerate_element_pairs,
    log_pair_routing,
    pair_label,
    parse_pair_counts,
    resolve_n_elements,
    route_element_pairs,
)
from goal.ml.nn.blocks.embedding import AtomicNumberEmbedding
from goal.ml.nn.blocks.fragment_interaction import (
    EquivariantFragmentInteraction,
    FragmentGeometry,
)
from goal.ml.nn.blocks.symmetric_contraction import SymmetricContraction
from goal.ml.nn.models.simurgh.geometry import differentiable_edges
from goal.ml.nn.primitives.linear import EquivariantLinear
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY

# ``_AracePairArtisan`` historically lived in this module; it now sits in
# ``goal.ml.nn.blocks.artisans`` next to the other artisan implementations
# and is re-exported here for backward compatibility.
__all__ = [
    "_AracePairArtisan",
    "AraceBlock",
    "AraceBackbone",
    "MonolithicArace",
]


class AraceBlock(nn.Module):
    """One artisan layer + one ACE message-passing block.

    The artisan layer routes every edge to its element-pair artisan with
    the same static-shape zero-masking schedule as the standard SIMURGH
    bank (inputs masked before the forward, outputs masked after — every
    artisan runs every step, DDP-safe).  The masked per-edge features of
    all artisans are summed into a single ``h_pair`` tensor (each edge
    matches exactly one pair type, so the sum is a routed select).

    The ACE block aggregates ``h_pair`` onto the *destination* nodes
    (``agg_i = Σ_j h_pair_{j→i} / N̄``) and updates the node features
    with an equivariant linear, a residual connection and a smooth
    RMS norm.

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model (defines the pair routing).
    irreps_node : Irreps or str
        Irreps of the node features entering and leaving the block.
        Must equal the artisans' ``hidden_irreps`` so the edge → node
        aggregation and the residual connection are well-typed.
    cutoff : float
        Cutoff radius (Angstrom) for the radial basis and the cosine
        taper on pair energies.
    artisan_kwargs : dict
        Forwarded to every :class:`_AracePairArtisan`.
    artisans : nn.ModuleDict, optional
        Pre-built artisan bank to **share** across blocks
        (``share_artisan_weights=True``).  When ``None`` the block
        builds its own independent bank.  With routing on it must hold
        the *dedicated* pairs only.
    pair_counts : mapping, optional
        ``{(Z_A, Z_B): directed_edge_count}`` over the training set.
        Required for routing; ``None`` dedicates every pair.
    rare_pair_enabled : bool
        Turn frequency routing on.  Needs ``pair_counts``.
    min_pair_frequency : float
        Share-of-edges threshold for a dedicated artisan; see
        :func:`~goal.ml.nn.blocks.artisans.route_element_pairs`.
    rare_pair_embed_dim : int
        Element-embedding width inside the shared rare artisan.
    rare_artisan : _AraceRarePairArtisan, optional
        Pre-built rare artisan to **share** across blocks.
    avg_num_neighbors : float, optional
        Mean neighbour count used to normalise the edge → node
        aggregation.  ``None`` disables normalisation.
    fragment_interaction_config : dict, optional
        ``None`` (default) → no fragment channel, zero overhead.  A dict
        builds
        :class:`~goal.ml.nn.blocks.fragment_interaction.EquivariantFragmentInteraction`
        with those hyperparameters.
    adaptive_gate_config : dict, optional
        ``None`` (default) → no depth gate, zero overhead.  A dict builds
        :class:`~goal.ml.nn.blocks.adaptive_gate.AdaptiveDepthGate` with
        those hyperparameters.
    """

    def __init__(
        self,
        elements: typing.Sequence[int],
        irreps_node: Irreps | str,
        cutoff: float,
        artisan_kwargs: dict[str, typing.Any] | None = None,
        artisans: nn.ModuleDict | None = None,
        avg_num_neighbors: float | None = None,
        fragment_interaction_config: dict[str, typing.Any] | None = None,
        adaptive_gate_config: dict[str, typing.Any] | None = None,
        pair_counts: typing.Mapping[tuple[int, int], int] | None = None,
        rare_pair_enabled: bool = False,
        min_pair_frequency: float = 0.01,
        rare_pair_embed_dim: int = 16,
        rare_artisan: _AraceRarePairArtisan | None = None,
        correlation: int | None = None,
        num_elements: int = 120,
    ) -> None:
        super().__init__()
        self.irreps_node: Irreps = Irreps(irreps_node)
        self._cutoff: float = float(cutoff)
        all_pairs: tuple[tuple[int, int], ...] = tuple(enumerate_element_pairs(elements))
        self._pairs, self._rare_pairs = route_element_pairs(
            all_pairs,
            pair_counts if rare_pair_enabled else None,
            min_pair_frequency,
        )
        self._pair_keys: list[str] = [f"z{a}_z{b}" for a, b in self._pairs]
        self._rare_enabled: bool = bool(rare_pair_enabled)

        kwargs: dict[str, typing.Any] = dict(artisan_kwargs or {})
        if artisans is not None:
            self.artisans: nn.ModuleDict = artisans
        else:
            self.artisans = nn.ModuleDict(
                {
                    key: _AracePairArtisan(irreps_node=self.irreps_node, cutoff=cutoff, **kwargs)
                    for key in self._pair_keys
                }
            )

        # Built only when the threshold actually routes something.  The
        # declared ``elements`` list is the model's supported chemistry,
        # and a shared artisan with no pairs assigned to it would be
        # parameters that never see an edge — so a bank whose every pair
        # clears the threshold is exactly the bank it was before routing
        # existed.
        self.rare_artisan: _AraceRarePairArtisan | None = None
        if self._rare_enabled and self._rare_pairs:
            self.rare_artisan = rare_artisan or _AraceRarePairArtisan(
                irreps_node=self.irreps_node,
                cutoff=cutoff,
                pair_embed_dim=rare_pair_embed_dim,
                **kwargs,
            )

        reference: _AracePairArtisan | None = (
            typing.cast(_AracePairArtisan, self.artisans[self._pair_keys[0]])
            if self._pair_keys
            else self.rare_artisan
        )
        if reference is None:
            raise ValueError(
                "AraceBlock has neither a dedicated artisan nor a rare one. "
                "Either `elements` is empty, or min_pair_frequency routed every "
                "pair to a rare artisan that was not enabled."
            )
        if reference.irreps_hidden != self.irreps_node:
            raise ValueError(
                f"Artisan hidden_irreps ({reference.irreps_hidden}) must equal the "
                f"node irreps ({self.irreps_node}) so the ACE aggregation and "
                "residual are well-typed."
            )
        self.irreps_edge: Irreps = reference.irreps_edge

        # ── Many-body contraction (MACE's A → B step) ─────────────────
        # The edge sum below produces the one-particle basis A_i.  Left
        # alone, a round is a two-body message and a stack of rounds is a
        # plain equivariant MPNN.  A symmetric contraction raises the
        # body order *within* a round: B_i = Σ_n W_n[Z_i] · CG_n(A_i),
        # with per-element couplings, which is the ingredient MACE
        # credits for its rMD17 accuracy.
        #
        # ``None`` builds nothing and the round is bit-for-bit what it
        # was before this existed.  1/2/3 are the supported orders;
        # 3 is MACE's standard.
        #
        # SIZED BY THE DECLARED ELEMENTS, not by ``num_elements``.  The
        # coupling table is indexed by atomic number, so it only has to
        # span the elements the model declares — and the difference is
        # not marginal: on a three-element molecule a 120-row table is
        # 27 M parameters against 5.4 M, forty times more rows than can
        # ever be gathered.  ``num_elements`` still sizes the *embedding*
        # table, which is cheap.
        self._contraction_elements: int = max(int(z) for z in elements) + 1
        self.symmetric_contraction: SymmetricContraction | None = None
        if correlation is not None:
            self.symmetric_contraction = SymmetricContraction(
                irreps_in=self.irreps_node,
                irreps_out=self.irreps_node,
                correlation=int(correlation),
                num_elements=self._contraction_elements,
            )

        # ACE block: aggregate edges → nodes, then linear + residual + norm.
        self.ace_linear: EquivariantLinear = EquivariantLinear(
            self.irreps_node, self.irreps_node, biases=False
        )
        self.ace_norm: _PairRMSNorm = _PairRMSNorm(self.irreps_node)

        norm_scale: float = (
            1.0 / float(avg_num_neighbors)
            if avg_num_neighbors is not None and float(avg_num_neighbors) > 0.0
            else 1.0
        )
        self.register_buffer(
            "agg_norm_scale",
            torch.tensor(norm_scale, dtype=torch.get_default_dtype()),
        )

        # ----- Optional add-ons (None = strict no-op, no parameters) -----
        self.fragment_interaction: EquivariantFragmentInteraction | None = None
        if fragment_interaction_config is not None:
            frag_kwargs: dict[str, typing.Any] = {
                str(k): v for k, v in dict(fragment_interaction_config).items()
            }
            frag_kwargs.setdefault("cutoff", float(cutoff))
            self.fragment_interaction = EquivariantFragmentInteraction(
                irreps_node=self.irreps_node, **frag_kwargs
            )
        self.adaptive_gate: AdaptiveDepthGate | None = (
            AdaptiveDepthGate(
                irreps_node=self.irreps_node,
                **{str(k): v for k, v in dict(adaptive_gate_config).items()},
            )
            if adaptive_gate_config is not None
            else None
        )

    def _apply_addons(
        self,
        h: torch.Tensor,
        h_local: torch.Tensor,
        fragment_geometry: FragmentGeometry | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Mix in the fragment-CA correction and apply the depth gate.

        Shared by the normal and the edge-free forward paths so both
        behave identically.  Returns ``(h_new, gate_scores, aux_loss)``;
        with both add-ons disabled it returns ``h_local`` untouched, a
        ``None`` gate and a zero aux loss.
        """
        if self.fragment_interaction is not None:
            if fragment_geometry is None:
                raise ValueError(
                    "The fragment interaction is enabled but no fragment "
                    "geometry was supplied.  Set data.compute_fragment_index: "
                    "true in the config (and rebuild the dataset cache) so the "
                    "graphs carry fragment labels."
                )
            delta_h: torch.Tensor = self.fragment_interaction(h, fragment_geometry)
            # Plain equivariant sum: delta_h spans the full node irreps, so
            # every angular channel gets the correction.
            h_new: torch.Tensor = h_local + delta_h
            # Re-normalise after mixing, reusing the block's own norm: with
            # its gains at their init value of 1 this pass is idempotent, so
            # an ``init_zero`` fragment module starts out exactly equal to plain
            # ARACE.
            h_new = self.ace_norm(h_new)
        else:
            h_new = h_local

        gate_scores: torch.Tensor | None = None
        aux_loss: torch.Tensor = torch.zeros((), device=h.device, dtype=h.dtype)
        if self.adaptive_gate is not None:
            h_new, aux_loss, gate_scores = self.adaptive_gate(
                h_new=h_new,
                h_prev=h,
                training=self.training,
            )
        return h_new, gate_scores, aux_loss

    def _contract(
        self,
        agg: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """Raise the body order of the aggregated basis, when configured.

        Applied to ``agg`` — the one-particle basis A_i — and before the
        linear + residual, which is where MACE puts it.  A no-op that
        returns its input when ``correlation`` was not set.
        """
        if self.symmetric_contraction is None:
            return agg
        # Clamped for the same reason ``_AracePairArtisan`` clamps its
        # element conditioning: with rare-pair routing on, an atom whose
        # element is outside the declared list can reach this block, and
        # the coupling table has no row for it.  It borrows the last
        # row rather than raising from inside a tensor gather.
        return self.symmetric_contraction(
            agg, atomic_numbers.clamp(0, self._contraction_elements - 1)
        )

    def forward(
        self,
        h: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_lengths: torch.Tensor,
        fragment_geometry: FragmentGeometry | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Run one artisan + ACE round, plus any enabled add-on.

        Parameters
        ----------
        h : Tensor ``(N, irreps_node.dim)``
            Current equivariant node features.
        atomic_numbers : Tensor ``(N,)``
        edge_index : Tensor ``(2, E)``
        edge_sh : Tensor ``(E, irreps_edge.dim)``
            Pre-computed edge spherical harmonics (geometry only, shared
            across all blocks).
        edge_lengths : Tensor ``(E,)``
        fragment_geometry : FragmentGeometry, optional
            Fragment graph (centroids, pairs, spherical harmonics) — built
            once per forward and required only when the fragment channel
            is enabled.

        Returns
        -------
        tuple
            ``(h_new, e_atom, gate_scores, aux_loss)`` — updated node
            features ``(N, irreps_node.dim)``, this layer's per-atom
            energy ``(N,)``, per-atom gate values ``(N,)`` (``None`` when
            the gate is disabled) and this round's auxiliary loss (0-d,
            exactly zero unless the gate is enabled and training).
        """
        num_atoms: int = h.shape[0]
        device: torch.device = h.device
        dtype: torch.dtype = h.dtype
        row, col = edge_index
        E: int = int(edge_lengths.shape[0])

        e_atom: torch.Tensor = torch.zeros(num_atoms, device=device, dtype=dtype)

        # ----- Degenerate case: no edges — keep every artisan reachable.
        if E == 0:
            dummy_feats: torch.Tensor = torch.zeros(
                1, self.irreps_node.dim, device=device, dtype=dtype
            )
            dummy_sh: torch.Tensor = torch.zeros(
                1, self.irreps_edge.dim, device=device, dtype=dtype
            )
            dummy_dist: torch.Tensor = torch.zeros(1, device=device, dtype=dtype)
            dummy_z: torch.Tensor = torch.zeros(1, dtype=torch.long, device=device)
            dummy_sum: torch.Tensor = torch.zeros((), device=device, dtype=dtype)
            for key in self._pair_keys:
                artisan = typing.cast(_AracePairArtisan, self.artisans[key])
                h_p, e_p = artisan(dummy_feats, dummy_feats, dummy_sh, dummy_dist, dummy_z)
                dummy_sum = dummy_sum + 0.0 * (h_p.sum() + e_p.sum())
            if self.rare_artisan is not None:
                h_p, e_p = self.rare_artisan(
                    dummy_feats,
                    dummy_feats,
                    dummy_sh,
                    dummy_dist,
                    dummy_z,
                    torch.zeros(1, 2, dtype=torch.long, device=device),
                )
                dummy_sum = dummy_sum + 0.0 * (h_p.sum() + e_p.sum())
            agg: torch.Tensor = torch.zeros_like(h) + dummy_sum
            agg = self._contract(agg, atomic_numbers)
            h_local_empty: torch.Tensor = self.ace_norm(self.ace_linear(agg) + h)
            h_new, gate_scores, aux_loss = self._apply_addons(h, h_local_empty, fragment_geometry)
            return h_new, e_atom + dummy_sum, gate_scores, aux_loss

        edge_lengths = edge_lengths.to(dtype)
        cut_env: torch.Tensor = cosine_cutoff(edge_lengths, self._cutoff).to(dtype)

        z_row: torch.Tensor = atomic_numbers[row]
        z_col: torch.Tensor = atomic_numbers[col]
        z_lo: torch.Tensor = torch.minimum(z_row, z_col)
        z_hi: torch.Tensor = torch.maximum(z_row, z_col)

        # ----- Artisan layer: route every edge to its pair artisan.
        h_pair_total: torch.Tensor = torch.zeros(
            E, self.irreps_node.dim, device=device, dtype=dtype
        )
        claimed: torch.Tensor = torch.zeros(E, device=device, dtype=dtype)
        for (a, b), key in zip(self._pairs, self._pair_keys):
            artisan = typing.cast(_AracePairArtisan, self.artisans[key])
            mask: torch.Tensor = ((z_lo == a) & (z_hi == b)).to(dtype)
            claimed = claimed + mask
            mask_col: torch.Tensor = mask.unsqueeze(-1)
            feats_a: torch.Tensor = h[row] * mask_col
            feats_b: torch.Tensor = h[col] * mask_col
            sh_in: torch.Tensor = edge_sh.to(dtype) * mask_col
            dist_in: torch.Tensor = edge_lengths * mask
            h_p, e_p = artisan(feats_a, feats_b, sh_in, dist_in, z_col)
            h_pair_total = h_pair_total + h_p * mask_col
            tapered: torch.Tensor = e_p * mask * cut_env
            e_atom = e_atom.index_add(0, row, 0.5 * tapered)

        if self.rare_artisan is not None:
            # Every edge matches exactly one pair and the dedicated masks
            # are disjoint, so "claimed by nobody" IS the rare set — one
            # test instead of one per rare pair.  It also catches pairs
            # outside `elements` entirely, which previously contributed a
            # silent zero.
            rare_mask: torch.Tensor = (1.0 - claimed).clamp_min(0.0)
            rare_col: torch.Tensor = rare_mask.unsqueeze(-1)
            h_p, e_p = self.rare_artisan(
                h[row] * rare_col,
                h[col] * rare_col,
                edge_sh.to(dtype) * rare_col,
                edge_lengths * rare_mask,
                z_col,
                torch.stack([z_lo, z_hi], dim=-1),
            )
            h_pair_total = h_pair_total + h_p * rare_col
            e_atom = e_atom.index_add(0, row, 0.5 * (e_p * rare_mask * cut_env))

        # ----- ACE block: aggregate edge features onto destination nodes.
        agg = torch.zeros_like(h)
        agg = agg.index_add(0, col, h_pair_total)
        agg = agg * self.agg_norm_scale.to(dtype)
        agg = self._contract(agg, atomic_numbers)
        h_local: torch.Tensor = self.ace_norm(self.ace_linear(agg) + h)

        # ----- Optional add-ons (fragment branch, then depth gate).
        h_new, gate_scores, aux_loss = self._apply_addons(h, h_local, fragment_geometry)
        return h_new, e_atom, gate_scores, aux_loss


class AraceBackbone(nn.Module):
    """Stacks ``num_rounds`` :class:`AraceBlock` rounds.

    Node features are initialised from the atomic-number embedding table
    (no separate environment-dressing phase) and projected into the
    artisans' equivariant space.  Edge spherical harmonics are computed
    once and shared across all rounds.

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model.
    num_rounds : int
        Number of artisan + ACE rounds.
    share_artisan_weights : bool
        ``True`` → one artisan bank shared by every round;
        ``False`` (default) → an independent bank per round.
    artisan : dict, optional
        Artisan sub-config.  ``architecture`` must be ``"equivariant"``
        (the ARACE variant is equivariant throughout); the
        remaining keys are forwarded to
        :class:`_AracePairArtisan` (``hidden_irreps``,
        ``num_layers``, ``num_rbf``, ``radial_hidden``,
        ``n_scalar_out``, ``final_hidden``, ``element_conditioned``).
    cutoff : float
        Neighbour cutoff radius (Angstrom).
    embedding_dim : int
        Width of the initial scalar embedding.
    num_elements : int
        Size of the embedding table (must exceed the largest Z).
    avg_num_neighbors : float, optional
        Edge → node aggregation normaliser for every ACE block.
    fragment_interaction : dict, optional
        Equivariant fragment-channel sub-config, applied to every block.
        ``None`` (default) → disabled.
    adaptive_gate : dict, optional
        Adaptive depth-gate sub-config, applied to every block.
        ``None`` (default) → disabled.
    """

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
        fragment_interaction: dict[str, typing.Any] | None = None,
        adaptive_gate: dict[str, typing.Any] | None = None,
        correlation: int | None = None,
    ) -> None:
        super().__init__()
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {num_rounds}.")
        self._elements: tuple[int, ...] = tuple(sorted({int(z) for z in elements}))
        self._num_rounds: int = int(num_rounds)
        self._share_artisan_weights: bool = bool(share_artisan_weights)
        self._cutoff: float = float(cutoff)

        artisan_cfg: dict[str, typing.Any] = dict(artisan or {})
        if "n_elements" in artisan_cfg:
            artisan_cfg["n_elements"] = resolve_n_elements(
                artisan_cfg["n_elements"], self._elements
            )
        architecture: str = str(artisan_cfg.pop("architecture", "equivariant"))
        if architecture != "equivariant":
            raise ValueError(
                "The ARACE variant is equivariant throughout; "
                f"artisan.architecture must be 'equivariant', got '{architecture}'."
            )

        # Frequency routing.  Popped before the rest of the sub-config is
        # forwarded to `_AracePairArtisan`, which would reject the keys.
        # Same layout and names as the legacy backbone's `artisan_config`,
        # so one spelling covers both model families:
        #   artisan.rare_artisan.{enabled, min_pair_frequency, pair_embed_dim}
        #   artisan.pair_counts   (injected by train.py)
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
        self._artisan_kwargs: dict[str, typing.Any] = artisan_cfg
        self._rare_pair_enabled: bool = rare_pair_enabled

        hidden_irreps: Irreps = Irreps(artisan_cfg.get("hidden_irreps", "16x0e + 16x1o + 16x2e"))
        self._irreps_hidden: Irreps = hidden_irreps

        # Optional add-on sub-configs — normalised the same way as in the
        # modular backbone (empty section == disabled).
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

        # Routing decided once, here, so every round agrees on it (and so
        # a shared bank holds exactly the dedicated pairs).
        all_pairs: tuple[tuple[int, int], ...] = tuple(enumerate_element_pairs(self._elements))
        self._dedicated_pairs, self._rare_pairs = route_element_pairs(
            all_pairs, pair_counts if rare_pair_enabled else None, min_pair_frequency
        )
        log_pair_routing(
            "AraceBackbone",
            self._elements,
            all_pairs,
            self._dedicated_pairs,
            rare_pair_enabled,
            min_pair_frequency,
        )

        # Shared bank (or None → each block builds its own).
        shared_bank: nn.ModuleDict | None = None
        shared_rare: _AraceRarePairArtisan | None = None
        if share_artisan_weights:
            shared_bank = nn.ModuleDict(
                {
                    f"z{a}_z{b}": _AracePairArtisan(
                        irreps_node=hidden_irreps, cutoff=cutoff, **artisan_cfg
                    )
                    for a, b in self._dedicated_pairs
                }
            )
            if rare_pair_enabled and self._rare_pairs:
                shared_rare = _AraceRarePairArtisan(
                    irreps_node=hidden_irreps,
                    cutoff=cutoff,
                    pair_embed_dim=rare_pair_embed_dim,
                    **artisan_cfg,
                )

        self.blocks: nn.ModuleList = nn.ModuleList(
            AraceBlock(
                elements=self._elements,
                irreps_node=hidden_irreps,
                cutoff=cutoff,
                artisan_kwargs=artisan_cfg,
                artisans=shared_bank,
                avg_num_neighbors=avg_num_neighbors,
                fragment_interaction_config=self._fragment_cfg,
                adaptive_gate_config=self._adaptive_gate_cfg,
                pair_counts=pair_counts,
                rare_pair_enabled=rare_pair_enabled,
                min_pair_frequency=min_pair_frequency,
                rare_pair_embed_dim=rare_pair_embed_dim,
                rare_artisan=shared_rare,
                correlation=correlation,
                num_elements=num_elements,
            )
            for _ in range(self._num_rounds)
        )
        self._correlation: int | None = correlation
        if correlation is not None:
            print(
                f"[AraceBackbone] many-body contraction: correlation={correlation} "
                f"per round (MACE-style A → B)"
            )
        self.irreps_edge: Irreps = typing.cast(AraceBlock, self.blocks[0]).irreps_edge

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def elements(self) -> tuple[int, ...]:
        return self._elements

    @property
    def num_rounds(self) -> int:
        return self._num_rounds

    @property
    def share_artisan_weights(self) -> bool:
        return self._share_artisan_weights

    @property
    def irreps_out(self) -> Irreps:
        return self._irreps_hidden

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def fragment_interaction_enabled(self) -> bool:
        """Whether the equivariant fragment channel is active in every block."""
        return self._fragment_cfg is not None

    @property
    def adaptive_gate_enabled(self) -> bool:
        """Whether the adaptive depth gate is active in every block."""
        return self._adaptive_gate_cfg is not None

    @property
    def correlation(self) -> int | None:
        """Body order of the per-round contraction, or ``None`` when off."""
        return self._correlation

    @property
    def rare_pairs(self) -> tuple[tuple[int, int], ...]:
        """Pairs routed to the shared rare artisan (empty when routing is off)."""
        return self._rare_pairs

    @property
    def dedicated_pairs(self) -> tuple[tuple[int, int], ...]:
        """Pairs that own a dedicated artisan."""
        return self._dedicated_pairs

    def gates(self) -> dict[str, torch.Tensor]:
        """Snapshot of every artisan gate, keyed ``layer{L}/{pair}``.

        With routing on, the shared rare artisan reports under
        ``layer{L}/rare`` — one curve standing for every pair below the
        frequency threshold.
        """
        result: dict[str, torch.Tensor] = {}
        for layer_idx, block in enumerate(self.blocks):
            blk = typing.cast(AraceBlock, block)
            for (a, b), key in zip(blk._pairs, blk._pair_keys):
                artisan = typing.cast(_AracePairArtisan, blk.artisans[key])
                result[f"layer{layer_idx}/{pair_label(a, b)}"] = artisan.gate.detach()
            if blk.rare_artisan is not None:
                result[f"layer{layer_idx}/rare"] = blk.rare_artisan.gate.detach()
        return result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_lengths: torch.Tensor,
        fragment_geometry: FragmentGeometry | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor | None]]:
        """Run all rounds.

        Parameters
        ----------
        atomic_numbers, edge_index, edge_vectors, edge_lengths
            As before.
        fragment_geometry : FragmentGeometry, optional
            Fragment graph (centroids, pairs, spherical harmonics) — built
            once per forward and required only when the fragment channel
            is enabled.

        Returns
        -------
        tuple
            ``(node_energies, layer_energies, total_aux_loss,
            gate_scores_per_round)`` — the summed per-atom artisan energy
            ``(N,)``, the per-round per-atom contributions
            ``(num_rounds, N)``, the summed auxiliary loss over rounds
            (0-d, zero when the gate is disabled) and one gate-score
            tensor ``(N,)`` per round (``None`` entries when disabled).
        """
        h: torch.Tensor = self.input_linear(self.embedding(atomic_numbers))

        # Edge SH — geometry only, computed once and shared by every round.
        edge_sh: torch.Tensor = spherical_harmonics(
            self.irreps_edge,
            edge_vectors.to(h.dtype),
            normalize=True,
            normalization="component",
        )

        per_layer: list[torch.Tensor] = []
        gate_scores_per_round: list[torch.Tensor | None] = []
        total_aux_loss: torch.Tensor = torch.zeros((), device=h.device, dtype=h.dtype)
        for block in self.blocks:
            h, e_layer, gate_scores, aux_loss = block(
                h,
                atomic_numbers,
                edge_index,
                edge_sh,
                edge_lengths,
                fragment_geometry=fragment_geometry,
            )
            per_layer.append(e_layer)
            gate_scores_per_round.append(gate_scores)
            total_aux_loss = total_aux_loss + aux_loss

        layer_energies: torch.Tensor = torch.stack(per_layer, dim=0)  # (L, N)
        return (
            layer_energies.sum(dim=0),
            layer_energies,
            total_aux_loss,
            gate_scores_per_round,
        )


@MODEL_REGISTRY.register("monolithic_arace")
@BACKBONE_REGISTRY.register("monolithic_arace")
class MonolithicArace(nn.Module):
    """Self-contained ARACE SIMURGH model.

    Satisfies the ``MonolithicModel`` protocol — returns a property
    dict with ``"energy"``, ``"forces"``, ``"num_atoms"`` and
    ``"layer_energies"`` directly.  Configure with ``head: null``.

    ``E_total = E_atomic + scale × Σ_L E_artisan_L`` and
    ``F = -∂E_total/∂r`` via autograd.

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model (injected by ``train.py``
        from the training set).
    num_rounds : int
        Number of artisan + ACE rounds.
    share_artisan_weights : bool
        Share one artisan bank across all rounds (default ``False``).
    artisan : dict, optional
        Artisan sub-config — see :class:`AraceBackbone`.
    cutoff : float
        Neighbour cutoff radius (Angstrom); must match ``data.cutoff``.
    embedding_dim, num_elements :
        Initial embedding table configuration.
    avg_num_neighbors : float, optional
        ACE aggregation normaliser.
    atomic_energies : dict, optional
        Per-element baseline with the same three modes as the standard
        SIMURGH backbone: ``learned`` (default, ``nn.Parameter``),
        ``dataset`` / ``provided`` (fixed buffer seeded from
        ``values``).
    scale : float, optional
        Multiplicative gain on the artisan interaction energy.
    num_elements_table : int
        Size of the Z-indexed atomic-energy parameter/buffer.
    fragment_interaction : dict, optional
        Equivariant fragment-channel sub-config — see
        :class:`~goal.ml.nn.models.simurgh.backbone_arace.SimurghAraceBackbone`.
        ``None`` (default) → disabled.
    adaptive_gate : dict, optional
        Adaptive depth-gate sub-config.  ``None`` (default) → disabled.
        When enabled, ``forward`` returns an extra ``"aux_loss"`` entry
        that ``GOALModule`` adds to the total training loss.
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
        artisan_cfg: dict[str, typing.Any] | None = (
            {str(k): v for k, v in dict(artisan).items()} if artisan is not None else None
        )
        self.backbone: AraceBackbone = AraceBackbone(
            elements=elements,
            num_rounds=num_rounds,
            share_artisan_weights=share_artisan_weights,
            artisan=artisan_cfg,
            cutoff=cutoff,
            embedding_dim=embedding_dim,
            num_elements=num_elements,
            avg_num_neighbors=(
                float(avg_num_neighbors) if avg_num_neighbors is not None else None
            ),
            fragment_interaction=fragment_interaction,
            adaptive_gate=adaptive_gate,
        )

        # ----- Per-element atomic-energy baseline (3-mode contract) -----
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
        else:
            values: typing.Any = ae_cfg.get("values")
            if values is None:
                raise ValueError(
                    f"atomic_energies.mode={mode!r} requires 'values' to be a "
                    "{Z: e_Z} dict (train.py injects it for mode='dataset')."
                )
            buf: torch.Tensor = torch.zeros(num_elements_table, dtype=torch.get_default_dtype())
            for z_key, e_val in dict(values).items():
                buf[int(z_key)] = float(e_val)
            self.register_buffer("atomic_energies", buf)
        self._atomic_energies_mode: str = mode

        scale_val: float = float(scale) if scale is not None else 1.0
        self.register_buffer(
            "scale",
            torch.tensor(scale_val, dtype=torch.get_default_dtype()),
        )

    # ------------------------------------------------------------------
    # MonolithicModel protocol
    # ------------------------------------------------------------------

    @property
    def output_keys(self) -> list[str]:
        """Return list of output keys produced by the model."""
        return ["energy", "forces"]

    @property
    def elements(self) -> tuple[int, ...]:
        return self.backbone.elements

    @property
    def num_rounds(self) -> int:
        return self.backbone.num_rounds

    @property
    def share_artisan_weights(self) -> bool:
        return self.backbone.share_artisan_weights

    @property
    def atomic_energies_mode(self) -> str:
        return self._atomic_energies_mode

    def gates(self) -> dict[str, torch.Tensor]:
        """Return artisan gating weights keyed by layer and pair type."""
        return self.backbone.gates()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _fragment_geometry(
        self,
        graph: AtomicGraph,
        positions: torch.Tensor,
    ) -> FragmentGeometry | None:
        """Fragment centroids, pairs and SH for this batch, or ``None``.

        Built once and shared by every block — the decomposition depends
        only on geometry.  ``positions`` must be the tensor the energy is
        differentiated against, or the fragment channel contributes
        nothing to the forces.
        """
        if not self.backbone.fragment_interaction_enabled:
            return None
        fragment_index: torch.Tensor | None = graph.get("fragment_index", None)
        if fragment_index is None:
            raise ValueError(
                "model.backbone.fragment_interaction is enabled but the graphs "
                "carry no 'fragment_index'.  Set data.compute_fragment_index: "
                "true in the config (and rebuild any cached dataset) so the "
                "fragment labels are attached at load time."
            )
        module = typing.cast(AraceBlock, self.backbone.blocks[0]).fragment_interaction
        assert module is not None  # guarded by fragment_interaction_enabled
        return module.build_geometry(
            positions=positions,
            fragment_index=fragment_index,
            batch=graph.get("batch", None),
            cell=graph.get("cell", None),
            pbc=graph.get("pbc", None),
        )

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        """Forward pass: dict with 'energy', 'forces', 'num_atoms', 'layer_energies'.

        With the adaptive depth gate enabled the dict also carries
        ``"aux_loss"`` (0-d, grad-enabled — ``GOALModule`` adds it to the
        total loss) and ``"gate_scores"`` (detached ``(L, N)`` per-round
        gate values, logging only).  Both keys are absent when the gate is
        disabled, so the loss and metric paths see exactly the dict they
        saw before.
        """
        positions: torch.Tensor = graph.pos
        positions.requires_grad_(True)

        edge_vectors: torch.Tensor
        edge_lengths: torch.Tensor
        edge_vectors, edge_lengths = differentiable_edges(graph, positions)
        edge_index: torch.Tensor = typing.cast(torch.Tensor, graph.edge_index)

        node_energies: torch.Tensor
        layer_energies: torch.Tensor
        total_aux_loss: torch.Tensor
        gate_scores_per_round: list[torch.Tensor | None]
        node_energies, layer_energies, total_aux_loss, gate_scores_per_round = self.backbone(
            atomic_numbers=graph.atomic_numbers,
            edge_index=edge_index,
            edge_vectors=edge_vectors,
            edge_lengths=edge_lengths,
            fragment_geometry=self._fragment_geometry(graph, positions),
        )

        scale_v: torch.Tensor = self.scale.to(node_energies.dtype)
        baseline: torch.Tensor = self.atomic_energies[graph.atomic_numbers].to(node_energies.dtype)
        node_total: torch.Tensor = scale_v * node_energies + baseline  # (N,)

        batch: torch.Tensor = (
            graph.batch
            if graph.batch is not None
            else torch.zeros(graph.num_atoms, dtype=torch.long, device=node_total.device)
        )
        energy: torch.Tensor = scatter(node_total, batch, dim=0, reduce="sum")  # (B,)
        num_atoms: torch.Tensor = scatter(
            torch.ones_like(node_total), batch, dim=0, reduce="sum"
        )  # (B,)

        # Per-round energy totals (L, B) — diagnostic for layer
        # contributions; detached so it never interferes with training.
        layer_totals: torch.Tensor = scatter(
            scale_v * layer_energies, batch, dim=1, reduce="sum"
        ).detach()  # (L, B)

        grad_outputs: tuple[torch.Tensor, ...] = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=positions,
            create_graph=self.training,
            retain_graph=True,
        )
        forces: torch.Tensor = -grad_outputs[0]  # (N, 3)

        out: dict[str, torch.Tensor] = {
            "energy": energy,
            "forces": forces,
            "num_atoms": num_atoms,
            "layer_energies": layer_totals,
        }
        if self.backbone.adaptive_gate_enabled:
            out["aux_loss"] = total_aux_loss
            # (L, N) detached — mean per round is what gets logged.
            out["gate_scores"] = torch.stack(
                [
                    g.detach() if g is not None else torch.zeros_like(node_energies)
                    for g in gate_scores_per_round
                ],
                dim=0,
            )
        return out
