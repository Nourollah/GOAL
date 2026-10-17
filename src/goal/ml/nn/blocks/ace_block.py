"""ACE message-passing block for the ARACE architecture.

In ARACE (ARtisan + Atomic Cluster Expansion) the potential artisans
are the primary computation at every layer: each round starts with an
:class:`~goal.ml.nn.blocks.artisans.ArtisanLayer` that builds per-edge
equivariant representations ``h_pair`` and a per-layer pair energy.
The :class:`AceBlock` then aggregates those *artisan edge features*
back onto the destination nodes::

    agg_i  = Σ_j h_pair_{j→i} / N̄
    h_new  = RMSNorm(EquivLinear(agg) + h)

Note this differs from MACE-style message passing: the aggregated
quantity is the artisan output (an edge feature), not a node→node
message built inside the block itself.

:class:`AraceRound` is the convenience wrapper combining one
``ArtisanLayer`` and one ``AceBlock`` into one full round::

    nodes → edges (artisan) → nodes (ACE)

Two **optional** add-ons hang off the round, each disabled by default and
each configured by its own config section
(``model.backbone.fragment_interaction`` /
``model.backbone.adaptive_gate``).  Both are equivariant at every angular
order:

    ┌──────────────────────────────────────────────────────────┐
    │ ARTISAN LAYER L → h_pair, E_L                            │
    └──────────────────────────────────────────────────────────┘
       ↓ h_pair                  ↓ h  (same pre-update features)
    ┌──────────────┐   ┌──────────────────────────────────────┐
    │  ACE BRANCH  │   │  FRAGMENT BRANCH (optional)          │
    │  h_local     │   │  Δh = Σ_l TP(f_l, Y(r̂_kl); w(d))    │
    │              │   │  full irreps — l=0, l=1, l=2         │
    └──────────────┘   └──────────────────────────────────────┘
       └────────────── MIXING ──────────────┘
                       ↓ h_new
    ┌──────────────────────────────────────────────────────────┐
    │  ADAPTIVE DEPTH GATE (optional)                          │
    │  h_next = g·h_new + (1-g)·h  (soft train / hard eval)   │
    │  g invariant, per irrep block                            │
    └──────────────────────────────────────────────────────────┘

With both sections absent, ``AraceRound.forward`` runs exactly the
computation it ran before the add-ons existed — same ops, same order,
same numbers — and neither module contributes a single parameter.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import Irreps

from goal.ml.nn.blocks.adaptive_gate import AdaptiveDepthGate
from goal.ml.nn.blocks.artisans import ArtisanLayer, _PairRMSNorm
from goal.ml.nn.blocks.fragment_interaction import (
    EquivariantFragmentInteraction,
    FragmentGeometry,
)
from goal.ml.nn.primitives.linear import EquivariantLinear


def normalise_addon_config(cfg: typing.Any) -> dict[str, typing.Any] | None:
    """Normalise an optional add-on sub-config to a plain dict or ``None``.

    Accepts ``None``, a plain dict, or an OmegaConf ``DictConfig`` (which
    is what arrives from Hydra).  An empty mapping is treated as ``None``:
    commenting out every key under ``fragment_ca:`` in a YAML file leaves
    an empty section behind, and the intent there is plainly "disabled",
    not "build it with defaults".

    Shared by both ARACE backbones so ``fragment_ca`` / ``adaptive_gate``
    behave identically in the modular and monolithic variants.
    """
    if cfg is None:
        return None
    normalised: dict[str, typing.Any] = {str(k): v for k, v in dict(cfg).items()}
    return normalised or None


class AceBlock(nn.Module):
    """ACE message passing block for ARACE.

    Aggregates equivariant edge features (``h_pair``) onto destination
    nodes and updates the node features with an equivariant linear, a
    residual connection and a smooth RMS norm.

    All maps are bias-free so zero inputs produce exactly zero updates
    (required by the artisan bank's static-shape zero-masking schedule).

    Parameters
    ----------
    irreps_node : Irreps or str
        Irreps of the node features entering and leaving the block.
        Must equal the irreps of ``h_pair``.
    avg_num_neighbors : float, optional
        Mean neighbour count used to normalise the edge → node
        aggregation.  ``None`` disables normalisation.
    """

    def __init__(
        self,
        irreps_node: Irreps | str,
        avg_num_neighbors: float | None = None,
    ) -> None:
        super().__init__()
        self.irreps_node: Irreps = Irreps(irreps_node)

        self.linear: EquivariantLinear = EquivariantLinear(
            self.irreps_node, self.irreps_node, biases=False
        )
        self.norm: _PairRMSNorm = _PairRMSNorm(self.irreps_node)

        norm_scale: float = (
            1.0 / float(avg_num_neighbors)
            if avg_num_neighbors is not None and float(avg_num_neighbors) > 0.0
            else 1.0
        )
        self.register_buffer(
            "agg_norm_scale",
            torch.tensor(norm_scale, dtype=torch.get_default_dtype()),
        )

    def set_avg_num_neighbors(self, avg_num_neighbors: float | None) -> None:
        """Re-set the aggregation normaliser in place (post-construction).

        Used by the training entry point when ``avg_num_neighbors`` is
        computed from the dataset after the model has been built.
        """
        norm_scale: float = (
            1.0 / float(avg_num_neighbors)
            if avg_num_neighbors is not None and float(avg_num_neighbors) > 0.0
            else 1.0
        )
        with torch.no_grad():
            self.agg_norm_scale.fill_(norm_scale)

    def forward(
        self,
        h: torch.Tensor,
        h_pair: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate artisan edge features and update node features.

        Parameters
        ----------
        h : Tensor ``(N, irreps_node.dim)``
            Current node features.
        h_pair : Tensor ``(E, irreps_node.dim)``
            Artisan edge features (may be ``(0, dim)`` for edge-free
            batches).
        edge_index : Tensor ``(2, E)``

        Returns
        -------
        Tensor
            Updated node features ``h_new`` of shape
            ``(N, irreps_node.dim)``.
        """
        col: torch.Tensor = edge_index[1]
        agg: torch.Tensor = torch.zeros_like(h)
        if h_pair.shape[0] > 0:
            agg = agg.index_add(0, col, h_pair.to(h.dtype))
        agg = agg * self.agg_norm_scale.to(h.dtype)
        return self.norm(self.linear(agg) + h)


class AraceRound(nn.Module):
    """One full ARACE round: artisan layer + ACE block.

    ``nodes → edges (artisan) → nodes (ACE)`` with the per-layer pair
    energy returned alongside the updated node features.

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model.
    irreps_node : Irreps or str
        Node feature irreps (must equal the artisans' ``hidden_irreps``).
    cutoff : float
        Cutoff radius (Angstrom).
    artisan_kwargs : dict, optional
        Forwarded to every :class:`_AracePairArtisan` in the layer.
    artisans : nn.ModuleDict, optional
        Pre-built artisan bank shared across rounds
        (``share_artisan_weights=True``).
    avg_num_neighbors : float, optional
        Edge → node aggregation normaliser for the ACE block.
    fragment_interaction_config : dict, optional
        ``None`` (default) → no fragment channel, zero overhead.  A dict
        builds :class:`EquivariantFragmentInteraction` with those
        hyperparameters (``irreps_hidden``, ``num_rbf``, ``radial_hidden``,
        ``num_layers``, ``init_zero``, ``max_fragments``, and ``cutoff``
        which defaults to the round's neighbour cutoff).
    adaptive_gate_config : dict, optional
        ``None`` (default) → no depth gate, zero overhead.  A dict builds
        :class:`AdaptiveDepthGate` with those hyperparameters
        (``n_scalar``, ``hard_threshold``, ``aux_loss_weight``,
        ``init_bias``, ``per_irrep``).
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
        rare_artisan: typing.Any = None,
    ) -> None:
        super().__init__()
        self.artisan_layer: ArtisanLayer = ArtisanLayer(
            elements=elements,
            irreps_node=irreps_node,
            cutoff=cutoff,
            artisan_kwargs=artisan_kwargs,
            artisans=artisans,
            pair_counts=pair_counts,
            rare_pair_enabled=rare_pair_enabled,
            min_pair_frequency=min_pair_frequency,
            rare_pair_embed_dim=rare_pair_embed_dim,
            rare_artisan=rare_artisan,
        )
        self.ace_block: AceBlock = AceBlock(
            irreps_node=irreps_node,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.irreps_node: Irreps = self.artisan_layer.irreps_node
        self.irreps_edge: Irreps = self.artisan_layer.irreps_edge

        # ----- Optional add-ons (None = strict no-op, no parameters) -----
        # The fragment cutoff defaults to the round's neighbour cutoff so
        # the model stays local unless the config says otherwise.
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

    @property
    def artisans(self) -> nn.ModuleDict:
        return self.artisan_layer.artisans

    def gates(self) -> dict[str, torch.Tensor]:
        """Snapshot of this round's artisan gates, keyed by pair label."""
        return self.artisan_layer.gates()

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
        h, atomic_numbers, edge_index, edge_sh, edge_lengths
            As before — see :class:`AceBlock` and
            :class:`~goal.ml.nn.blocks.artisans.ArtisanLayer`.
        fragment_geometry : FragmentGeometry, optional
            Fragment graph (centroids, pairs, spherical harmonics), built
            once per forward by the backbone and shared across rounds —
            it depends only on geometry, exactly like ``edge_sh``.
            Required only when the fragment channel is enabled.

        Returns
        -------
        tuple
            ``(h_new, e_atom, gate_scores, aux_loss)`` — updated node
            features ``(N, irreps_node.dim)``, this round's per-atom
            energy ``(N,)``, the per-atom gate values ``(N,)`` (``None``
            when the gate is disabled) and the round's auxiliary loss
            (0-d, exactly zero unless the gate is enabled and training).
        """
        h_pair, e_atom = self.artisan_layer(h, atomic_numbers, edge_index, edge_sh, edge_lengths)
        # ----- ACE branch (unchanged) -----
        h_local: torch.Tensor = self.ace_block(h, h_pair, edge_index)

        # ----- Fragment branch (optional, parallel to ACE) -----
        # Both branches consume the SAME pre-update ``h``, not h_local.
        if self.fragment_interaction is not None:
            if fragment_geometry is None:
                raise ValueError(
                    "The fragment interaction is enabled but no fragment "
                    "geometry was supplied.  Set data.compute_fragment_index: "
                    "true in the config (and rebuild the dataset cache) so the "
                    "graphs carry fragment labels."
                )
            delta_h: torch.Tensor = self.fragment_interaction(h, fragment_geometry)
            # Mixing: a plain equivariant sum.  ``delta_h`` spans the full
            # node irreps, so every angular channel gets the correction —
            # no scalar-only slice, and adding two tensors of the same
            # irreps is equivariant blockwise.
            h_new: torch.Tensor = h_local + delta_h
            # Re-normalise after mixing, reusing the ACE block's norm rather
            # than adding a second one: with its gains at their init value of
            # 1 this pass is idempotent, so an ``init_zero`` fragment module
            # starts out exactly equal to plain ARACE.
            h_new = self.ace_block.norm(h_new)
        else:
            h_new = h_local

        # ----- Adaptive depth gate (optional) -----
        gate_scores: torch.Tensor | None = None
        aux_loss: torch.Tensor = torch.zeros((), device=h.device, dtype=h.dtype)
        if self.adaptive_gate is not None:
            h_new, aux_loss, gate_scores = self.adaptive_gate(
                h_new=h_new,
                h_prev=h,
                training=self.training,
            )

        return h_new, e_atom, gate_scores, aux_loss
