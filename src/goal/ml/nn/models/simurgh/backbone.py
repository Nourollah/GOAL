"""Modular SIMURGH backbone.

SIMURGH — equivariant force field with an **element-pair bank of
potential artisans** (a routed mixture of pairwise energy modules).

The backbone produces ``NodeFeatures`` where ``node_energies`` is
already populated with the per-atom artisan-bank energy contributions.  Heads
that respect this (``energy``, ``energy_forces``, ``dual_forces``)
simply sum the populated channel; otherwise heads fall back to the
standard scalar-readout pathway.

This is the **legacy ACE-dressing-first** variant of SIMURGH
(registered ``"simurgh"`` and, for clarity, also
``"simurgh_ace_first"``).  The project **default** is now the ARACE
architecture in ``goal.ml.nn.models.simurgh.backbone_arace``
(``"simurgh_arace"``), where artisans are the primary computation at
every layer.  A fully self-contained monolithic counterpart of this
class lives in ``goal.ml.nn.models.simurgh.monolithic``.
"""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.nn.blocks.artisans import (
    ArtisanConfig,
    SimurghArtisanBank,
    parse_pair_counts,
    resolve_n_elements,
)
from goal.ml.nn.blocks.env_dressing import EnvironmentDressing
from goal.ml.nn.models.simurgh.geometry import differentiable_edges
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY


@MODEL_REGISTRY.register("simurgh")
@BACKBONE_REGISTRY.register("simurgh")
@MODEL_REGISTRY.register("simurgh_ace_first")
@BACKBONE_REGISTRY.register("simurgh_ace_first")
class SimurghBackbone(nn.Module):
    """SIMURGH backbone — environment dressing + element-pair artisan bank.

    Two-stage architecture:

    1. **Environment dressing** — one or more rounds of ACE-style
       equivariant message passing produce the one-particle basis
       ``A_i``.  Optional body-order expansion (``B²``, ``B³``)
       captures bond angles / dihedrals.  Output irreps for body
       orders ≥ 2 are derived programmatically from the CG
       decomposition (see :func:`cg_product_irreps`) and compressed
       back to ``irreps_hidden`` so the expert interface is
       unchanged.
    2. **Element-pair experts** — one expert per unordered element
       pair contributes a scalar pair energy with a learnable gate
       ``P_{AB}`` and a cosine-cutoff envelope.  Static-shape
       zero-masking keeps the autograd graph identical across ranks
       (DDP-safe).

    The block emits ``NodeFeatures`` whose ``node_energies`` field
    already holds the per-atom artisan-bank energy contributions, so downstream
    heads can either pick them up directly (``EnergyHead`` /
    ``EnergyForcesHead`` / ``DualForcesHead`` recognise the field) or
    apply their own readout to the dressed features.

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model.  Determines the number
        and identity of expert modules.
    dressing_kwargs : dict
        Forwarded to :class:`EnvironmentDressing` — see its docstring.
        Includes ``body_order`` for the ACE body-order expansion.
    artisan_config : dict
        Forwarded to :class:`ArtisanConfig` — see its docstring.
    cutoff : float
        Cosine-cutoff radius for the pairwise experts (Angstrom).
        Defaults to the dressing cutoff.
    atomic_energies : dict, optional
        Per-element baseline-energy specification with three modes
        (chosen by the ``mode`` field):

        * ``mode="learned"`` (default) — register
          ``self.atomic_energies`` as an ``nn.Parameter`` of length
          ``num_elements_table``, zero-initialised, learned jointly
          with the rest of the model.  The fallback when no prior is
          provided.
        * ``mode="dataset"`` — register ``self.atomic_energies`` as a
          buffer (non-learnable) seeded from ``values``, where
          ``values`` is the output of
          :func:`goal.ml.data.statistics.compute_atomic_references`
          on the training set (the training entry point computes it
          and injects it here).
        * ``mode="provided"`` — register ``self.atomic_energies`` as
          a buffer seeded from ``values``, a user-provided
          ``{Z: e_Z}`` dict (e.g. isolated-atom DFT energies).

        Keys of the sub-config: ``mode`` (str), ``values`` (dict or
        null), ``compute_from_dataset`` (bool, redundant convenience
        that mirrors ``mode == "dataset"``).
    scale : float, optional
        Multiplicative gain applied to the artisan bank interaction energy
        before it is added to the atomic baseline.  Set from
        :func:`goal.ml.data.statistics.compute_energy_scale` on the
        training set.  Defaults to ``1.0`` (no gain).
    num_elements_table : int
        Size of the Z-indexed buffer / parameter.  ``120`` covers the
        whole periodic table.
    """

    # Type annotations for the parameter / buffer registered in
    # __init__.  ``nn.Parameter`` is a subclass of ``torch.Tensor``,
    # so a single ``Tensor`` annotation covers both the "learned"
    # (Parameter) and "dataset"/"provided" (buffer) modes.
    atomic_energies: torch.Tensor
    scale: torch.Tensor

    _ATOMIC_ENERGIES_MODES: typing.ClassVar[tuple[str, ...]] = (
        "learned",
        "dataset",
        "provided",
    )

    def __init__(
        self,
        elements: typing.Sequence[int] = (1, 6, 7, 8),
        dressing_kwargs: dict[str, typing.Any] | None = None,
        artisan_config: dict[str, typing.Any] | None = None,
        cutoff: float | None = None,
        atomic_energies: dict[str, typing.Any] | None = None,
        scale: float | None = None,
        num_elements_table: int = 120,
        compute_pairwise_forces: bool = False,
    ) -> None:
        super().__init__()
        dressing_cfg: dict[str, typing.Any] = dict(dressing_kwargs or {})
        artisan_cfg: dict[str, typing.Any] = dict(artisan_config or {})

        self.dressing: EnvironmentDressing = EnvironmentDressing(**dressing_cfg)
        bank_cutoff: float = cutoff if cutoff is not None else self.dressing.cutoff

        # Extract nested rare-pair config (CHANGE 4).
        # Config layout: artisan_config.rare_artisan.{enabled, pair_embed_dim,
        # min_pair_frequency}.  Pop it before building ArtisanConfig so the
        # dataclass constructor doesn't see the unknown key.
        ge_cfg: dict[str, typing.Any] = dict(artisan_cfg.pop("rare_artisan", {}) or {})
        rare_pair_enabled: bool = bool(ge_cfg.get("enabled", False))
        min_pair_frequency: float = float(ge_cfg.get("min_pair_frequency", 0.01))
        rare_pair_embed_dim: int = int(ge_cfg.get("pair_embed_dim", 16))
        artisan_cfg["rare_pair_embed_dim"] = rare_pair_embed_dim

        # Normalise tuple-typed config entries that may arrive as ListConfig
        if "hidden_dims" in artisan_cfg:
            artisan_cfg["hidden_dims"] = tuple(int(x) for x in artisan_cfg["hidden_dims"])

        # Normalise the nested equivariant sub-config (may arrive as an
        # OmegaConf DictConfig) to a plain dict for ArtisanConfig.
        if artisan_cfg.get("equivariant") is not None:
            artisan_cfg["equivariant"] = {
                str(k): v for k, v in dict(artisan_cfg["equivariant"]).items()
            }
            if "n_elements" in artisan_cfg["equivariant"]:
                artisan_cfg["equivariant"]["n_elements"] = resolve_n_elements(
                    artisan_cfg["equivariant"]["n_elements"], elements
                )

        # pair_counts injected by train.py when rare_pair_enabled is True
        pair_counts: dict[tuple[int, int], int] | None = parse_pair_counts(
            artisan_cfg.pop("pair_counts", None)
        )

        self.artisan_bank: SimurghArtisanBank = SimurghArtisanBank(
            elements=elements,
            irreps_in=self.dressing.irreps_out,
            artisan_config=ArtisanConfig(**artisan_cfg),
            cutoff=bank_cutoff,
            pair_counts=pair_counts,
            rare_pair_enabled=rare_pair_enabled,
            min_pair_frequency=min_pair_frequency,
        )

        self._elements: tuple[int, ...] = self.artisan_bank.elements
        self._num_interactions: int = len(self.dressing.interactions)
        self._irreps_out: Irreps = self.dressing.irreps_out

        # Whether to route the artisan bank through ``forward_pairwise`` and
        # populate ``NodeFeatures.node_forces`` (consumed by force
        # heads in pairwise mode — see ``DualForcesHead``).
        self._compute_pairwise_forces: bool = bool(compute_pairwise_forces)

        # ----- Per-element atomic-energy baseline -----
        self._atomic_energies_mode: str = self._init_atomic_energies(
            atomic_energies, num_elements_table, list(elements)
        )

        # ----- ScaleShift on the interaction-energy output -----
        #
        # Pure multiplicative gain (the shift is folded into the
        # atomic-energy baseline above, matching MACE's
        # ``ScaleShiftBlock`` with ``shift=0``).  Registered as a
        # buffer so the optimiser never touches it.
        scale_val: float = float(scale) if scale is not None else 1.0
        self.register_buffer(
            "scale",
            torch.tensor(scale_val, dtype=torch.get_default_dtype()),
        )

        # ---- Startup summary ----
        d = self.dressing
        rank_zero_info(
            f"[SIMURGH] backbone initialised\n"
            f"  elements      : {list(self._elements)}\n"
            f"  irreps_hidden : {d.irreps_out}  (lmax={d.lmax}, "
            f"channels={d.hidden_channels})\n"
            f"  interactions  : {self._num_interactions}  "
            f"element_conditioned={d.element_conditioned}  "
            f"per_layer_readout={d.per_layer_readout}\n"
            f"  body_order    : {d.body_order}  "
            f"sym_contraction={d.symmetric_contraction}\n"
            f"  avg_num_neigh : {d._avg_num_neighbors}  "
            f"norm_exponent={d._agg_norm_exponent}\n"
            f"  atomic_E mode : {self._atomic_energies_mode}  "
            f"scale={scale_val:.4g}\n"
            f"  experts       : {self.artisan_bank.num_artisans} modules  "
            f"(dedicated={len(self.artisan_bank._dedicated_pairs)}, "
            f"rare={len(self.artisan_bank._rare_pairs)})"
        )

    def _init_atomic_energies(
        self,
        atomic_energies: dict[str, typing.Any] | None,
        num_elements_table: int,
        dataset_elements: list[int],
    ) -> str:
        """Register ``self.atomic_energies`` per the three-mode contract.

        Parameters
        ----------
        atomic_energies :
            Sub-config dict with keys ``mode``, ``values``,
            ``compute_from_dataset``.
        num_elements_table :
            Size of the Z-indexed buffer/parameter (default 120).
        dataset_elements :
            Sorted list of unique atomic numbers found in the training set.
            Used to validate coverage in ``provided`` mode.

        Returns
        -------
        str
            The resolved mode string for introspection.
        """
        from goal.ml.data.statistics import ATOMIC_SYMBOLS

        cfg: dict[str, typing.Any] = dict(atomic_energies or {})
        mode: str = str(cfg.get("mode", "learned"))
        values: typing.Any = cfg.get("values", None)
        # ``compute_from_dataset`` is a user-facing convenience; if
        # ``mode`` is left default it forces "dataset" semantics.
        if cfg.get("compute_from_dataset", False) and mode == "learned":
            mode = "dataset"

        if mode not in self._ATOMIC_ENERGIES_MODES:
            raise ValueError(
                f"atomic_energies.mode must be one of {self._ATOMIC_ENERGIES_MODES}, got {mode!r}."
            )

        if mode == "learned":
            # Learnable per-element offset, zero-init.  Gradient flows
            # through ``self.atomic_energies[atomic_numbers]`` exactly
            # like any other parameter indexed by an integer tensor.
            self.atomic_energies = nn.Parameter(
                torch.zeros(num_elements_table, dtype=torch.get_default_dtype())
            )
            return mode

        # ``dataset`` and ``provided`` modes are identical at runtime:
        # both register a Z-indexed buffer.  The provenance differs
        # (LSQ vs hand-set) but the model interface is the same.
        if values is None:
            raise ValueError(
                f"atomic_energies.mode={mode!r} requires 'values' to be "
                "set to a {{Z: e_Z}} dict; got None.  For mode='dataset' "
                "the training entry point populates this from "
                "compute_atomic_references()."
            )

        # Validate coverage: every element in the dataset must have a value.
        if mode == "provided" and dataset_elements:
            provided_z: set[int] = {int(k) for k in dict(values).keys()}
            missing: list[int] = [z for z in dataset_elements if z not in provided_z]
            if missing:
                missing_names: list[str] = [f"{ATOMIC_SYMBOLS.get(z, '?')}({z})" for z in missing]
                raise ValueError(
                    f"atomic_energies.mode is 'provided' but the following "
                    f"elements found in the dataset have no reference energy: "
                    f"{missing_names}.\n"
                    f"Fix one of:\n"
                    f"  1. Add the missing values under atomic_energies.values "
                    f"in your config.\n"
                    f"  2. Change atomic_energies.mode to 'dataset' — reference "
                    f"energies are computed automatically from the training set.\n"
                    f"  3. Change atomic_energies.mode to 'learned' — reference "
                    f"energies are jointly optimised from zero (slowest)."
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
    # EquivariantBackbone protocol
    # ------------------------------------------------------------------

    @property
    def irreps_out(self) -> Irreps:
        return self._irreps_out

    @property
    def num_interactions(self) -> int:
        return self._num_interactions

    @property
    def elements(self) -> tuple[int, ...]:
        return self._elements

    @property
    def num_artisans(self) -> int:
        return self.artisan_bank.num_artisans

    @property
    def body_order(self) -> int:
        """ACE body-order parameter inherited from the dressing block."""
        return self.dressing.body_order

    @property
    def atomic_energies_mode(self) -> str:
        """Resolved atomic-energies mode (``learned``, ``dataset`` or ``provided``)."""
        return self._atomic_energies_mode

    def gates(self) -> dict[str, torch.Tensor]:
        """Snapshot of every expert's gate parameter."""
        return self.artisan_bank.gates()

    def compute_artisan_loads(
        self,
        graph: AtomicGraph,
    ) -> dict[str, torch.Tensor]:
        """Delegate to :meth:`SimurghArtisanBank.compute_artisan_loads` for the given batch.

        Recomputes edge geometry (without requiring grad) so this can be
        called standalone after the main forward without touching the
        training graph.
        """
        import torch

        from goal.ml.nn.models.simurgh.geometry import differentiable_edges

        with torch.no_grad():
            edge_vectors, edge_lengths = differentiable_edges(graph, graph.pos)
            edge_index = typing.cast(torch.Tensor, graph.edge_index)
            return self.artisan_bank.compute_artisan_loads(
                atom_features=self.dressing(
                    atomic_numbers=graph.atomic_numbers,
                    edge_index=edge_index,
                    edge_vectors=edge_vectors,
                    edge_lengths=edge_lengths,
                )[
                    0
                ],  # take only the features, discard layer_energies
                atomic_numbers=graph.atomic_numbers,
                edge_index=edge_index,
                edge_lengths=edge_lengths,
                edge_vectors=edge_vectors,
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, graph: AtomicGraph) -> NodeFeatures:
        """Dress the atoms, run the artisan bank and add the atomic baseline.

        Parameters
        ----------
        graph : AtomicGraph
            Batched graph with ``pos`` (Å), ``atomic_numbers`` and
            ``edge_index``.  If ``graph.pos`` does not require grad it is
            replaced by a detached, grad-enabled copy, so energy-derived
            force heads can differentiate against it.

        Returns
        -------
        NodeFeatures
            ``node_feats`` — ``(N, irreps_out.dim)`` dressed equivariant
            features; ``node_energies`` — ``(N,)`` invariant per-atom energy
            ``scale · E_artisan + E0(Z)``, plus the unscaled per-layer
            readout energies when the dressing has them; ``node_forces`` —
            ``(N, 3)`` scaled pairwise forces when ``compute_pairwise_forces``
            is on, else ``None``.
        """
        # Re-derive edge geometry from positions so the autograd graph
        # links energy back to ``graph.pos`` (required by autograd-based
        # force heads).  Enable ``requires_grad`` on ``graph.pos`` when
        # the caller hasn't already done so — without this,
        # :class:`DualForcesHead` in autograd / hybrid mode falls back
        # to all-zero forces (see its ``if not graph.pos.requires_grad``
        # branch), which silently makes the forces cosine-similarity
        # metric collapse to 0 and the forces loss equal the magnitude
        # of the labels themselves.  We mutate ``graph.pos`` (and not
        # a clone) so the downstream head sees the same grad-enabled
        # leaf tensor when it inspects ``graph.pos.requires_grad`` and
        # when it calls ``torch.autograd.grad(energy, graph.pos)``.
        positions: torch.Tensor = typing.cast(torch.Tensor, graph.pos)
        if not positions.requires_grad:
            # ``detach`` first to break any prior graph (e.g. a previous
            # batch's autograd graph that would otherwise be retained),
            # then enable grad tracking in place on the leaf tensor.
            positions = positions.detach().requires_grad_(True)
            graph.pos = positions
        edge_vectors: torch.Tensor
        edge_lengths: torch.Tensor
        edge_vectors, edge_lengths = differentiable_edges(graph, positions)

        # Dressed equivariant features + optional per-layer energies.
        # EnvironmentDressing now returns (features, layer_energies) where
        # layer_energies is the sum of per-layer readout contributions (N,)
        # or None when per_layer_readout=False.
        dressed_out = self.dressing(
            atomic_numbers=graph.atomic_numbers,
            edge_index=graph.edge_index,
            edge_vectors=edge_vectors,
            edge_lengths=edge_lengths,
        )
        dressed: torch.Tensor = dressed_out[0]  # (N, irreps_out.dim)
        layer_energies: torch.Tensor | None = dressed_out[1]  # (N,) or None

        # ``graph.edge_index`` is statically typed as ``Tensor | None``
        # by PyG but is always populated for our atomic graphs (built
        # in ``AtomicGraph.from_ase`` / ``from_dict``).  Narrow the
        # type once so downstream signatures stay tight.
        edge_index: torch.Tensor = typing.cast(torch.Tensor, graph.edge_index)

        # Pairwise expert energies (and, optionally, pairwise forces)
        # → per-atom interaction residual.  In the pairwise branch the
        # artisan bank also returns ``F_per_atom`` built from per-pair
        # ``-∂E_ij/∂r_ij`` with Newton's third law applied at scatter
        # time.
        node_forces: torch.Tensor | None = None
        if self._compute_pairwise_forces:
            interaction_energies, interaction_forces = self.artisan_bank.forward_pairwise(
                atom_features=dressed,
                atomic_numbers=graph.atomic_numbers,
                edge_index=edge_index,
                edge_vectors=edge_vectors,
            )  # (N,), (N, 3)
            scale_v: torch.Tensor = self.scale.to(interaction_energies.dtype)
            node_forces = scale_v * interaction_forces  # (N, 3)
        else:
            interaction_energies = self.artisan_bank(
                atom_features=dressed,
                atomic_numbers=graph.atomic_numbers,
                edge_index=edge_index,
                edge_lengths=edge_lengths,
                edge_vectors=edge_vectors,
            )  # (N,)
            scale_v = self.scale.to(interaction_energies.dtype)

        # Apply the dataset-derived scale (MACE-style ScaleShiftBlock with
        # shift=0 — the shift is folded into the atomic-energy baseline).
        scaled_residual: torch.Tensor = scale_v * interaction_energies

        # Per-layer readout energies (CHANGE 3).  These are NOT scaled by
        # the interaction scale — they are absolute energy contributions from
        # the readout MLPs attached to each interaction layer.  Add them to
        # the artisan bank-derived interaction energy before the atomic baseline.
        if layer_energies is not None:
            scaled_residual = scaled_residual + layer_energies.to(scaled_residual.dtype)

        # Add the per-element baseline e(Z).
        atomic_baseline: torch.Tensor = self.atomic_energies[graph.atomic_numbers].to(
            interaction_energies.dtype
        )  # (N,)
        node_energies: torch.Tensor = scaled_residual + atomic_baseline  # (N,)

        return NodeFeatures(
            node_feats=dressed,
            irreps=str(self._irreps_out),
            node_energies=node_energies,
            node_forces=node_forces,
        )
