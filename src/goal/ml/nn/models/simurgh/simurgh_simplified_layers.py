"""SIMURGH Simplified — the shared skeleton of the three-line models.

SIMURGH Simplified replaces SIMURGH's sequential pipeline (artisans →
virtual nodes → experts → readout) with **three parallel lines** that
share one input embedding and meet only at the readout::

                       shared input (embedding, RBF, SH, envelope)
                 ┌───────────────────┼─────────────────────┐
             atom line           pair line           fragment line
     L1 node-wise, L2+ MP    edges, i↔j symmetric   virtual nodes + MP
                 │                   │                     │
            e_i (N,)            ε_ij (E,)             E_k (K,)
                 └──────── readout / combination ──────────┘
                     E = Σ E0 + scale · (E_atom + E_pair + E_frag)
                     F = -∂E/∂r   (autograd, no force head)

This module holds everything the two registered models share — the
shared input, the three line skeletons, the processing units, the
readouts and the base model — so that

* ``simurgh_simplified_monolithic``
  (:mod:`~goal.ml.nn.models.simurgh.simurgh_simplified_monolithic`) and
* ``simurgh_simplified_routed``
  (:mod:`~goal.ml.nn.models.simurgh.simurgh_simplified_routed`)

differ in **exactly one thing**: the processor plugged into each line's
slot.  The monolithic model fills every slot with a single stack; the
routed model fills them with frequency-routed artisans (atom, pair) and
a mixture of experts (fragment).  Every other line of arithmetic is this
file's, executed identically by both.

Design invariants
-----------------
These are properties of the skeleton, not options:

1. **Equivariant until the last moment.**  Every line carries e3nn
   irreps with ``l > 0`` channels through every layer
   (:func:`validate_line_irreps` refuses a scalar-only line).  The only
   ``irreps → 0e`` reductions are the per-line :class:`LineReadout` s and
   the learned-combination gate, all at the very end.
2. **Equivariant nonlinearities only.**  :class:`EquivariantGate`
   (``e3nn.nn.Gate``): SiLU on ``0e`` channels, ``l > 0`` channels scaled
   by sigmoid gates computed from ``0e`` channels.  No pointwise
   nonlinearity ever touches an ``l > 0`` component.
3. **Scalar decisions from ``0e`` only, applied as scalar multipliers.**
   Type modulation, MoE gate weights and combination weights are all
   invariants multiplying equivariant tensors.
4. **Lines never read each other's features.**  Each builds what it needs
   from :class:`SharedInputs`.  The learned combination reads the atom
   line's final features — that is the readout, where lines may meet.
5. **Mean pooling** for every angular order when atoms are pooled onto
   virtual nodes (member ``l > 0`` vectors partially cancel; a sum would
   shrink with fragment size as an artifact).
6. **Zero-initialised corrections.**  The fragment readout and the
   learned-combination gate start at exactly zero, so a fresh model's
   fragment line contributes nothing and ``learned`` equals ``additive``
   at initialisation.  Atom and pair readouts are the primary energy
   terms and use default initialisation.
7. **Bias-free learnable maps** inside every unit and readout, so zero
   input maps to exactly zero output (a pair feature vanishes at the
   cutoff, a lone fragment with no partners gets no message).
8. **Smoothness.**  Every radial weight is multiplied by a polynomial
   envelope; the pair energy is multiplied by it again.  The fragment
   pair set is combinatorial (every ``k ≠ l`` in a structure), so pairs
   never enter or leave as atoms move; the envelope alone sets the range.
9. **Every parameter reaches the loss on every step** (DDP static graph):
   no unit is skipped — a routed unit with no members runs on an empty
   gather, which still registers its parameters with autograd.
10. **Size-extensivity.**  The energy is a sum of local per-atom,
    per-edge and per-fragment terms; no nonlinearity acts on a total.

Known structural fact (approved design, Phase 0 item 9a)
--------------------------------------------------------
The shared input carries ``0e`` channels only (an element embedding), so
the atom line's node-wise layer 1 *declares* ``l > 0`` blocks that are
identically zero: an equivariant map from scalars cannot create a
vector.  They become populated at layer 2, the first message-passing
layer.  The pair line's ``1e`` block is likewise empty at its first edge
layer (``0e ⊗ Y_even`` reaches only ``0e`` and ``2e``) and is populated
from the second.

Isolation
---------
Purely additive: imports stable primitives and blocks, modifies nothing.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.nn import Gate
from e3nn.o3 import Irrep, Irreps, spherical_harmonics
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.blocks.artisans import _PairRMSNorm, build_artisan_edge_irreps
from goal.ml.nn.blocks.embedding import AtomicNumberEmbedding
from goal.ml.nn.blocks.fragment_interaction import build_fragment_geometry
from goal.ml.nn.models.simurgh.geometry import differentiable_edges
from goal.ml.nn.primitives.linear import EquivariantLinear
from goal.ml.nn.primitives.radial import BesselBasis, PolynomialEnvelope, RadialMLP
from goal.ml.nn.primitives.tp import WeightedTensorProduct

__all__ = [
    "AtomLine",
    "AtomLineProcessor",
    "EquivariantGate",
    "EquivariantMessageUnit",
    "FragmentLine",
    "FragmentLineProcessor",
    "LearnedCombination",
    "LineReadout",
    "NodewiseUnit",
    "PairEdgeUnit",
    "PairLine",
    "PairLineProcessor",
    "SharedInput",
    "SharedInputs",
    "SimplifiedSettings",
    "SimurghSimplifiedModel",
    "TypeModulation",
    "validate_line_irreps",
]

#: Floor on distances entering a normalisation or a ``1/d`` basis.
_EPS: float = 1e-6

#: The three lines, in the order every per-line tensor uses.
LINE_NAMES: tuple[str, str, str] = ("atom", "pair", "fragment")

#: A recorded intermediate feature: ``(name, irreps, tensor)``.
FeatureRecord = tuple[str, Irreps, torch.Tensor]


# ---------------------------------------------------------------------------
# Irrep bookkeeping
# ---------------------------------------------------------------------------


def validate_line_irreps(
    irreps: Irreps | str,
    line: str,
    even_parity_only: bool = False,
) -> Irreps:
    """Check a line's hidden irreps against the global invariants.

    * sorted and simplified (``e3nn.nn.Gate`` re-orders otherwise);
    * a leading ``0e`` block (scalars for the readout and the gates);
    * **at least one ``l > 0`` block** — the equivariance-until-readout
      invariant made into an error;
    * for the pair line, even parity throughout: its edge features are
      built from even-``l`` harmonics only (that is what makes
      ``ε_ij == ε_ji`` exact), so an odd-parity block could never be
      populated and would be dead weight.
    """
    parsed: Irreps = Irreps(irreps)
    if parsed != parsed.sort().irreps.simplify():
        raise ValueError(
            f"{line}_line.hidden_irreps must be sorted and simplified, e.g. "
            f"'{parsed.sort().irreps.simplify()}', got '{parsed}'."
        )
    if len(parsed) == 0 or parsed[0].ir != Irrep("0e"):
        raise ValueError(f"{line}_line.hidden_irreps must lead with a 0e block, got '{parsed}'.")
    if any(ir.l == 0 and ir.p != 1 for _, ir in parsed):
        raise ValueError(
            f"{line}_line.hidden_irreps: only 0e scalars are supported, got '{parsed}'."
        )
    if not any(ir.l > 0 for _, ir in parsed):
        raise ValueError(
            f"{line}_line.hidden_irreps='{parsed}' has no l > 0 channels.  SIMURGH "
            f"Simplified keeps every line equivariant until the final readout."
        )
    if even_parity_only and any(ir.p != 1 for _, ir in parsed):
        raise ValueError(
            f"pair_line.hidden_irreps='{parsed}' contains odd-parity blocks.  The pair "
            f"line uses even-l spherical harmonics only (exact i↔j symmetry), so odd "
            f"blocks can never be populated.  Use e.g. '16x0e + 16x1e + 16x2e'."
        )
    return parsed


def _channel_expansion(irreps: Irreps) -> torch.Tensor:
    """``(irreps.dim,)`` index mapping every component to its channel.

    A per-channel scalar ``s (M, num_irreps)`` expanded with
    ``s[:, index]`` multiplies all ``2l+1`` components of a channel by
    the same invariant, which is what keeps the product equivariant.
    """
    repeats: list[int] = [ir.dim for mul, ir in irreps for _ in range(mul)]
    return torch.repeat_interleave(
        torch.arange(len(repeats), dtype=torch.long), torch.tensor(repeats, dtype=torch.long)
    )


def _even_sh_columns(irreps_sh: Irreps) -> torch.Tensor:
    """Column indices of the even-``l`` blocks of a spherical-harmonic irreps."""
    columns: list[int] = []
    for (_, ir), block in zip(irreps_sh, irreps_sh.slices()):
        if ir.l % 2 == 0:
            columns.extend(range(block.start, block.stop))
    return torch.tensor(columns, dtype=torch.long)


# ---------------------------------------------------------------------------
# Units — the building blocks every processor is made of
# ---------------------------------------------------------------------------


class EquivariantGate(nn.Module):
    """e3nn gated nonlinearity onto *irreps_out*.

    Input layout is ``scalars + gates + gated`` (``irreps_in``): SiLU acts
    on the ``0e`` scalars, sigmoid on one extra ``0e`` gate per ``l > 0``
    channel, and each gate multiplies its channel.  Zero in, zero out.
    """

    def __init__(self, irreps_out: Irreps | str) -> None:
        super().__init__()
        target: Irreps = Irreps(irreps_out)
        scalars: Irreps = Irreps([(mul, ir) for mul, ir in target if ir.l == 0])
        gated: Irreps = Irreps([(mul, ir) for mul, ir in target if ir.l > 0])
        self.gate: Gate = Gate(
            scalars, [F.silu], Irreps(f"{gated.num_irreps}x0e"), [torch.sigmoid], gated
        )
        self.irreps_in: Irreps = self.gate.irreps_in
        self.irreps_out: Irreps = self.gate.irreps_out
        if self.irreps_out != target:
            raise ValueError(f"Gate output {self.irreps_out} != requested irreps {target}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated nonlinearity.

        Parameters
        ----------
        x : torch.Tensor
            ``(M, irreps_in.dim)`` features laid out ``scalars + gates + gated``.

        Returns
        -------
        torch.Tensor
            ``(M, irreps_out.dim)``; equivariant, since only ``0e`` channels
            pass through a pointwise nonlinearity.
        """
        return self.gate(x)


class TypeModulation(nn.Module):
    """``1 + Linear(Σ embed(Z))`` — the rare-artisan type code.

    The embedding is **zero-initialised**: a type that receives gradient
    learns a code, while a type never seen in training keeps a zero code
    and the module returns exactly ``1`` — the plain shared rare artisan,
    not a random embedding row.  Out-of-table atomic numbers map to the
    last row, which is never trained and therefore also neutral.

    Accepts ``(M,)`` element keys or ``(M, 2)`` element-pair keys (summed,
    so the code of an unordered pair is symmetric).
    """

    def __init__(self, num_elements: int, embed_dim: int, num_out: int) -> None:
        super().__init__()
        self._num_elements: int = int(num_elements)
        self.embedding: nn.Embedding = nn.Embedding(self._num_elements, int(embed_dim))
        self.linear: nn.Linear = nn.Linear(int(embed_dim), int(num_out), bias=False)
        with torch.no_grad():
            self.embedding.weight.zero_()

    def forward(self, keys: torch.Tensor) -> torch.Tensor:
        """Per-channel multiplicative type code ``1 + Linear(Σ embed(Z))``.

        Parameters
        ----------
        keys : torch.Tensor
            ``(M,)`` atomic numbers or ``(M, 2)`` unordered element pairs
            (int64), clamped into the embedding table.

        Returns
        -------
        torch.Tensor
            ``(M, num_out)`` invariant multipliers; exactly ``1`` for a type
            whose embedding row is still zero.
        """
        code: torch.Tensor = self.embedding(keys.clamp(0, self._num_elements - 1))
        if code.dim() == 3:
            code = code.sum(dim=1)
        return 1.0 + self.linear(code)


class NodewiseUnit(nn.Module):
    """Atom-line layer 1: one atom at a time, no message passing.

    ``u_i = Gate(Linear(h_i) [⊙ type modulation])``.
    """

    _channels: torch.Tensor

    def __init__(
        self,
        irreps: Irreps | str,
        num_elements: int = 120,
        type_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.irreps: Irreps = Irreps(irreps)
        self.gate: EquivariantGate = EquivariantGate(self.irreps)
        self.linear: EquivariantLinear = EquivariantLinear(
            self.irreps, self.gate.irreps_in, biases=False
        )
        self.modulation: TypeModulation | None = None
        if type_embed_dim:
            self.modulation = TypeModulation(
                num_elements, type_embed_dim, self.gate.irreps_in.num_irreps
            )
            self.register_buffer(
                "_channels", _channel_expansion(self.gate.irreps_in), persistent=False
            )

    def forward(self, h: torch.Tensor, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Update each atom from its own features only.

        Parameters
        ----------
        h : torch.Tensor
            ``(M, irreps.dim)`` node features.
        atomic_numbers : torch.Tensor
            ``(M,)`` int64 atomic numbers; read only when the unit is
            type-modulated.

        Returns
        -------
        torch.Tensor
            ``(M, irreps.dim)`` equivariant update (no residual, no norm).
        """
        x: torch.Tensor = self.linear(h)
        if self.modulation is not None:
            x = x * self.modulation(atomic_numbers).to(x.dtype)[:, self._channels]
        return self.gate(x)


class EquivariantMessageUnit(nn.Module):
    """One equivariant message-passing update (atom line L2+, fragment line).

    ::

        w_e   = RadialMLP(rbf_e) · env_e  [⊙ type modulation]
        m_e   = TP( Linear(h_src(e)), Y(r̂_e); w_e )
        agg_i = agg_scale · Σ_{e → i} m_e
        u_i   = Gate( Linear(agg_i) + Linear(h_i) )

    The receiving node owns the unit: when routed, the unit chosen by
    the receiver's type processes every message that receiver gets.
    """

    def __init__(
        self,
        irreps: Irreps | str,
        irreps_sh: Irreps | str,
        num_rbf: int,
        radial_hidden: int,
        num_elements: int = 120,
        type_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.irreps: Irreps = Irreps(irreps)
        self.irreps_sh: Irreps = Irreps(irreps_sh)
        self.source_linear: EquivariantLinear = EquivariantLinear(
            self.irreps, self.irreps, biases=False
        )
        self.tp: WeightedTensorProduct = WeightedTensorProduct(
            self.irreps, self.irreps_sh, self.irreps
        )
        self.radial_mlp: RadialMLP = RadialMLP(
            num_basis=int(num_rbf), hidden_dim=int(radial_hidden), num_out=self.tp.weight_numel
        )
        self.gate: EquivariantGate = EquivariantGate(self.irreps)
        self.message_linear: EquivariantLinear = EquivariantLinear(
            self.irreps, self.gate.irreps_in, biases=False
        )
        self.self_linear: EquivariantLinear = EquivariantLinear(
            self.irreps, self.gate.irreps_in, biases=False
        )
        self.modulation: TypeModulation | None = (
            TypeModulation(num_elements, type_embed_dim, self.tp.weight_numel)
            if type_embed_dim
            else None
        )

    def forward(
        self,
        h_source: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_rbf: torch.Tensor,
        edge_env: torch.Tensor,
        receiver: torch.Tensor,
        h_receiver: torch.Tensor,
        edge_types: torch.Tensor | None = None,
        agg_scale: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """Update ``(R, irreps.dim)`` for the ``R`` receivers in *h_receiver*.

        *receiver* indexes rows of *h_receiver* (local indices).
        """
        dtype: torch.dtype = h_receiver.dtype
        weights: torch.Tensor = self.radial_mlp(edge_rbf).to(dtype) * edge_env.to(dtype).unsqueeze(
            -1
        )
        if self.modulation is not None:
            if edge_types is None:
                raise ValueError("A type-modulated unit needs edge_types.")
            weights = weights * self.modulation(edge_types).to(dtype)
        messages: torch.Tensor = self.tp(self.source_linear(h_source), edge_sh.to(dtype), weights)
        agg: torch.Tensor = torch.zeros(
            h_receiver.shape[0], self.irreps.dim, device=h_receiver.device, dtype=dtype
        ).index_add(0, receiver, messages)
        agg = agg * agg_scale
        return self.gate(self.message_linear(agg) + self.self_linear(h_receiver))


class PairEdgeUnit(nn.Module):
    """One pair-line edge update, symmetric under ``i ↔ j`` by construction.

    ::

        u_ij = Gate( Linear( TP( L(s_i) + L(s_j), Y_even(r̂_ij); w(d_ij) ) ) )

    ``L(s_i) + L(s_j)`` is symmetric, ``Y_even(-r̂) = Y_even(r̂)`` exactly,
    ``d`` and the unordered pair type are symmetric — so ``u_ij == u_ji``
    bit for bit.  (Symmetrising ``TP(·, Y(r̂_ij)) + TP(·, Y(r̂_ji))`` is the
    same thing: the TP is linear in ``Y`` and the odd blocks cancel.)
    """

    def __init__(
        self,
        irreps: Irreps | str,
        irreps_sh_even: Irreps | str,
        num_rbf: int,
        radial_hidden: int,
        num_elements: int = 120,
        type_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.irreps: Irreps = Irreps(irreps)
        self.irreps_sh: Irreps = Irreps(irreps_sh_even)
        self.endpoint_linear: EquivariantLinear = EquivariantLinear(
            self.irreps, self.irreps, biases=False
        )
        self.tp: WeightedTensorProduct = WeightedTensorProduct(
            self.irreps, self.irreps_sh, self.irreps
        )
        self.radial_mlp: RadialMLP = RadialMLP(
            num_basis=int(num_rbf), hidden_dim=int(radial_hidden), num_out=self.tp.weight_numel
        )
        self.gate: EquivariantGate = EquivariantGate(self.irreps)
        self.output_linear: EquivariantLinear = EquivariantLinear(
            self.irreps, self.gate.irreps_in, biases=False
        )
        self.modulation: TypeModulation | None = (
            TypeModulation(num_elements, type_embed_dim, self.tp.weight_numel)
            if type_embed_dim
            else None
        )

    def forward(
        self,
        s_a: torch.Tensor,
        s_b: torch.Tensor,
        edge_sh_even: torch.Tensor,
        edge_rbf: torch.Tensor,
        edge_env: torch.Tensor,
        pair_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Symmetric edge update ``u_ij`` for a batch of edges.

        Parameters
        ----------
        s_a, s_b : torch.Tensor
            ``(E, irreps.dim)`` pair-line node states of the two endpoints;
            swapping them leaves the output unchanged.
        edge_sh_even : torch.Tensor
            ``(E, irreps_sh.dim)`` even-``l`` spherical harmonics of the edge
            direction.
        edge_rbf : torch.Tensor
            ``(E, num_rbf)`` radial basis of the edge length.
        edge_env : torch.Tensor
            ``(E,)`` smooth cutoff envelope.
        pair_types : torch.Tensor, optional
            ``(E, 2)`` unordered element pairs; required when the unit is
            type-modulated.

        Returns
        -------
        torch.Tensor
            ``(E, irreps.dim)`` equivariant edge update, identical for both
            directions of an edge.
        """
        dtype: torch.dtype = s_a.dtype
        weights: torch.Tensor = self.radial_mlp(edge_rbf).to(dtype) * edge_env.to(dtype).unsqueeze(
            -1
        )
        if self.modulation is not None:
            if pair_types is None:
                raise ValueError("A type-modulated unit needs pair_types.")
            weights = weights * self.modulation(pair_types).to(dtype)
        endpoints: torch.Tensor = self.endpoint_linear(s_a) + self.endpoint_linear(s_b)
        return self.gate(self.output_linear(self.tp(endpoints, edge_sh_even.to(dtype), weights)))


# ---------------------------------------------------------------------------
# Readouts — the only irreps → 0e reductions in the model
# ---------------------------------------------------------------------------


class LineReadout(nn.Module):
    """``irreps → n_scalar×0e → bias-free MLP → (M,)`` — a line's final collapse."""

    def __init__(self, irreps: Irreps | str, n_scalar: int, hidden: int, zero_init: bool) -> None:
        super().__init__()
        self.zero_init: bool = bool(zero_init)
        self.to_scalars: EquivariantLinear = EquivariantLinear(
            Irreps(irreps), Irreps(f"{int(n_scalar)}x0e"), biases=False
        )
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(int(n_scalar), int(hidden), bias=False),
            nn.SiLU(),
            nn.Linear(int(hidden), 1, bias=False),
        )
        if self.zero_init:
            with torch.no_grad():
                typing.cast(nn.Linear, self.mlp[-1]).weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse ``(M, irreps.dim)`` features to ``(M,)`` invariant energies.

        Only the ``0e`` channels survive the projection to scalars, so the
        output is rotation invariant.
        """
        return self.mlp(self.to_scalars(x)).squeeze(-1)


class LearnedCombination(nn.Module):
    """Per-atom line weights ``g = 1 + tanh(MLP(0e of the atom line))``, ``(N, 3)``.

    Bounded in ``(0, 2)`` and applied to *per-atom* line energies, so the
    total stays a sum of local terms (size-extensive).  Zero-initialised:
    ``g ≡ 1`` at construction, so ``learned`` starts equal to ``additive``.
    """

    def __init__(self, irreps: Irreps | str, n_scalar: int, hidden: int) -> None:
        super().__init__()
        self.to_scalars: EquivariantLinear = EquivariantLinear(
            Irreps(irreps), Irreps(f"{int(n_scalar)}x0e"), biases=False
        )
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(int(n_scalar), int(hidden), bias=False),
            nn.SiLU(),
            nn.Linear(int(hidden), len(LINE_NAMES), bias=False),
        )
        with torch.no_grad():
            typing.cast(nn.Linear, self.mlp[-1]).weight.zero_()

    def forward(self, h_atom: torch.Tensor) -> torch.Tensor:
        """Per-atom line weights from the atom line's final features.

        Parameters
        ----------
        h_atom : torch.Tensor
            ``(N, atom_irreps.dim)`` final atom-line features.

        Returns
        -------
        torch.Tensor
            ``(N, 3)`` invariant weights in ``(0, 2)``, columns ordered as
            ``LINE_NAMES`` (atom, pair, fragment).
        """
        return 1.0 + torch.tanh(self.mlp(self.to_scalars(h_atom)))


# ---------------------------------------------------------------------------
# Shared input
# ---------------------------------------------------------------------------


@dataclass
class SharedInputs:
    """What every line receives.  Geometry, not learned features."""

    atomic_numbers: torch.Tensor
    """(N,) int64."""
    node_embedding: torch.Tensor
    """(N, embedding_dim) — ``0e`` element embedding."""
    batch: torch.Tensor
    """(N,) int64 structure index."""
    num_graphs: int
    positions: torch.Tensor
    """(N, 3) — the tensor the energy is differentiated against."""
    edge_index: torch.Tensor
    """(2, E) — ``row`` = source, ``col`` = receiver."""
    edge_sh: torch.Tensor
    """(E, irreps_sh.dim) — harmonics of ``r_col - r_row``, component norm."""
    edge_rbf: torch.Tensor
    """(E, num_rbf) — Bessel basis of the clamped edge length."""
    edge_env: torch.Tensor
    """(E,) — polynomial envelope, zero at the cutoff."""
    fragment_index: torch.Tensor
    """(N,) int64 contiguous fragment labels."""
    cell: torch.Tensor | None
    pbc: torch.Tensor | None


class SharedInput(nn.Module):
    """The one input embedding shared by all three lines.

    Its only learnable part is the element embedding table (imported
    :class:`~goal.ml.nn.blocks.embedding.AtomicNumberEmbedding`); the radial
    basis, envelope and spherical harmonics are fixed functions of the
    geometry.  There is no message passing here.
    """

    def __init__(
        self,
        cutoff: float,
        embedding_dim: int,
        num_elements: int,
        num_rbf: int,
        sh_lmax: int,
    ) -> None:
        super().__init__()
        self.embedding: AtomicNumberEmbedding = AtomicNumberEmbedding(
            num_elements=int(num_elements), embedding_dim=int(embedding_dim)
        )
        self.radial_basis: BesselBasis = BesselBasis(num_basis=int(num_rbf), cutoff=float(cutoff))
        self.envelope: PolynomialEnvelope = PolynomialEnvelope(cutoff=float(cutoff))
        self.irreps_sh: Irreps = build_artisan_edge_irreps(int(sh_lmax))
        self.irreps_embedding: Irreps = Irreps(f"{int(embedding_dim)}x0e")

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_lengths: torch.Tensor,
        positions: torch.Tensor,
        fragment_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
    ) -> SharedInputs:
        """Embed the elements and expand the edge geometry once for all lines.

        Everything is computed in the parameters' dtype; the geometry casts
        are differentiable, so forces still reach *positions*.

        Parameters
        ----------
        atomic_numbers : torch.Tensor
            ``(N,)`` int64 atomic numbers.
        edge_index : torch.Tensor
            ``(2, E)`` ``(source, receiver)`` indices.
        edge_vectors : torch.Tensor
            ``(E, 3)`` edge vectors ``r_receiver - r_source`` (Å).
        edge_lengths : torch.Tensor
            ``(E,)`` edge lengths (Å), floored at ``1e-6`` before the radial
            basis and the envelope.
        positions : torch.Tensor
            ``(N, 3)`` positions (Å) the energy is differentiated against.
        fragment_index : torch.Tensor
            ``(N,)`` fragment label of each atom.
        batch : torch.Tensor
            ``(N,)`` structure index of each atom.
        num_graphs : int
            Number of structures ``B`` in the batch.
        cell, pbc : torch.Tensor, optional
            Periodic cell and boundary flags, passed through unchanged.

        Returns
        -------
        SharedInputs
            Element embedding, edge harmonics, radial basis and envelope,
            plus the passed-through geometry.
        """
        # Compute in the PARAMETERS' dtype and cast the geometry to it (the
        # ARACE convention).  Datasets ship float64 positions while the model
        # is usually float32; following the positions instead would feed
        # float64 features into float32 weights.  The casts are
        # differentiable, so forces still reach ``positions``.
        dtype: torch.dtype = self.embedding.embedding.weight.dtype
        lengths: torch.Tensor = edge_lengths.to(dtype).clamp_min(_EPS)
        return SharedInputs(
            atomic_numbers=atomic_numbers,
            node_embedding=self.embedding(atomic_numbers),
            batch=batch,
            num_graphs=num_graphs,
            positions=positions,
            edge_index=edge_index,
            edge_sh=spherical_harmonics(
                self.irreps_sh, edge_vectors.to(dtype), normalize=True, normalization="component"
            ),
            edge_rbf=self.radial_basis(lengths).to(dtype),
            edge_env=self.envelope(lengths).to(dtype),
            fragment_index=fragment_index.to(device=positions.device, dtype=torch.long),
            cell=cell,
            pbc=pbc,
        )


# ---------------------------------------------------------------------------
# Processor slots
# ---------------------------------------------------------------------------


class AtomLineProcessor(nn.Module):
    """Processor slot of the atom line.

    ``nodewise`` is layer 1 (each atom alone); ``message`` is layer
    ``layer + 2`` (``layer`` counts message-passing layers from 0).  Both
    return an update of the line's irreps; the skeleton adds the residual
    and the norm.
    """

    def nodewise(self, h: torch.Tensor, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Layer-1 update of every atom from its own features (no messages).

        Parameters
        ----------
        h : torch.Tensor
            ``(N, irreps.dim)`` atom-line features after the input linear.
        atomic_numbers : torch.Tensor
            ``(N,)`` int64 atomic numbers (for routing / type modulation).

        Returns
        -------
        torch.Tensor
            ``(N, irreps.dim)`` equivariant update.
        """
        raise NotImplementedError

    def message(
        self,
        layer: int,
        h: torch.Tensor,
        atomic_numbers: torch.Tensor,
        inputs: SharedInputs,
        agg_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Message-passing update of atom-line layer ``layer + 2``.

        Parameters
        ----------
        layer : int
            Message-passing layer index, counted from 0.
        h : torch.Tensor
            ``(N, irreps.dim)`` current atom-line features.
        atomic_numbers : torch.Tensor
            ``(N,)`` int64 atomic numbers.
        inputs : SharedInputs
            Shared edge geometry (``edge_index``, ``edge_sh``, ``edge_rbf``,
            ``edge_env``).
        agg_scale : torch.Tensor
            Scalar aggregation normaliser (``1 / avg_num_neighbors``, or 1).

        Returns
        -------
        torch.Tensor
            ``(N, irreps.dim)`` equivariant update.
        """
        raise NotImplementedError


class PairLineProcessor(nn.Module):
    """Processor slot of the pair line: one edge update ``(E, irreps.dim)`` per layer."""

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
        """Edge update of pair-line layer *layer*.

        Parameters
        ----------
        layer : int
            Edge-layer index, counted from 0.
        s : torch.Tensor
            ``(N, irreps.dim)`` the pair line's private node state.
        edge_index : torch.Tensor
            ``(2, E)`` ``(source, receiver)`` indices.
        edge_sh_even : torch.Tensor
            ``(E, irreps_sh_even.dim)`` even-``l`` edge harmonics.
        edge_rbf : torch.Tensor
            ``(E, num_rbf)`` radial basis of the edge lengths.
        edge_env : torch.Tensor
            ``(E,)`` cutoff envelope.
        pair_types : torch.Tensor
            ``(E, 2)`` unordered element pair ``(Z_lo, Z_hi)`` of each edge.

        Returns
        -------
        torch.Tensor
            ``(E, irreps.dim)`` equivariant update, symmetric under ``i ↔ j``.
        """
        raise NotImplementedError


class FragmentLineProcessor(nn.Module):
    """Processor slot of the fragment line.

    ``prepare`` runs once per forward on the geometry-independent
    fragment composition and returns whatever routing state the
    processor needs; ``forward`` is one virtual-node update.
    """

    def prepare(self, composition: torch.Tensor) -> typing.Any:
        """Per-forward routing state from the fragment composition codes.

        Parameters
        ----------
        composition : torch.Tensor
            ``(K, embedding_dim + 1)`` geometry-independent fragment codes
            (see :meth:`FragmentLine.composition`).

        Returns
        -------
        typing.Any
            State handed to every :meth:`forward` call and to
            :meth:`aux_loss`; this base implementation returns ``None``.
        """
        return None

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
        """One virtual-node update of fragment-line layer *layer*.

        Parameters
        ----------
        layer : int
            Layer index, counted from 0.
        v : torch.Tensor
            ``(K, irreps.dim)`` equivariant virtual-node features.
        state : typing.Any
            Whatever :meth:`prepare` returned for this forward.
        pair_index : torch.Tensor
            ``(2, P)`` ``(source, receiver)`` fragment indices (every ordered
            ``k ≠ l`` of a structure).
        pair_sh : torch.Tensor
            ``(P, irreps_sh.dim)`` harmonics of the centroid-centroid vectors.
        pair_rbf : torch.Tensor
            ``(P, num_rbf)`` radial basis of the centroid distances.
        pair_env : torch.Tensor
            ``(P,)`` envelope, zero at ``vnode_cutoff``.
        agg_scale : torch.Tensor
            Scalar aggregation normaliser.

        Returns
        -------
        torch.Tensor
            ``(K, irreps.dim)`` equivariant update; the line adds the
            residual and the norm.
        """
        raise NotImplementedError

    def aux_loss(self, state: typing.Any) -> torch.Tensor | None:
        """Scalar auxiliary training loss from *state*, or ``None`` (the default)."""
        return None


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------


class AtomLine(nn.Module):
    """Atom line: node-wise layer 1, message-passing layers 2..L, per-atom energy.

    ``h ← RMSNorm(h + processor update)`` after every layer.
    """

    agg_scale: torch.Tensor

    def __init__(
        self,
        processor: AtomLineProcessor,
        irreps: Irreps,
        irreps_embedding: Irreps,
        num_layers: int,
        avg_num_neighbors: float | None,
        n_scalar: int,
        hidden: int,
    ) -> None:
        super().__init__()
        if int(num_layers) < 2:
            raise ValueError(
                f"atom_line.num_layers must be >= 2 (layer 1 is node-wise, the rest pass "
                f"messages; without one the l > 0 channels stay empty), got {num_layers}."
            )
        self.irreps: Irreps = irreps
        self.num_layers: int = int(num_layers)
        self.processor: AtomLineProcessor = processor
        self.input_linear: EquivariantLinear = EquivariantLinear(irreps_embedding, irreps)
        self.norms: nn.ModuleList = nn.ModuleList(
            _PairRMSNorm(irreps) for _ in range(self.num_layers)
        )
        self.readout: LineReadout = LineReadout(irreps, n_scalar, hidden, zero_init=False)
        self.register_buffer("agg_scale", _inverse_or_one(avg_num_neighbors))

    def forward(
        self, inputs: SharedInputs
    ) -> tuple[torch.Tensor, torch.Tensor, list[FeatureRecord]]:
        """``(e_atom (N,), h (N, irreps), features)``."""
        z: torch.Tensor = inputs.atomic_numbers
        h: torch.Tensor = self.input_linear(inputs.node_embedding)
        features: list[FeatureRecord] = []

        h = self.norms[0](h + self.processor.nodewise(h, z))
        features.append(("atom/layer1", self.irreps, h))
        scale: torch.Tensor = self.agg_scale.to(h.dtype)
        for layer in range(1, self.num_layers):
            h = self.norms[layer](h + self.processor.message(layer - 1, h, z, inputs, scale))
            features.append((f"atom/layer{layer + 1}", self.irreps, h))
        return self.readout(h), h, features


class PairLine(nn.Module):
    """Pair line: symmetric edge features with a private node state.

    The line keeps its own node state ``s`` — fed only by its own edges —
    so a deeper edge layer sees context without reading the atom line::

        h_ij ← h_ij + processor.edge(s, ...)        (no norm: h → 0 at the cutoff)
        s_i  ← RMSNorm(s_i + Linear(agg_scale · Σ_j h_ij))   (between edge layers)
        ε_ij  = LineReadout(h_ij) · env(d_ij)

    Both directed copies of an edge carry identical features, so the
    structure's pair energy is ``½ Σ_directed ε = Σ_{i<j} ε_ij``.
    """

    agg_scale: torch.Tensor
    even_columns: torch.Tensor

    def __init__(
        self,
        processor: PairLineProcessor,
        irreps: Irreps,
        irreps_embedding: Irreps,
        irreps_sh: Irreps,
        num_layers: int,
        avg_num_neighbors: float | None,
        n_scalar: int,
        hidden: int,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError(f"pair_line.num_layers must be >= 1, got {num_layers}.")
        self.irreps: Irreps = irreps
        self.num_layers: int = int(num_layers)
        self.processor: PairLineProcessor = processor
        self.irreps_sh_even: Irreps = Irreps([(m, ir) for m, ir in irreps_sh if ir.l % 2 == 0])
        self.register_buffer("even_columns", _even_sh_columns(irreps_sh), persistent=False)
        self.input_linear: EquivariantLinear = EquivariantLinear(irreps_embedding, irreps)
        # Node updates exist only BETWEEN edge layers: one after the last
        # edge layer would feed nothing and leave parameters without
        # gradient (the trap MonolithicArace's last round falls into).
        self.node_linears: nn.ModuleList = nn.ModuleList(
            EquivariantLinear(irreps, irreps, biases=False) for _ in range(self.num_layers - 1)
        )
        self.node_norms: nn.ModuleList = nn.ModuleList(
            _PairRMSNorm(irreps) for _ in range(self.num_layers - 1)
        )
        self.readout: LineReadout = LineReadout(irreps, n_scalar, hidden, zero_init=False)
        self.register_buffer("agg_scale", _inverse_or_one(avg_num_neighbors))

    @staticmethod
    def pair_types(atomic_numbers: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """``(E, 2)`` unordered element pair ``(Z_lo, Z_hi)`` per edge."""
        z_row: torch.Tensor = atomic_numbers[edge_index[0]]
        z_col: torch.Tensor = atomic_numbers[edge_index[1]]
        return torch.stack([torch.minimum(z_row, z_col), torch.maximum(z_row, z_col)], dim=-1)

    def forward(self, inputs: SharedInputs) -> tuple[torch.Tensor, list[FeatureRecord]]:
        """``(ε (E,) per directed edge, features)``."""
        edge_index: torch.Tensor = inputs.edge_index
        col: torch.Tensor = edge_index[1]
        types: torch.Tensor = self.pair_types(inputs.atomic_numbers, edge_index)
        sh_even: torch.Tensor = inputs.edge_sh[:, self.even_columns]
        s: torch.Tensor = self.input_linear(inputs.node_embedding)
        scale: torch.Tensor = self.agg_scale.to(s.dtype)
        features: list[FeatureRecord] = []

        h_edge: torch.Tensor | None = None
        for layer in range(self.num_layers):
            update: torch.Tensor = self.processor.edge(
                layer, s, edge_index, sh_even, inputs.edge_rbf, inputs.edge_env, types
            )
            h_edge = update if h_edge is None else h_edge + update
            features.append((f"pair/edge{layer + 1}", self.irreps, h_edge))
            if layer < self.num_layers - 1:
                agg: torch.Tensor = torch.zeros_like(s).index_add(0, col, h_edge) * scale
                s = self.node_norms[layer](s + self.node_linears[layer](agg))
                features.append((f"pair/node{layer + 1}", self.irreps, s))
        assert h_edge is not None
        return self.readout(h_edge) * inputs.edge_env, features


@dataclass
class FragmentLineOutputs:
    """What :class:`FragmentLine` hands to the model's readout."""

    energies: torch.Tensor
    """(K,) per-fragment energy."""
    fragment_index: torch.Tensor
    """(N,) fragment of each atom."""
    counts: torch.Tensor
    """(K, 1) atoms per fragment."""
    fragment_graph: torch.Tensor
    """(K,) structure of each fragment."""
    state: typing.Any
    """The processor's routing state (MoE weights, or ``None``)."""
    features: list[FeatureRecord]


class FragmentLine(nn.Module):
    """Fragment line: equivariant virtual nodes and fragment-level message passing.

    ::

        rel_i = r_i - c_k(i)                         (translation-invariant)
        V_k   = mean_{i∈k} TP( emb_i, Y(rel_i / R_f); w(|rel_i|) )
        V_k  ← RMSNorm(V_k + processor(V, pairs))    × num_layers
        E_k   = LineReadout(V_k)                      zero-initialised

    The atom → virtual-node map uses *unnormalised* harmonics
    (``|x|^l Y_l(x̂)``, polynomials), so a one-atom fragment (``rel = 0``)
    is regular, and a smoothed ``|rel|`` keeps the radial gradient finite
    there.  Fragment pairs are every ordered ``k ≠ l`` of a structure
    (reusing :func:`~goal.ml.nn.blocks.fragment_interaction.build_fragment_geometry`
    with ``cutoff=inf``); ``vnode_cutoff`` enters only through the smooth
    envelope.  Aggregation is a **sum with a fixed scale** (no softmax, no
    per-fragment normalisation), so two copies farther apart than
    ``vnode_cutoff`` are exactly separable.

    Centroids are plain position means, as in the existing fragment
    channel: a fragment straddling a periodic boundary gets an
    approximate centroid.
    """

    agg_scale: torch.Tensor

    def __init__(
        self,
        processor: FragmentLineProcessor,
        irreps: Irreps,
        irreps_embedding: Irreps,
        irreps_sh: Irreps,
        num_layers: int,
        num_rbf: int,
        radial_hidden: int,
        vnode_cutoff: float,
        fragment_radius: float,
        avg_num_fragment_neighbors: float | None,
        max_fragments: int | None,
        n_scalar: int,
        hidden: int,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError(f"fragment_line.num_layers must be >= 1, got {num_layers}.")
        self.irreps: Irreps = irreps
        self.irreps_sh: Irreps = irreps_sh
        self.num_layers: int = int(num_layers)
        self.processor: FragmentLineProcessor = processor
        self._fragment_radius: float = float(fragment_radius)
        self._vnode_cutoff: float = float(vnode_cutoff)
        self._max_fragments: int | None = int(max_fragments) if max_fragments else None

        self.embed_tp: WeightedTensorProduct = WeightedTensorProduct(
            irreps_embedding, irreps_sh, irreps
        )
        self.embed_basis: BesselBasis = BesselBasis(
            num_basis=int(num_rbf), cutoff=self._fragment_radius
        )
        self.embed_radial: RadialMLP = RadialMLP(
            num_basis=int(num_rbf),
            hidden_dim=int(radial_hidden),
            num_out=self.embed_tp.weight_numel,
        )
        self.pair_basis: BesselBasis = BesselBasis(
            num_basis=int(num_rbf), cutoff=self._vnode_cutoff
        )
        self.pair_envelope: PolynomialEnvelope = PolynomialEnvelope(cutoff=self._vnode_cutoff)
        self.norms: nn.ModuleList = nn.ModuleList(
            _PairRMSNorm(irreps) for _ in range(self.num_layers)
        )
        self.readout: LineReadout = LineReadout(irreps, n_scalar, hidden, zero_init=True)
        self.register_buffer("agg_scale", _inverse_or_one(avg_num_fragment_neighbors))

    @staticmethod
    def composition(
        node_embedding: torch.Tensor,
        fragment_index: torch.Tensor,
        n_fragments: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Geometry-independent fragment code ``[mean emb ‖ log1p(n)]`` and counts.

        A function of the element labels and the fragment partition only —
        no positions — which is what lets the MoE gate use hard top-k
        without making the energy discontinuous.
        """
        dtype: torch.dtype = node_embedding.dtype
        device: torch.device = node_embedding.device
        counts: torch.Tensor = torch.zeros(n_fragments, 1, device=device, dtype=dtype).index_add(
            0, fragment_index, torch.ones(fragment_index.shape[0], 1, device=device, dtype=dtype)
        )
        mean_emb: torch.Tensor = torch.zeros(
            n_fragments, node_embedding.shape[-1], device=device, dtype=dtype
        ).index_add(0, fragment_index, node_embedding) / counts.clamp(min=1.0)
        return torch.cat([mean_emb, torch.log1p(counts)], dim=-1), counts

    def forward(self, inputs: SharedInputs) -> FragmentLineOutputs:
        """Build the virtual nodes, run the fragment layers and read out ``E_k``.

        Parameters
        ----------
        inputs : SharedInputs
            Shared input; uses the node embedding, positions (Å), fragment
            labels, batch and ``cell`` / ``pbc``.

        Returns
        -------
        FragmentLineOutputs
            ``(K,)`` invariant per-fragment energies, the fragment labels and
            counts, the fragment-to-structure map, the processor's routing
            state and the recorded features.
        """
        positions: torch.Tensor = inputs.positions
        # Model compute dtype; the geometry below is built from the
        # full-precision positions and cast once it is relative.
        dtype: torch.dtype = inputs.node_embedding.dtype
        geometry = build_fragment_geometry(
            positions=positions,
            fragment_index=inputs.fragment_index,
            irreps_sh=self.irreps_sh,
            cutoff=float("inf"),
            batch=inputs.batch,
            cell=inputs.cell,
            pbc=inputs.pbc,
            max_fragments=self._max_fragments,
        )
        index: torch.Tensor = geometry.fragment_index
        n_fragments: int = geometry.n_fragments
        composition, counts = self.composition(inputs.node_embedding, index, n_fragments)

        # ----- Atoms → equivariant virtual nodes (mean over every order).
        rel: torch.Tensor = (positions - geometry.centroids[index]).to(dtype)
        rel_length: torch.Tensor = torch.sqrt(rel.pow(2).sum(dim=-1) + _EPS**2)
        atom_sh: torch.Tensor = spherical_harmonics(
            self.irreps_sh, rel / self._fragment_radius, normalize=False, normalization="component"
        ).to(dtype)
        weights: torch.Tensor = self.embed_radial(self.embed_basis(rel_length)).to(dtype)
        atom_terms: torch.Tensor = self.embed_tp(inputs.node_embedding, atom_sh, weights)
        v: torch.Tensor = torch.zeros(
            n_fragments, self.irreps.dim, device=positions.device, dtype=dtype
        ).index_add(0, index, atom_terms) / counts.clamp(min=1.0)
        features: list[FeatureRecord] = [("fragment/vnode", self.irreps, v)]

        # ----- Fragment-fragment message passing.
        pair_index: torch.Tensor = geometry.pair_index  # (2, P): source, receiver
        pair_length: torch.Tensor = geometry.pair_lengths.to(dtype).clamp_min(_EPS)
        pair_rbf: torch.Tensor = self.pair_basis(pair_length).to(dtype)
        pair_env: torch.Tensor = self.pair_envelope(pair_length).to(dtype)
        state: typing.Any = self.processor.prepare(composition)
        scale: torch.Tensor = self.agg_scale.to(dtype)
        for layer in range(self.num_layers):
            update: torch.Tensor = self.processor(
                layer, v, state, pair_index, geometry.pair_sh.to(dtype), pair_rbf, pair_env, scale
            )
            v = self.norms[layer](v + update)
            features.append((f"fragment/layer{layer + 1}", self.irreps, v))

        fragment_graph: torch.Tensor = torch.zeros(
            n_fragments, dtype=torch.long, device=positions.device
        )
        if n_fragments > 0:
            fragment_graph[index] = inputs.batch.to(torch.long)
        return FragmentLineOutputs(
            energies=self.readout(v),
            fragment_index=index,
            counts=counts,
            fragment_graph=fragment_graph,
            state=state,
            features=features,
        )


def _inverse_or_one(value: float | None) -> torch.Tensor:
    scale: float = 1.0 / float(value) if value is not None and float(value) > 0.0 else 1.0
    return torch.tensor(scale, dtype=torch.get_default_dtype())


# ---------------------------------------------------------------------------
# Settings — one parser for both models, so their configs cannot drift
# ---------------------------------------------------------------------------

_LINE_DEFAULTS: dict[str, dict[str, typing.Any]] = {
    "atom": {"hidden_irreps": "16x0e + 16x1o + 16x2e", "num_layers": 3},
    "pair": {"hidden_irreps": "16x0e + 16x1e + 16x2e", "num_layers": 2},
    "fragment": {
        "hidden_irreps": "16x0e + 16x1o + 16x2e",
        "num_layers": 2,
        "vnode_cutoff": 15.0,
        "fragment_radius": 3.0,
        "avg_num_fragment_neighbors": None,
        "max_fragments": 512,
    },
}


def _merge_section(
    name: str, given: typing.Mapping[str, typing.Any] | None, defaults: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    section: dict[str, typing.Any] = {str(k): v for k, v in dict(given or {}).items()}
    unknown: set[str] = set(section) - set(defaults)
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in {name}; accepted: {sorted(defaults)}."
        )
    return {**defaults, **section}


@dataclass(frozen=True)
class SimplifiedSettings:
    """Validated hyperparameters shared by both SIMURGH Simplified models."""

    cutoff: float
    embedding_dim: int
    num_elements: int
    num_rbf: int
    radial_hidden: int
    sh_lmax: int
    avg_num_neighbors: float | None
    combination: str
    atom_irreps: Irreps
    atom_layers: int
    pair_irreps: Irreps
    pair_layers: int
    fragment_irreps: Irreps
    fragment_layers: int
    vnode_cutoff: float
    fragment_radius: float
    avg_num_fragment_neighbors: float | None
    max_fragments: int | None
    readout_scalars: int
    readout_hidden: int

    @property
    def irreps_sh(self) -> Irreps:
        return build_artisan_edge_irreps(self.sh_lmax)

    @property
    def irreps_sh_even(self) -> Irreps:
        return Irreps([(m, ir) for m, ir in self.irreps_sh if ir.l % 2 == 0])

    @property
    def composition_dim(self) -> int:
        return self.embedding_dim + 1

    @classmethod
    def build(
        cls,
        cutoff: float,
        embedding_dim: int,
        num_elements: int,
        num_rbf: int,
        radial_hidden: int,
        sh_lmax: int,
        avg_num_neighbors: float | None,
        combination: str,
        atom_line: typing.Mapping[str, typing.Any] | None,
        pair_line: typing.Mapping[str, typing.Any] | None,
        fragment_line: typing.Mapping[str, typing.Any] | None,
        readout: typing.Mapping[str, typing.Any] | None,
    ) -> SimplifiedSettings:
        """Validate raw config values and merge in the per-section defaults.

        Parameters
        ----------
        cutoff, embedding_dim, num_elements, num_rbf, radial_hidden, sh_lmax :
            Shared-input hyperparameters (``cutoff`` in Å; ``sh_lmax >= 2``).
        avg_num_neighbors : float, optional
            Atom/pair aggregation normaliser.
        combination : {"additive", "learned"}
            How the line energies are combined.
        atom_line, pair_line, fragment_line, readout : mapping, optional
            Per-section overrides; unknown keys raise, missing keys take the
            defaults.

        Returns
        -------
        SimplifiedSettings
            Frozen settings with every line's irreps checked by
            :func:`validate_line_irreps`.

        Raises
        ------
        ValueError
            On an unknown ``combination``, ``sh_lmax < 2``, an unknown
            section key or invalid line irreps.
        """
        if str(combination) not in ("additive", "learned"):
            raise ValueError(f"combination must be 'additive' or 'learned', got {combination!r}.")
        if int(sh_lmax) < 2:
            raise ValueError(
                f"sh_lmax must be >= 2 (the pair line needs an even l > 0 harmonic), "
                f"got {sh_lmax}."
            )
        atom: dict[str, typing.Any] = _merge_section(
            "atom_line", atom_line, _LINE_DEFAULTS["atom"]
        )
        pair: dict[str, typing.Any] = _merge_section(
            "pair_line", pair_line, _LINE_DEFAULTS["pair"]
        )
        frag: dict[str, typing.Any] = _merge_section(
            "fragment_line", fragment_line, _LINE_DEFAULTS["fragment"]
        )
        head: dict[str, typing.Any] = _merge_section(
            "readout", readout, {"n_scalar": 16, "hidden": 16}
        )
        return cls(
            cutoff=float(cutoff),
            embedding_dim=int(embedding_dim),
            num_elements=int(num_elements),
            num_rbf=int(num_rbf),
            radial_hidden=int(radial_hidden),
            sh_lmax=int(sh_lmax),
            avg_num_neighbors=float(avg_num_neighbors) if avg_num_neighbors is not None else None,
            combination=str(combination),
            atom_irreps=validate_line_irreps(atom["hidden_irreps"], "atom"),
            atom_layers=int(atom["num_layers"]),
            pair_irreps=validate_line_irreps(pair["hidden_irreps"], "pair", even_parity_only=True),
            pair_layers=int(pair["num_layers"]),
            fragment_irreps=validate_line_irreps(frag["hidden_irreps"], "fragment"),
            fragment_layers=int(frag["num_layers"]),
            vnode_cutoff=float(frag["vnode_cutoff"]),
            fragment_radius=float(frag["fragment_radius"]),
            avg_num_fragment_neighbors=(
                float(frag["avg_num_fragment_neighbors"])
                if frag["avg_num_fragment_neighbors"] is not None
                else None
            ),
            max_fragments=int(frag["max_fragments"]) if frag["max_fragments"] else None,
            readout_scalars=int(head["n_scalar"]),
            readout_hidden=int(head["hidden"]),
        )


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class SimurghSimplifiedModel(nn.Module):
    """Shared base: embedding, three lines, readout, E0/scale, autograd forces.

    Subclasses build the three processors and pass them in; nothing else
    differs between the registered models.  Satisfies the
    ``MonolithicModel`` protocol (``head: null``).

    Returns
    -------
    dict
        ``energy`` (B,), ``forces`` (N, 3), ``num_atoms`` (B,), the
        per-line energies ``energy_{atom,pair,fragment}_line`` (B,,
        detached, already multiplied by ``scale`` and, in ``learned``
        mode, by the combination weights — so in both modes
        ``energy == Σ lines + energy_baseline``), ``energy_baseline``
        (B,), and ``aux_loss`` when the fragment processor supplies one.
    """

    atomic_energies: torch.Tensor
    scale: torch.Tensor

    def __init__(
        self,
        settings: SimplifiedSettings,
        atom_processor: AtomLineProcessor,
        pair_processor: PairLineProcessor,
        fragment_processor: FragmentLineProcessor,
        atomic_energies: typing.Mapping[str, typing.Any] | None = None,
        scale: float | None = None,
        num_elements_table: int = 120,
    ) -> None:
        super().__init__()
        self.settings: SimplifiedSettings = settings
        s: SimplifiedSettings = settings
        self.shared_input: SharedInput = SharedInput(
            cutoff=s.cutoff,
            embedding_dim=s.embedding_dim,
            num_elements=s.num_elements,
            num_rbf=s.num_rbf,
            sh_lmax=s.sh_lmax,
        )
        irreps_embedding: Irreps = self.shared_input.irreps_embedding
        self.atom_line: AtomLine = AtomLine(
            processor=atom_processor,
            irreps=s.atom_irreps,
            irreps_embedding=irreps_embedding,
            num_layers=s.atom_layers,
            avg_num_neighbors=s.avg_num_neighbors,
            n_scalar=s.readout_scalars,
            hidden=s.readout_hidden,
        )
        self.pair_line: PairLine = PairLine(
            processor=pair_processor,
            irreps=s.pair_irreps,
            irreps_embedding=irreps_embedding,
            irreps_sh=s.irreps_sh,
            num_layers=s.pair_layers,
            avg_num_neighbors=s.avg_num_neighbors,
            n_scalar=s.readout_scalars,
            hidden=s.readout_hidden,
        )
        self.fragment_line: FragmentLine = FragmentLine(
            processor=fragment_processor,
            irreps=s.fragment_irreps,
            irreps_embedding=irreps_embedding,
            irreps_sh=s.irreps_sh,
            num_layers=s.fragment_layers,
            num_rbf=s.num_rbf,
            radial_hidden=s.radial_hidden,
            vnode_cutoff=s.vnode_cutoff,
            fragment_radius=s.fragment_radius,
            avg_num_fragment_neighbors=s.avg_num_fragment_neighbors,
            max_fragments=s.max_fragments,
            n_scalar=s.readout_scalars,
            hidden=s.readout_hidden,
        )
        # Built only in ``learned`` mode: in ``additive`` mode it would be
        # parameters without a gradient.
        self.combination: LearnedCombination | None = (
            LearnedCombination(s.atom_irreps, s.readout_scalars, s.readout_hidden)
            if s.combination == "learned"
            else None
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
        self.register_buffer(
            "scale",
            torch.tensor(
                float(scale) if scale is not None else 1.0, dtype=torch.get_default_dtype()
            ),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def output_keys(self) -> list[str]:
        return [
            "energy",
            "forces",
            "energy_atom_line",
            "energy_pair_line",
            "energy_fragment_line",
        ]

    @property
    def atomic_energies_mode(self) -> str:
        return self._atomic_energies_mode

    @property
    def cutoff(self) -> float:
        return self.settings.cutoff

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def shared_inputs(self, graph: AtomicGraph) -> SharedInputs:
        """Build the shared input (sets ``requires_grad`` on ``graph.pos``)."""
        fragment_index: torch.Tensor | None = graph.get("fragment_index", None)
        if fragment_index is None:
            raise ValueError(
                "SIMURGH Simplified's fragment line needs a 'fragment_index' on every "
                "graph.  Set data.compute_fragment_index: true in the config (and "
                "rebuild any cached dataset).  There is no fallback without it: all "
                "three lines always run."
            )
        positions: torch.Tensor = graph.pos
        positions.requires_grad_(True)
        edge_vectors, edge_lengths = differentiable_edges(graph, positions)
        batch: torch.Tensor = (
            graph.batch
            if graph.batch is not None
            else torch.zeros(graph.num_atoms, dtype=torch.long, device=positions.device)
        )
        num_graphs: int = int(batch.max().item()) + 1 if batch.numel() else 0
        return self.shared_input(
            atomic_numbers=graph.atomic_numbers,
            edge_index=typing.cast(torch.Tensor, graph.edge_index),
            edge_vectors=edge_vectors,
            edge_lengths=edge_lengths,
            positions=positions,
            fragment_index=fragment_index,
            batch=batch,
            num_graphs=num_graphs,
            cell=graph.get("cell", None),
            pbc=graph.get("pbc", None),
        )

    def _run(
        self, graph: AtomicGraph
    ) -> tuple[dict[str, torch.Tensor], dict[str, list[FeatureRecord]]]:
        inputs: SharedInputs = self.shared_inputs(graph)
        batch: torch.Tensor = inputs.batch
        n_graphs: int = inputs.num_graphs
        row: torch.Tensor = inputs.edge_index[0]

        # ----- The three lines — independent, all unconditional.
        e_atom, h_atom, atom_features = self.atom_line(inputs)
        e_pair, pair_features = self.pair_line(inputs)
        frag: FragmentLineOutputs = self.fragment_line(inputs)
        dtype: torch.dtype = e_atom.dtype

        # ----- Readout / combination.
        if self.combination is None:
            line_totals: torch.Tensor = torch.stack(
                [
                    scatter(e_atom, batch, dim=0, dim_size=n_graphs, reduce="sum"),
                    scatter(0.5 * e_pair, batch[row], dim=0, dim_size=n_graphs, reduce="sum"),
                    scatter(
                        frag.energies, frag.fragment_graph, dim=0, dim_size=n_graphs, reduce="sum"
                    ),
                ],
                dim=-1,
            )  # (B, 3)
        else:
            # Per-atom line energies: each directed edge hands half its
            # energy to its source (both copies of an edge exist), each
            # fragment shares its energy equally among its atoms.
            pair_per_atom: torch.Tensor = torch.zeros_like(e_atom).index_add(0, row, 0.5 * e_pair)
            frag_per_atom: torch.Tensor = (
                frag.energies[frag.fragment_index] / frag.counts[frag.fragment_index, 0]
            )
            per_atom: torch.Tensor = torch.stack([e_atom, pair_per_atom, frag_per_atom], dim=-1)
            line_totals = scatter(
                per_atom * self.combination(h_atom), batch, dim=0, dim_size=n_graphs, reduce="sum"
            )

        scale_v: torch.Tensor = self.scale.to(dtype)
        baseline_atoms: torch.Tensor = self.atomic_energies[inputs.atomic_numbers].to(dtype)
        baseline: torch.Tensor = scatter(
            baseline_atoms, batch, dim=0, dim_size=n_graphs, reduce="sum"
        )
        energy: torch.Tensor = scale_v * line_totals.sum(dim=-1) + baseline  # (B,)

        forces: torch.Tensor = -torch.autograd.grad(
            outputs=energy.sum(),
            inputs=inputs.positions,
            create_graph=self.training,
            retain_graph=True,
        )[0]

        scaled: torch.Tensor = (scale_v * line_totals).detach()
        out: dict[str, torch.Tensor] = {
            "energy": energy,
            "forces": forces,
            "num_atoms": scatter(
                torch.ones_like(baseline_atoms), batch, dim=0, dim_size=n_graphs, reduce="sum"
            ),
            "energy_atom_line": scaled[:, 0],
            "energy_pair_line": scaled[:, 1],
            "energy_fragment_line": scaled[:, 2],
            "energy_baseline": baseline.detach(),
        }
        aux: torch.Tensor | None = self.fragment_line.processor.aux_loss(frag.state)
        if aux is not None:
            out["aux_loss"] = aux
        features: dict[str, list[FeatureRecord]] = {
            "atom": atom_features,
            "pair": pair_features,
            "fragment": frag.features,
        }
        return out, features

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        """Energy and autograd forces for a batch of structures.

        Parameters
        ----------
        graph : AtomicGraph
            Batched graph with ``pos`` (Å), ``atomic_numbers``,
            ``edge_index`` and ``fragment_index``; ``requires_grad`` is set
            on ``graph.pos`` in place.

        Returns
        -------
        dict[str, torch.Tensor]
            As in the class docstring: ``energy`` (B,), ``forces``
            ``-∂E/∂r`` (N, 3), ``num_atoms`` (B,), the detached per-line
            energies, ``energy_baseline`` and, if supplied, ``aux_loss``.
        """
        return self._run(graph)[0]

    def line_features(self, graph: AtomicGraph) -> dict[str, list[FeatureRecord]]:
        """Every intermediate equivariant feature, per line (for diagnostics/tests)."""
        return self._run(graph)[1]
