"""NF_SIMURGH — Native Fragment SIMURGH.

**NF_SIMURGH = ARACE rounds + fragment virtual nodes.**  A monolithic
research model whose fundamental unit of computation is the *virtual
node*: one node per molecular fragment, carrying that fragment's pooled
representation and its centroid geometry.

It is **not** ARACE with a fragment feature switched on.  There is no key
that disables the virtual-node mechanism and no fallback path that runs
plain ARACE when fragment labels are missing — NF_SIMURGH without virtual
nodes is exactly ``monolithic_arace``, which already exists.  A graph
without ``fragment_index`` raises.

Pipeline::

    1. ARACE rounds (unchanged)      e_atom (N,), h (N, irreps)
    2. Fragment base energy          E_base[k]  = Σ_{i∈k} e_atom[i]
    3. Fragment representation       s_k (invariant), centroid geometry
    4. Fragment-fragment interaction distance-modulated attention (Stage 1)
                                     CG message passing      (Stage 2)
    5. Interaction energy            E_int[k] = MLP(...)      zero-init
    6. Totals                        E_vnode[k] = E_base[k] + E_int[k]
    7. Per graph                     E = Σ_k E_vnode[k]
                                       = Σ_i e_atom[i] + Σ_k E_int[k]
    8. Forces                        F = -∂E/∂r  (autograd)

Two properties carry the design:

* **Exact conservation.**  Step 2 is a plain ``index_add``, so
  ``Σ_k E_base[k] == Σ_i e_atom[i]`` holds *by construction* — summation
  is associative, and nothing is learned or approximated there.  The
  atom-level energy is preserved exactly, not approximately.

* **Only the correction is new physics.**  Step 5 is zero-initialised, so
  a freshly built NF_SIMURGH reproduces ``MonolithicArace`` numerically.
  ``vnode_interaction_energies`` growing away from zero during training is
  the signal that the fragment channel is being used.

Why the invariant collapse is not a symmetry violation: the contract is
that the energy is ``0e`` and the forces are ``1o``.  Force equivariance
follows from energy invariance by differentiating ``E(Rx) = E(x)``, so
the only thing that must be right is that the energy is a true scalar.
Collapsing equivariant features to invariants *at the readout* is the
standard equivariant design — the same thing each ARACE artisan does.
Stage 2 simply moves the collapse later, after fragment-level message
passing, so nothing upstream is starved of directional information.

Isolation
---------
This module is purely additive.  It imports stable primitives, blocks and
:class:`~goal.ml.nn.models.simurgh.arace.AraceBackbone` unchanged and
modifies nothing.  Deleting this file restores the repository exactly.

Registered as ``"nf_simurgh"`` in both ``MODEL_REGISTRY`` and
``BACKBONE_REGISTRY``, mirroring ``monolithic_arace``.  Discovery is
automatic (``goal.ml.nn.models.__init__`` calls ``auto_discover``), so no
registry file needs editing.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, TensorProduct, spherical_harmonics
from torch_geometric.utils import scatter

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.blocks.ace_block import normalise_addon_config
from goal.ml.nn.blocks.artisans import build_artisan_edge_irreps
from goal.ml.nn.blocks.fragment_interaction import build_fragment_geometry
from goal.ml.nn.models.simurgh.arace import AraceBackbone
from goal.ml.nn.models.simurgh.geometry import differentiable_edges
from goal.ml.nn.primitives.linear import EquivariantLinear
from goal.ml.nn.primitives.radial import BesselBasis, PolynomialEnvelope, RadialMLP
from goal.ml.nn.primitives.tp import WeightedTensorProduct
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY

__all__ = [
    "MonolithicNFSimurgh",
    "NFSimurghBackbone",
    "NFSimurghEquivariantVirtualNodeLayer",
    "NFSimurghOutputs",
    "NFSimurghVirtualNodeInteraction",
]

#: Floor on centroid separations and softmax denominators.  ``r_hat`` is
#: undefined at ``d = 0`` and its gradient carries a ``1/d`` factor, so
#: every distance entering a normalisation is clamped by this.
_EPS: float = 1e-6

#: The virtual-node geometry always carries spherical harmonics up to at
#: least ``l = 2``: Stage 1's optional quadrupole invariant needs the
#: ``2e`` block even when Stage 2 is off.
_MIN_SH_LMAX: int = 2


# ---------------------------------------------------------------------------
# Irrep bookkeeping
# ---------------------------------------------------------------------------


def _blocks_with_l(irreps: Irreps, l_value: int) -> tuple[list[slice], int]:
    """Column slices and total multiplicity of every ``l``-block of *irreps*.

    Also enforces **natural parity** (``p = (-1)^l``) on the blocks it
    returns.  This is implementation trap #3 made into an error rather
    than a comment: with ``16x1o`` features the contraction
    ``mu_k · r_hat_kl`` is ``0e``, a true scalar and a valid energy
    contribution.  If the feature irreps ever become ``1e``, that same
    contraction is a **pseudoscalar** — it flips sign under inversion —
    and adding it to an energy silently breaks parity.  Rather than let
    that happen quietly, refuse to build the module.
    """
    natural_parity: int = 1 if l_value % 2 == 0 else -1
    slices: list[slice] = []
    mul_total: int = 0
    for (mul, ir), column_slice in zip(irreps, irreps.slices()):
        if ir.l != l_value:
            continue
        if ir.p != natural_parity:
            raise ValueError(
                f"NF_SIMURGH's virtual-node invariants contract l={l_value} "
                f"features against the spherical harmonics of the "
                f"inter-fragment direction, which requires natural parity "
                f"p=(-1)^l={natural_parity:+d}.  Got {mul}x{ir} instead: that "
                f"contraction would be a pseudoscalar, not a valid energy "
                f"contribution.  Use '{mul}x{l_value}"
                f"{'e' if natural_parity > 0 else 'o'}' in "
                f"artisan.hidden_irreps / virtual_node.vnode_irreps."
            )
        slices.append(column_slice)
        mul_total += mul
    return slices, mul_total


def _gather_l(x: torch.Tensor, slices: typing.Sequence[slice]) -> torch.Tensor:
    """Concatenate the given column slices of ``x`` along the feature axis."""
    if len(slices) == 1:
        return x[:, slices[0]]
    return torch.cat([x[:, s] for s in slices], dim=-1)


# ---------------------------------------------------------------------------
# Fragment (virtual-node) geometry
# ---------------------------------------------------------------------------


@dataclass
class _VNodePairGeometry:
    """Virtual-node graph for one batch — built once, shared by both stages.

    Deliberately carries **no absolute centroid**: see
    :func:`_vnode_pair_geometry`.
    """

    fragment_index: torch.Tensor
    """(N,) int64 — fragment label per atom."""

    n_fragments: int
    """K — number of fragments in the batch (contiguous labels ``0..K-1``)."""

    fragment_graph: torch.Tensor
    """(K,) int64 — which structure each fragment belongs to."""

    pair_k: torch.Tensor
    """(P,) int64 — receiving fragment of each pair."""

    pair_l: torch.Tensor
    """(P,) int64 — sending fragment of each pair."""

    distance: torch.Tensor
    """(P,) — centroid separations, clamped, differentiable.  [INVARIANT]"""

    pair_sh: torch.Tensor
    """(P, irreps_sh.dim) — spherical harmonics of the k → l direction."""

    irreps_sh: Irreps
    """Irreps of :attr:`pair_sh` (``1x0e + 1x1o + 1x2e + ...``)."""

    def sh_block(self, l_value: int) -> torch.Tensor:
        """The ``l``-block of :attr:`pair_sh`, e.g. ``r_hat`` for ``l = 1``."""
        return self.pair_sh[:, self.irreps_sh.slices()[l_value]]

    def sh_prefix(self, lmax: int) -> torch.Tensor:
        """Columns of :attr:`pair_sh` covering ``l = 0 .. lmax``.

        The harmonics are ordered by ascending ``l``, so the requested
        range is a leading prefix — no gather needed.
        """
        return self.pair_sh[:, : build_artisan_edge_irreps(lmax).dim]


def _vnode_pair_geometry(
    positions: torch.Tensor,
    fragment_index: torch.Tensor,
    irreps_sh: Irreps,
    batch: torch.Tensor | None,
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
    max_fragments: int | None,
) -> _VNodePairGeometry:
    """Build the virtual-node graph — **differences only, by construction**.

    Structural guarantee
    --------------------
    Absolute centroids are computed here and *die with this function's
    scope*.  The returned object exposes only pairwise quantities
    (separation, direction), so no downstream code can develop an
    absolute-position dependency even by accident: to do so it would have
    to be written inside this one small function, which is short enough
    to review by eye.  That makes translation invariance a structural
    property rather than a convention someone has to remember.

    Masking reuse
    -------------
    The pair enumeration, the per-structure mask (no fragment ever sees a
    fragment from another graph in the same batch) and the minimum-image
    convention all come from
    :func:`~goal.ml.nn.blocks.fragment_interaction.build_fragment_geometry`
    — v1's implementation, imported unchanged rather than reimplemented.

    Why ``cutoff=inf``
    ------------------
    A structure has a handful of fragments, so the full ordered pair set
    is cheap, and reaching every fragment regardless of separation is the
    entire point of the model — it is exactly the information atom-level
    message passing cannot see past its cutoff.  Two consequences:

    * The pair set becomes purely combinatorial (all ordered ``k != l``
      inside one structure).  It does not depend on geometry, so no pair
      can enter or leave as atoms move.  That keeps the attention softmax
      — whose denominator runs over a fragment's pairs — a *smooth*
      function of the positions.  Truncating the set at a hard radius
      instead would make the denominator jump as a fragment crossed it,
      putting a discontinuity in ``E`` and therefore a wrong ``F``.
    * ``vnode_cutoff`` then enters only through the smooth polynomial
      envelope, which is what makes the interaction decay with
      separation and vanish beyond the cutoff.

    Enumeration is O(K²) in the batch's fragment count; ``max_fragments``
    guards the pathological ``K = N`` case (a gas of single atoms).
    """
    geometry = build_fragment_geometry(
        positions=positions,
        fragment_index=fragment_index,
        irreps_sh=irreps_sh,
        cutoff=float("inf"),
        batch=batch,
        cell=cell,
        pbc=pbc,
        max_fragments=max_fragments,
    )
    # ``geometry.centroids`` is never read, and ``geometry`` itself does not
    # escape this function — only the pairwise fields below do.
    source, destination = geometry.pair_index  # (P,), (P,)

    device: torch.device = positions.device
    n_fragments: int = geometry.n_fragments
    fragment_graph: torch.Tensor = torch.zeros(n_fragments, dtype=torch.long, device=device)
    if batch is not None and n_fragments > 0:
        fragment_graph[geometry.fragment_index] = batch.to(device=device, dtype=torch.long)

    return _VNodePairGeometry(
        fragment_index=geometry.fragment_index,
        n_fragments=n_fragments,
        fragment_graph=fragment_graph,
        pair_k=destination,
        pair_l=source,
        distance=geometry.pair_lengths.clamp_min(_EPS),
        pair_sh=geometry.pair_sh,
        irreps_sh=irreps_sh,
    )


# ---------------------------------------------------------------------------
# Pooling and softmax helpers
# ---------------------------------------------------------------------------


def _fragment_counts(
    fragment_index: torch.Tensor,
    n_fragments: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """(K, 1) atom count per fragment."""
    device: torch.device = fragment_index.device
    return torch.zeros(n_fragments, 1, device=device, dtype=dtype).index_add(
        0,
        fragment_index,
        torch.ones(fragment_index.shape[0], 1, device=device, dtype=dtype),
    )


def _segment_softmax(
    logits: torch.Tensor,
    segment: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Softmax over the rows sharing a segment id, numerically stabilised.

    Used both for attention pooling (segments = fragments, rows = atoms)
    and for the fragment-fragment attention (segments = receiving
    fragments, rows = pairs).  In both cases segment membership is fixed
    by the fragment labelling, not by geometry, so this is smooth in the
    positions.

    A segment with no rows contributes nothing and is never indexed, so
    the zero its ``max``/``sum`` slots hold is harmless; the denominator
    clamp guards only against underflow.
    """
    if logits.shape[0] == 0:
        return logits
    peak: torch.Tensor = scatter(logits, segment, dim=0, dim_size=num_segments, reduce="max")
    weights: torch.Tensor = torch.exp(logits - peak[segment])
    total: torch.Tensor = scatter(weights, segment, dim=0, dim_size=num_segments, reduce="sum")
    return weights / total[segment].clamp_min(_EPS)


# ---------------------------------------------------------------------------
# Stage 1 — invariant virtual-node interaction
# ---------------------------------------------------------------------------


class NFSimurghVirtualNodeInteraction(nn.Module):
    """Stage 1: invariant pooling, centroid geometry, distance-modulated
    attention, zero-init interaction-energy MLP.

    Per fragment ``k`` and every other fragment ``l`` of the same
    structure::

        s_k        = MLP( pool_{i∈k} h_i[l=0]  ‖  log1p(n_k)  [‖ type] )
        rbf_kl     = VNodeRadialBasis(d_kl)          [INVARIANT]
        env_kl     = VNodeEnvelope(d_kl)             [INVARIANT]
        logit_kl   = MLP( s_k ‖ s_l ‖ rbf_kl [‖ l>0 invariants] )
        p_kl       = softmax_l( logit_kl )
        w_kl       = p_kl · env_kl
        context_k  = Σ_l w_kl · s_l
        E_int[k]   = MLP( context_k )                [INVARIANT scalar]

    The envelope multiplies the *normalised* attention, so the distance
    modulates the interaction **strength**, not merely its shape: the
    total weight mass ``Σ_l p_kl · env_kl`` decays as fragments separate
    and reaches exactly zero beyond ``vnode_cutoff``.  Attention over
    chemical summaries alone would keep full strength at any separation,
    which is precisely the failure this model exists to fix.

    Design invariants — deliberately not configurable
    -------------------------------------------------
    * The final readout layer is **zero-initialised**, so the correction
      starts at exactly zero and has to be learned into existence.  There
      is no ``init_zero`` key; the reduction-to-ARACE test depends on it.
    * The whole energy MLP is **bias-free**, so a fragment with no
      neighbours (``context = 0``) contributes exactly zero rather than a
      spurious learned constant per lone fragment.
    * ``pooling`` selects only how the *representation* is pooled.  The
      fragment **energy** pooling in Step 2 is always an exact sum.

    Parameters
    ----------
    irreps_node : Irreps or str
        Node feature irreps leaving the ARACE rounds.
    vnode_cutoff : float
        Separation (Angstrom) at which the envelope reaches zero.  Much
        larger than the atom cutoff; set very large for effectively
        unbounded small molecules.
    pooling : {"mean", "sum", "attention"}
        How the ``l = 0`` channels are pooled per fragment.  ``mean``
        (default) stays scale-consistent across fragments of very
        different sizes, which matters for generalising to fragment sizes
        not seen in training.
    use_size_scalar : bool
        Concatenate ``log1p(n_atoms_k)``.  This resolves the mean-vs-sum
        tension by providing both: the mean answers *what does this stuff
        look like*, the size scalar answers *how much of it is there*,
        without sum's unbounded out-of-distribution scaling.  ``false``
        only for ablation studies.
    n_scalar : int
        Width of the fragment scalar ``s_k``.
    vnode_num_rbf, vnode_radial_hidden : int
        Bessel basis size and hidden width for the fragment radial branch.
        These build a **separate** basis/envelope instance from the atom
        level — same primitives, fragment-appropriate cutoff.
    n_heads : int
        Attention heads.  ``context_k`` is their concatenation.
    final_hidden : int
        Hidden width of the interaction-energy MLP.
    dropout : float
        Dropout inside the attention-logit MLP.  ``0.0`` disables it.
    pool_l1l2 : bool
        Feed pooled ``l = 1`` / ``l = 2`` invariants to the logit MLP.
        **STAGING: remove once Stage 2 is validated.**
    l1l2_invariants : sequence of str
        Which of ``norm`` / ``axial`` / ``mutual`` / ``quad`` to include.
    vnode_type_conditioned : bool
        Condition ``s_k`` on a pooled learned element embedding, so one
        fragment module can specialise by fragment chemistry.  This is the
        continuous, DDP-safe alternative to routing fragments to separate
        experts — the same idea as ``element_conditioned`` on the
        artisans.
    n_elements : int
        Element-embedding table size (must exceed the largest Z).
    """

    def __init__(
        self,
        irreps_node: Irreps | str,
        vnode_cutoff: float,
        pooling: str = "mean",
        use_size_scalar: bool = True,
        n_scalar: int = 16,
        vnode_num_rbf: int = 8,
        vnode_radial_hidden: int = 32,
        n_heads: int = 4,
        final_hidden: int = 16,
        dropout: float = 0.0,
        pool_l1l2: bool = False,
        l1l2_invariants: typing.Sequence[str] = ("norm", "axial", "mutual", "quad"),
        vnode_type_conditioned: bool = False,
        n_elements: int = 120,
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "sum", "attention"):
            raise ValueError(
                f"virtual_node.pooling must be 'mean', 'sum' or 'attention', got {pooling!r}."
            )
        if int(n_heads) < 1:
            raise ValueError(f"virtual_node.n_heads must be >= 1, got {n_heads}.")

        self.irreps_node: Irreps = Irreps(irreps_node)
        self._cutoff: float = float(vnode_cutoff)
        self._pooling: str = str(pooling)
        self._use_size_scalar: bool = bool(use_size_scalar)
        self._n_scalar: int = int(n_scalar)
        self._n_heads: int = int(n_heads)
        self._pool_l1l2: bool = bool(pool_l1l2)
        self._n_elements: int = int(n_elements)
        self._num_rbf: int = int(vnode_num_rbf)

        self._l0_slices, n_l0 = _blocks_with_l(self.irreps_node, 0)
        if n_l0 == 0:
            raise ValueError(
                f"NF_SIMURGH pools the l=0 channels of the node features, but "
                f"{self.irreps_node} has none."
            )

        # ----- Fragment scalar s_k -----
        scalar_in: int = n_l0 + (1 if self._use_size_scalar else 0)
        self.pool_score: nn.Linear | None = (
            nn.Linear(n_l0, 1, bias=False) if self._pooling == "attention" else None
        )
        self.type_embedding: nn.Embedding | None = None
        if vnode_type_conditioned:
            type_dim: int = max(4, self._n_scalar // 2)
            self.type_embedding = nn.Embedding(self._n_elements, type_dim)
            scalar_in += type_dim
        self.scalar_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(scalar_in, self._n_scalar),
            nn.SiLU(),
            nn.Linear(self._n_scalar, self._n_scalar),
        )

        # ----- Fragment radial branch (separate instances from atom level) -----
        self.radial_basis: BesselBasis = BesselBasis(num_basis=self._num_rbf, cutoff=self._cutoff)
        self.envelope: PolynomialEnvelope = PolynomialEnvelope(cutoff=self._cutoff)

        # ----- Optional l>0 pooled invariants (STAGING) -----
        # Every contraction below goes through ``o3.TensorProduct`` rather
        # than a hand-rolled dot product.  e3nn's l=1 components are not in
        # Cartesian (x, y, z) order, so indexing a pooled l=1 feature against
        # a hand-built direction vector is silently NOT invariant; letting
        # e3nn do the bookkeeping makes invariance true by construction.
        self._invariant_names: tuple[str, ...] = tuple(l1l2_invariants)
        self.tp_axial: TensorProduct | None = None
        self.tp_mutual: TensorProduct | None = None
        self.tp_quad: TensorProduct | None = None
        self._l1_slices: list[slice] = []
        self._l2_slices: list[slice] = []
        n_extra: int = 0
        if self._pool_l1l2:
            self._l1_slices, n_l1 = _blocks_with_l(self.irreps_node, 1)
            self._l2_slices, n_l2 = _blocks_with_l(self.irreps_node, 2)
            if "norm" in self._invariant_names and n_l1:
                n_extra += 2 * n_l1
            if "axial" in self._invariant_names and n_l1:
                self.tp_axial = TensorProduct(
                    f"{n_l1}x1o", "1x1o", f"{n_l1}x0e", [(0, 0, 0, "uvu", False)]
                )
                n_extra += 2 * n_l1
            if "mutual" in self._invariant_names and n_l1:
                self.tp_mutual = TensorProduct(
                    f"{n_l1}x1o", f"{n_l1}x1o", f"{n_l1}x0e", [(0, 0, 0, "uuu", False)]
                )
                n_extra += n_l1
            if self.tp_axial is not None and self.tp_mutual is not None:
                # 3·axial_k·axial_l − mutual_kl is the angular part of the
                # dipole-dipole tensor.  Handing the MLP that product
                # explicitly gives it the correct functional form and leaves
                # only the radial dependence to be learned from rbf_kl — a
                # one-hidden-layer MLP does not get a product of two of its
                # own inputs for free.
                n_extra += n_l1
            if "quad" in self._invariant_names and n_l2:
                self.tp_quad = TensorProduct(
                    f"{n_l2}x2e", "1x2e", f"{n_l2}x0e", [(0, 0, 0, "uvu", False)]
                )
                n_extra += 2 * n_l2
        self._n_extra_invariants: int = n_extra

        # ----- Attention over fragment pairs -----
        attn_in: int = 2 * self._n_scalar + self._num_rbf + n_extra
        attn_layers: list[nn.Module] = [
            nn.Linear(attn_in, int(vnode_radial_hidden)),
            nn.SiLU(),
        ]
        if float(dropout) > 0.0:
            attn_layers.append(nn.Dropout(float(dropout)))
        attn_layers.append(nn.Linear(int(vnode_radial_hidden), self._n_heads))
        self.attn_mlp: nn.Sequential = nn.Sequential(*attn_layers)

        # ----- Interaction energy readout -----
        # Bias-free throughout so context = 0 maps to exactly 0, and
        # zero-initialised at the output so the whole correction starts at
        # exactly zero.  Both are design invariants, not options.
        self.energy_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(self._n_heads * self._n_scalar, int(final_hidden), bias=False),
            nn.SiLU(),
            nn.Linear(int(final_hidden), 1, bias=False),
        )
        with torch.no_grad():
            typing.cast(nn.Linear, self.energy_mlp[-1]).weight.zero_()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def n_scalar(self) -> int:
        return self._n_scalar

    # ------------------------------------------------------------------
    # Fragment scalars
    # ------------------------------------------------------------------

    def fragment_scalars(
        self,
        h: torch.Tensor,
        atomic_numbers: torch.Tensor,
        geometry: _VNodePairGeometry,
    ) -> torch.Tensor:
        """Pool the invariant fragment representation ``s_k``, ``(K, n_scalar)``.

        Only the ``l = 0`` channels are pooled here — the representation
        pooling mode applies to scalars alone.  ``l > 0`` channels are
        pooled with **mean only** (see :meth:`_pool_mean_l`), never sum:
        member-atom vectors partially cancel, so a summed result would
        decay with atom count as a pure cancellation artifact rather than
        a physical quantity.
        """
        dtype: torch.dtype = h.dtype
        fragment_index: torch.Tensor = geometry.fragment_index
        n_fragments: int = geometry.n_fragments

        scalars: torch.Tensor = _gather_l(h, self._l0_slices)  # (N, n_l0)
        counts: torch.Tensor = _fragment_counts(fragment_index, n_fragments, dtype)

        pooled: torch.Tensor
        if self._pooling == "attention":
            assert self.pool_score is not None
            weights: torch.Tensor = _segment_softmax(
                self.pool_score(scalars), fragment_index, n_fragments
            )  # (N, 1)
            pooled = torch.zeros(
                n_fragments, scalars.shape[-1], device=h.device, dtype=dtype
            ).index_add(0, fragment_index, scalars * weights)
        else:
            pooled = torch.zeros(
                n_fragments, scalars.shape[-1], device=h.device, dtype=dtype
            ).index_add(0, fragment_index, scalars)
            if self._pooling == "mean":
                pooled = pooled / counts.clamp(min=1.0)

        features: list[torch.Tensor] = [pooled]
        if self._use_size_scalar:
            features.append(torch.log1p(counts))
        if self.type_embedding is not None:
            embedded: torch.Tensor = self.type_embedding(
                atomic_numbers.clamp(0, self._n_elements - 1)
            ).to(dtype)
            pooled_type: torch.Tensor = torch.zeros(
                n_fragments, embedded.shape[-1], device=h.device, dtype=dtype
            ).index_add(0, fragment_index, embedded) / counts.clamp(min=1.0)
            features.append(pooled_type)
        return self.scalar_mlp(torch.cat(features, dim=-1))

    def _pool_mean_l(
        self,
        h: torch.Tensor,
        slices: typing.Sequence[slice],
        geometry: _VNodePairGeometry,
    ) -> torch.Tensor:
        """Mean-pool the given ``l > 0`` blocks per fragment, ``(K, sum(mul)*(2l+1))``."""
        dtype: torch.dtype = h.dtype
        chunk: torch.Tensor = _gather_l(h, slices)
        counts: torch.Tensor = _fragment_counts(
            geometry.fragment_index, geometry.n_fragments, dtype
        )
        pooled: torch.Tensor = torch.zeros(
            geometry.n_fragments, chunk.shape[-1], device=h.device, dtype=dtype
        ).index_add(0, geometry.fragment_index, chunk)
        return pooled / counts.clamp(min=1.0)

    def _pair_invariants(
        self,
        h: torch.Tensor,
        geometry: _VNodePairGeometry,
    ) -> torch.Tensor:
        """Pooled ``l>0`` contractions for every fragment pair, ``(P, n_extra)``.

        STAGING: remove once Stage 2 is validated.
        """
        dtype: torch.dtype = h.dtype
        k, l_idx = geometry.pair_k, geometry.pair_l
        columns: list[torch.Tensor] = []

        if self._l1_slices:
            mu: torch.Tensor = self._pool_mean_l(h, self._l1_slices, geometry)  # (K, 3M)
            r_hat: torch.Tensor = geometry.sh_block(1).to(dtype)  # (P, 3)
            axial_k: torch.Tensor | None = None
            axial_l: torch.Tensor | None = None
            mutual: torch.Tensor | None = None
            if "norm" in self._invariant_names:
                # ‖mu‖ is a sum of squares — invariant regardless of e3nn's
                # component ordering.  Clamped before the sqrt so the
                # gradient stays finite where a fragment's l=1 features
                # vanish by symmetry.
                norms: torch.Tensor = (
                    mu.reshape(mu.shape[0], -1, 3).pow(2).sum(-1).clamp_min(_EPS**2).sqrt()
                )
                columns.extend([norms[k], norms[l_idx]])
            if self.tp_axial is not None:
                axial_k = self.tp_axial(mu[k], r_hat)
                axial_l = self.tp_axial(mu[l_idx], r_hat)
                columns.extend([axial_k, axial_l])
            if self.tp_mutual is not None:
                mutual = self.tp_mutual(mu[k], mu[l_idx])
                columns.append(mutual)
            if axial_k is not None and axial_l is not None and mutual is not None:
                columns.append(3.0 * axial_k * axial_l - mutual)

        if self.tp_quad is not None and self._l2_slices:
            quadrupole: torch.Tensor = self._pool_mean_l(h, self._l2_slices, geometry)
            sh2: torch.Tensor = geometry.sh_block(2).to(dtype)  # (P, 5)
            columns.extend(
                [self.tp_quad(quadrupole[k], sh2), self.tp_quad(quadrupole[l_idx], sh2)]
            )

        if not columns:
            return torch.zeros(k.shape[0], 0, device=h.device, dtype=dtype)
        return torch.cat(columns, dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        h: torch.Tensor,
        atomic_numbers: torch.Tensor,
        geometry: _VNodePairGeometry,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-fragment interaction energy and fragment scalars.

        Returns
        -------
        tuple
            ``(E_int, s)`` with shapes ``(K,)`` and ``(K, n_scalar)``.
            ``s`` is handed to Stage 2 so both stages share one
            representation.
        """
        dtype: torch.dtype = h.dtype
        n_fragments: int = geometry.n_fragments
        scalars: torch.Tensor = self.fragment_scalars(h, atomic_numbers, geometry)
        n_pairs: int = int(geometry.pair_k.shape[0])

        if n_pairs == 0:
            return self._zero_masked(scalars, h.device, dtype, n_fragments), scalars

        k, l_idx = geometry.pair_k, geometry.pair_l
        distance: torch.Tensor = geometry.distance.to(dtype)  # (P,)
        rbf: torch.Tensor = self.radial_basis(distance).to(dtype)  # (P, num_rbf)
        envelope: torch.Tensor = self.envelope(distance).to(dtype).unsqueeze(-1)  # (P, 1)

        logit_input: list[torch.Tensor] = [scalars[k], scalars[l_idx], rbf]
        if self._pool_l1l2:
            logit_input.append(self._pair_invariants(h, geometry))
        logits: torch.Tensor = self.attn_mlp(torch.cat(logit_input, dim=-1))  # (P, H)

        # Normalised attention shape, then distance-modulated magnitude.
        attention: torch.Tensor = _segment_softmax(logits, k, n_fragments) * envelope

        messages: torch.Tensor = attention.unsqueeze(-1) * scalars[l_idx].unsqueeze(
            1
        )  # (P, H, n_scalar)
        context: torch.Tensor = scatter(
            messages, k, dim=0, dim_size=n_fragments, reduce="sum"
        ).reshape(n_fragments, self._n_heads * self._n_scalar)

        return self.energy_mlp(context).squeeze(-1), scalars

    def _zero_masked(
        self,
        scalars: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        n_fragments: int,
    ) -> torch.Tensor:
        """Zero energy for a batch with no fragment pairs, DDP-safely.

        Every structure in the batch has a single fragment, so there is
        genuinely nothing to interact with and the correction is zero.  It
        still has to be *computed* rather than skipped: DDP with
        ``find_unused_parameters: false`` requires every parameter to
        participate in every step, and ``torch.compile`` wants static
        shapes.  Same contract, and the same idiom, as the artisan bank's
        zero-masking schedule — this is a correctness requirement, not an
        optimisation.
        """
        ones: torch.Tensor = torch.ones(1, device=device, dtype=dtype)
        rbf: torch.Tensor = self.radial_basis(ones).to(dtype)
        logit_input: torch.Tensor = torch.cat(
            [
                torch.zeros(1, 2 * self._n_scalar, device=device, dtype=dtype),
                rbf,
                torch.zeros(1, self._n_extra_invariants, device=device, dtype=dtype),
            ],
            dim=-1,
        )
        logits: torch.Tensor = self.attn_mlp(logit_input)
        context: torch.Tensor = (
            torch.zeros(1, self._n_heads * self._n_scalar, device=device, dtype=dtype)
            + logits.sum() * 0.0
        )
        dummy: torch.Tensor = self.energy_mlp(context).sum() * 0.0
        # Keep the pooling / scalar branch reachable too.
        dummy = dummy + scalars.sum() * 0.0
        return torch.zeros(n_fragments, device=device, dtype=dtype) + dummy

    def extra_repr(self) -> str:
        """Show node irreps, vnode cutoff, pooling, scalar width, head count and ``pool_l1l2``."""
        return (
            f"irreps_node={self.irreps_node}, vnode_cutoff={self._cutoff}, "
            f"pooling={self._pooling!r}, n_scalar={self._n_scalar}, "
            f"n_heads={self._n_heads}, pool_l1l2={self._pool_l1l2}"
        )


# ---------------------------------------------------------------------------
# Stage 2 — equivariant virtual nodes  (STAGING: remove once validated)
# ---------------------------------------------------------------------------


class NFSimurghEquivariantVirtualNodeLayer(nn.Module):
    """Stage 2: equivariant fragment virtual nodes, CG-coupled
    fragment-fragment message passing, scalar readout.

    STAGING: remove once Stage 2 is validated — this and Stage 1's
    readout are alternatives to be collapsed into one unconditional path
    once the comparison has been made.

    ::

        H_k   = mean_{i∈k} h_i                     full irreps, equivariant
        m_kl  = TP( H_l, Y(r_hat_kl); w(d_kl) )    CG-coupled message
        H'_k  = H_k + Σ_l gate_kl · m_kl           gate is a scalar
        E[k]  = MLP( scalar_readout(H'_k) )        collapse, last

    Structurally this is the artisan layer one level up: the same
    ``WeightedTensorProduct`` primitive against the same kind of radial
    weights, with fragments in place of atoms.

    Mean pooling for **every** angular order, never sum: member-atom
    ``l > 0`` features partially cancel, so a summed virtual node would
    shrink with fragment size as an artifact of that cancellation rather
    than because of any physics.

    Gates are scalars built from invariants only (``s_k``, ``s_l``,
    ``rbf_kl``) multiplying an equivariant message — never derived from a
    raw ``l > 0`` component, which would break equivariance.  ``H'_k``
    stays equivariant; the collapse to a scalar happens only at the
    readout.

    Exactly **one** round of fragment-level message passing.  Multiple
    rounds are out of scope, and that single-round design is what makes
    the readout-time collapse lossless: with one hop there is no later
    fragment-level layer that could have used the directional information
    the collapse discards.
    """

    def __init__(
        self,
        irreps_node: Irreps | str,
        vnode_irreps: Irreps | str,
        vnode_sh_lmax: int,
        vnode_cutoff: float,
        n_scalar: int = 16,
        vnode_num_rbf: int = 8,
        vnode_radial_hidden: int = 32,
        final_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.irreps_node: Irreps = Irreps(irreps_node)
        self.irreps_vnode: Irreps = Irreps(vnode_irreps)
        self._sh_lmax: int = int(vnode_sh_lmax)
        self.irreps_sh: Irreps = build_artisan_edge_irreps(self._sh_lmax)
        self._cutoff: float = float(vnode_cutoff)
        self._num_rbf: int = int(vnode_num_rbf)

        # Pooled node features → virtual-node space (bias-free, so a zero
        # fragment pools to exactly zero).
        self.vnode_embed: EquivariantLinear = EquivariantLinear(
            self.irreps_node, self.irreps_vnode, biases=False
        )
        self.tp: WeightedTensorProduct = WeightedTensorProduct(
            irreps_in1=self.irreps_vnode,
            irreps_in2=self.irreps_sh,
            irreps_out=self.irreps_vnode,
        )
        self.radial_basis: BesselBasis = BesselBasis(num_basis=self._num_rbf, cutoff=self._cutoff)
        self.envelope: PolynomialEnvelope = PolynomialEnvelope(cutoff=self._cutoff)
        self.radial_mlp: RadialMLP = RadialMLP(
            num_basis=self._num_rbf,
            hidden_dim=int(vnode_radial_hidden),
            num_out=self.tp.weight_numel,
        )
        self.gate_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(2 * int(n_scalar) + self._num_rbf, int(vnode_radial_hidden)),
            nn.SiLU(),
            nn.Linear(int(vnode_radial_hidden), 1),
        )

        self.to_scalars: EquivariantLinear = EquivariantLinear(
            self.irreps_vnode, Irreps(f"{int(n_scalar)}x0e"), biases=False
        )
        self.energy_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(int(n_scalar), int(final_hidden), bias=False),
            nn.SiLU(),
            nn.Linear(int(final_hidden), 1, bias=False),
        )
        # Zero-init: unconditional, exactly as in Stage 1, so the model
        # still reduces to plain ARACE at initialisation with the
        # equivariant path switched on.
        with torch.no_grad():
            typing.cast(nn.Linear, self.energy_mlp[-1]).weight.zero_()

    @property
    def cutoff(self) -> float:
        return self._cutoff

    def forward(
        self,
        h: torch.Tensor,
        scalars: torch.Tensor,
        geometry: _VNodePairGeometry,
    ) -> torch.Tensor:
        """Per-fragment equivariant interaction energy, ``(K,)``."""
        dtype: torch.dtype = h.dtype
        device: torch.device = h.device
        n_fragments: int = geometry.n_fragments

        counts: torch.Tensor = _fragment_counts(geometry.fragment_index, n_fragments, dtype)
        pooled: torch.Tensor = torch.zeros(
            n_fragments, self.irreps_node.dim, device=device, dtype=dtype
        ).index_add(0, geometry.fragment_index, h) / counts.clamp(min=1.0)
        vnodes: torch.Tensor = self.vnode_embed(pooled)  # (K, irreps_vnode.dim)

        n_pairs: int = int(geometry.pair_k.shape[0])
        if n_pairs == 0:
            return self._zero_masked(vnodes, scalars, device, dtype, n_fragments)

        k, l_idx = geometry.pair_k, geometry.pair_l
        distance: torch.Tensor = geometry.distance.to(dtype)
        rbf: torch.Tensor = self.radial_basis(distance)
        envelope: torch.Tensor = self.envelope(distance).to(dtype).unsqueeze(-1)
        weights: torch.Tensor = self.radial_mlp(rbf).to(dtype) * envelope

        messages: torch.Tensor = self.tp(
            vnodes[l_idx], geometry.sh_prefix(self._sh_lmax).to(dtype), weights
        )
        gate: torch.Tensor = self.gate_mlp(
            torch.cat([scalars[k], scalars[l_idx], rbf.to(dtype)], dim=-1)
        )  # (P, 1) — invariant scalars only
        aggregated: torch.Tensor = scatter(
            gate * messages, k, dim=0, dim_size=n_fragments, reduce="sum"
        )

        updated: torch.Tensor = vnodes + aggregated  # still equivariant
        return self.energy_mlp(self.to_scalars(updated)).squeeze(-1)

    def _zero_masked(
        self,
        vnodes: torch.Tensor,
        scalars: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        n_fragments: int,
    ) -> torch.Tensor:
        """Zero energy with every parameter still reachable — see Stage 1."""
        ones: torch.Tensor = torch.ones(1, device=device, dtype=dtype)
        rbf: torch.Tensor = self.radial_basis(ones)
        weights: torch.Tensor = self.radial_mlp(rbf).to(dtype)
        message: torch.Tensor = self.tp(
            torch.zeros(1, self.irreps_vnode.dim, device=device, dtype=dtype),
            torch.zeros(1, self.irreps_sh.dim, device=device, dtype=dtype),
            weights,
        )
        gate: torch.Tensor = self.gate_mlp(
            torch.cat(
                [
                    torch.zeros(1, 2 * scalars.shape[-1], device=device, dtype=dtype),
                    rbf.to(dtype),
                ],
                dim=-1,
            )
        )
        updated: torch.Tensor = vnodes[:1] * 0.0 + (message * gate) * 0.0
        dummy: torch.Tensor = self.energy_mlp(self.to_scalars(updated)).sum() * 0.0
        dummy = dummy + vnodes.sum() * 0.0
        return torch.zeros(n_fragments, device=device, dtype=dtype) + dummy

    def extra_repr(self) -> str:
        """Show the virtual-node irreps, the harmonic irreps and the virtual-node cutoff."""
        return (
            f"irreps_vnode={self.irreps_vnode}, irreps_sh={self.irreps_sh}, "
            f"vnode_cutoff={self._cutoff}"
        )


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


@dataclass
class NFSimurghOutputs:
    """Everything the backbone produces, before ``scale`` and the baseline.

    Kept as raw, un-scaled tensors so the conservation identity
    (``vnode_base.sum() == node_energies.sum()``) can be checked directly.
    """

    node_energies: torch.Tensor
    """(N,) summed per-atom artisan energy across rounds."""

    layer_energies: torch.Tensor
    """(L, N) per-round per-atom contributions."""

    vnode_base: torch.Tensor
    """(K,) exact per-fragment sum of ``node_energies``."""

    vnode_interaction: torch.Tensor
    """(K,) the learned fragment-fragment correction, zero at init."""

    fragment_graph: torch.Tensor
    """(K,) int64 — structure each fragment belongs to."""

    n_fragments: int
    """K."""


class NFSimurghBackbone(nn.Module):
    """Runs standard ARACE rounds unchanged, then adds the
    fragment-base + fragment-interaction energy correction.

    The ARACE stack is :class:`~goal.ml.nn.models.simurgh.arace.AraceBackbone`
    imported as-is with both of its optional add-ons off, so the rounds
    are bit-identical to ``monolithic_arace``'s and per-round energies
    stay available as ``energy_round_{L}``.

    Construction order matters: the ARACE stack is built **before** the
    virtual-node modules so that, under the same seed, its parameters draw
    exactly the same random numbers as a standalone ``MonolithicArace``.
    That is what makes the reduction-to-ARACE comparison exact rather than
    merely close.

    Parameters
    ----------
    virtual_node : dict, optional
        Hyperparameters of the virtual-node mechanism.  Every key tunes a
        mechanism that **always runs**; none of them switches it off.
        ``None`` or an empty section means "use the defaults", *not*
        "disable" — there is no way to disable it.
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
        virtual_node: dict[str, typing.Any] | None = None,
        correlation: int | None = None,
    ) -> None:
        super().__init__()
        artisan_cfg: dict[str, typing.Any] | None = (
            {str(k): v for k, v in dict(artisan).items()} if artisan is not None else None
        )
        # Built first — see the class docstring on RNG ordering.
        self.arace: AraceBackbone = AraceBackbone(
            elements=elements,
            num_rounds=num_rounds,
            share_artisan_weights=share_artisan_weights,
            artisan=artisan_cfg,
            cutoff=cutoff,
            embedding_dim=embedding_dim,
            num_elements=num_elements,
            avg_num_neighbors=avg_num_neighbors,
            fragment_interaction=None,
            adaptive_gate=None,
            correlation=correlation,
        )
        irreps_node: Irreps = self.arace.irreps_out

        vn: dict[str, typing.Any] = normalise_addon_config(virtual_node) or {}
        for forbidden in (
            "use_fragments",
            "enable_virtual_nodes",
            "use_fragment_ca",
            "enabled",
            "init_zero",
        ):
            if forbidden in vn:
                raise ValueError(
                    f"virtual_node.{forbidden} is not a NF_SIMURGH key.  The "
                    f"virtual-node mechanism is mandatory and its zero-init is a "
                    f"design invariant — NF_SIMURGH without virtual nodes is "
                    f"'monolithic_arace', which already exists."
                )

        self._vnode_equivariant: bool = bool(vn.get("vnode_equivariant", False))
        self._vnode_sh_lmax: int = int(vn.get("vnode_sh_lmax", 2))
        self._max_fragments: int | None = (
            int(vn["vnode_max_fragments"]) if vn.get("vnode_max_fragments") else 512
        )
        vnode_cutoff: float = float(vn.get("vnode_cutoff", 15.0))
        n_scalar: int = int(vn.get("n_scalar", 16))

        self.vnode_interaction: NFSimurghVirtualNodeInteraction = NFSimurghVirtualNodeInteraction(
            irreps_node=irreps_node,
            vnode_cutoff=vnode_cutoff,
            pooling=str(vn.get("pooling", "mean")),
            use_size_scalar=bool(vn.get("use_size_scalar", True)),
            n_scalar=n_scalar,
            vnode_num_rbf=int(vn.get("vnode_num_rbf", 8)),
            vnode_radial_hidden=int(vn.get("vnode_radial_hidden", 32)),
            n_heads=int(vn.get("n_heads", 4)),
            final_hidden=int(vn.get("final_hidden", 16)),
            dropout=float(vn.get("dropout", 0.0)),
            pool_l1l2=bool(vn.get("pool_l1l2", False)),
            l1l2_invariants=tuple(vn.get("l1l2_invariants", ("norm", "axial", "mutual", "quad"))),
            vnode_type_conditioned=bool(vn.get("vnode_type_conditioned", False)),
            n_elements=int(num_elements),
        )

        # STAGING: remove once Stage 2 is validated.
        #
        # Stage 2 is *additive* rather than a replacement for Stage 1's
        # readout.  Replacing it would leave Stage 1's attention and energy
        # MLP without gradients whenever the flag is on, which breaks DDP's
        # static graph for exactly the reason the zero-masking schedule
        # exists.  Additive keeps both paths live, and the difference
        # between flag settings then isolates Stage 2's own contribution —
        # which is the point of staging.
        self.vnode_equivariant_layer: NFSimurghEquivariantVirtualNodeLayer | None = None
        if self._vnode_equivariant:
            self.vnode_equivariant_layer = NFSimurghEquivariantVirtualNodeLayer(
                irreps_node=irreps_node,
                vnode_irreps=str(vn.get("vnode_irreps", str(irreps_node))),
                vnode_sh_lmax=self._vnode_sh_lmax,
                vnode_cutoff=vnode_cutoff,
                n_scalar=n_scalar,
                vnode_num_rbf=int(vn.get("vnode_num_rbf", 8)),
                vnode_radial_hidden=int(vn.get("vnode_radial_hidden", 32)),
                final_hidden=int(vn.get("final_hidden", 16)),
            )

        # One SH expansion shared by both stages: wide enough for Stage 1's
        # quadrupole invariant and for Stage 2's message passing.
        self.irreps_sh: Irreps = build_artisan_edge_irreps(max(_MIN_SH_LMAX, self._vnode_sh_lmax))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def elements(self) -> tuple[int, ...]:
        return self.arace.elements

    @property
    def num_rounds(self) -> int:
        return self.arace.num_rounds

    @property
    def share_artisan_weights(self) -> bool:
        return self.arace.share_artisan_weights

    @property
    def irreps_out(self) -> Irreps:
        return self.arace.irreps_out

    @property
    def cutoff(self) -> float:
        return self.arace.cutoff

    @property
    def vnode_equivariant(self) -> bool:
        """Whether the Stage 2 equivariant virtual-node layer is built."""
        return self._vnode_equivariant

    def gates(self) -> dict[str, torch.Tensor]:
        """Snapshot of every artisan gate, keyed ``layer{L}/{pair}``."""
        return self.arace.gates()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_lengths: torch.Tensor,
        positions: torch.Tensor,
        fragment_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        pbc: torch.Tensor | None = None,
    ) -> NFSimurghOutputs:
        """Run the ARACE rounds, then pool and correct on virtual nodes.

        ``positions`` must be the tensor the energy is differentiated
        against — the centroids are built from it, so the correction
        contributes to the forces only if that autograd chain is intact.
        """
        node_energies: torch.Tensor
        layer_energies: torch.Tensor
        h: torch.Tensor
        node_energies, layer_energies, h = self._run_rounds(
            atomic_numbers, edge_index, edge_vectors, edge_lengths
        )

        geometry: _VNodePairGeometry = _vnode_pair_geometry(
            positions=positions,
            fragment_index=fragment_index,
            irreps_sh=self.irreps_sh,
            batch=batch,
            cell=cell,
            pbc=pbc,
            max_fragments=self._max_fragments,
        )

        # ----- Step 2: fragment base energy.  Exact by construction.
        # A plain index_add: Σ_k E_base[k] == Σ_i e_atom[i] to the last bit,
        # because summation is associative.  Not learned, not approximate,
        # not configurable.
        vnode_base: torch.Tensor = torch.zeros(
            geometry.n_fragments, device=node_energies.device, dtype=node_energies.dtype
        ).index_add(0, geometry.fragment_index, node_energies)

        # ----- Steps 3-5: the correction.
        interaction, scalars = self.vnode_interaction(h, atomic_numbers, geometry)
        if self.vnode_equivariant_layer is not None:
            interaction = interaction + self.vnode_equivariant_layer(h, scalars, geometry)

        return NFSimurghOutputs(
            node_energies=node_energies,
            layer_energies=layer_energies,
            vnode_base=vnode_base,
            vnode_interaction=interaction,
            fragment_graph=geometry.fragment_graph,
            n_fragments=geometry.n_fragments,
        )

    def _run_rounds(
        self,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the ARACE rounds, returning energies **and** node features.

        # vendored from models/simurgh/arace.py::AraceBackbone.forward
        # @ 152d8e8, modified: also returns the final node features ``h``.
        #
        # Only the loop is vendored — the embedding, the input linear and
        # every round are ``self.arace``'s own modules, used unchanged.  The
        # arithmetic is therefore identical to ``monolithic_arace``'s, which
        # is what keeps the reduction-to-ARACE comparison exact.
        #
        # Why vendor at all: ``AraceBackbone.forward`` returns energies but
        # discards ``h``, and NF_SIMURGH pools ``h`` onto the virtual nodes.
        # Calling it and then replaying the rounds to recover ``h`` would run
        # the whole artisan stack twice per step.  Adding an ``h``-returning
        # flag to the shared class is the other way out, and the one this
        # model is not allowed to take.

        Returns
        -------
        tuple
            ``(node_energies (N,), layer_energies (L, N), h (N, irreps))``.
        """
        h: torch.Tensor = self.arace.input_linear(self.arace.embedding(atomic_numbers))

        # Edge SH — geometry only, computed once and shared by every round.
        edge_sh: torch.Tensor = spherical_harmonics(
            self.arace.irreps_edge,
            edge_vectors.to(h.dtype),
            normalize=True,
            normalization="component",
        )

        per_layer: list[torch.Tensor] = []
        for block in self.arace.blocks:
            h, e_layer, _gate, _aux = block(
                h,
                atomic_numbers,
                edge_index,
                edge_sh,
                edge_lengths,
                fragment_geometry=None,
            )
            per_layer.append(e_layer)

        layer_energies: torch.Tensor = torch.stack(per_layer, dim=0)  # (L, N)
        return layer_energies.sum(dim=0), layer_energies, h


# ---------------------------------------------------------------------------
# Monolithic model
# ---------------------------------------------------------------------------


@MODEL_REGISTRY.register("nf_simurgh")
@BACKBONE_REGISTRY.register("nf_simurgh")
class MonolithicNFSimurgh(nn.Module):
    """Self-contained NF_SIMURGH model.

    Satisfies the ``MonolithicModel`` protocol exactly as
    ``MonolithicArace`` does — returns the property dict directly.
    Configure with ``head: null``.

    ``E_total = E_atomic + scale × (Σ_i e_atom_i + Σ_k E_int_k)`` and
    ``F = -∂E_total/∂r`` via autograd.

    The correction sits **inside** ``scale``, alongside the artisan
    energy: ``scale`` is the ScaleShift gain that maps the network's
    dimensionless interaction output onto the physical spread of
    ``E - baseline``, and the fragment correction is an interaction
    energy in exactly that sense.  Keeping it inside also means zero-init
    reduces to plain ARACE regardless of what ``scale`` happens to be.

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model.  **Unlike ``monolithic_arace``
        this must be set in the config**: ``train.py``'s auto-extraction
        from the training set is keyed on a hard-coded list of backbone
        names that does not include ``nf_simurgh``.
    num_rounds, share_artisan_weights, artisan, cutoff, embedding_dim,
    num_elements, avg_num_neighbors :
        As :class:`~goal.ml.nn.models.simurgh.arace.MonolithicArace`.
    virtual_node : dict, optional
        Virtual-node hyperparameters — see :class:`NFSimurghBackbone`.
        Cannot disable the mechanism.
    atomic_energies : dict, optional
        Per-element baseline, same three modes as ``MonolithicArace``:
        ``learned`` (default, ``nn.Parameter``), ``dataset`` / ``provided``
        (fixed buffer seeded from ``values``).
    scale : float, optional
        Multiplicative gain on the interaction energy.
    num_elements_table : int
        Size of the Z-indexed atomic-energy parameter/buffer.
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
        virtual_node: dict[str, typing.Any] | None = None,
        atomic_energies: dict[str, typing.Any] | None = None,
        scale: float | None = None,
        num_elements_table: int = 120,
        correlation: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone: NFSimurghBackbone = NFSimurghBackbone(
            elements=elements,
            num_rounds=num_rounds,
            share_artisan_weights=share_artisan_weights,
            artisan=artisan,
            cutoff=cutoff,
            embedding_dim=embedding_dim,
            num_elements=num_elements,
            avg_num_neighbors=(
                float(avg_num_neighbors) if avg_num_neighbors is not None else None
            ),
            virtual_node=virtual_node,
            correlation=correlation,
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
        """Keys this model produces.

        ``vnode_interaction_energies`` is the one to watch: it is
        numerically zero at initialisation, and its growth during training
        is the signal that the fragment channel is being used at all.
        """
        return [
            "energy",
            "forces",
            "vnode_energies",
            "vnode_interaction_energies",
        ]

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

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        """Forward pass returning energy, forces and virtual-node diagnostics.

        Raises
        ------
        ValueError
            When the graph carries no ``fragment_index``.  There is no
            fallback to an atom-level-only path: silently degrading to
            plain ARACE would produce entirely plausible numbers from the
            wrong model, with nothing in the logs to say so.
        """
        fragment_index: torch.Tensor | None = graph.get("fragment_index", None)
        if fragment_index is None:
            raise ValueError(
                "NF_SIMURGH computes on fragment virtual nodes, so every graph "
                "must carry a 'fragment_index'.  Set data.compute_fragment_index: "
                "true in the config (and rebuild any cached dataset) so the "
                "fragment labels are attached at load time.  There is no "
                "atom-level-only fallback: that would be 'monolithic_arace' "
                "wearing this model's name."
            )

        positions: torch.Tensor = graph.pos
        positions.requires_grad_(True)

        edge_vectors: torch.Tensor
        edge_lengths: torch.Tensor
        edge_vectors, edge_lengths = differentiable_edges(graph, positions)
        edge_index: torch.Tensor = typing.cast(torch.Tensor, graph.edge_index)

        batch: torch.Tensor = (
            graph.batch
            if graph.batch is not None
            else torch.zeros(graph.num_atoms, dtype=torch.long, device=positions.device)
        )
        num_graphs: int = int(batch.max().item()) + 1 if batch.numel() else 0

        outputs: NFSimurghOutputs = self.backbone(
            atomic_numbers=graph.atomic_numbers,
            edge_index=edge_index,
            edge_vectors=edge_vectors,
            edge_lengths=edge_lengths,
            positions=positions,
            fragment_index=fragment_index,
            batch=batch,
            cell=graph.get("cell", None),
            pbc=graph.get("pbc", None),
        )

        dtype: torch.dtype = outputs.node_energies.dtype
        scale_v: torch.Tensor = self.scale.to(dtype)

        # ----- Steps 6-7: totals.
        # E_vnode_total[k] = E_base[k] + E_int[k], summed per structure.
        # The base term collapses back to exactly Σ_i e_atom[i]; the only
        # new physics is the interaction correction.
        vnode_total: torch.Tensor = outputs.vnode_base + outputs.vnode_interaction
        interaction_energy: torch.Tensor = scatter(
            vnode_total,
            outputs.fragment_graph,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )  # (B,)

        baseline: torch.Tensor = self.atomic_energies[graph.atomic_numbers].to(dtype)
        energy: torch.Tensor = scale_v * interaction_energy + scatter(
            baseline, batch, dim=0, dim_size=num_graphs, reduce="sum"
        )  # (B,)
        num_atoms: torch.Tensor = scatter(
            torch.ones_like(baseline), batch, dim=0, dim_size=num_graphs, reduce="sum"
        )  # (B,)

        # Per-round energy totals (L, B) — the existing collapsed-round
        # diagnostic, unchanged, logged as energy_round_{L}.
        layer_totals: torch.Tensor = scatter(
            scale_v * outputs.layer_energies,
            batch,
            dim=1,
            dim_size=num_graphs,
            reduce="sum",
        ).detach()  # (L, B)

        grad_outputs: tuple[torch.Tensor, ...] = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=positions,
            create_graph=self.training,
            retain_graph=True,
        )
        forces: torch.Tensor = -grad_outputs[0]  # (N, 3)

        vnode_interaction: torch.Tensor = (scale_v * outputs.vnode_interaction).detach()
        # Reported against the *learned* energy, i.e. E_total minus the fixed
        # per-element baseline.  Against E_total itself this ratio would be
        # ~1e-6 for any dataset with real atomic references (O is about
        # -2041 eV) and would tell nobody anything; the interesting question
        # is what fraction of the energy the model actually predicts comes
        # from the fragment channel.
        learned_energy: torch.Tensor = (scale_v * interaction_energy).detach()

        return {
            "energy": energy,
            "forces": forces,
            "num_atoms": num_atoms,
            "layer_energies": layer_totals,
            "vnode_energies": (scale_v * vnode_total).detach(),
            "vnode_interaction_energies": vnode_interaction,
            "vnode_interaction_abs_mean": vnode_interaction.abs().mean(),
            "vnode_interaction_frac": (
                vnode_interaction.abs().sum() / learned_energy.abs().sum().clamp_min(_EPS)
            ),
        }
