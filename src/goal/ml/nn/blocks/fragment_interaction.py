"""Equivariant fragment interaction — optional add-on for ARACE rounds.

Message passing reaches only as far as the bonded graph within
``num_rounds x cutoff``.  Two fragments that sit inside the neighbour
cutoff but share no covalent path therefore exchange information only
one bond at a time, through the pair artisans.  This module adds an
explicit fragment-level channel.

It is a **convolution over fragments**, built from the same e3nn
primitives as the pair artisans — not an attention head.  Per round::

    f_k  = mean_{i in k} h_i                      pool full irreps
    c_k  = mean_{i in k} r_i                      fragment centroid
    m_k  = Σ_l TP(f_l, Y^l(r_kl); w(d_kl)) / n_k  equivariant message
    Δh_i = Linear(m_{frag(i)})                    per-atom correction

**Equivariant throughout, at every angular order.**  Pooling is a sum of
features that already share the global frame, so ``f_k`` transforms as
``D^l(R)``.  The tensor product of an equivariant feature with the
spherical harmonics of the inter-centroid direction, weighted by a
radial function of the (invariant) distance, is the standard e3nn
convolution and is equivariant by construction.  Unlike a scalars-only
correction, the output carries ``l = 1`` and ``l = 2`` components, so the
module can tell an atom *where* the other fragment is, not merely what it
looks like.

Why a convolution rather than attention: the fragment count K is
typically 2 (the GMD FragmentDuplication case), where a softmax over two
tokens carries almost no information, and attention logits are invariant
by necessity — they cannot see direction at all.  The radial weights
``w(d_kl)`` already provide a learned, continuous, distance-dependent
weighting, and the spherical harmonics add the direction that attention
would have thrown away.

Locality: fragment pairs are cut off at ``cutoff`` (the model's neighbour
radius by default) and tapered by the same polynomial envelope as the
radial basis, so energies stay local and forces stay smooth.  Without
that, a structure's energy would depend on fragments arbitrarily far away
and would no longer be reproducible under the neighbour-list
approximation used at inference.

Enabled per config section (``model.backbone.fragment_interaction``);
absent section → the ARACE round never builds the module and the forward
path is bit-identical to plain ARACE.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, spherical_harmonics

from goal.ml.nn.blocks.artisans import _PairRMSNorm, build_artisan_edge_irreps
from goal.ml.nn.primitives.linear import EquivariantLinear
from goal.ml.nn.primitives.radial import BesselBasis, PolynomialEnvelope, RadialMLP
from goal.ml.nn.primitives.tp import WeightedTensorProduct


@dataclass
class FragmentGeometry:
    """Geometry of the fragment graph — computed once, shared by all rounds.

    The fragment decomposition depends only on positions, so (exactly like
    the edge spherical harmonics) it is built once per forward and reused
    by every round.
    """

    fragment_index: torch.Tensor
    """(N,) int64 — fragment label per atom."""

    n_fragments: int
    """Number of fragments in the batch (contiguous labels ``0..K-1``)."""

    pair_index: torch.Tensor
    """(2, P) int64 — ``[source, destination]`` fragment pairs within cutoff."""

    pair_sh: torch.Tensor
    """(P, irreps_sh.dim) — spherical harmonics of the source→destination direction."""

    pair_lengths: torch.Tensor
    """(P,) — centroid separations, differentiable w.r.t. positions."""

    centroids: torch.Tensor
    """(K, 3) — fragment centroids, kept for diagnostics."""


def _mic_pair_displacement(
    displacement: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    pair_graph: torch.Tensor | None,
) -> torch.Tensor:
    """Minimum image convention on fragment-pair displacements.

    Each pair may belong to a different structure with a different cell,
    so the wrap is done per pair rather than with one shared cell.  PyG
    collation gives ``cell`` shape ``(G, 3, 3)`` and ``pbc`` shape
    ``(3G,)`` for a batch, and ``(1, 3, 3)`` / ``(3,)`` for a lone graph —
    both are normalised here.  Molecular graphs carry an all-zero
    (singular) cell, which is skipped: inverting it would give NaNs and
    there is nothing to wrap.
    """
    if displacement.numel() == 0:
        return displacement

    cells: torch.Tensor = cell if cell.dim() == 3 else cell.unsqueeze(0)  # (G, 3, 3)
    flags: torch.Tensor = pbc.reshape(-1, 3)  # (G, 3)
    if cells.shape[0] != flags.shape[0]:
        # Defensive: a single cell paired with per-graph flags, or vice versa.
        n_graphs: int = max(cells.shape[0], flags.shape[0])
        cells = cells.expand(n_graphs, 3, 3) if cells.shape[0] == 1 else cells
        flags = flags.expand(n_graphs, 3) if flags.shape[0] == 1 else flags

    index: torch.Tensor = (
        pair_graph
        if pair_graph is not None
        else torch.zeros(displacement.shape[0], dtype=torch.long, device=displacement.device)
    )
    cell_p: torch.Tensor = cells.to(displacement.dtype)[index]  # (P, 3, 3)
    pbc_p: torch.Tensor = flags.to(torch.bool)[index]  # (P, 3)

    active: torch.Tensor = pbc_p.any(dim=-1) & (cell_p.abs().sum(dim=(-2, -1)) > 0)
    if not bool(active.any()):
        return displacement

    # Fractional coordinates, wrap the periodic axes to [-0.5, 0.5), back
    # to Cartesian — the batched form of ``goal.ml.data.graph._apply_mic``.
    inv_cell: torch.Tensor = torch.linalg.inv(cell_p[active])  # (A, 3, 3)
    disp_a: torch.Tensor = displacement[active]
    frac: torch.Tensor = torch.einsum("pi,pji->pj", disp_a, inv_cell)
    frac = torch.where(pbc_p[active], frac - torch.round(frac), frac)
    wrapped: torch.Tensor = torch.einsum("pj,pjk->pk", frac, cell_p[active])

    out: torch.Tensor = displacement.clone()
    out[active] = wrapped
    return out


def build_fragment_geometry(
    positions: torch.Tensor,
    fragment_index: torch.Tensor,
    irreps_sh: Irreps,
    cutoff: float,
    batch: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    max_fragments: int | None = None,
) -> FragmentGeometry:
    """Build the fragment graph: centroids, pairs within *cutoff*, and SH.

    Parameters
    ----------
    positions : Tensor ``(N, 3)``
        Atomic positions.  **Must be the same tensor the energy is
        differentiated against** — the centroids are built from it, so the
        fragment channel contributes to the forces only if the autograd
        chain is intact here.
    fragment_index : Tensor ``(N,)``
        Contiguous fragment labels (``AtomicGraph.__inc__`` guarantees
        contiguity across a batch).
    irreps_sh : Irreps
        Spherical-harmonic irreps for the inter-fragment direction.
    cutoff : float
        Maximum centroid separation for a fragment pair (Angstrom).
    batch : Tensor ``(N,)``, optional
        Per-atom graph index.  Pairs are confined to a single structure,
        so no structure's energy can depend on its batch neighbours.
    cell, pbc : Tensor, optional
        Applied as the minimum image convention to centroid
        displacements.  Note the centroid of a molecule straddling a
        periodic boundary is itself ill-defined; for such systems prefer
        a fragment decomposition that does not wrap, or accept that the
        centroid is approximate.
    max_fragments : int, optional
        Guard against the ``K = N`` case (every atom its own fragment),
        where the pair enumeration is O(K²).  ``None`` disables the check.

    Returns
    -------
    FragmentGeometry
    """
    device: torch.device = positions.device
    dtype: torch.dtype = positions.dtype
    fragment_index = fragment_index.to(device=device, dtype=torch.long)
    n_fragments: int = int(fragment_index.max().item()) + 1 if fragment_index.numel() else 0

    if max_fragments is not None and n_fragments > int(max_fragments):
        raise ValueError(
            f"Fragment interaction: {n_fragments} fragments exceeds "
            f"max_fragments={int(max_fragments)}.  Pair enumeration is O(K²), "
            f"so a structure decomposed into many tiny fragments (e.g. a gas "
            f"of single atoms) gets expensive.  Raise max_fragments, lower "
            f"data.fragment_covalent_cutoff so fewer fragments are found, or "
            f"reduce the pair cutoff."
        )

    # ----- Centroids (differentiable w.r.t. positions) -----
    counts: torch.Tensor = torch.zeros(n_fragments, 1, device=device, dtype=dtype).index_add(
        0, fragment_index, torch.ones(positions.shape[0], 1, device=device, dtype=dtype)
    )
    centroids: torch.Tensor = torch.zeros(n_fragments, 3, device=device, dtype=dtype).index_add(
        0, fragment_index, positions
    ) / counts.clamp(min=1.0)

    # ----- Candidate pairs: every ordered k != l inside one structure -----
    idx: torch.Tensor = torch.arange(n_fragments, device=device)
    src: torch.Tensor = idx.repeat(n_fragments)
    dst: torch.Tensor = idx.repeat_interleave(n_fragments)
    keep: torch.Tensor = src != dst

    if batch is not None and n_fragments > 0:
        frag_graph: torch.Tensor = torch.zeros(n_fragments, dtype=torch.long, device=device)
        frag_graph[fragment_index] = batch.to(device=device, dtype=torch.long)
        keep = keep & (frag_graph[src] == frag_graph[dst])

    src, dst = src[keep], dst[keep]
    displacement: torch.Tensor = centroids[src] - centroids[dst]  # dst receives

    if cell is not None and pbc is not None:
        pair_graph: torch.Tensor | None = None
        if batch is not None and n_fragments > 0:
            frag_graph_full: torch.Tensor = torch.zeros(
                n_fragments, dtype=torch.long, device=device
            )
            frag_graph_full[fragment_index] = batch.to(device=device, dtype=torch.long)
            pair_graph = frag_graph_full[dst]  # every pair lives in one graph
        displacement = _mic_pair_displacement(displacement, cell, pbc, pair_graph)

    lengths: torch.Tensor = displacement.norm(dim=-1)
    within: torch.Tensor = lengths < float(cutoff)
    src, dst, displacement, lengths = (
        src[within],
        dst[within],
        displacement[within],
        lengths[within],
    )

    pair_sh: torch.Tensor = spherical_harmonics(
        irreps_sh, displacement, normalize=True, normalization="component"
    ).to(dtype)

    return FragmentGeometry(
        fragment_index=fragment_index,
        n_fragments=n_fragments,
        pair_index=torch.stack([src, dst], dim=0),
        pair_sh=pair_sh,
        pair_lengths=lengths,
        centroids=centroids,
    )


class EquivariantFragmentInteraction(nn.Module):
    """Fragment-level equivariant convolution producing a full-irreps correction.

    Parameters
    ----------
    irreps_node : Irreps or str
        Node feature irreps.  Both the input pooling and the output
        correction live in this space.
    cutoff : float
        Maximum centroid separation for a fragment pair (Angstrom).
        Defaults to the round's neighbour cutoff, which keeps the model
        local: a fragment beyond it contributes exactly nothing, and the
        polynomial envelope takes the contribution smoothly to zero at the
        boundary so forces stay continuous.
    irreps_hidden : Irreps or str, optional
        Internal width of the fragment convolution.  ``None`` (default)
        reuses ``irreps_node``.
    num_rbf : int
        Bessel radial basis functions for the centroid separation.
    radial_hidden : int
        Hidden width of the radial MLP producing the tensor-product weights.
    num_layers : int
        ``num_layers - 1`` residual ``EquivLinear + RMSNorm`` refinements
        follow the convolution (same idiom as the pair artisans).
    init_zero : bool
        Zero-initialise the output projection so the module starts as an
        exact no-op and learns to activate.  Strongly recommended
        (default ``True``) — enabling it cannot degrade a converged model
        at step 0.
    max_fragments : int, optional
        Safety cap on K (pair enumeration is O(K²)).
    """

    def __init__(
        self,
        irreps_node: Irreps | str,
        cutoff: float,
        irreps_hidden: Irreps | str | None = None,
        num_rbf: int = 8,
        radial_hidden: int = 32,
        num_layers: int = 1,
        init_zero: bool = True,
        max_fragments: int | None = 512,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        self.irreps_node: Irreps = Irreps(irreps_node)
        self.irreps_hidden: Irreps = (
            Irreps(irreps_hidden) if irreps_hidden is not None else self.irreps_node
        )
        self._cutoff: float = float(cutoff)
        self.max_fragments: int | None = int(max_fragments) if max_fragments else None

        lmax: int = max((ir.l for _, ir in self.irreps_hidden), default=0)
        self.irreps_sh: Irreps = build_artisan_edge_irreps(lmax)

        # Pooled fragment features → convolution space (bias-free, so an
        # empty fragment graph produces exactly zero).
        self.fragment_embed: EquivariantLinear = EquivariantLinear(
            self.irreps_node, self.irreps_hidden, biases=False
        )

        # The convolution itself: CG product of the neighbouring fragment's
        # features with the spherical harmonics of the direction to it,
        # weighted by a radial function of the separation.
        self.tp: WeightedTensorProduct = WeightedTensorProduct(
            irreps_in1=self.irreps_hidden,
            irreps_in2=self.irreps_sh,
            irreps_out=self.irreps_hidden,
        )
        self.radial_basis: BesselBasis = BesselBasis(num_basis=num_rbf, cutoff=self._cutoff)
        self.envelope: PolynomialEnvelope = PolynomialEnvelope(cutoff=self._cutoff)
        self.radial_mlp: RadialMLP = RadialMLP(
            num_basis=num_rbf,
            hidden_dim=radial_hidden,
            num_out=self.tp.weight_numel,
        )

        self.layers: nn.ModuleList = nn.ModuleList(
            EquivariantLinear(self.irreps_hidden, self.irreps_hidden, biases=False)
            for _ in range(int(num_layers) - 1)
        )
        self.norms: nn.ModuleList = nn.ModuleList(
            _PairRMSNorm(self.irreps_hidden) for _ in range(int(num_layers) - 1)
        )

        # Fragment message → per-atom correction, in the node irreps.
        self.out_linear: EquivariantLinear = EquivariantLinear(
            self.irreps_hidden, self.irreps_node, biases=False
        )
        if init_zero:
            with torch.no_grad():
                for param in self.out_linear.parameters():
                    param.zero_()

    @property
    def cutoff(self) -> float:
        return self._cutoff

    def build_geometry(
        self,
        positions: torch.Tensor,
        fragment_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        pbc: torch.Tensor | None = None,
    ) -> FragmentGeometry:
        """Convenience wrapper binding this module's ``irreps_sh`` and cutoff."""
        return build_fragment_geometry(
            positions=positions,
            fragment_index=fragment_index,
            irreps_sh=self.irreps_sh,
            cutoff=self._cutoff,
            batch=batch,
            cell=cell,
            pbc=pbc,
            max_fragments=self.max_fragments,
        )

    def forward(self, h: torch.Tensor, geometry: FragmentGeometry) -> torch.Tensor:
        """Per-atom equivariant correction from the neighbouring fragments.

        Parameters
        ----------
        h : Tensor ``(N, irreps_node.dim)``
            Equivariant node features (the pre-update features, the same
            tensor the ACE branch consumes).
        geometry : FragmentGeometry
            Fragment graph for this batch — see
            :func:`build_fragment_geometry`.

        Returns
        -------
        Tensor ``(N, irreps_node.dim)``
            Equivariant correction spanning **all** angular channels.
        """
        device: torch.device = h.device
        dtype: torch.dtype = h.dtype
        fragment_index: torch.Tensor = geometry.fragment_index
        n_fragments: int = geometry.n_fragments

        # ----- Pool node features per fragment (equivariant: the summands
        # already share the global frame).
        counts: torch.Tensor = torch.zeros(n_fragments, 1, device=device, dtype=dtype).index_add(
            0, fragment_index, torch.ones(h.shape[0], 1, device=device, dtype=dtype)
        )
        pooled: torch.Tensor = torch.zeros(
            n_fragments, self.irreps_node.dim, device=device, dtype=dtype
        ).index_add(0, fragment_index, h) / counts.clamp(min=1.0)

        src, dst = geometry.pair_index
        n_pairs: int = int(src.shape[0])

        # ----- Degenerate case: a single fragment, or every pair beyond the
        # cutoff.  There is genuinely nothing to interact with, so the
        # correction is zero — but every parameter must still receive a
        # gradient or DDP's static graph breaks (same contract as the
        # artisan bank's zero-mask schedule).  Run one dummy pair and
        # multiply it out.
        if n_pairs == 0:
            dummy_feats: torch.Tensor = torch.zeros(
                1, self.irreps_hidden.dim, device=device, dtype=dtype
            )
            dummy_sh: torch.Tensor = torch.zeros(1, self.irreps_sh.dim, device=device, dtype=dtype)
            dummy_w: torch.Tensor = self.radial_mlp(
                self.radial_basis(torch.ones(1, device=device, dtype=dtype))
            )
            dummy_msg: torch.Tensor = self.tp(dummy_feats, dummy_sh, dummy_w)
            for lin, norm in zip(self.layers, self.norms):
                dummy_msg = norm(lin(dummy_msg) + dummy_msg)
            dummy: torch.Tensor = self.out_linear(dummy_msg).sum() * 0.0
            # Keep ``fragment_embed`` reachable too.
            dummy = dummy + self.fragment_embed(pooled).sum() * 0.0
            return torch.zeros_like(h) + dummy

        # ----- Radial weights, tapered so the contribution vanishes at the
        # cutoff (forces stay smooth as a fragment drifts across it).
        lengths: torch.Tensor = geometry.pair_lengths.to(dtype).clamp_min(1e-6)
        rbf: torch.Tensor = self.radial_basis(lengths)
        env: torch.Tensor = self.envelope(lengths).to(dtype).unsqueeze(-1)
        weights: torch.Tensor = self.radial_mlp(rbf).to(dtype) * env

        # ----- Equivariant convolution over the fragment graph.
        messages: torch.Tensor = self.tp(
            self.fragment_embed(pooled)[src], geometry.pair_sh.to(dtype), weights
        )
        neighbour_counts: torch.Tensor = torch.zeros(
            n_fragments, 1, device=device, dtype=dtype
        ).index_add(0, dst, torch.ones(n_pairs, 1, device=device, dtype=dtype))
        aggregated: torch.Tensor = torch.zeros(
            n_fragments, self.irreps_hidden.dim, device=device, dtype=dtype
        ).index_add(0, dst, messages) / neighbour_counts.clamp(min=1.0)

        for lin, norm in zip(self.layers, self.norms):
            aggregated = norm(lin(aggregated) + aggregated)

        # ----- Broadcast the fragment message back to its atoms.
        return self.out_linear(aggregated)[fragment_index]

    def extra_repr(self) -> str:
        """Summarise the node / hidden / SH irreps and cutoff for ``repr``."""
        return (
            f"irreps_node={self.irreps_node}, irreps_hidden={self.irreps_hidden}, "
            f"irreps_sh={self.irreps_sh}, cutoff={self._cutoff}"
        )


__all__ = [
    "EquivariantFragmentInteraction",
    "FragmentGeometry",
    "build_fragment_geometry",
]
