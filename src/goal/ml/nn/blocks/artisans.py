"""SIMURGH element-pair bank of potential artisans.

For a configurable set of chemical elements ``E = {Z_1, ..., Z_K}`` this
module instantiates one artisan ``E_{AB}`` for every *unordered* pair of
elements ``(A, B)`` (so ``K * (K + 1) // 2`` artisans in total).

Each artisan outputs a single scalar energy contribution per pair which
is multiplied by a learnable scalar gate ``P_{AB}`` and smoothly tapered
by a cosine cutoff envelope ``f_cut(d)`` so forces are continuous at the
cutoff.

Two artisan *architectures* are configurable from Hydra
(``architecture`` key):

* ``"scalar"`` (default) — each atom's features are projected to
  invariant scalars (``o3.Linear`` keeping only ``l = 0`` outputs);
  the artisan reads ``[scalars_A, scalars_B, d_ij]``.  Two scalar
  backbones exist (``expert_type`` key):

  * ``"linear"`` — ``Linear → SiLU → Linear → SiLU → Linear``;
  * ``"transformer"`` — an *equivariance-safe* mini-transformer that
    operates exclusively on the **invariant scalar** branch, so the
    E(3) symmetry of the model is preserved.

* ``"equivariant"`` — the artisan operates on the full equivariant
  node features.  Both endpoints are mapped through a **shared**
  equivariant linear and summed (symmetric by construction, so
  ``E(A, B) = E(B, A)``), combined with the bond direction via a CG
  tensor product weighted by a radial MLP of the bond length, and
  finally read out through invariant scalars.  See
  :class:`_EquivariantArtisanCore`.

Static-shape zero-masking schedule
----------------------------------
The block follows a strict **input-masking, output-masking** schedule
for DDP safety:

1. For every expert ``E_{AB}`` we build a per-edge mask
   ``m[e] ∈ {0.0, 1.0}`` that selects edges of pair type ``(A, B)``.
2. The *inputs* (scalars_A, scalars_B, distance) are multiplied by the
   mask **before** the expert forward pass — there is no ``if/else``
   that skips an expert.
3. The expert output ``E_{AB}(masked_inputs)`` is multiplied by the
   mask **again** before being added to ``atom_energies``, so the
   contribution from non-matching edges is exactly ``0.0``.

This keeps the autograd graph perfectly static across all ranks (no
DDP deadlock, no ``find_unused_parameters``, no stale AdamW moments),
and every expert touches a deterministic number of edges every step.
The only branch left is the (very rare) ``E == 0`` degenerate case
where the whole batch carries no edges at all; we then run each
expert once on a 1-edge dummy with mask = 0 so its parameters remain
reachable from the loss.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.o3 import Irreps

from goal.ml.nn.primitives.linear import EquivariantLinear
from goal.ml.nn.primitives.radial import BesselBasis, PolynomialEnvelope, RadialMLP
from goal.ml.nn.primitives.tp import WeightedTensorProduct

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def enumerate_element_pairs(elements: typing.Sequence[int]) -> list[tuple[int, int]]:
    """Return every unordered pair ``(A, B)`` for the given elements, including
    self-pairs.  Order is deterministic: ``(A, B)`` with ``A <= B``.

    Example
    -------
    >>> enumerate_element_pairs([1, 6, 7, 8])
    [(1, 1), (1, 6), (1, 7), (1, 8), (6, 6), (6, 7), (6, 8), (7, 7), (7, 8), (8, 8)]
    """
    sorted_el: list[int] = sorted(elements)
    pairs: list[tuple[int, int]] = []
    for i, a in enumerate(sorted_el):
        for b in sorted_el[i:]:
            pairs.append((a, b))
    return pairs


#: Width of the element-conditioning tables when ``n_elements`` is not
#: given.  Kept for checkpoints whose saved config predates the key: they
#: were trained with this size and must rebuild with it to load.
LEGACY_N_ELEMENTS: int = 120


def resolve_n_elements(value: typing.Any, elements: typing.Sequence[int]) -> int:
    """Size of an artisan's element-conditioning table.

    The tables (``element_linear``, the rare artisan's ``pair_embed``) are
    indexed by **atomic number** — ``one_hot(Z)`` / ``embedding(Z)`` — so
    they need ``max(Z) + 1`` rows, not one row per distinct element:
    H/C/O needs 9 rows, not 3.  Rows for atomic numbers that never occur
    receive no gradient and are dead parameters.

    Parameters
    ----------
    value : int, "auto" or None
        ``"auto"`` → ``max(elements) + 1`` (the smallest table that covers
        every declared element).  ``None`` → :data:`LEGACY_N_ELEMENTS`.
        An int is used as given and must exceed ``max(elements)``.
    elements : sequence of int
        The backbone's declared elements.

    Raises
    ------
    ValueError
        For an int that cannot index every declared element, or an
        unknown string.
    """
    max_z: int = max((int(z) for z in elements), default=0)
    if value is None:
        return LEGACY_N_ELEMENTS
    if isinstance(value, str):
        if value.strip().lower() != "auto":
            raise ValueError(f"artisan.n_elements must be an int or 'auto', got {value!r}.")
        return max_z + 1
    n: int = int(value)
    if n <= max_z:
        raise ValueError(
            f"artisan.n_elements={n} cannot index atomic number {max_z}: the "
            f"conditioning tables are looked up by Z, so they need at least "
            f"max(Z) + 1 = {max_z + 1} rows.  Use 'auto' to size them exactly."
        )
    return n


def pair_label(a: int, b: int, symbols: typing.Mapping[int, str] | None = None) -> str:
    """Pretty label for a pair, e.g. ``"H-C"`` (used in logging)."""
    sym: typing.Mapping[int, str] = symbols or _DEFAULT_SYMBOLS
    lo, hi = sorted((a, b))
    return f"{sym.get(lo, str(lo))}-{sym.get(hi, str(hi))}"


_DEFAULT_SYMBOLS: dict[int, str] = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


def cosine_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Smooth cosine cutoff envelope, ``f(d) = 0.5 (cos(pi d / r_c) + 1)``.

    Zero outside ``cutoff``.  ``distances`` of shape ``(...,)``; output
    same shape.
    """
    mask: torch.Tensor = (distances < cutoff).to(distances.dtype)
    arg: torch.Tensor = math.pi * (distances / cutoff)
    return 0.5 * (torch.cos(arg) + 1.0) * mask


# ---------------------------------------------------------------------------
# Per-expert backbones
# ---------------------------------------------------------------------------


@dataclass
class ArtisanConfig:
    """Static configuration carried by every pairwise expert.

    Attributes
    ----------
    architecture : str
        ``"scalar"`` (default — invariant-scalar artisan, original
        behaviour) or ``"equivariant"`` (full equivariant artisan, see
        :class:`_EquivariantArtisanCore`).
    scalar_channels : int
        Width of the invariant projection of each atom's features
        (``"scalar"`` architecture only).
    hidden_dims : tuple of int
        Hidden widths of the artisan MLP (excluding input + output).
    expert_type : str
        ``"linear"`` (default) or ``"transformer"`` — scalar-architecture
        backbone selector.
    transformer_heads : int
        Multi-head attention heads when ``expert_type == "transformer"``.
    transformer_layers : int
        Number of transformer encoder layers.
    dropout_rate : float
        Dropout probability inserted between every ``Linear`` layer in
        the ``"linear"`` backbone MLP.  ``0.0`` disables dropout (default).
        ``nn.Dropout`` is used so dropout is automatically disabled during
        ``model.eval()`` / validation and inference.
    equivariant : dict, optional
        Sub-config for ``architecture="equivariant"`` forwarded to
        :class:`_EquivariantArtisanCore` — keys ``hidden_irreps``,
        ``num_layers``, ``num_rbf``, ``radial_hidden``, ``n_scalar_out``,
        ``final_hidden``, ``element_conditioned``, ``n_elements``.
        ``None`` uses the core's defaults.
    """

    scalar_channels: int = 16
    hidden_dims: tuple[int, ...] = (64, 32)
    expert_type: str = "linear"
    transformer_heads: int = 2
    transformer_layers: int = 1
    dropout_rate: float = 0.0
    # Rare-pair artisan (CHANGE 4)
    rare_pair_embed_dim: int = 16
    # Artisan architecture: "scalar" (default) or "equivariant"
    architecture: str = "scalar"
    equivariant: dict[str, typing.Any] | None = None


class _LinearExpert(nn.Module):
    """Standard MLP expert ``[in] → SiLU → [Dropout] → ... → 1``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: typing.Sequence[int],
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev: int = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        """``(P, in_dim) → (P,)``."""
        return self.net(pair_features).squeeze(-1)


class _TransformerExpert(nn.Module):
    """E(3)-safe transformer expert.

    The transformer operates purely on the *invariant scalar* track —
    we treat the per-pair feature vector as a sequence of three tokens
    ``[scalars_A, scalars_B, distance_embedding]`` (each projected to
    ``token_dim``).  Because every input token is already an invariant
    scalar, the entire encoder is automatically invariant under
    rotations of the input positions, and thus the larger network
    composed of dressing → invariant projection → transformer expert
    remains E(3)-equivariant.
    """

    def __init__(
        self,
        scalar_channels: int,
        hidden_dims: typing.Sequence[int],
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        token_dim: int = hidden_dims[0] if len(hidden_dims) > 0 else 64

        # Project per-atom scalars and the distance scalar to the same
        # token dimension so they can be concatenated as a sequence.
        self.atom_proj: nn.Linear = nn.Linear(scalar_channels, token_dim)
        self.distance_proj: nn.Linear = nn.Linear(1, token_dim)

        # Adjust num_heads so that token_dim is divisible
        adjusted_heads: int = max(1, num_heads)
        while token_dim % adjusted_heads != 0 and adjusted_heads > 1:
            adjusted_heads -= 1

        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=adjusted_heads,
            dim_feedforward=hidden_dims[-1] if len(hidden_dims) > 0 else token_dim,
            activation="gelu",
            batch_first=True,
            dropout=0.0,
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Final readout: average tokens + scalar head
        self.readout: nn.Linear = nn.Linear(token_dim, 1)

    def forward(
        self,
        pair_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """``([P, S], [P, S], [P, 1]) → (P,)`` — invariant-only inputs."""
        scalars_a, scalars_b, distance = pair_features
        token_a: torch.Tensor = self.atom_proj(scalars_a)  # (P, token_dim)
        token_b: torch.Tensor = self.atom_proj(scalars_b)  # (P, token_dim)
        token_d: torch.Tensor = self.distance_proj(distance)  # (P, token_dim)

        # Stack into sequence (P, 3, token_dim)
        sequence: torch.Tensor = torch.stack([token_a, token_b, token_d], dim=1)
        encoded: torch.Tensor = self.encoder(sequence)  # (P, 3, token_dim)
        pooled: torch.Tensor = encoded.mean(dim=1)  # (P, token_dim)
        return self.readout(pooled).squeeze(-1)  # (P,)


# ---------------------------------------------------------------------------
# Equivariant artisan core
# ---------------------------------------------------------------------------


def build_artisan_edge_irreps(lmax: int) -> Irreps:
    """Spherical-harmonics irreps for artisan edges, one channel per ``l``.

    Same convention as :func:`goal.ml.nn.blocks.env_dressing.build_edge_irreps`
    (duplicated here to keep ``blocks.artisans`` free of intra-``blocks``
    imports): even ``l`` → parity ``e``, odd ``l`` → parity ``o``.
    """
    parts: list[str] = []
    for l_val in range(lmax + 1):
        parity: str = "e" if l_val % 2 == 0 else "o"
        parts.append(f"1x{l_val}{parity}")
    return Irreps("+".join(parts))


class _PairRMSNorm(nn.Module):
    """Smooth equivariant RMS normalisation for pair features.

    Divides the whole feature vector by its global root-mean-square
    (an invariant scalar), with one learnable gain per irrep block:

        ``y_block = γ_block · x_block / sqrt(mean(x²) + ε²)``

    Why not :class:`goal.ml.nn.primitives.norm.EquivariantLayerNorm`?
    Its per-block ``x / max(‖x‖, ε)`` normalisation re-scales
    *symmetry-suppressed* blocks (e.g. the ``l = 1`` features of an
    atom in a tetrahedral environment, which vanish by symmetry) with
    a gain of up to ``1/ε`` — turning numerical-cancellation residue
    into O(1) feature noise with enormous position gradients, which
    destroys force smoothness exactly at high-symmetry geometries.
    The global RMS denominator is bounded away from zero by the
    scalar channels, so this norm is smooth and well-conditioned
    everywhere, and zero input maps to exactly zero output.
    """

    def __init__(self, irreps: Irreps | str, eps: float = 1e-6) -> None:
        super().__init__()
        self.irreps: Irreps = Irreps(irreps)
        self.eps: float = float(eps)
        # One learnable gain per (mul, ir) block, initialised to 1.
        self.weight: nn.Parameter = nn.Parameter(torch.ones(len(self.irreps)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, irreps.dim) → (N, irreps.dim)``, equivariant."""
        msq: torch.Tensor = x.pow(2).mean(dim=-1, keepdim=True)  # (N, 1) invariant
        inv: torch.Tensor = torch.rsqrt(msq + self.eps**2)  # (N, 1)
        outputs: list[torch.Tensor] = []
        idx: int = 0
        for block_i, (mul, ir) in enumerate(self.irreps):
            dim: int = mul * ir.dim
            outputs.append(x[:, idx : idx + dim] * (self.weight[block_i] * inv))
            idx += dim
        return torch.cat(outputs, dim=-1)


class _EquivariantArtisanCore(nn.Module):
    """Equivariant pairwise energy network (``architecture="equivariant"``).

    Per edge ``(i → j)`` the core computes

    1. **Symmetric node combination** — both endpoints' equivariant
       features are mapped through a *shared* equivariant linear and
       summed: ``h_AB = L(h_A) + L(h_B)``.  Sharing the map (rather
       than using separate ``L_A`` / ``L_B``) is what makes the sum
       genuinely symmetric, so ``E(A, B) = E(B, A)`` holds by
       construction.
    2. **CG tensor product with bond geometry** —
       ``h = TP(h_AB, Y^l(r̂_ij); w(d_ij))`` where the per-edge TP
       weights come from a radial MLP on a Bessel expansion of the
       bond length, tapered by a polynomial envelope, and optionally
       modulated per destination element
       (``element_conditioned=True``).
    3. **Optional deeper equivariant layers** — ``num_layers - 1``
       rounds of ``h ← RMSNorm(EquivLinear(h) + h)`` using the smooth
       :class:`_PairRMSNorm` (see its docstring for why the hard
       per-block layer norm is unsuitable here).
    4. **Scalar readout** — invariant ``l = 0`` scalars →
       ``Linear → SiLU → Linear → 1``.

    Every learnable map is bias-free, so a zero input produces exactly
    zero output — required by the bank's static-shape zero-masking
    schedule (masked edges feed zeros in and must contribute nothing).

    Parameters
    ----------
    irreps_in : Irreps
        Irreps of the per-atom node features fed to the artisan.
    cutoff : float
        Cutoff radius (Angstrom) for the radial basis and envelope.
    hidden_irreps : str or Irreps
        Internal equivariant width, e.g. ``"16x0e + 16x1o + 16x2e"``.
    num_layers : int
        Total equivariant depth; ``num_layers - 1`` residual
        linear+norm layers follow the tensor product.
    num_rbf : int
        Number of Bessel radial basis functions.
    radial_hidden : int
        Hidden width of the radial MLP.
    n_scalar_out : int
        Number of invariant scalars extracted before the energy MLP.
    final_hidden : int
        Hidden width of the final energy MLP.
    element_conditioned : bool
        Modulate the radial TP weights by a per-element linear map of
        the destination atom's element (one-hot of ``Z[col]``).
    n_elements : int
        One-hot vocabulary size for element conditioning.
    """

    def __init__(
        self,
        irreps_in: Irreps | str,
        cutoff: float,
        hidden_irreps: Irreps | str = "16x0e + 16x1o + 16x2e",
        num_layers: int = 1,
        num_rbf: int = 8,
        radial_hidden: int = 32,
        n_scalar_out: int = 16,
        final_hidden: int = 16,
        element_conditioned: bool = True,
        n_elements: int = 120,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        self.irreps_in: Irreps = Irreps(irreps_in)
        self.irreps_hidden: Irreps = Irreps(hidden_irreps)
        lmax: int = max((ir.l for _, ir in self.irreps_hidden), default=0)
        self.irreps_edge: Irreps = build_artisan_edge_irreps(lmax)
        self._element_conditioned: bool = bool(element_conditioned)
        self._n_elements: int = int(n_elements)

        # Step 1 — shared symmetric node embedding (bias-free).
        self.node_embed: EquivariantLinear = EquivariantLinear(
            self.irreps_in, self.irreps_hidden, biases=False
        )

        # Step 2 — CG tensor product with bond geometry.
        self.tp: WeightedTensorProduct = WeightedTensorProduct(
            irreps_in1=self.irreps_hidden,
            irreps_in2=self.irreps_edge,
            irreps_out=self.irreps_hidden,
        )
        self.radial_basis: BesselBasis = BesselBasis(num_basis=num_rbf, cutoff=cutoff)
        self.envelope: PolynomialEnvelope = PolynomialEnvelope(cutoff=cutoff)
        self.radial_mlp: RadialMLP = RadialMLP(
            num_basis=num_rbf,
            hidden_dim=radial_hidden,
            num_out=self.tp.weight_numel,
        )
        self.element_linear: nn.Linear | None = (
            nn.Linear(self._n_elements, self.tp.weight_numel, bias=False)
            if element_conditioned
            else None
        )

        # Step 3 — optional deeper equivariant layers (bias-free).
        self.layers: nn.ModuleList = nn.ModuleList(
            EquivariantLinear(self.irreps_hidden, self.irreps_hidden, biases=False)
            for _ in range(num_layers - 1)
        )
        self.norms: nn.ModuleList = nn.ModuleList(
            _PairRMSNorm(self.irreps_hidden) for _ in range(num_layers - 1)
        )

        # Step 4 — invariant scalar readout (bias-free).
        scalar_irreps: Irreps = Irreps(f"{int(n_scalar_out)}x0e")
        self.to_scalars: EquivariantLinear = EquivariantLinear(
            self.irreps_hidden, scalar_irreps, biases=False
        )
        self.energy_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(int(n_scalar_out), int(final_hidden), bias=False),
            nn.SiLU(),
            nn.Linear(int(final_hidden), 1, bias=False),
        )

    def forward(
        self,
        feats_a: torch.Tensor,
        feats_b: torch.Tensor,
        edge_sh: torch.Tensor,
        distances: torch.Tensor,
        z_dst: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-edge invariant energies (un-gated, un-tapered).

        Parameters
        ----------
        feats_a, feats_b : Tensor ``(E, irreps_in.dim)``
            Equivariant node features of the source / destination atom.
        edge_sh : Tensor ``(E, irreps_edge.dim)``
            Spherical harmonics of the edge direction.
        distances : Tensor ``(E,)``
            Bond lengths.  Zeros (masked edges) are clamped to a tiny
            positive value so the Bessel basis stays finite; the bank
            re-masks the output, so the dummy value never contributes.
        z_dst : Tensor ``(E,)``, optional
            Atomic numbers of the destination atom — required when
            ``element_conditioned=True``.

        Returns
        -------
        Tensor ``(E,)``
            Per-edge scalar energies.
        """
        h_ab: torch.Tensor = self.node_embed(feats_a) + self.node_embed(feats_b)

        # Radial TP weights.  Clamp avoids 0/0 in the Bessel basis on
        # zero-masked edges; the envelope and the bank's output mask
        # make the clamped value irrelevant.
        d_safe: torch.Tensor = distances.clamp_min(1e-6)
        rbf: torch.Tensor = self.radial_basis(d_safe)  # (E, num_rbf)
        env: torch.Tensor = self.envelope(d_safe).to(h_ab.dtype).unsqueeze(-1)  # (E, 1)
        radial_w: torch.Tensor = self.radial_mlp(rbf) * env  # (E, weight_numel)

        if self._element_conditioned and self.element_linear is not None:
            if z_dst is None:
                raise ValueError("element_conditioned=True requires z_dst in forward()")
            one_hot: torch.Tensor = F.one_hot(
                z_dst.clamp(0, self._n_elements - 1),
                num_classes=self._n_elements,
            ).to(radial_w.dtype)
            radial_w = radial_w * self.element_linear(one_hot)  # (E, weight_numel)

        h: torch.Tensor = self.tp(h_ab, edge_sh.to(h_ab.dtype), radial_w)

        for lin, norm in zip(self.layers, self.norms):
            h = norm(lin(h) + h)

        scalars: torch.Tensor = self.to_scalars(h)  # (E, n_scalar_out)
        return self.energy_mlp(scalars).squeeze(-1)  # (E,)


# ---------------------------------------------------------------------------
# Single artisan wrapper
# ---------------------------------------------------------------------------


class PotentialArtisan(nn.Module):
    """One ``E_{AB}`` artisan for a single element pair.

    Holds:

    * a learnable scalar gate ``P_{AB}``;
    * for ``architecture="scalar"`` (default): an invariant projection
      ``o3.Linear`` mapping atom features to ``scalar_channels x 0e``
      scalars (E(3)-safe because only ``l = 0`` outputs are retained)
      plus the scalar backbone (Linear MLP or Transformer);
    * for ``architecture="equivariant"``: an
      :class:`_EquivariantArtisanCore` operating on the full
      equivariant node features and the bond direction.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        config: ArtisanConfig,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__()
        self.architecture: str = str(config.architecture)
        if self.architecture not in ("scalar", "equivariant"):
            raise ValueError(
                f"Unknown architecture '{config.architecture}'. Use 'scalar' or 'equivariant'."
            )
        self.scalar_channels: int = config.scalar_channels

        # Learnable gate, initialised to 1.0
        self.gate: nn.Parameter = nn.Parameter(torch.tensor(1.0))

        # ----- Equivariant architecture -----
        self.equivariant_core: _EquivariantArtisanCore | None = None
        if self.architecture == "equivariant":
            eq_kwargs: dict[str, typing.Any] = dict(config.equivariant or {})
            self.equivariant_core = _EquivariantArtisanCore(
                irreps_in=irreps_in,
                cutoff=cutoff,
                **eq_kwargs,
            )
            self._backbone_type: str = "equivariant"
            return

        # ----- Scalar architecture (original behaviour) -----
        scalar_irreps: Irreps = Irreps(f"{config.scalar_channels}x0e")

        # Project arbitrary equivariant atom features → scalars (l=0)
        self.scalar_proj: EquivariantLinear = EquivariantLinear(irreps_in, scalar_irreps)

        # Backbone
        in_dim: int = 2 * config.scalar_channels + 1
        if config.expert_type == "linear":
            self._backbone_type = "linear"
            self.backbone: nn.Module = _LinearExpert(
                in_dim=in_dim,
                hidden_dims=config.hidden_dims,
                dropout_rate=config.dropout_rate,
            )
        elif config.expert_type == "transformer":
            self._backbone_type = "transformer"
            self.backbone = _TransformerExpert(
                scalar_channels=config.scalar_channels,
                hidden_dims=config.hidden_dims,
                num_heads=config.transformer_heads,
                num_layers=config.transformer_layers,
            )
        else:
            raise ValueError(
                f"Unknown expert_type '{config.expert_type}'. Use 'linear' or 'transformer'."
            )

    def project(self, atom_features: torch.Tensor) -> torch.Tensor:
        """Invariant projection ``(N, irreps_in.dim) → (N, scalar_channels)``.

        Scalar architecture only — the equivariant artisan consumes the
        full node features without an invariant projection.
        """
        if self.architecture != "scalar":
            raise RuntimeError(
                "project() is only available for architecture='scalar'; "
                "the equivariant artisan consumes full node features."
            )
        return self.scalar_proj(atom_features)

    def forward_equivariant(
        self,
        feats_a: torch.Tensor,
        feats_b: torch.Tensor,
        edge_sh: torch.Tensor,
        distances: torch.Tensor,
        z_dst: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute gated pair energies for ``architecture="equivariant"``.

        Parameters
        ----------
        feats_a, feats_b : Tensor ``(P, irreps_in.dim)``
            Equivariant node features of both endpoints.
        edge_sh : Tensor ``(P, irreps_edge.dim)``
            Spherical harmonics of the edge direction.
        distances : Tensor ``(P,)``
            Pair distances.
        z_dst : Tensor ``(P,)``, optional
            Destination atomic numbers (element conditioning).

        Returns
        -------
        Tensor
            ``(P,)`` per-pair scalar energy (gate × core output).
        """
        if self.equivariant_core is None:
            raise RuntimeError("forward_equivariant() requires architecture='equivariant'.")
        raw: torch.Tensor = self.equivariant_core(
            feats_a, feats_b, edge_sh, distances, z_dst
        )  # (P,)
        return self.gate * raw

    def forward(
        self,
        scalars_a: torch.Tensor,
        scalars_b: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gated pair energies (un-tapered) — scalar architecture.

        Parameters
        ----------
        scalars_a : Tensor
            Invariant scalars of atom ``A``, shape ``(P, scalar_channels)``.
        scalars_b : Tensor
            Invariant scalars of atom ``B``, shape ``(P, scalar_channels)``.
        distances : Tensor
            Pair distances, shape ``(P,)``.

        Returns
        -------
        Tensor
            ``(P,)`` per-pair scalar energy (gate × backbone output).
        """
        if self.architecture != "scalar":
            raise RuntimeError(
                "forward() implements the scalar architecture; use "
                "forward_equivariant() for architecture='equivariant'."
            )
        if self._backbone_type == "linear":
            pair_features: torch.Tensor = torch.cat(  # (P, 2 * S + 1)
                [scalars_a, scalars_b, distances.unsqueeze(-1)], dim=-1
            )
            raw: torch.Tensor = self.backbone(pair_features)  # (P,)
        else:  # transformer
            raw = self.backbone((scalars_a, scalars_b, distances.unsqueeze(-1)))  # (P,)
        return self.gate * raw

    def forward_pairwise(
        self,
        scalars_a: torch.Tensor,
        scalars_b: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``(E_ij, F_ij)`` for the pairwise force mode.

        ``F_ij = -∂E_ij/∂r_ij`` is taken via ``torch.autograd.grad`` so
        Newton's third law is satisfied per directed edge when the
        caller scatters ``F_ij`` to source and ``-F_ij`` to destination
        atoms.

        Parameters
        ----------
        scalars_a, scalars_b : Tensor
            Invariant scalars of source / destination atom, shape
            ``(P, scalar_channels)``.
        edge_vectors : Tensor
            Per-edge displacement vectors ``r_ij = pos[j] - pos[i]``
            of shape ``(P, 3)``.  Treated as the leaf of the local
            autograd graph — its ``requires_grad`` is enabled inside
            this method without affecting the caller's tensor.

        Returns
        -------
        tuple of Tensor
            ``(E_ij, F_ij)`` with shapes ``(P,)`` and ``(P, 3)``.
        """
        if self.architecture != "scalar":
            raise RuntimeError(
                "forward_pairwise() is only implemented for "
                "architecture='scalar'; equivariant artisans rely on the "
                "autograd force path through graph positions."
            )
        # Use a local leaf so the gradient stops at edge_vectors (we
        # only want the per-pair derivative, not a derivative back
        # through positions — the head can still take a separate
        # autograd path on ``graph.pos`` if it wants combined mode).
        r_ij: torch.Tensor = edge_vectors.detach().requires_grad_(True)
        distances: torch.Tensor = r_ij.norm(dim=-1)  # (P,)

        if self._backbone_type == "linear":
            pair_features: torch.Tensor = torch.cat(
                [scalars_a, scalars_b, distances.unsqueeze(-1)], dim=-1
            )
            raw: torch.Tensor = self.backbone(pair_features)  # (P,)
        else:  # transformer
            raw = self.backbone((scalars_a, scalars_b, distances.unsqueeze(-1)))  # (P,)
        e_ij: torch.Tensor = self.gate * raw  # (P,)

        # F_ij = -dE_ij/dr_ij
        grad_outputs: tuple[torch.Tensor, ...] = torch.autograd.grad(
            outputs=e_ij.sum(),
            inputs=r_ij,
            create_graph=self.training,
            retain_graph=True,
        )
        f_ij: torch.Tensor = -grad_outputs[0]  # (P, 3)
        return e_ij, f_ij


# ---------------------------------------------------------------------------
# Rare-pair expert (CHANGE 4)
# ---------------------------------------------------------------------------


class RarePotentialArtisan(nn.Module):
    """Shared expert for rare/infrequent element pairs.

    Unlike :class:`PotentialArtisan` — which dedicates one full MLP per pair —
    this module handles *all* rare pairs in a single network by injecting a
    per-pair learned embedding.  Each rare pair gets its own
    ``pair_embed_dim``-dimensional embedding vector and its own scalar gate,
    so the network can still specialize per pair.

    The ``n_rare`` learned gate parameters and embedding rows remain in the
    autograd graph regardless of whether the corresponding pair appears in
    the current batch (same zero-masking contract as :class:`SimurghArtisanBank`).

    Parameters
    ----------
    irreps_in : Irreps
        Irreps of the atom features fed into this expert.
    pairs : list of (int, int)
        The rare unordered element pairs this expert handles, in canonical
        ``(lo, hi)`` order.
    config : ArtisanConfig
        Shared expert config; ``rare_pair_embed_dim`` controls the embedding width.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        pairs: list[tuple[int, int]],
        config: ArtisanConfig,
    ) -> None:
        super().__init__()
        if not pairs:
            raise ValueError("RarePotentialArtisan requires at least one pair.")
        self._pairs: tuple[tuple[int, int], ...] = tuple(pairs)
        self.scalar_channels: int = config.scalar_channels
        self._embed_dim: int = config.rare_pair_embed_dim

        scalar_irreps: Irreps = Irreps(f"{config.scalar_channels}x0e")
        self.scalar_proj: EquivariantLinear = EquivariantLinear(irreps_in, scalar_irreps)

        # Per-pair embedding and gates
        n: int = len(pairs)
        self.pair_embed: nn.Embedding = nn.Embedding(n, config.rare_pair_embed_dim)
        self.gates: nn.Parameter = nn.Parameter(torch.ones(n))

        # Shared MLP: [s_A, s_B, pair_embed, dist] → 1
        in_dim: int = 2 * config.scalar_channels + config.rare_pair_embed_dim + 1
        self.backbone: _LinearExpert = _LinearExpert(
            in_dim=in_dim,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate,
        )

    def project(self, atom_features: torch.Tensor) -> torch.Tensor:
        """Invariant projection ``(N, irreps_in.dim) → (N, scalar_channels)``."""
        return self.scalar_proj(atom_features)

    def forward(
        self,
        scalars_a: torch.Tensor,
        scalars_b: torch.Tensor,
        distances: torch.Tensor,
        pair_local_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gated pair energies for a batch of edges (same rare pair type).

        Parameters
        ----------
        scalars_a, scalars_b : Tensor ``(E, scalar_channels)``
            Masked atom scalars for source / destination.
        distances : Tensor ``(E,)``
            Masked pair distances.
        pair_local_idx : Tensor ``(E,)`` int
            Local index into this expert's pair list (same value for all
            edges of the same rare pair type; varies when batching multiple
            rare pairs, but caller loops over pairs separately).

        Returns
        -------
        Tensor ``(E,)``
            Gated per-pair energy contributions.
        """
        embed: torch.Tensor = self.pair_embed(
            pair_local_idx.clamp(0, len(self._pairs) - 1)
        )  # (E, embed_dim)
        feats: torch.Tensor = torch.cat(
            [scalars_a, scalars_b, embed, distances.unsqueeze(-1)], dim=-1
        )  # (E, 2*S + D + 1)
        raw: torch.Tensor = self.backbone(feats)  # (E,)
        gate: torch.Tensor = self.gates[pair_local_idx]  # (E,)
        return gate * raw


# ---------------------------------------------------------------------------
# Artisan-bank container
# ---------------------------------------------------------------------------


class SimurghArtisanBank(nn.Module):
    """SIMURGH Element-Pair Mixture-of-Experts with optional data-driven routing.

    By default, instantiates one :class:`PotentialArtisan` per unordered element
    pair (original behaviour, ``rare_pair_enabled=False``).

    When ``rare_pair_enabled=True`` and ``pair_counts`` is provided, pairs whose
    frequency (fraction of total edges) falls below ``min_pair_frequency`` are
    routed to a single shared :class:`RarePotentialArtisan` that uses a learned
    pair-type embedding to distinguish them.  Common pairs still get their own
    dedicated :class:`PotentialArtisan`.  A routing summary is logged at init.

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model.
    irreps_in : Irreps or str
        Irreps of the per-atom features fed into the artisan bank.
    artisan_config : ArtisanConfig
        Config shared by all expert modules.
    cutoff : float
        Cosine-cutoff radius (Angstrom) applied to each pair energy.
    pair_counts : dict, optional
        ``{(Z_A, Z_B): edge_count}`` over the training set.  Required for
        data-driven routing (used when ``rare_pair_enabled=True``).
    rare_pair_enabled : bool
        When ``True`` and ``pair_counts`` is provided, apply frequency-based
        routing.  Pairs below ``min_pair_frequency`` go to
        :class:`RarePotentialArtisan`.
    min_pair_frequency : float
        Minimum fraction of total edges for a pair to get a dedicated expert.
        Pairs below this threshold are handled by :class:`RarePotentialArtisan`.
    pair_symbols : optional mapping
        Optional override for the pretty pair labels.  Purely cosmetic.

    Notes
    -----
    The block emits **per-atom** scalar energies.  Neighbour lists are
    bidirectional (both ``(i, j)`` and ``(j, i)`` edges are present),
    so the standard MLIP convention is followed: a directed edge
    ``(i, j)`` contributes ``E_pair / 2`` to atom ``i`` only.
    """

    def __init__(
        self,
        elements: typing.Sequence[int],
        irreps_in: Irreps | str,
        artisan_config: ArtisanConfig,
        cutoff: float = 5.0,
        pair_counts: typing.Mapping[tuple[int, int], int] | None = None,
        rare_pair_enabled: bool = False,
        min_pair_frequency: float = 0.01,
        pair_symbols: typing.Mapping[int, str] | None = None,
    ) -> None:
        super().__init__()
        if len(elements) == 0:
            raise ValueError("`elements` must contain at least one atomic number.")
        self._elements: tuple[int, ...] = tuple(sorted({int(z) for z in elements}))
        self._irreps_in: Irreps = Irreps(irreps_in)
        self._cutoff: float = cutoff
        self._symbols: typing.Mapping[int, str] = pair_symbols or _DEFAULT_SYMBOLS
        self._rare_pair_enabled: bool = bool(rare_pair_enabled)

        # Enumerate all pairs deterministically
        pairs: list[tuple[int, int]] = enumerate_element_pairs(self._elements)
        self._pairs: tuple[tuple[int, int], ...] = tuple(pairs)

        # Data-driven routing (CHANGE 4)
        dedicated_pairs: list[tuple[int, int]] = []
        rare_pairs: list[tuple[int, int]] = []

        if rare_pair_enabled and pair_counts is not None:
            total_edges: int = max(1, sum(pair_counts.values()))
            for p in pairs:
                cnt: int = int(pair_counts.get(p, pair_counts.get((p[1], p[0]), 0)))
                freq: float = cnt / total_edges
                if freq >= min_pair_frequency:
                    dedicated_pairs.append(p)
                else:
                    rare_pairs.append(p)
        else:
            dedicated_pairs = list(pairs)

        self._dedicated_pairs: tuple[tuple[int, int], ...] = tuple(dedicated_pairs)
        self._rare_pairs: tuple[tuple[int, int], ...] = tuple(rare_pairs)

        # Artisan architecture ("scalar" or "equivariant") — uniform
        # across the bank; validated by the first PotentialArtisan.
        self._architecture: str = str(artisan_config.architecture)

        # Build dedicated artisans
        artisans: dict[str, PotentialArtisan] = {}
        self._pair_keys: list[str] = []
        for a, b in self._dedicated_pairs:
            key: str = self._key(a, b)
            self._pair_keys.append(key)
            artisans[key] = PotentialArtisan(self._irreps_in, artisan_config, cutoff=cutoff)
        self.artisans: nn.ModuleDict = nn.ModuleDict(artisans)

        # Edge SH irreps shared by all equivariant artisans (None for scalar).
        self._irreps_edge: Irreps | None = None
        if self._architecture == "equivariant" and self._pair_keys:
            first = typing.cast(PotentialArtisan, self.artisans[self._pair_keys[0]])
            assert first.equivariant_core is not None
            self._irreps_edge = first.equivariant_core.irreps_edge

        # Build shared rare-pair expert (None when no rare pairs)
        self.rare_artisan_module: RarePotentialArtisan | None = (
            RarePotentialArtisan(self._irreps_in, list(rare_pairs), artisan_config)
            if rare_pairs
            else None
        )

        # Log routing table at construction time
        self._log_routing_table(pair_counts, min_pair_frequency)

    # ------------------------------------------------------------------
    # Naming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(a: int, b: int) -> str:
        lo, hi = sorted((a, b))
        return f"z{lo}_z{hi}"

    def pair_label(self, a: int, b: int) -> str:
        """Order-independent ``"A-B"`` label for pair ``(a, b)`` using the bank's symbols."""
        return pair_label(a, b, self._symbols)

    # ------------------------------------------------------------------
    # Routing-table logger
    # ------------------------------------------------------------------

    def _log_routing_table(
        self,
        pair_counts: typing.Mapping[tuple[int, int], int] | None,
        min_pair_frequency: float,
    ) -> None:
        """Print expert routing summary using rich (falls back to plain text)."""
        from lightning.pytorch.utilities.rank_zero import rank_zero_info

        total_edges: int = max(1, sum(pair_counts.values())) if pair_counts is not None else 0
        rare_labels: list[str] = [self.pair_label(a, b) for a, b in self._rare_pairs]

        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(
                title="[bold cyan]SIMURGH Artisan Routing[/bold cyan]",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Pair", style="cyan", no_wrap=True)
            table.add_column("Count", justify="right")
            table.add_column("Frequency", justify="right")
            table.add_column("Routing", justify="left")

            for a, b in self._pairs:
                lbl = self.pair_label(a, b)
                cnt: int = 0
                freq_str: str = "n/a"
                if pair_counts is not None:
                    cnt = int(pair_counts.get((a, b), pair_counts.get((b, a), 0)))
                    freq = cnt / total_edges
                    freq_str = f"{freq * 100:.2f}%"

                is_rare = (a, b) in self._rare_pairs
                if is_rare:
                    routing = "[yellow]→ RarePotentialArtisan[/yellow]"
                else:
                    routing = "[green]PotentialArtisan (dedicated)[/green]"
                table.add_row(lbl, str(cnt), freq_str, routing)

            import io

            buf = io.StringIO()
            Console(file=buf, no_color=True, width=90).print(table)
            rank_zero_info("\n" + buf.getvalue())

        except ImportError:
            # Plain-text fallback
            lines: list[str] = ["SIMURGH Artisan Routing:"]
            lines.append(f"  {'Pair':<10} {'Count':>8} {'Freq':>8}  Routing")
            lines.append("  " + "-" * 50)
            for a, b in self._pairs:
                lbl = self.pair_label(a, b)
                cnt = 0
                freq_str = "n/a"
                if pair_counts is not None:
                    cnt = int(pair_counts.get((a, b), pair_counts.get((b, a), 0)))
                    freq = cnt / total_edges
                    freq_str = f"{freq * 100:.2f}%"
                is_rare = (a, b) in self._rare_pairs
                routing = "→ RarePotentialArtisan" if is_rare else "PotentialArtisan (dedicated)"
                lines.append(f"  {lbl:<10} {cnt:>8} {freq_str:>8}  {routing}")
            lines.append("")
            rank_zero_info("\n".join(lines))

        # Summary line
        n_dedicated = len(self._dedicated_pairs)
        n_rare = len(self._rare_pairs)
        if n_rare > 0:
            rank_zero_info(
                f"[SIMURGH ArtisanBank] {n_dedicated} dedicated PotentialArtisans | "
                f"{n_rare} rare pair(s) handled by RarePotentialArtisan: "
                f"{', '.join(rare_labels)}"
            )
        else:
            rank_zero_info(
                f"[SIMURGH ArtisanBank] {n_dedicated} dedicated PotentialArtisans "
                f"(no rare-pair routing active)"
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def elements(self) -> tuple[int, ...]:
        return self._elements

    @property
    def num_artisans(self) -> int:
        """Total number of expert modules (dedicated + 1 shared if rare pairs exist)."""
        return len(self._dedicated_pairs) + (1 if self.rare_artisan_module is not None else 0)

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        return self._pairs

    @property
    def cutoff(self) -> float:
        return self._cutoff

    def gates(self) -> dict[str, torch.Tensor]:
        """Return a dict ``{pair_label: gate_value}`` for logging."""
        result: dict[str, torch.Tensor] = {
            self.pair_label(a, b): self.artisans[self._key(a, b)].gate.detach()
            for a, b in self._dedicated_pairs
        }
        if self.rare_artisan_module is not None:
            for local_idx, (a, b) in enumerate(self._rare_pairs):
                result[self.pair_label(a, b)] = self.rare_artisan_module.gates[local_idx].detach()
        return result

    @torch.no_grad()
    def compute_artisan_loads(
        self,
        atom_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_lengths: torch.Tensor,
        edge_vectors: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute mean absolute energy contribution per expert over a batch.

        For each expert ``E_{AB}``:
            ``load_AB = mean(|P_AB * E_AB_output(masked_inputs)|)``

        averaged over all edges in the batch (masked edges contribute 0 and
        are included in the denominator, consistent with the forward pass).

        Returns
        -------
        dict
            ``{pair_label: scalar_tensor}`` of per-expert loads (detached).
            Also includes ``"load_variance"`` — the coefficient of variation
            ``std(loads) / mean(loads)`` across all artisans.  A value > 2.0
            indicates expert collapse.  When E == 0 all loads are 0 and
            ``load_variance`` is 0.
        """
        dtype: torch.dtype = atom_features.dtype
        device: torch.device = atom_features.device
        E: int = int(edge_lengths.shape[0])

        result: dict[str, torch.Tensor] = {}

        if E == 0:
            for a, b in self._pairs:
                result[self.pair_label(a, b)] = torch.zeros((), device=device, dtype=dtype)
            result["load_variance"] = torch.zeros((), device=device, dtype=dtype)
            return result

        edge_lengths_dt = edge_lengths.to(dtype)
        cut_env: torch.Tensor = cosine_cutoff(edge_lengths_dt, self._cutoff)
        row, col = edge_index
        z_row: torch.Tensor = atomic_numbers[row]
        z_col: torch.Tensor = atomic_numbers[col]
        z_lo: torch.Tensor = torch.minimum(z_row, z_col)
        z_hi: torch.Tensor = torch.maximum(z_row, z_col)

        equivariant: bool = self._architecture == "equivariant"
        edge_sh: torch.Tensor | None = None
        if equivariant and self._irreps_edge is not None:
            if edge_vectors is None:
                raise ValueError(
                    "architecture='equivariant' requires edge_vectors in compute_artisan_loads()."
                )
            from e3nn.o3 import spherical_harmonics

            edge_sh = spherical_harmonics(
                self._irreps_edge,
                edge_vectors.to(dtype),
                normalize=True,
                normalization="component",
            )

        load_values: list[torch.Tensor] = []
        for (a, b), key in zip(self._pairs, self._pair_keys):
            expert = typing.cast(PotentialArtisan, self.artisans[key])
            mask: torch.Tensor = ((z_lo == a) & (z_hi == b)).to(dtype)
            mask_col = mask.unsqueeze(-1)

            if equivariant:
                feats_a: torch.Tensor = atom_features[row] * mask_col
                feats_b: torch.Tensor = atom_features[col] * mask_col
                sh_in: torch.Tensor = typing.cast(torch.Tensor, edge_sh) * mask_col
                dist_in: torch.Tensor = edge_lengths_dt * mask
                pair_e: torch.Tensor = expert.forward_equivariant(
                    feats_a, feats_b, sh_in, dist_in, z_col
                )
            else:
                scalars: torch.Tensor = expert.project(atom_features)
                scalars_a: torch.Tensor = scalars[row] * mask_col
                scalars_b: torch.Tensor = scalars[col] * mask_col
                dist_in = edge_lengths_dt * mask
                pair_e = expert(scalars_a, scalars_b, dist_in)
            tapered: torch.Tensor = pair_e * mask * cut_env
            load: torch.Tensor = tapered.abs().mean()
            result[self.pair_label(a, b)] = load
            load_values.append(load)

        loads_t: torch.Tensor = torch.stack(load_values)
        mean_load: torch.Tensor = loads_t.mean().clamp_min(1e-12)
        result["load_variance"] = loads_t.std() / mean_load
        return result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        atom_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_lengths: torch.Tensor,
        edge_vectors: torch.Tensor | None = None,
        return_per_pair: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute per-atom scalar energy contributions.

        Static-shape schedule:

        1. For every artisan build a per-edge **float** mask selecting
           edges of its pair type.
        2. Multiply the inputs (scalars and distance for the scalar
           architecture; node features, edge SH and distance for the
           equivariant architecture) by the mask *before* the forward.
        3. Multiply the artisan output by the mask *again* before adding
           to ``atom_energies`` — non-matching edges contribute
           **exactly** ``0.0`` to the total energy.

        Parameters
        ----------
        atom_features : Tensor
            Dressed per-atom equivariant features, shape
            ``(N, irreps_in.dim)``.
        atomic_numbers : Tensor
            Atomic numbers ``(N,)`` int64.
        edge_index : Tensor
            Edge index ``(2, E)``.
        edge_lengths : Tensor
            Pair distances ``(E,)``.
        edge_vectors : Tensor, optional
            Edge displacement vectors ``(E, 3)``.  Required when the
            bank uses ``architecture="equivariant"`` (spherical
            harmonics of the bond direction); ignored otherwise.
        return_per_pair : bool
            If ``True``, return ``(atom_energies, per_pair_total_energy)``
            where ``per_pair_total_energy`` is a dict keyed by
            ``pair_label`` whose values are detached scalar tensors —
            useful for testing the zero-mask correctness.

        Returns
        -------
        Tensor or tuple
            Per-atom scalar energy ``(N,)``, optionally followed by a
            dict of detached per-pair total contributions.
        """
        num_atoms: int = atom_features.shape[0]
        device: torch.device = atom_features.device
        dtype: torch.dtype = atom_features.dtype

        row, col = edge_index  # both shape (E,)
        E: int = int(edge_lengths.shape[0])

        equivariant: bool = self._architecture == "equivariant"
        if equivariant and edge_vectors is None and E > 0:
            raise ValueError("architecture='equivariant' requires edge_vectors in forward().")

        # Boundary cast: align edge_lengths with the per-atom feature dtype
        # so the downstream concat in ``_LinearExpert`` (scalars ⊕ distance)
        # and the per-edge mask multiplication stay in a single precision.
        # Same MACE / NequIP pattern as ``radial.RadialMLP`` / interaction.
        edge_lengths = edge_lengths.to(dtype)

        # Accumulator for per-atom interaction energy.  The per-element
        # baseline lives on the ``SimurghBackbone`` as a fixed buffer
        # (``atomic_energies``, computed from data via least-squares
        # regression) and is added there — the artisan bank only models the
        # local interaction residual.
        atom_energies: torch.Tensor = torch.zeros(num_atoms, device=device, dtype=dtype)
        per_pair_total: dict[str, torch.Tensor] = {}

        # ----- Degenerate case: no edges at all -----
        if E == 0:
            # Keep every expert in the autograd graph via a zero-contribution dummy.
            first_key: str = self._pair_keys[0] if self._pair_keys else ""
            scalar_channels: int = (
                self.artisans[first_key].scalar_channels
                if first_key
                else (self.rare_artisan_module.scalar_channels if self.rare_artisan_module else 1)
            )
            dummy_scalars: torch.Tensor = torch.zeros(
                1, scalar_channels, device=device, dtype=dtype
            )
            dummy_dist: torch.Tensor = torch.zeros(1, device=device, dtype=dtype)
            dummy_feats: torch.Tensor | None = None
            dummy_sh: torch.Tensor | None = None
            dummy_z: torch.Tensor | None = None
            if equivariant and self._irreps_edge is not None:
                dummy_feats = torch.zeros(1, self._irreps_in.dim, device=device, dtype=dtype)
                dummy_sh = torch.zeros(1, self._irreps_edge.dim, device=device, dtype=dtype)
                dummy_z = torch.zeros(1, dtype=torch.long, device=device)
            for (a, b), key in zip(self._dedicated_pairs, self._pair_keys):
                expert = typing.cast(PotentialArtisan, self.artisans[key])
                if equivariant:
                    dummy_out: torch.Tensor = expert.forward_equivariant(
                        dummy_feats, dummy_feats, dummy_sh, dummy_dist, dummy_z
                    )
                else:
                    _ = expert.project(atom_features)
                    dummy_out = expert(dummy_scalars, dummy_scalars, dummy_dist)
                atom_energies = atom_energies + 0.0 * dummy_out.sum()
                if return_per_pair:
                    per_pair_total[self.pair_label(a, b)] = torch.zeros(
                        (), device=device, dtype=dtype
                    )
            if self.rare_artisan_module is not None:
                _ = self.rare_artisan_module.project(atom_features)
                dummy_idx: torch.Tensor = torch.zeros(1, dtype=torch.long, device=device)
                dummy_re: torch.Tensor = self.rare_artisan_module(
                    dummy_scalars, dummy_scalars, dummy_dist, dummy_idx
                )
                atom_energies = atom_energies + 0.0 * dummy_re.sum()
                for a, b in self._rare_pairs:
                    if return_per_pair:
                        per_pair_total[self.pair_label(a, b)] = torch.zeros(
                            (), device=device, dtype=dtype
                        )
            return (atom_energies, per_pair_total) if return_per_pair else atom_energies

        # ----- Normal case: E > 0 -----
        cut_env: torch.Tensor = cosine_cutoff(edge_lengths, self._cutoff).to(dtype)

        z_row: torch.Tensor = atomic_numbers[row]
        z_col: torch.Tensor = atomic_numbers[col]
        z_lo: torch.Tensor = torch.minimum(z_row, z_col)
        z_hi: torch.Tensor = torch.maximum(z_row, z_col)

        # Edge spherical harmonics — geometry only, computed once and
        # shared by every equivariant artisan in the bank.
        edge_sh: torch.Tensor | None = None
        if equivariant and self._irreps_edge is not None:
            from e3nn.o3 import spherical_harmonics

            edge_sh = spherical_harmonics(  # (E, irreps_edge.dim)
                self._irreps_edge,
                typing.cast(torch.Tensor, edge_vectors).to(dtype),
                normalize=True,
                normalization="component",
            )

        # ---- Dedicated pair loop ----
        for (a, b), key in zip(self._dedicated_pairs, self._pair_keys):
            expert = typing.cast(PotentialArtisan, self.artisans[key])
            mask: torch.Tensor = ((z_lo == a) & (z_hi == b)).to(dtype)
            mask_col: torch.Tensor = mask.unsqueeze(-1)
            if equivariant:
                feats_a: torch.Tensor = atom_features[row] * mask_col
                feats_b: torch.Tensor = atom_features[col] * mask_col
                sh_in: torch.Tensor = typing.cast(torch.Tensor, edge_sh) * mask_col
                dist_in: torch.Tensor = edge_lengths * mask
                pair_e: torch.Tensor = expert.forward_equivariant(
                    feats_a, feats_b, sh_in, dist_in, z_col
                )
            else:
                scalars: torch.Tensor = expert.project(atom_features)
                scalars_a: torch.Tensor = scalars[row] * mask_col
                scalars_b: torch.Tensor = scalars[col] * mask_col
                dist_in = edge_lengths * mask
                pair_e = expert(scalars_a, scalars_b, dist_in)
            tapered: torch.Tensor = pair_e * mask * cut_env
            atom_energies = atom_energies.index_add(0, row, 0.5 * tapered)
            if return_per_pair:
                per_pair_total[self.pair_label(a, b)] = 0.5 * tapered.sum().detach()

        # ---- Rare pairs via shared RarePotentialArtisan ----
        if self.rare_artisan_module is not None:
            rare_scalars: torch.Tensor = self.rare_artisan_module.project(atom_features)  # (N, S)
            for local_idx, (a, b) in enumerate(self._rare_pairs):
                mask = ((z_lo == a) & (z_hi == b)).to(dtype)
                mask_col = mask.unsqueeze(-1)
                sa: torch.Tensor = rare_scalars[row] * mask_col
                sb: torch.Tensor = rare_scalars[col] * mask_col
                di: torch.Tensor = edge_lengths * mask
                pidx: torch.Tensor = torch.full((E,), local_idx, dtype=torch.long, device=device)
                pair_e = self.rare_artisan_module(sa, sb, di, pidx)
                tapered = pair_e * mask * cut_env
                atom_energies = atom_energies.index_add(0, row, 0.5 * tapered)
                if return_per_pair:
                    per_pair_total[self.pair_label(a, b)] = 0.5 * tapered.sum().detach()

        return (atom_energies, per_pair_total) if return_per_pair else atom_energies

    # ------------------------------------------------------------------
    # Pairwise force mode (TASK 4)
    # ------------------------------------------------------------------

    def forward_pairwise(
        self,
        atom_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-atom energies AND per-atom forces via pairwise mode.

        Each :class:`PotentialArtisan` produces ``(E_ij, F_ij)`` with
        ``F_ij = -∂E_ij/∂r_ij`` via autograd; the artisan bank applies the
        per-pair mask and cosine-cutoff envelope and scatter-sums to
        atoms with Newton's third law (``+0.5 F`` to row, ``-0.5 F``
        to col).  The factor of ``0.5`` accounts for the bidirectional
        edge convention — each undirected pair appears twice as a
        directed edge.

        Returns
        -------
        tuple of Tensor
            ``(atom_energies, atom_forces)`` with shapes ``(N,)`` and
            ``(N, 3)``.  Newton's third law: ``sum_i F_i ≡ 0`` per
            molecule by construction.
        """
        if self._architecture != "scalar":
            raise RuntimeError(
                "forward_pairwise() is only implemented for "
                "architecture='scalar'; use the autograd force path for "
                "equivariant artisans."
            )
        num_atoms: int = atom_features.shape[0]
        device: torch.device = atom_features.device
        dtype: torch.dtype = atom_features.dtype

        row, col = edge_index
        E: int = int(edge_vectors.shape[0])

        atom_energies: torch.Tensor = torch.zeros(num_atoms, device=device, dtype=dtype)
        atom_forces: torch.Tensor = torch.zeros((num_atoms, 3), device=device, dtype=dtype)

        # Degenerate case: keep all artisans in the autograd graph.
        if E == 0:
            first_key: str = self._pair_keys[0] if self._pair_keys else ""
            scalar_channels: int = (
                self.artisans[first_key].scalar_channels
                if first_key
                else (self.rare_artisan_module.scalar_channels if self.rare_artisan_module else 1)
            )
            dummy_scalars: torch.Tensor = torch.zeros(
                1, scalar_channels, device=device, dtype=dtype
            )
            dummy_dist: torch.Tensor = torch.zeros(1, device=device, dtype=dtype)
            for _, key in zip(self._dedicated_pairs, self._pair_keys):
                expert = typing.cast(PotentialArtisan, self.artisans[key])
                _ = expert.project(atom_features)
                dummy_out: torch.Tensor = expert(dummy_scalars, dummy_scalars, dummy_dist)
                atom_energies = atom_energies + 0.0 * dummy_out.sum()
            if self.rare_artisan_module is not None:
                _ = self.rare_artisan_module.project(atom_features)
                dummy_idx_re: torch.Tensor = torch.zeros(1, dtype=torch.long, device=device)
                dummy_re: torch.Tensor = self.rare_artisan_module(
                    dummy_scalars, dummy_scalars, dummy_dist, dummy_idx_re
                )
                atom_energies = atom_energies + 0.0 * dummy_re.sum()
            return atom_energies, atom_forces

        edge_vectors_dt: torch.Tensor = edge_vectors.to(dtype)
        edge_lengths: torch.Tensor = edge_vectors_dt.norm(dim=-1)
        cut_env: torch.Tensor = cosine_cutoff(edge_lengths, self._cutoff).to(dtype)
        cut_env_v: torch.Tensor = cut_env.unsqueeze(-1)

        z_row: torch.Tensor = atomic_numbers[row]
        z_col: torch.Tensor = atomic_numbers[col]
        z_lo: torch.Tensor = torch.minimum(z_row, z_col)
        z_hi: torch.Tensor = torch.maximum(z_row, z_col)

        # ---- Dedicated pair loop ----
        for (a, b), key in zip(self._dedicated_pairs, self._pair_keys):
            expert = typing.cast(PotentialArtisan, self.artisans[key])
            mask: torch.Tensor = ((z_lo == a) & (z_hi == b)).to(dtype)
            mask_v: torch.Tensor = mask.unsqueeze(-1)
            scalars: torch.Tensor = expert.project(atom_features)
            scalars_a: torch.Tensor = scalars[row] * mask_v
            scalars_b: torch.Tensor = scalars[col] * mask_v
            e_ij, f_ij = expert.forward_pairwise(scalars_a, scalars_b, edge_vectors_dt)
            tapered_e: torch.Tensor = e_ij * mask * cut_env
            tapered_f: torch.Tensor = f_ij * mask_v * cut_env_v
            atom_energies = atom_energies.index_add(0, row, 0.5 * tapered_e)
            atom_forces = atom_forces.index_add(0, row, 0.5 * tapered_f)
            atom_forces = atom_forces.index_add(0, col, -0.5 * tapered_f)

        # ---- Rare pairs via shared RarePotentialArtisan (energy only, no pairwise forces) ----
        # RarePotentialArtisan doesn't implement forward_pairwise; use regular forward
        # and let the backbone's autograd head derive forces from positions.
        if self.rare_artisan_module is not None:
            rare_scalars: torch.Tensor = self.rare_artisan_module.project(atom_features)
            for local_idx, (a, b) in enumerate(self._rare_pairs):
                mask = ((z_lo == a) & (z_hi == b)).to(dtype)
                mask_col_re = mask.unsqueeze(-1)
                sa_re: torch.Tensor = rare_scalars[row] * mask_col_re
                sb_re: torch.Tensor = rare_scalars[col] * mask_col_re
                di_re: torch.Tensor = edge_lengths * mask
                pidx_re: torch.Tensor = torch.full(
                    (E,), local_idx, dtype=torch.long, device=device
                )
                pair_e_re: torch.Tensor = self.rare_artisan_module(sa_re, sb_re, di_re, pidx_re)
                tapered_re: torch.Tensor = pair_e_re * mask * cut_env
                atom_energies = atom_energies.index_add(0, row, 0.5 * tapered_re)

        return atom_energies, atom_forces


# ---------------------------------------------------------------------------
# ARACE artisans (ARtisan + Atomic Cluster Expansion)
# ---------------------------------------------------------------------------


class _AracePairArtisan(nn.Module):
    """One equivariant element-pair artisan for the ARACE variant.

    Identical in spirit to :class:`_EquivariantArtisanCore`, but in
    addition to the per-edge scalar energy it also *returns the
    equivariant pair representation* ``h_pair`` so the ACE block can
    aggregate it back onto nodes.  All learnable maps are bias-free so
    zero-masked edges contribute exactly zero.

    Every architectural dimension is exposed as a constructor argument
    (``hidden_irreps``, ``num_layers``, ``num_rbf``, ``radial_hidden``,
    ``n_scalar_out``, ``final_hidden``) so all of them are reachable
    from Hydra configs and hyperparameter-search spaces.

    Parameters mirror the ``artisan`` sub-config of the ARACE models;
    see :class:`goal.ml.nn.models.simurgh.arace.MonolithicArace`.
    """

    def __init__(
        self,
        irreps_node: Irreps | str,
        cutoff: float,
        hidden_irreps: Irreps | str = "16x0e + 16x1o + 16x2e",
        num_layers: int = 1,
        num_rbf: int = 8,
        radial_hidden: int = 32,
        n_scalar_out: int = 16,
        final_hidden: int = 16,
        element_conditioned: bool = True,
        n_elements: int = 120,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        self.irreps_node: Irreps = Irreps(irreps_node)
        self.irreps_hidden: Irreps = Irreps(hidden_irreps)
        lmax: int = max((ir.l for _, ir in self.irreps_hidden), default=0)
        self.irreps_edge: Irreps = build_artisan_edge_irreps(lmax)
        self._element_conditioned: bool = bool(element_conditioned)
        self._n_elements: int = int(n_elements)

        # Shared symmetric node embedding — one map applied to both
        # endpoints guarantees E(A, B) = E(B, A) under the sum.
        self.node_embed: EquivariantLinear = EquivariantLinear(
            self.irreps_node, self.irreps_hidden, biases=False
        )

        # CG tensor product with bond geometry.
        self.tp: WeightedTensorProduct = WeightedTensorProduct(
            irreps_in1=self.irreps_hidden,
            irreps_in2=self.irreps_edge,
            irreps_out=self.irreps_hidden,
        )
        self.radial_basis: BesselBasis = BesselBasis(num_basis=num_rbf, cutoff=cutoff)
        self.envelope: PolynomialEnvelope = PolynomialEnvelope(cutoff=cutoff)
        self.radial_mlp: RadialMLP = RadialMLP(
            num_basis=num_rbf,
            hidden_dim=radial_hidden,
            num_out=self.tp.weight_numel,
        )
        self.element_linear: nn.Linear | None = (
            nn.Linear(self._n_elements, self.tp.weight_numel, bias=False)
            if element_conditioned
            else None
        )

        # Optional deeper equivariant layers.
        self.layers: nn.ModuleList = nn.ModuleList(
            EquivariantLinear(self.irreps_hidden, self.irreps_hidden, biases=False)
            for _ in range(num_layers - 1)
        )
        self.norms: nn.ModuleList = nn.ModuleList(
            _PairRMSNorm(self.irreps_hidden) for _ in range(num_layers - 1)
        )

        # Invariant scalar readout (bias-free).
        scalar_irreps: Irreps = Irreps(f"{int(n_scalar_out)}x0e")
        self.to_scalars: EquivariantLinear = EquivariantLinear(
            self.irreps_hidden, scalar_irreps, biases=False
        )
        self.energy_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(int(n_scalar_out), int(final_hidden), bias=False),
            nn.SiLU(),
            nn.Linear(int(final_hidden), 1, bias=False),
        )

        # Learnable gate, initialised to 1.0.
        self.gate: nn.Parameter = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        feats_a: torch.Tensor,
        feats_b: torch.Tensor,
        edge_sh: torch.Tensor,
        distances: torch.Tensor,
        z_dst: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-edge pair representation and gated scalar energy.

        Parameters
        ----------
        feats_a, feats_b : Tensor ``(E, irreps_node.dim)``
            Equivariant node features of source / destination atom.
        edge_sh : Tensor ``(E, irreps_edge.dim)``
            Spherical harmonics of the edge direction.
        distances : Tensor ``(E,)``
            Bond lengths (zeros on masked edges are clamped internally).
        z_dst : Tensor ``(E,)``, optional
            Destination atomic numbers (element conditioning).

        Returns
        -------
        tuple of Tensor
            ``(h_pair, e_pair)`` with shapes ``(E, irreps_hidden.dim)``
            and ``(E,)``; ``e_pair`` is already multiplied by the gate.
        """
        h_ab: torch.Tensor = self._embed_endpoints(feats_a, feats_b)
        radial_w: torch.Tensor = self._radial_weights(distances, z_dst, h_ab.dtype)
        h_pair: torch.Tensor = self._pair_features(h_ab, edge_sh, radial_w)

        return h_pair, self._readout(h_pair)

    # ------------------------------------------------------------------
    # Forward, in pieces
    # ------------------------------------------------------------------
    #
    # ``forward`` is split so :class:`_AraceRarePairArtisan` can reuse
    # every stage and insert exactly one extra term (a pair-type
    # modulation of the tensor-product weights) without copying the
    # equivariant core.  Each helper is a straight extract — the
    # composition below is the original forward, line for line.

    def _embed_endpoints(self, feats_a: torch.Tensor, feats_b: torch.Tensor) -> torch.Tensor:
        """Symmetric node embedding: one map on both endpoints, summed."""
        return self.node_embed(feats_a) + self.node_embed(feats_b)

    def _radial_weights(
        self,
        distances: torch.Tensor,
        z_dst: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Envelope-damped radial weights for the tensor product."""
        d_safe: torch.Tensor = distances.clamp_min(1e-6)
        rbf: torch.Tensor = self.radial_basis(d_safe)  # (E, num_rbf)
        env: torch.Tensor = self.envelope(d_safe).to(dtype).unsqueeze(-1)  # (E, 1)
        radial_w: torch.Tensor = self.radial_mlp(rbf) * env  # (E, weight_numel)

        if self._element_conditioned and self.element_linear is not None:
            if z_dst is None:
                raise ValueError("element_conditioned=True requires z_dst in forward()")
            one_hot: torch.Tensor = F.one_hot(
                z_dst.clamp(0, self._n_elements - 1),
                num_classes=self._n_elements,
            ).to(radial_w.dtype)
            radial_w = radial_w * self.element_linear(one_hot)
        return radial_w

    def _pair_features(
        self,
        h_ab: torch.Tensor,
        edge_sh: torch.Tensor,
        radial_w: torch.Tensor,
    ) -> torch.Tensor:
        """CG tensor product with the bond geometry, plus optional depth."""
        h_pair: torch.Tensor = self.tp(h_ab, edge_sh.to(h_ab.dtype), radial_w)
        for lin, norm in zip(self.layers, self.norms):
            h_pair = norm(lin(h_pair) + h_pair)
        return h_pair

    def _readout(self, h_pair: torch.Tensor) -> torch.Tensor:
        """Invariant scalar readout → gated per-edge energy."""
        scalars: torch.Tensor = self.to_scalars(h_pair)  # (E, n_scalar_out)
        return self.gate * self.energy_mlp(scalars).squeeze(-1)  # (E,)


class _AraceRarePairArtisan(_AracePairArtisan):
    """One shared ARACE artisan for every pair that has no dedicated one.

    The ARACE counterpart of :class:`RarePotentialArtisan`, and it exists
    because the two cannot be the same class: a rare artisan on this path
    must return ``(h_pair, e_pair)`` — the ACE block aggregates the
    equivariant pair representation back onto nodes — whereas
    :class:`RarePotentialArtisan` returns a scalar energy alone.

    Specialisation comes from a learned **element** embedding rather than
    a per-rare-pair table:

        ``pair_code = embed(Z_lo) + embed(Z_hi)``

    and ``pair_code`` modulates the tensor-product weights exactly as
    ``element_linear`` does for the destination element.  Indexing by
    element rather than by pair means the module is defined for any pair,
    including one it never saw — so while it exists it also absorbs edges
    whose elements are outside the model's declared ``elements`` list.
    That is a side effect, not the contract: this module is built only
    when the frequency threshold actually routes pairs to it, and a
    structure outside the declared chemistry is outside what the bank
    covers either way.  The sum is symmetric, so ``E(A, B) = E(B, A)``
    survives.

    Parameters
    ----------
    irreps_node, cutoff, **artisan_kwargs
        Exactly as :class:`_AracePairArtisan` — the rare artisan has the
        same architecture as a dedicated one, so a pair does not change
        shape when it crosses the routing threshold.
    pair_embed_dim : int
        Width of the element embedding whose sum forms the pair code.
    """

    def __init__(
        self,
        irreps_node: Irreps | str,
        cutoff: float,
        pair_embed_dim: int = 16,
        **artisan_kwargs: typing.Any,
    ) -> None:
        super().__init__(irreps_node=irreps_node, cutoff=cutoff, **artisan_kwargs)
        self._pair_embed_dim: int = int(pair_embed_dim)
        self.pair_embed: nn.Embedding = nn.Embedding(self._n_elements, self._pair_embed_dim)
        # Modulates the same weights ``element_linear`` does, and is built
        # the same way (bias-free linear onto ``tp.weight_numel``), so the
        # two conditioning paths compose rather than fight.
        self.pair_linear: nn.Linear = nn.Linear(
            self._pair_embed_dim, self.tp.weight_numel, bias=False
        )

    def forward(  # type: ignore[override]
        self,
        feats_a: torch.Tensor,
        feats_b: torch.Tensor,
        edge_sh: torch.Tensor,
        distances: torch.Tensor,
        z_dst: torch.Tensor | None = None,
        z_pair: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """As :meth:`_AracePairArtisan.forward`, plus pair-type conditioning.

        Parameters
        ----------
        z_pair : Tensor ``(E, 2)`` int, required
            ``(Z_lo, Z_hi)`` per edge, canonically ordered.  Any element
            pair is accepted, including one absent from the model's
            ``elements`` list.
        """
        if z_pair is None:
            raise ValueError("_AraceRarePairArtisan.forward() requires z_pair")

        h_ab: torch.Tensor = self._embed_endpoints(feats_a, feats_b)
        radial_w: torch.Tensor = self._radial_weights(distances, z_dst, h_ab.dtype)

        codes: torch.Tensor = self.pair_embed(z_pair.clamp(0, self._n_elements - 1))
        pair_code: torch.Tensor = codes.sum(dim=1).to(radial_w.dtype)  # (E, embed_dim)
        radial_w = radial_w * self.pair_linear(pair_code)

        h_pair: torch.Tensor = self._pair_features(h_ab, edge_sh, radial_w)
        return h_pair, self._readout(h_pair)


def route_element_pairs(
    pairs: typing.Sequence[tuple[int, int]],
    pair_counts: typing.Mapping[tuple[int, int], int] | None,
    min_pair_frequency: float,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Split *pairs* into ``(dedicated, rare)`` by training-set frequency.

    A pair earns a dedicated artisan when its share of the training set's
    directed edges reaches *min_pair_frequency*.  Everything else — rare
    pairs **and pairs that never occur at all**, which have frequency
    zero — goes to the shared rare artisan.

    Because frequencies sum to one, at most ``1 / min_pair_frequency``
    pairs can be dedicated; the actual count depends on how skewed the
    corpus is.  Measured on OMol25 at a 6 Å cutoff: the default 0.01
    keeps 9 pairs (91% of edges), 1e-4 keeps 104 (99.1%), and 5e-5 keeps
    159 (99.5%).

    ``pair_counts=None`` means "no routing information", and every pair
    is dedicated — the pre-routing behaviour.

    Parameters
    ----------
    pairs : sequence of (int, int)
        All unordered element pairs, canonically ordered.
    pair_counts : mapping, optional
        ``{(Z_A, Z_B): directed_edge_count}`` over the training set, as
        produced by :func:`goal.ml.data.statistics.compute_pair_counts`.
    min_pair_frequency : float
        Fraction-of-edges threshold.

    Returns
    -------
    tuple
        ``(dedicated, rare)``, each a tuple of pairs, together covering
        *pairs* exactly once.
    """
    if pair_counts is None:
        return tuple(pairs), ()

    total: int = max(1, sum(int(v) for v in pair_counts.values()))
    dedicated: list[tuple[int, int]] = []
    rare: list[tuple[int, int]] = []
    for pair in pairs:
        count: int = int(pair_counts.get(pair, pair_counts.get((pair[1], pair[0]), 0)))
        (dedicated if count / total >= min_pair_frequency else rare).append(pair)
    return tuple(dedicated), tuple(rare)


def parse_pair_counts(raw: typing.Any) -> dict[tuple[int, int], int] | None:
    """Normalise a configured ``pair_counts`` into ``{(Z_lo, Z_hi): count}``.

    Element pairs are tuples, and YAML has no tuple keys — so the value
    that reaches a config file cannot be the value the router wants.
    Three spellings are accepted, and they are the three that occur:

    * ``{(1, 6): 12}`` — tuple keys, what
      :func:`goal.ml.data.statistics.compute_pair_counts` returns and what
      ``train.py`` passes in memory;
    * ``{"1_6": 12}`` — string keys, the form written to the run's saved
      ``config.yaml`` so that rebuilding the model from it reproduces the
      same artisan bank (routing changes *which modules exist*, so a
      checkpoint saved with routing cannot load into a model built
      without it);
    * ``[[1, 6, 12], ...]`` — triples, for hand-written configs.

    ``None`` passes through as ``None`` ("no routing information").
    """
    if raw is None:
        return None

    counts: dict[tuple[int, int], int] = {}
    if isinstance(raw, typing.Mapping):
        for key, value in dict(raw).items():
            if isinstance(key, str):
                lo_text, _, hi_text = key.partition("_")
                if not hi_text:
                    raise ValueError(f"pair_counts key {key!r} is not of the form 'Zlo_Zhi'.")
                pair: tuple[int, int] = (int(lo_text), int(hi_text))
            else:
                pair = (int(key[0]), int(key[1]))
            counts[pair] = counts.get(pair, 0) + int(value)
        return counts

    for entry in raw:
        items: list[typing.Any] = list(entry)
        if len(items) != 3:
            raise ValueError(f"pair_counts entry {entry!r} must be [Z_lo, Z_hi, count].")
        pair = (int(items[0]), int(items[1]))
        counts[pair] = counts.get(pair, 0) + int(items[2])
    return counts


def encode_pair_counts(counts: typing.Mapping[tuple[int, int], int]) -> dict[str, int]:
    """``{(1, 6): 12}`` → ``{"1_6": 12}``, the YAML-safe form."""
    return {f"{int(a)}_{int(b)}": int(v) for (a, b), v in counts.items()}


def log_pair_routing(
    owner: str,
    elements: typing.Sequence[int],
    all_pairs: typing.Sequence[tuple[int, int]],
    dedicated_pairs: typing.Sequence[tuple[int, int]],
    enabled: bool,
    min_pair_frequency: float,
) -> None:
    """Declare the bank's supported chemistry and how its pairs are covered.

    Two lines, not the per-pair table :class:`SimurghArtisanBank` prints:
    the whole reason routing exists is element counts where the table
    would be thousands of rows.

    The first line is the model's **contract** — the elements it was
    built for.  A structure containing anything else is outside what the
    bank covers: those edges match no artisan and contribute nothing.
    Printing the supported set at construction is how that boundary stays
    visible instead of being discovered from a suspiciously low energy.
    """
    from ase.data import chemical_symbols

    symbols: str = " ".join(chemical_symbols[int(z)] for z in elements)
    if len(symbols) > 96:
        symbols = symbols[:93] + "..."
    total: int = len(all_pairs)
    n_dedicated: int = len(dedicated_pairs)
    print(f"[{owner}] supports {len(elements)} elements ({symbols}) = {total:,} pairs")

    if not enabled:
        print(f"[{owner}] {total:,} dedicated artisans (rare-pair routing off)")
        return
    if n_dedicated == total:
        print(
            f"[{owner}] {total:,} dedicated artisans, no rare artisan "
            f"(every pair clears min_pair_frequency={min_pair_frequency:g})"
        )
        return
    print(
        f"[{owner}] {n_dedicated:,} dedicated artisans + 1 shared rare artisan "
        f"({total - n_dedicated:,} pairs below min_pair_frequency="
        f"{min_pair_frequency:g}, {1.0 - n_dedicated / max(1, total):.1%} "
        f"of the bank saved)"
    )


class ArtisanLayer(nn.Module):
    """Routes all edges to their element-pair ARACE artisan.

    One :class:`_AracePairArtisan` per unordered element pair — or, with
    frequency routing on (``pair_counts`` given and
    ``rare_pair_enabled=True``), one per *frequent* pair plus a single
    shared :class:`_AraceRarePairArtisan` for the rest.  That matters at
    scale: the pair count is quadratic in the element count (83 elements
    = 3,486 pairs), and because of the zero-masking schedule below every
    artisan costs both parameters *and* a full-edge forward pass on every
    step.  Routing turns both into a constant.

    The shared artisan is built **only when the threshold routes
    something**.  A bank whose every pair clears it is bit-for-bit the
    bank it was before routing existed — same modules, same parameter
    count, same state-dict keys.

    Every
    edge is routed to exactly one artisan with the same static-shape
    zero-masking schedule as :class:`SimurghArtisanBank` (inputs masked
    before the forward, outputs masked after — every artisan runs every
    step, DDP-safe).  The masked per-edge features of all artisans are
    summed into a single ``h_pair`` tensor (each edge matches exactly
    one pair type, so the sum is a routed select).

    Returns ``(h_pair, e_atom)`` per call — the equivariant edge
    features for the downstream ACE aggregation and this layer's
    per-atom energy contribution (gated, cosine-tapered, halved for the
    bidirectional edge convention).

    Parameters
    ----------
    elements : sequence of int
        Atomic numbers covered by the model (defines the pair routing).
    irreps_node : Irreps or str
        Irreps of the node features entering the layer.  Must equal the
        artisans' ``hidden_irreps`` so the edge → node aggregation and
        the residual connection downstream are well-typed.
    cutoff : float
        Cutoff radius (Angstrom) for the radial basis and the cosine
        taper on pair energies.
    artisan_kwargs : dict, optional
        Forwarded to every :class:`_AracePairArtisan` (``hidden_irreps``,
        ``num_layers``, ``num_rbf``, ``radial_hidden``, ``n_scalar_out``,
        ``final_hidden``, ``element_conditioned``, ``n_elements``).
    artisans : nn.ModuleDict, optional
        Pre-built artisan bank to **share** across layers
        (``share_artisan_weights=True``).  When ``None`` the layer
        builds its own independent bank.  With routing on it must hold
        the *dedicated* pairs only.
    pair_counts : mapping, optional
        ``{(Z_A, Z_B): directed_edge_count}`` over the training set.
        Required for routing; ``None`` dedicates every pair.
    rare_pair_enabled : bool
        Turn frequency routing on.  Needs ``pair_counts``.
    min_pair_frequency : float
        Share-of-edges threshold for a dedicated artisan.  See
        :func:`route_element_pairs`.
    rare_pair_embed_dim : int
        Element-embedding width inside the shared rare artisan.
    rare_artisan : _AraceRarePairArtisan, optional
        Pre-built rare artisan to **share** across layers, alongside
        ``artisans``.
    """

    def __init__(
        self,
        elements: typing.Sequence[int],
        irreps_node: Irreps | str,
        cutoff: float,
        artisan_kwargs: dict[str, typing.Any] | None = None,
        artisans: nn.ModuleDict | None = None,
        pair_counts: typing.Mapping[tuple[int, int], int] | None = None,
        rare_pair_enabled: bool = False,
        min_pair_frequency: float = 0.01,
        rare_pair_embed_dim: int = 16,
        rare_artisan: _AraceRarePairArtisan | None = None,
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
                "ArtisanLayer has neither a dedicated artisan nor a rare one. "
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

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        return self._pairs

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def rare_pairs(self) -> tuple[tuple[int, int], ...]:
        """Pairs routed to the shared rare artisan (empty when routing is off)."""
        return self._rare_pairs

    def gates(self) -> dict[str, torch.Tensor]:
        """Snapshot of every artisan gate, keyed by pretty pair label."""
        result: dict[str, torch.Tensor] = {}
        for (a, b), key in zip(self._pairs, self._pair_keys):
            artisan = typing.cast(_AracePairArtisan, self.artisans[key])
            result[pair_label(a, b)] = artisan.gate.detach()
        if self.rare_artisan is not None:
            result["rare"] = self.rare_artisan.gate.detach()
        return result

    def forward(
        self,
        h: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run all pair artisans over the edges.

        Parameters
        ----------
        h : Tensor ``(N, irreps_node.dim)``
            Current equivariant node features.
        atomic_numbers : Tensor ``(N,)``
        edge_index : Tensor ``(2, E)``
        edge_sh : Tensor ``(E, irreps_edge.dim)``
            Pre-computed edge spherical harmonics (geometry only, shared
            across all layers).
        edge_lengths : Tensor ``(E,)``

        Returns
        -------
        tuple of Tensor
            ``(h_pair, e_atom)`` — routed equivariant edge features
            ``(E, irreps_node.dim)`` and this layer's per-atom energy
            contribution ``(N,)``.
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
            h_pair_empty: torch.Tensor = torch.zeros(
                0, self.irreps_node.dim, device=device, dtype=dtype
            )
            return h_pair_empty, e_atom + dummy_sum

        edge_lengths = edge_lengths.to(dtype)
        cut_env: torch.Tensor = cosine_cutoff(edge_lengths, self._cutoff).to(dtype)

        z_row: torch.Tensor = atomic_numbers[row]
        z_col: torch.Tensor = atomic_numbers[col]
        z_lo: torch.Tensor = torch.minimum(z_row, z_col)
        z_hi: torch.Tensor = torch.maximum(z_row, z_col)

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
            # Every edge matches exactly one pair, and the dedicated masks
            # are disjoint — so "not claimed by anyone" is precisely the
            # rare set, computed in one step instead of re-testing every
            # rare pair.  It also catches pairs outside `elements`
            # altogether, which used to contribute a silent zero.
            h_pair_total, e_atom = self._apply_rare(
                h=h,
                row=row,
                col=col,
                z_col=z_col,
                z_lo=z_lo,
                z_hi=z_hi,
                edge_sh=edge_sh,
                edge_lengths=edge_lengths,
                cut_env=cut_env,
                mask=(1.0 - claimed).clamp_min(0.0),
                h_pair_total=h_pair_total,
                e_atom=e_atom,
                dtype=dtype,
            )

        return h_pair_total, e_atom

    def _apply_rare(
        self,
        h: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
        z_col: torch.Tensor,
        z_lo: torch.Tensor,
        z_hi: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_lengths: torch.Tensor,
        cut_env: torch.Tensor,
        mask: torch.Tensor,
        h_pair_total: torch.Tensor,
        e_atom: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One masked pass of the shared rare artisan over the leftover edges.

        Runs unconditionally, like every dedicated artisan — an all-zero
        mask still produces gradients of zero rather than no gradient,
        which is what keeps the graph static for DDP.
        """
        assert self.rare_artisan is not None
        mask_col: torch.Tensor = mask.unsqueeze(-1)
        h_p, e_p = self.rare_artisan(
            h[row] * mask_col,
            h[col] * mask_col,
            edge_sh.to(dtype) * mask_col,
            edge_lengths * mask,
            z_col,
            torch.stack([z_lo, z_hi], dim=-1),
        )
        h_pair_total = h_pair_total + h_p * mask_col
        return h_pair_total, e_atom.index_add(0, row, 0.5 * (e_p * mask * cut_env))
