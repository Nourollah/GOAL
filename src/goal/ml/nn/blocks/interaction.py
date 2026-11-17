"""Equivariant message-passing interaction blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.o3 import Irreps, spherical_harmonics

from goal.ml.nn.primitives.linear import EquivariantLinear
from goal.ml.nn.primitives.norm import EquivariantLayerNorm
from goal.ml.nn.primitives.radial import BesselBasis, PolynomialEnvelope, RadialMLP
from goal.ml.nn.primitives.tp import WeightedTensorProduct


class EquivariantInteractionBlock(nn.Module):
    """One layer of equivariant message passing.

    Messages are constructed via a tensor product of sender node features
    with spherical harmonics of edge vectors, weighted by a radial MLP.
    Optionally, the TP weights are further modulated by a per-element linear
    map conditioned on the *destination* (central) atom's element type
    (``element_conditioned=True``).

    Parameters
    ----------
    irreps_node : str or Irreps
        Irreps of input and output node features.
    irreps_edge : str or Irreps
        Irreps of edge spherical harmonics (e.g. ``'1x0e + 1x1o + 1x2e'``).
    num_basis : int
        Number of radial basis functions.
    cutoff : float
        Cutoff radius for the radial basis and envelope.
    hidden_dim : int
        Width of the radial MLP hidden layers.
    avg_num_neighbors : float, optional
        Mean number of neighbours per atom over the training set.
        Used to normalise the sum-aggregated message so message magnitude
        is invariant to dataset density.  ``None`` skips normalisation.
    agg_norm_exponent : float, optional
        Exponent applied to ``avg_num_neighbors`` in the normalisation:
        ``scale = 1 / N̄^exponent``.

        * ``1.0`` (default) — divide by ``N̄``, matching MACE exactly.
        * ``0.5`` — divide by ``sqrt(N̄)``, the original SIMURGH behaviour.

        Ignored when ``avg_num_neighbors`` is ``None``.
    element_conditioned : bool, optional
        When ``True``, multiply the radial TP weights by a per-element
        linear map ``Linear(n_elements, weight_numel, bias=False)``
        indexed on the destination atom's element type.  This lets a
        carbon central atom and a nitrogen central atom produce different
        messages from identical geometric environments.  Requires
        ``n_elements`` to also be set.
    n_elements : int, optional
        Size of the one-hot element vocabulary.  Only used when
        ``element_conditioned=True``.
    """

    def __init__(
        self,
        irreps_node: str | Irreps,
        irreps_edge: str | Irreps,
        num_basis: int = 8,
        cutoff: float = 5.0,
        hidden_dim: int = 64,
        avg_num_neighbors: float | None = None,
        agg_norm_exponent: float = 1.0,
        element_conditioned: bool = False,
        n_elements: int = 120,
    ) -> None:
        super().__init__()
        self.irreps_node = Irreps(irreps_node)
        self.irreps_edge = Irreps(irreps_edge)
        self._element_conditioned: bool = bool(element_conditioned)
        self._n_elements: int = int(n_elements)

        # Radial basis and envelope
        self.radial_basis = BesselBasis(num_basis=num_basis, cutoff=cutoff)
        self.envelope = PolynomialEnvelope(cutoff=cutoff)

        # Tensor product: node_feats ⊗ edge_sh → message
        self.tp = WeightedTensorProduct(
            irreps_in1=self.irreps_node,
            irreps_in2=self.irreps_edge,
            irreps_out=self.irreps_node,
        )

        # Radial MLP: rbf → (E, weight_numel)
        self.radial_mlp = RadialMLP(
            num_basis=num_basis,
            hidden_dim=hidden_dim,
            num_out=self.tp.weight_numel,
        )

        # Element-conditioned weight modulation (CHANGE 1).
        # element_linear: one_hot(dst_Z) → (N, weight_numel), no bias so
        # the radial profile is unchanged on average at init.
        self.element_linear: nn.Linear | None = (
            nn.Linear(n_elements, self.tp.weight_numel, bias=False)
            if element_conditioned
            else None
        )

        # Post-message linear + layer norm
        self.linear = EquivariantLinear(self.irreps_node, self.irreps_node)
        self.norm = EquivariantLayerNorm(self.irreps_node)

        # Aggregation normalisation stored as a scalar buffer so the
        # forward path is a single multiply.  Shape () so torch.compile
        # and DDP see a constant-shape buffer regardless of mode.
        _exp: float = float(agg_norm_exponent)
        norm_scale: float = (
            1.0 / (float(avg_num_neighbors) ** _exp)
            if avg_num_neighbors is not None and float(avg_num_neighbors) > 0.0
            else 1.0
        )
        self.register_buffer(
            "agg_norm_scale",
            torch.tensor(norm_scale, dtype=torch.get_default_dtype()),
        )
        self._avg_num_neighbors: float | None = (
            float(avg_num_neighbors) if avg_num_neighbors is not None else None
        )
        self._agg_norm_exponent: float = _exp

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_lengths: torch.Tensor,
        edge_sh: torch.Tensor | None = None,
        atomic_numbers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute one round of equivariant message passing.

        Parameters
        ----------
        node_feats : Tensor
            Node features, shape ``(N, irreps_node.dim)``.
        edge_index : Tensor
            Edge indices, shape ``(2, E)``.
        edge_vectors : Tensor
            Edge displacement vectors, shape ``(E, 3)``.
        edge_lengths : Tensor
            Edge lengths, shape ``(E,)``.
        edge_sh : Tensor, optional
            Pre-computed spherical harmonics, shape ``(E, irreps_edge.dim)``.
            When provided, skip the internal SH computation (the caller
            computes them once and reuses them across layers).
            When ``None``, SH are computed from ``edge_vectors`` internally
            (backward-compatible behaviour).
        atomic_numbers : Tensor, optional
            Integer atomic numbers ``(N,)`` for element-conditioned TP
            weights.  Required when ``element_conditioned=True``.

        Returns
        -------
        Tensor
            Updated node features with residual connection, shape
            ``(N, irreps_node.dim)``.
        """
        row, col = edge_index  # row = src (sender), col = dst (receiver)

        # Boundary cast: align edge geometry to node_feats dtype.
        edge_vectors = edge_vectors.to(node_feats.dtype)
        edge_lengths = edge_lengths.to(node_feats.dtype)

        # ---- Spherical harmonics (CHANGE 2) ----
        # Use pre-computed SH if supplied; otherwise compute inline.
        if edge_sh is None:
            edge_sh = spherical_harmonics(  # (E, irreps_edge.dim)
                self.irreps_edge,
                edge_vectors,
                normalize=True,
                normalization="component",
            )
        else:
            edge_sh = edge_sh.to(node_feats.dtype)

        # ---- Radial weights ----
        rbf = self.radial_basis(edge_lengths)  # (E, num_basis)
        env = self.envelope(edge_lengths).unsqueeze(-1)  # (E, 1)
        tp_weights = self.radial_mlp(rbf) * env  # (E, weight_numel)

        # ---- Element conditioning on TP weights (CHANGE 1) ----
        # Multiply radial weights by a per-element scale derived from the
        # destination (central) atom's element type.  Each element gets its
        # own weight_numel-dimensional modulation vector; the modulation is
        # a pointwise scale so the radial profile is preserved in direction.
        if self._element_conditioned and self.element_linear is not None:
            if atomic_numbers is None:
                raise ValueError("element_conditioned=True requires atomic_numbers in forward()")
            one_hot = F.one_hot(
                atomic_numbers.clamp(0, self._n_elements - 1),
                num_classes=self._n_elements,
            ).to(
                tp_weights.dtype
            )  # (N, n_elements)
            elem_scale = self.element_linear(one_hot)  # (N, weight_numel)
            tp_weights = tp_weights * elem_scale[col]  # (E, weight_numel)

        # ---- Messages via tensor product ----
        sender_feats = node_feats[row]  # (E, irreps_node.dim)
        messages = self.tp(sender_feats, edge_sh, tp_weights)  # (E, irreps_node.dim)

        # ---- Aggregate + normalise ----
        agg = torch.zeros_like(node_feats)
        agg.index_add_(0, col, messages)
        agg = agg * self.agg_norm_scale.to(agg.dtype)  # 1 / N̄^exponent

        # ---- Post-process + residual ----
        out = self.linear(agg)
        out = self.norm(out)
        return node_feats + out
