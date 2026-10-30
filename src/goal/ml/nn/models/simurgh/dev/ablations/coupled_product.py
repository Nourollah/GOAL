"""Element-dependent channel mixing at every correlation order of the product basis.

A recursive channel-wise Clebsch-Gordan product with an element-indexed
linear map at each order (``φq = W_q[Z] (φ_{q-1} ⊗ A)``), added to the
symmetric product basis; its output map starts at zero, so a fresh model
equals SimurghDev.  The parameter cost is reported by
:attr:`SimurghCoupledProduct.extra_parameters`.
"""

from __future__ import annotations

import torch
from torch import nn

from goal.ml.nn.models.simurgh.dev.monolithic import (
    SimurghDev,
    _from_channels,
    _LadderProductBasis,
    _to_channels,
)
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY


class _SymmetricPlusCoupled(nn.Module):
    """``B = B_symmetric + B_coupled`` (channel layout in and out)."""

    def __init__(self, symmetric: nn.Module, irreps_a, irreps_out, correlation, num_elements):
        super().__init__()
        self.symmetric = symmetric
        self.coupled = _LadderProductBasis(irreps_a, irreps_out, correlation, False, num_elements)
        with torch.no_grad():
            self.coupled.out.weight.zero_()
        self.irreps_a, self.irreps_out = irreps_a, irreps_out

    def forward(self, a: torch.Tensor, elem: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Sum of both product bases for channel-layout ``A``."""
        coupled = self.coupled(_from_channels(a, self.irreps_a), elem, emb)
        return self.symmetric(a, elem, emb) + _to_channels(coupled, self.irreps_out)


@MODEL_REGISTRY.register("simurgh_dev_abl_coupled_product")
@BACKBONE_REGISTRY.register("simurgh_dev_abl_coupled_product")
class SimurghCoupledProduct(SimurghDev):
    """SIMURGH whose product basis adds an element-coupled correlation ladder."""

    def __init__(self, **config) -> None:
        super().__init__(**config)
        if self.config["product_basis"] != "symmetric":
            raise ValueError("the coupled-product ablation extends product_basis: symmetric")
        before = sum(p.numel() for p in self.parameters())
        for layer in self.atom_line.layers:
            layer.product = _SymmetricPlusCoupled(
                layer.product,
                layer.irreps_a,
                layer.irreps_out,
                self.config["correlation"],
                len(self.elements),
            ).to(self.scale.dtype)
        self.extra_parameters = sum(p.numel() for p in self.parameters()) - before
