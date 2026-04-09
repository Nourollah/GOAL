"""Tensor product wrappers — the core equivariant operation."""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import FullyConnectedTensorProduct, Irreps


class WeightedTensorProduct(nn.Module):
    """Weighted fully-connected tensor product.

    Wraps ``e3nn.o3.FullyConnectedTensorProduct`` with a clean interface.
    The weights are typically produced by a radial MLP that takes edge
    lengths as input.

    Parameters
    ----------
    irreps_in1 : str or Irreps
        Irreps of the first input (typically node features).
    irreps_in2 : str or Irreps
        Irreps of the second input (typically spherical harmonics of edges).
    irreps_out : str or Irreps
        Output irreps.
    """

    def __init__(
        self,
        irreps_in1: typing.Union[str, Irreps],
        irreps_in2: typing.Union[str, Irreps],
        irreps_out: typing.Union[str, Irreps],
    ) -> None:
        super().__init__()
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        self._tp = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out,
        )

    @property
    def weight_numel(self) -> int:
        """Number of weights needed from the radial network."""
        return self._tp.weight_numel

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weight: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the tensor product.

        Parameters
        ----------
        x1 : Tensor
            First input, shape ``(N, irreps_in1.dim)``.
        x2 : Tensor
            Second input, shape ``(N, irreps_in2.dim)``.
        weight : Tensor or None
            External weights from a radial network, shape ``(N, weight_numel)``.
        """
        return self._tp(x1, x2, weight)                                     # (N, irreps_out.dim)
