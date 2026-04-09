"""Equivariant linear layers — thin wrappers around ``e3nn.o3.Linear``."""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from e3nn.o3 import Linear as E3Linear


class EquivariantLinear(nn.Module):
    """Type-safe wrapper around ``e3nn.o3.Linear``.

    Maps between two irrep spaces without mixing angular momentum
    channels that would break equivariance.

    Parameters
    ----------
    irreps_in : str or Irreps
        Input irreducible representations, e.g. ``'128x0e + 64x1o'``.
    irreps_out : str or Irreps
        Output irreducible representations.
    biases : bool
        Whether to include biases (only allowed for scalars ``l=0``).
    """

    def __init__(
        self,
        irreps_in: str | Irreps,
        irreps_out: str | Irreps,
        biases: bool = True,
    ) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self._linear = E3Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            biases=biases,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply equivariant linear transformation."""
        return self._linear(x)  # (N, irreps_out.dim)
