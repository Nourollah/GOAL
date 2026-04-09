"""Equivariant normalisation layers."""

from __future__ import annotations

import typing

import torch
import torch.nn as nn
from e3nn.o3 import Irreps


class EquivariantLayerNorm(nn.Module):
    """Per-irrep layer normalisation that preserves equivariance.

    Scalars (l=0) are normalised with standard LayerNorm.
    Higher-order features (l>0) are normalised by their norm only
    (no learned shift), which preserves rotational equivariance.

    Parameters
    ----------
    irreps : str or Irreps
        The irrep specification of the input features.
    eps : float
        Small constant for numerical stability.
    """

    def __init__(self, irreps: typing.Union[str, Irreps], eps: float = 1e-5) -> None:
        super().__init__()
        self.irreps = Irreps(irreps)
        self.eps = eps

        num_scalars = sum(mul for mul, ir in self.irreps if ir.l == 0)
        self.scalar_norm = nn.LayerNorm(num_scalars, eps=eps) if num_scalars > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply equivariant normalisation."""
        outputs = []
        idx = 0
        scalar_features = []

        for mul, ir in self.irreps:
            dim = mul * ir.dim
            chunk = x[:, idx : idx + dim]                                   # (N, mul * ir.dim)
            idx += dim

            if ir.l == 0:
                scalar_features.append(chunk)                               # (N, mul)
            else:
                # Normalise by the norm of each irrep block
                chunk_view = chunk.reshape(-1, mul, ir.dim)                 # (N, mul, 2l+1)
                norms = chunk_view.norm(dim=-1, keepdim=True).clamp(min=self.eps)  # (N, mul, 1)
                chunk_view = chunk_view / norms                             # (N, mul, 2l+1)
                outputs.append(chunk_view.reshape(-1, dim))                 # (N, mul * ir.dim)

        if scalar_features and self.scalar_norm is not None:
            scalars = torch.cat(scalar_features, dim=-1)                    # (N, total_scalars)
            scalars = self.scalar_norm(scalars)                             # (N, total_scalars)
            outputs.insert(0, scalars)

        return torch.cat(outputs, dim=-1)                                   # (N, irreps.dim)
