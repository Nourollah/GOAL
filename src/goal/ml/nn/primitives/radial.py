"""Radial basis functions for encoding interatomic distances."""

from __future__ import annotations

import math
import typing

import torch
import torch.nn as nn


class BesselBasis(nn.Module):
    """Bessel radial basis functions with polynomial envelope.

    Used to expand scalar edge lengths into a rich feature vector before
    feeding into the radial MLP that produces tensor product weights.

    Parameters
    ----------
    num_basis : int
        Number of Bessel basis functions.
    cutoff : float
        Cutoff radius in Angstrom. Basis functions are zero beyond this.
    """

    def __init__(self, num_basis: int = 8, cutoff: float = 5.0) -> None:
        super().__init__()
        self.num_basis: int = num_basis
        self.cutoff: float = cutoff

        # Frequencies: n * pi / cutoff for n = 1, ..., num_basis
        freqs: torch.Tensor = (
            torch.arange(1, num_basis + 1, dtype=torch.float64) * math.pi / cutoff
        )
        self.register_buffer("freqs", freqs)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Expand distances into Bessel basis, shape ``(E,) → (E, num_basis)``."""
        d: torch.Tensor = distances.unsqueeze(-1)  # (E, 1)
        return (2.0 / self.cutoff) ** 0.5 * torch.sin(self.freqs * d) / d  # (E, num_basis)


class PolynomialEnvelope(nn.Module):
    """Smooth polynomial cutoff envelope that goes to zero at the cutoff.

    Ensures the radial features decay smoothly to zero at the cutoff
    distance, preventing discontinuities in forces.

    Parameters
    ----------
    cutoff : float
        Cutoff radius.
    p : int
        Polynomial order (higher = sharper decay near cutoff).
    """

    def __init__(self, cutoff: float, p: int = 6) -> None:
        super().__init__()
        self.cutoff: float = cutoff
        self.p: int = p

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute envelope values, shape ``(E,) → (E,)``."""
        d_scaled: torch.Tensor = distances / self.cutoff  # (E,)
        envelope: torch.Tensor = (  # (E,)
            1.0
            - ((self.p + 1) * (self.p + 2) / 2) * d_scaled.pow(self.p)
            + self.p * (self.p + 2) * d_scaled.pow(self.p + 1)
            - (self.p * (self.p + 1) / 2) * d_scaled.pow(self.p + 2)
        )
        return envelope * (distances < self.cutoff).float()  # (E,)


class RadialMLP(nn.Module):
    """MLP that maps radial basis features to tensor product weights.

    Parameters
    ----------
    num_basis : int
        Dimension of the radial basis input.
    hidden_dim : int
        Width of hidden layers.
    num_out : int
        Output dimension (must match ``WeightedTensorProduct.weight_numel``).
    num_layers : int
        Number of hidden layers.
    """

    def __init__(
        self,
        num_basis: int,
        hidden_dim: int,
        num_out: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim: int = num_basis
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_out))
        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, basis: torch.Tensor) -> torch.Tensor:
        """Map radial basis features to weights, shape ``(E, num_basis) → (E, num_out)``."""
        return self.net(basis)  # (E, num_out)
