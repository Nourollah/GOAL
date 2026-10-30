"""O(2)-equivariant primitives for features in an edge frame.

Layout ``[edge, channel, m]`` with ``m = -M..M`` at index ``m + M``.  The
degree ``l`` of a feature lives in the channel axis; frequency blocks are
padded to a uniform width (components with ``|m| > l`` are zero).  Under a
rotation by ``ψ`` about the edge, each ``(+m, -m)`` pair turns by ``m ψ``;
under a reflection through a plane containing the edge, one member of
each pair flips sign, the same one for every channel.

* :class:`O2Linear`: real weights shared by ``+m`` and ``-m``, free mixing of
  channels at fixed ``|m|``, bias on ``m = 0`` only.  Complex weights would
  break the reflection part of O(2), so there are none.
* :class:`FourierActivation`: inverse DFT to ``G`` angles, SiLU, DFT back;
  equivariant up to aliasing (exact for rotations by multiples of 2π/G).
* :class:`O2Product`: the band-limited square of the feature (products of
  frequency blocks, ``a, b -> |a - b|, a + b``, truncated at ``M``) followed
  by an :class:`O2Linear`; exactly equivariant.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["FourierActivation", "O2Linear", "O2Product", "fourier_basis"]


class O2Linear(nn.Module):
    """``y_{±m} = W_|m| x_{±m}`` (+ bias on ``m = 0``)."""

    def __init__(self, in_channels: int, out_channels: int, mmax: int, bias: bool = True) -> None:
        super().__init__()
        self.mmax = mmax
        scale = 1.0 / math.sqrt(in_channels)
        self.weight = nn.Parameter(torch.randn(mmax + 1, in_channels, out_channels) * scale)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(E, C_in, 2M+1)`` -> ``(E, C_out, 2M+1)``."""
        index = torch.arange(-self.mmax, self.mmax + 1, device=x.device).abs()
        y = torch.einsum("eim,mio->eom", x, self.weight[index])
        if self.bias is not None:
            y = y.clone()
            y[..., self.mmax] = y[..., self.mmax] + self.bias
        return y


def fourier_basis(mmax: int, points: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """``(G, 2M+1)`` values of ``[sin Mφ ... sin φ, 1, cos φ ... cos Mφ]`` on ``G`` angles."""
    phi = torch.arange(points, dtype=dtype) * (2 * math.pi / points)
    cols = [torch.sin(m * phi) for m in range(mmax, 0, -1)]
    cols += [torch.ones_like(phi)] + [torch.cos(m * phi) for m in range(1, mmax + 1)]
    return torch.stack(cols, dim=-1)


class FourierActivation(nn.Module):
    """``DFT ∘ SiLU ∘ IDFT`` over the azimuthal angle on ``points`` grid angles."""

    def __init__(self, mmax: int, points: int) -> None:
        super().__init__()
        if points < 2 * (2 * mmax + 1):
            raise ValueError(f"grid_points must be at least 2(2 mmax + 1) = {2 * (2 * mmax + 1)}")
        basis = fourier_basis(mmax, points)
        self.register_buffer("basis", basis)
        self.register_buffer("analysis", torch.linalg.pinv(basis))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pointwise SiLU of the function on the circle that ``x`` represents."""
        grid = torch.einsum("ecm,gm->ecg", x, self.basis.to(x.dtype))
        return torch.einsum("ecg,mg->ecm", nn.functional.silu(grid), self.analysis.to(x.dtype))


class O2Product(nn.Module):
    """``O2Linear(x²)``: the square of the band-limited function, truncated at ``M``.

    Evaluated on ``4M + 1`` angles, enough to resolve every frequency up to
    ``2M``, so the truncation is exact and the map exactly equivariant.
    """

    def __init__(self, channels: int, out_channels: int, mmax: int) -> None:
        super().__init__()
        basis = fourier_basis(mmax, 4 * mmax + 1)
        self.register_buffer("basis", basis)
        self.register_buffer("analysis", torch.linalg.pinv(basis))
        self.linear = O2Linear(channels, out_channels, mmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quadratic, exactly O(2)-equivariant nonlinearity."""
        grid = torch.einsum("ecm,gm->ecg", x, self.basis.to(x.dtype))
        square = torch.einsum("ecg,mg->ecm", grid.square(), self.analysis.to(x.dtype))
        return self.linear(square)
