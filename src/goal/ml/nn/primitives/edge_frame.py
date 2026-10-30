"""Edge frames: rotations taking each bond onto the polar axis, and their Wigner-D blocks.

In an edge's own frame only rotations about the bond and reflections
through planes containing it remain (O(2)).  The polar axis is e3nn's
``y`` axis: the real spherical harmonics ``Y_l`` of ``ŷ`` have weight only in
their middle component (``m = 0``), and every ``(+m, -m)`` component pair
rotates by ``m ψ`` under a rotation about ``ŷ``.

The rotation is the minimal one taking ``r̂`` to ``ŷ``,
``R = I + [v]× + [v]×² / (1 + c)`` with ``v = r̂ × ŷ``, ``c = r̂·ŷ``: a rational
function of ``r̂``, so differentiable everywhere except at the antipode
``r̂ = -ŷ`` (where a fixed half turn about ``z`` is used).  The Wigner-D
blocks are polynomials in ``R``, obtained by projecting spherical
harmonics at fixed sample directions (``D = Y(R X)ᵀ pinv(Y(X)ᵀ)``), so no
Euler angles and no angle singularities.
"""

from __future__ import annotations

import torch
from e3nn import o3
from torch import nn

__all__ = ["EdgeFrames", "edge_rotation"]


def _axis_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrices about a fixed unit ``axis`` by ``angle`` (batched)."""
    k = torch.zeros(3, 3, dtype=angle.dtype, device=angle.device)
    k[0, 1], k[0, 2], k[1, 0], k[1, 2], k[2, 0], k[2, 1] = (
        -axis[2],
        axis[1],
        axis[2],
        -axis[0],
        -axis[1],
        axis[0],
    )
    s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
    return eye + s * k + (1 - c) * (k @ k)


def edge_rotation(unit: torch.Tensor, gauge: torch.Tensor | None = None) -> torch.Tensor:
    """``(E, 3, 3)`` rotations ``R`` with ``R r̂ = ŷ`` for unit vectors ``unit``.

    ``gauge`` (radians, per edge) adds a rotation about ``ŷ`` after the
    alignment; edge-centred energies must not depend on it (gauge test).
    """
    y = torch.tensor([0.0, 1.0, 0.0], dtype=unit.dtype, device=unit.device)
    c = unit @ y  # (E,)
    v = torch.cross(unit, y.expand_as(unit), dim=-1)
    zero = torch.zeros_like(c)
    vx = torch.stack(
        [
            torch.stack([zero, -v[:, 2], v[:, 1]], -1),
            torch.stack([v[:, 2], zero, -v[:, 0]], -1),
            torch.stack([-v[:, 1], v[:, 0], zero], -1),
        ],
        dim=1,
    )
    near_antipode = 1.0 + c < 1e-12
    denom = torch.where(near_antipode, torch.ones_like(c), 1.0 + c)
    eye = torch.eye(3, dtype=unit.dtype, device=unit.device).expand(unit.shape[0], 3, 3)
    rot = eye + vx + vx @ vx / denom[:, None, None]
    half_turn = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=unit.dtype, device=unit.device))
    rot = torch.where(near_antipode[:, None, None], half_turn.expand_as(rot), rot)
    if gauge is not None:
        rot = _axis_rotation(y, gauge) @ rot
    return rot


class EdgeFrames(nn.Module):
    """Wigner-D blocks ``D^l(R_e)``, ``l = 0..lmax``, of each edge's alignment rotation.

    ``D^l`` acts on e3nn's real, component-normalised ``Y_l`` (and on any
    irrep of degree ``l`` and natural parity): ``Y_l(R x) = D^l(R) Y_l(x)``.
    Features ``f`` in the global frame become ``D^l f`` in the edge frame;
    ``D^lᵀ`` rotates them back.
    """

    def __init__(self, lmax: int, num_points: int = 32) -> None:
        super().__init__()
        self.lmax = lmax
        g = torch.Generator().manual_seed(0)
        points = torch.randn(num_points, 3, generator=g, dtype=torch.float64)
        points = points / points.norm(dim=-1, keepdim=True)
        self.register_buffer("points", points)
        for ell in range(lmax + 1):
            y = o3.spherical_harmonics(ell, points, normalize=True, normalization="component")
            self.register_buffer(f"pinv_{ell}", torch.linalg.pinv(y))  # (2l+1, P)

    def forward(
        self, vectors: torch.Tensor, gauge: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        """``[D^0, ..., D^lmax]``, each ``(E, 2l+1, 2l+1)``, for edge vectors ``(E, 3)``."""
        unit = vectors / vectors.norm(dim=-1, keepdim=True)
        rot = edge_rotation(unit, gauge)
        rotated = torch.einsum("pk,eck->epc", self.points.to(rot.dtype), rot)  # R x_p
        blocks = []
        for ell in range(self.lmax + 1):
            y = o3.spherical_harmonics(ell, rotated, normalize=False, normalization="component")
            pinv = getattr(self, f"pinv_{ell}").to(rot.dtype)
            blocks.append(
                torch.einsum("mp,epn->enm", pinv, y)
            )  # D[n, m] = Σ_p Y_n(R x_p) pinv[m, p]
        return blocks
