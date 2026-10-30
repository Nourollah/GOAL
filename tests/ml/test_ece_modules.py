"""Edge-frame, O(2) and edge-cluster-expansion modules (float64, CPU)."""

from __future__ import annotations

import math

import pytest
import torch
from e3nn import o3

from goal.ml.nn.blocks.ece import ECEBlock, from_edge_frame, to_edge_frame
from goal.ml.nn.heads.edge_energy import EdgeEnergyHead, reverse_edges
from goal.ml.nn.primitives.edge_frame import EdgeFrames, edge_rotation
from goal.ml.nn.primitives.o2 import FourierActivation, O2Linear, O2Product

M = 2


@pytest.fixture(autouse=True)
def _float64():
    """Float64 inside this module only (a module-level default would leak)."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def rotate_about_axis(x: torch.Tensor, psi: float) -> torch.Tensor:
    """Act on O(2) layout ``(E, C, 2M+1)``: the ``(+m, -m)`` pair turns by ``m ψ``."""
    y = x.clone()
    for m in range(1, (x.shape[-1] - 1) // 2 + 1):
        mm = (x.shape[-1] - 1) // 2
        a, b = x[..., mm + m], x[..., mm - m]
        c, s = math.cos(m * psi), math.sin(m * psi)
        y[..., mm + m], y[..., mm - m] = c * a - s * b, s * a + c * b
    return y


def reflect(x: torch.Tensor) -> torch.Tensor:
    """φ -> -φ: every ``-m`` (sine-like) component flips."""
    y = x.clone()
    mm = (x.shape[-1] - 1) // 2
    y[..., :mm] = -y[..., :mm]
    return y


# ---------------------------------------------------------------------------
# Edge frames
# ---------------------------------------------------------------------------


def test_frames_are_orthogonal_and_align_the_bond() -> None:
    vectors = torch.randn(50, 3)
    frames = EdgeFrames(3)(vectors)
    unit = vectors / vectors.norm(dim=-1, keepdim=True)
    for ell, d in enumerate(frames):
        eye = torch.eye(2 * ell + 1).expand_as(d)
        torch.testing.assert_close(d @ d.transpose(1, 2), eye, atol=1e-10, rtol=0)
        y = o3.spherical_harmonics(ell, unit, normalize=True, normalization="component")
        aligned = torch.einsum("enm,em->en", d, y)
        off = torch.cat([aligned[:, :ell], aligned[:, ell + 1 :]], dim=1)
        assert ell == 0 or off.abs().max() < 1e-10  # weight only in m = 0
    rot = edge_rotation(unit)
    torch.testing.assert_close(
        torch.einsum("eij,ej->ei", rot, unit), torch.tensor([0.0, 1.0, 0.0]).expand_as(unit)
    )


def test_frames_near_the_singular_direction_are_finite() -> None:
    theta = 1e-6
    vectors = torch.tensor(
        [[math.sin(theta), -math.cos(theta), 0.0], [0.0, -1.0, 0.0]], requires_grad=True
    )
    frames = EdgeFrames(2)(vectors)
    total = sum(d.square().sum() + d.sum() for d in frames)
    (grad,) = torch.autograd.grad(total, vectors)
    assert all(torch.isfinite(d).all() for d in frames) and torch.isfinite(grad).all()


# ---------------------------------------------------------------------------
# O(2) primitives
# ---------------------------------------------------------------------------


def test_o2_linear_commutes_with_rotations_and_reflections() -> None:
    lin = O2Linear(6, 4, M)
    x = torch.randn(3, 6, 2 * M + 1)
    torch.testing.assert_close(lin(rotate_about_axis(x, 0.7)), rotate_about_axis(lin(x), 0.7))
    torch.testing.assert_close(lin(reflect(x)), reflect(lin(x)))


def test_o2_product_is_exactly_equivariant() -> None:
    prod = O2Product(5, 5, M)
    x = torch.randn(3, 5, 2 * M + 1)
    torch.testing.assert_close(
        prod(rotate_about_axis(x, 1.234)), rotate_about_axis(prod(x), 1.234)
    )
    torch.testing.assert_close(prod(reflect(x)), reflect(prod(x)))


def test_fourier_activation_is_exact_on_grid_rotations_only_approximately_elsewhere() -> None:
    act = FourierActivation(M, 16)
    x = torch.randn(3, 5, 2 * M + 1)
    grid = 2 * math.pi / 16 * 3
    torch.testing.assert_close(act(rotate_about_axis(x, grid)), rotate_about_axis(act(x), grid))
    torch.testing.assert_close(act(reflect(x)), reflect(act(x)))
    off = (act(rotate_about_axis(x, 0.3)) - rotate_about_axis(act(x), 0.3)).abs().max()
    assert off < 0.1  # aliasing, shrinking with grid_points
    with pytest.raises(ValueError, match="grid_points"):
        FourierActivation(M, 8)


def test_edge_frame_round_trip() -> None:
    frames = EdgeFrames(2)(torch.randn(7, 3))
    blocks = [torch.randn(7, 3, 2 * ell + 1) for ell in range(3)]
    back = from_edge_frame(to_edge_frame(blocks, frames, mmax=2), frames, channels=3)
    for a, b in zip(blocks, back):
        torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# ECE block and edge-energy head
# ---------------------------------------------------------------------------


def _molecule_edges(n: int = 6, cutoff: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(n, 3, generator=g) * 1.2
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    mask = (i != j) & ((pos[i] - pos[j]).norm(dim=-1) < cutoff)
    return pos, torch.stack([i[mask], j[mask]])


@pytest.mark.parametrize("activation, tol", [("o2_product", 1e-12), ("fourier", 1e-3)])
def test_bond_invariants_do_not_depend_on_the_gauge(activation: str, tol: float) -> None:
    torch.manual_seed(1)
    pos, edge_index = _molecule_edges()
    vectors = pos[edge_index[1]] - pos[edge_index[0]]
    lmax, c = 2, 4
    block = ECEBlock(c, lmax, 2, 8, 8, 2, 2, 3, [8], 5, activation, 16)
    dens = [torch.randn(pos.shape[0], c, 2 * ell + 1) for ell in range(lmax + 1)]
    codes, rbf, env = (
        torch.randn(pos.shape[0], 3),
        torch.randn(len(vectors), 5),
        torch.rand(len(vectors)),
    )
    frames_layer = EdgeFrames(lmax)
    plain = block(dens, edge_index, frames_layer(vectors), codes, rbf, env, 3.0)[..., 2]
    gauge = torch.rand(len(vectors)) * 2 * math.pi
    turned = block(dens, edge_index, frames_layer(vectors, gauge), codes, rbf, env, 3.0)[..., 2]
    # relative: an untrained quadratic (o2_product) stack reaches large values
    assert (plain - turned).abs().max() / plain.abs().max() < tol


def test_edge_energy_head_split_and_symmetry() -> None:
    pos, edge_index = _molecule_edges()
    rev = reverse_edges(edge_index, pos.shape[0])
    torch.testing.assert_close(edge_index[:, rev], edge_index.flip(0))
    head = EdgeEnergyHead(5, [8])
    out = head(
        torch.randn(edge_index.shape[1], 5),
        torch.rand(edge_index.shape[1]),
        edge_index,
        pos.shape[0],
    )
    torch.testing.assert_close(out["atomic_energies"].sum(), out["edge_energies"].sum())
    torch.testing.assert_close(out["bond_energies"], out["bond_energies"][rev])
    zero = EdgeEnergyHead(5, [8], zero_init=True)
    assert not zero(torch.randn(3, 5), torch.ones(3), edge_index[:, :3], pos.shape[0])[
        "edge_energies"
    ].any()
