"""Huber threshold and the per-atom force-error-norm loss."""

from __future__ import annotations

import pytest
import torch
from scipy.spatial.transform import Rotation
from torch_geometric.data import Data

from goal.ml.training.loss import EnergyLoss, ForcesLoss


def _batch(n: int = 6, seed: int = 0) -> tuple[dict, Data]:
    g = torch.Generator().manual_seed(seed)
    ref = torch.randn(n, 3, generator=g, dtype=torch.float64)
    pred = ref + 0.05 * torch.randn(n, 3, generator=g, dtype=torch.float64)
    target = Data(forces=ref, batch=torch.tensor([0, 0, 0, 1, 1, 1]))
    return {"forces": pred}, target


def test_norm_huber_matches_its_definition() -> None:
    delta = 0.01
    pred, target = _batch()
    e = (pred["forces"] - target.forces).norm(dim=-1)
    h = torch.where(e <= delta, 0.5 * e**2, delta * (e - 0.5 * delta))
    loss = ForcesLoss("huber", normalize_by_n_atoms=False, error="norm", delta=delta)
    torch.testing.assert_close(loss(pred, target), h.mean())
    big = ForcesLoss("huber", normalize_by_n_atoms=False, error="norm", delta=10.0)
    torch.testing.assert_close(big(pred, target), (0.5 * e**2).mean())


@pytest.mark.parametrize("fn", ["huber", "mse", "mae", "rmse"])
def test_norm_loss_is_rotation_invariant(fn: str) -> None:
    pred, target = _batch()
    rot = torch.as_tensor(Rotation.random(random_state=1).as_matrix())
    rotated = Data(forces=target.forces @ rot.T, batch=target.batch)
    loss = ForcesLoss(fn, error="norm", delta=0.01)
    torch.testing.assert_close(
        loss(pred, target), loss({"forces": pred["forces"] @ rot.T}, rotated)
    )


def test_component_loss_is_not_rotation_invariant() -> None:
    """Why the recipe uses the norm: a per-component L1 changes under rotation."""
    pred, target = _batch()
    rot = torch.as_tensor(Rotation.random(random_state=1).as_matrix())
    rotated = Data(forces=target.forces @ rot.T, batch=target.batch)
    loss = ForcesLoss("mae", error="component")
    assert loss(pred, target) != pytest.approx(
        float(loss({"forces": pred["forces"] @ rot.T}, rotated)), rel=1e-6
    )


def test_norm_loss_per_structure_and_arms() -> None:
    pred, target = _batch()
    e = (pred["forces"] - target.forces).norm(dim=-1)
    per_structure = ForcesLoss("mae", error="norm")(pred, target)
    torch.testing.assert_close(per_structure, (e[:3].mean() + e[3:].mean()) / 2)
    mse = ForcesLoss("mse", normalize_by_n_atoms=False, error="norm")(pred, target)
    torch.testing.assert_close(mse, (e**2).mean())


def test_norm_loss_has_a_finite_gradient_at_zero_error() -> None:
    ref = torch.randn(4, 3, dtype=torch.float64)
    pred = ref.clone().requires_grad_(True)
    loss = ForcesLoss("huber", error="norm", delta=0.01)(
        {"forces": pred}, Data(forces=ref, batch=torch.zeros(4, dtype=torch.long))
    )
    (grad,) = torch.autograd.grad(loss, pred)
    assert torch.isfinite(grad).all() and not grad.any()


def test_energy_huber_threshold() -> None:
    pred = {"energy": torch.tensor([0.0, 10.0]), "num_atoms": torch.tensor([1, 2])}
    target = Data(energy=torch.tensor([0.004, 10.0 + 0.0005 * 2]))
    loss = EnergyLoss("huber", delta=0.001)(pred, target)
    expected = (0.001 * (0.004 - 0.0005) + 0.5 * 0.0005**2) / 2
    torch.testing.assert_close(loss, torch.tensor(expected))


def test_component_huber_with_custom_delta_is_refused() -> None:
    pred, target = _batch()
    with pytest.raises(ValueError, match="error: norm"):
        ForcesLoss("huber", delta=0.01)(pred, target)
