"""Regression tests for ``forces_cosine_similarity`` correctness.

Guards two specific failure modes:

1. **Always-zero from broken autograd path.**  When the SIMURGH
   backbone forgot to enable ``requires_grad`` on ``graph.pos``,
   :class:`DualForcesHead` fell back to all-zero forces, and the
   cosine metric collapsed to exactly ``0.0`` regardless of the
   target.  The autograd fix lives in
   ``goal.ml.nn.models.simurgh.legacy.backbone.SimurghBackbone.forward`` —
   this test enforces that forces survive a backbone+head forward
   with cold positions.

2. **Always-zero from zero-target bias.**  Datasets without force
   labels pad ``batch.forces`` with zeros.  Including those atoms in
   the cosine average would silently pull every reported value
   toward ``0`` — the metric is now expected to skip them.
"""

from __future__ import annotations

import typing

import torch
from torch_geometric.data import Batch

from goal.ml.nn.heads.dual_forces import DualForcesHead
from goal.ml.nn.models.simurgh.legacy.backbone import SimurghBackbone
from goal.ml.training.metrics import mlip_metrics


class _FakeBatch:
    """Bag-of-attributes stand-in for a PyG ``Batch`` for metric tests."""

    energy: torch.Tensor
    forces: torch.Tensor
    batch: torch.Tensor


def _make_batch(target_forces: torch.Tensor) -> _FakeBatch:
    n: int = target_forces.shape[0]
    b: _FakeBatch = _FakeBatch()
    b.energy = torch.tensor([0.0])
    b.forces = target_forces
    b.batch = torch.zeros(n, dtype=torch.long)
    return b


class TestCosineRange:
    """Cosine must respect the contract: every reported value in [-1, 1]."""

    def test_aligned_is_one(self) -> None:
        t: torch.Tensor = torch.randn(8, 3, dtype=torch.float64)
        preds: dict[str, torch.Tensor] = {
            "energy": torch.tensor([0.0]),
            "num_atoms": torch.tensor([8.0]),
            "forces": t.clone(),
        }
        m: dict[str, torch.Tensor] = mlip_metrics(preds, _make_batch(t))
        assert abs(m["forces_cosine_similarity"].item() - 1.0) < 1e-6

    def test_anti_aligned_is_minus_one(self) -> None:
        t: torch.Tensor = torch.randn(8, 3, dtype=torch.float64)
        preds: dict[str, torch.Tensor] = {
            "energy": torch.tensor([0.0]),
            "num_atoms": torch.tensor([8.0]),
            "forces": -t,
        }
        m: dict[str, torch.Tensor] = mlip_metrics(preds, _make_batch(t))
        assert abs(m["forces_cosine_similarity"].item() - (-1.0)) < 1e-6

    def test_random_in_range(self) -> None:
        torch.manual_seed(0)
        for _ in range(20):
            t: torch.Tensor = torch.randn(16, 3, dtype=torch.float64)
            p: torch.Tensor = torch.randn(16, 3, dtype=torch.float64)
            preds: dict[str, torch.Tensor] = {
                "energy": torch.tensor([0.0]),
                "num_atoms": torch.tensor([16.0]),
                "forces": p,
            }
            m: dict[str, torch.Tensor] = mlip_metrics(preds, _make_batch(t))
            v: float = m["forces_cosine_similarity"].item()
            assert -1.0 - 1e-7 <= v <= 1.0 + 1e-7, v


class TestCosineZeroTargetHandling:
    """Atoms with zero target force must not bias the average."""

    def test_all_zero_targets_omit_metric(self) -> None:
        """No directional info → metric is dropped, not reported as 0."""
        t: torch.Tensor = torch.zeros(8, 3, dtype=torch.float64)
        preds: dict[str, torch.Tensor] = {
            "energy": torch.tensor([0.0]),
            "num_atoms": torch.tensor([8.0]),
            "forces": torch.randn(8, 3, dtype=torch.float64),
        }
        m: dict[str, torch.Tensor] = mlip_metrics(preds, _make_batch(t))
        assert "forces_cosine_similarity" not in m

    def test_mixed_zero_targets_skipped(self) -> None:
        """Half-zero, half-aligned → reported cosine ≈ 1.0 (zeros dropped)."""
        torch.manual_seed(0)
        real: torch.Tensor = torch.randn(4, 3, dtype=torch.float64)
        t: torch.Tensor = torch.cat([torch.zeros(4, 3, dtype=torch.float64), real])
        preds: dict[str, torch.Tensor] = {
            "energy": torch.tensor([0.0]),
            "num_atoms": torch.tensor([8.0]),
            "forces": torch.cat([torch.randn(4, 3, dtype=torch.float64), real.clone()]),
        }
        m: dict[str, torch.Tensor] = mlip_metrics(preds, _make_batch(t))
        # The four aligned atoms give cos=1.0; the four zero-target
        # atoms are excluded, so the average is exactly 1.0.
        assert abs(m["forces_cosine_similarity"].item() - 1.0) < 1e-6


class TestBackboneEnablesGradOnPositions:
    """The SIMURGH backbone must enable ``requires_grad`` on positions
    so :class:`DualForcesHead` doesn't fall into its zero-force branch."""

    def test_pos_grad_enabled_after_forward(self, methane_batch: Batch) -> None:
        torch.manual_seed(0)
        backbone: SimurghBackbone = SimurghBackbone(
            elements=(1, 6),
            dressing_kwargs=dict(
                num_elements=9,
                embedding_dim=8,
                hidden_channels=8,
                lmax=1,
                num_radial_basis=4,
                cutoff=5.0,
                radial_mlp_hidden=8,
                num_message_passing=1,
            ),
            artisan_config=dict(hidden_dims=(8, 8)),
        ).double()
        # The batch's positions arrive cold (no autograd).
        assert methane_batch.pos.requires_grad is False
        _ = backbone(methane_batch)
        # After the forward, the backbone has set requires_grad so any
        # downstream head can run ``torch.autograd.grad(E, graph.pos)``.
        assert methane_batch.pos.requires_grad is True

    def test_dual_forces_head_produces_nonzero_forces(self, methane_batch: Batch) -> None:
        """End-to-end: cold batch → backbone+head → forces ≠ 0."""
        torch.manual_seed(0)
        backbone: SimurghBackbone = SimurghBackbone(
            elements=(1, 6),
            dressing_kwargs=dict(
                num_elements=9,
                embedding_dim=8,
                hidden_channels=8,
                lmax=1,
                num_radial_basis=4,
                cutoff=5.0,
                radial_mlp_hidden=8,
                num_message_passing=1,
            ),
            artisan_config=dict(hidden_dims=(8, 8)),
        ).double()
        head: DualForcesHead = DualForcesHead(
            irreps_in=str(backbone.irreps_out),
            hidden_dim=16,
            mode="autograd",
        ).double()
        features: typing.Any = backbone(methane_batch)
        preds: dict[str, torch.Tensor] = head(features, methane_batch)
        assert preds["forces"].abs().max().item() > 0.0, (
            "Forces are all zero after a cold-positions forward — the "
            "backbone's requires_grad-on-positions hand-off is broken"
        )
