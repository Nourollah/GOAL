"""Tests for the training subsystem — loss composition and module construction."""

from __future__ import annotations

import torch
import pytest


class TestLossSystem:
    """Verify the composable loss system."""

    def test_weighted_loss(self):
        """WeightedLoss should scale the inner loss by its weight."""
        from gmd.training.loss import WeightedLoss, EnergyLoss

        inner = EnergyLoss()
        weighted = WeightedLoss(inner, weight=4.0)

        pred = {"energy": torch.tensor([1.0]), "num_atoms": torch.tensor([10.0])}
        target = {"energy": torch.tensor([2.0]), "num_atoms": torch.tensor([10.0])}

        raw = inner(pred, target)
        scaled = weighted(pred, target)
        assert torch.allclose(scaled, 4.0 * raw)

    def test_composite_loss_via_addition(self):
        """WeightedLoss + WeightedLoss should create a CompositeLoss."""
        from gmd.training.loss import WeightedLoss, EnergyLoss, ForcesLoss, CompositeLoss

        a = WeightedLoss(EnergyLoss(), weight=1.0)
        b = WeightedLoss(ForcesLoss(), weight=1.0)
        composite = a + b

        assert isinstance(composite, CompositeLoss)
        assert len(composite.losses) == 2

    def test_composite_loss_forward(self):
        """CompositeLoss forward should return a dict with 'total' and individual keys."""
        from gmd.training.loss import WeightedLoss, EnergyLoss, ForcesLoss, CompositeLoss

        composite = CompositeLoss([
            WeightedLoss(EnergyLoss(), weight=4.0),
            WeightedLoss(ForcesLoss(), weight=100.0),
        ])

        pred = {
            "energy": torch.tensor([1.0]),
            "num_atoms": torch.tensor([5.0]),
            "forces": torch.randn(5, 3),
        }
        target = {
            "energy": torch.tensor([2.0]),
            "num_atoms": torch.tensor([5.0]),
            "forces": torch.randn(5, 3),
        }

        result = composite(pred, target)
        assert "total" in result
        assert "EnergyLoss" in result
        assert "ForcesLoss" in result
        assert result["total"] == result["EnergyLoss"] + result["ForcesLoss"]

    def test_loss_registry(self):
        """Built-in losses should be in the registry."""
        from gmd.registry import LOSS_REGISTRY

        # Force import to trigger @register decorators
        import gmd.training.loss  # noqa: F401

        assert "energy" in LOSS_REGISTRY
        assert "forces" in LOSS_REGISTRY
        assert "stress" in LOSS_REGISTRY


class TestConfigurableLossFn:
    """Verify that each loss class accepts and uses configurable loss_fn."""

    def test_energy_loss_mse_default(self):
        """EnergyLoss default should be MSE."""
        from gmd.training.loss import EnergyLoss

        loss = EnergyLoss()
        pred = {"energy": torch.tensor([1.0]), "num_atoms": torch.tensor([1.0])}
        target = {"energy": torch.tensor([2.0]), "num_atoms": torch.tensor([1.0])}
        result = loss(pred, target)
        expected = torch.nn.functional.mse_loss(torch.tensor([1.0]), torch.tensor([2.0]))
        assert torch.allclose(result, expected)

    def test_energy_loss_mae(self):
        """EnergyLoss with loss_fn='mae' should compute L1 loss."""
        from gmd.training.loss import EnergyLoss

        loss = EnergyLoss(loss_fn="mae")
        pred = {"energy": torch.tensor([1.0]), "num_atoms": torch.tensor([1.0])}
        target = {"energy": torch.tensor([3.0]), "num_atoms": torch.tensor([1.0])}
        result = loss(pred, target)
        expected = torch.nn.functional.l1_loss(torch.tensor([1.0]), torch.tensor([3.0]))
        assert torch.allclose(result, expected)

    def test_energy_loss_huber(self):
        """EnergyLoss with loss_fn='huber' should compute Huber loss."""
        from gmd.training.loss import EnergyLoss

        loss = EnergyLoss(loss_fn="huber")
        pred = {"energy": torch.tensor([1.0]), "num_atoms": torch.tensor([1.0])}
        target = {"energy": torch.tensor([10.0]), "num_atoms": torch.tensor([1.0])}
        result = loss(pred, target)
        expected = torch.nn.functional.huber_loss(torch.tensor([1.0]), torch.tensor([10.0]))
        assert torch.allclose(result, expected)

    def test_forces_loss_smooth_l1(self):
        """ForcesLoss with loss_fn='smooth_l1' should compute SmoothL1."""
        from gmd.training.loss import ForcesLoss

        loss = ForcesLoss(loss_fn="smooth_l1")
        pred_f = torch.randn(5, 3)
        target_f = torch.randn(5, 3)
        result = loss({"forces": pred_f}, {"forces": target_f})
        expected = torch.nn.functional.smooth_l1_loss(pred_f, target_f)
        assert torch.allclose(result, expected)

    def test_stress_loss_l1(self):
        """StressLoss with loss_fn='l1' should work (alias for mae)."""
        from gmd.training.loss import StressLoss

        loss = StressLoss(loss_fn="l1")
        pred_s = torch.randn(3, 3)
        target_s = torch.randn(3, 3)
        result = loss({"stress": pred_s}, {"stress": target_s})
        expected = torch.nn.functional.l1_loss(pred_s, target_s)
        assert torch.allclose(result, expected)

    def test_dipole_loss_configurable(self):
        """DipoleLoss should accept and use a custom loss_fn."""
        from gmd.training.loss import DipoleLoss

        loss = DipoleLoss(loss_fn="mae")
        pred_d = torch.tensor([[1.0, 2.0, 3.0]])
        target_d = torch.tensor([[4.0, 5.0, 6.0]])
        result = loss({"dipole": pred_d}, {"dipole": target_d})
        expected = torch.nn.functional.l1_loss(pred_d, target_d)
        assert torch.allclose(result, expected)

    def test_charge_loss_configurable(self):
        """ChargeLoss should accept a custom loss_fn."""
        from gmd.training.loss import ChargeLoss

        loss = ChargeLoss(loss_fn="mse")
        pred = {"total_charge": torch.tensor([0.5])}
        target = {"total_charge": torch.tensor([0.0])}
        result = loss(pred, target)
        expected = torch.nn.functional.mse_loss(torch.tensor([0.5]), torch.tensor([0.0]))
        assert torch.allclose(result, expected)

    def test_unknown_loss_fn_raises(self):
        """Unknown loss function name should raise ValueError."""
        from gmd.training.loss import EnergyLoss

        with pytest.raises(ValueError, match="Unknown loss function"):
            EnergyLoss(loss_fn="nonexistent")

    def test_resolve_loss_fn(self):
        """resolve_loss_fn should map names to callables."""
        from gmd.training.loss import resolve_loss_fn
        import torch.nn.functional as F

        assert resolve_loss_fn("mse") is F.mse_loss
        assert resolve_loss_fn("mae") is F.l1_loss
        assert resolve_loss_fn("l1") is F.l1_loss
        assert resolve_loss_fn("huber") is F.huber_loss
        assert resolve_loss_fn("smooth_l1") is F.smooth_l1_loss

    def test_different_fns_give_different_results(self):
        """MSE and MAE should give different values for the same inputs."""
        from gmd.training.loss import EnergyLoss

        pred = {"energy": torch.tensor([1.0]), "num_atoms": torch.tensor([1.0])}
        target = {"energy": torch.tensor([3.0]), "num_atoms": torch.tensor([1.0])}

        mse_loss = EnergyLoss(loss_fn="mse")(pred, target)
        mae_loss = EnergyLoss(loss_fn="mae")(pred, target)
        assert not torch.allclose(mse_loss, mae_loss)


class TestEMA:
    """Verify the EMA wrapper."""

    def test_ema_update(self):
        """EMA shadow should move towards current parameters."""
        from gmd.training.ema import EMAWrapper
        import torch.nn as nn

        model = nn.Linear(10, 1)
        ema = EMAWrapper(model.parameters(), decay=0.9)

        # Store initial shadow
        initial_shadow = [s.clone() for s in ema._shadow]

        # Change model params
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p))

        ema.update()

        # Shadow should have moved
        for s_old, s_new in zip(initial_shadow, ema._shadow):
            assert not torch.equal(s_old, s_new)

    def test_ema_context_manager(self):
        """average_parameters() context should swap and restore weights."""
        from gmd.training.ema import EMAWrapper
        import torch.nn as nn

        model = nn.Linear(10, 1)
        ema = EMAWrapper(model.parameters(), decay=0.9)

        original_weight = model.weight.data.clone()

        # Change model params
        with torch.no_grad():
            model.weight.data.fill_(999.0)

        ema.update()

        modified_weight = model.weight.data.clone()

        with ema.average_parameters():
            # Inside context: should be EMA weights (not the 999-filled ones)
            assert not torch.equal(model.weight.data, modified_weight)

        # After context: should be back to 999s
        assert torch.equal(model.weight.data, modified_weight)
