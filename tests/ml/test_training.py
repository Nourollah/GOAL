"""Tests for the training subsystem — loss composition and module construction."""

from __future__ import annotations

import pytest
import torch


class TestLossSystem:
    """Verify the composable loss system."""

    def test_weighted_loss(self):
        """WeightedLoss should scale the inner loss by its weight."""
        from goal.ml.training.loss import EnergyLoss, WeightedLoss

        inner = EnergyLoss()
        weighted = WeightedLoss(inner, weight=4.0)

        pred = {"energy": torch.tensor([1.0]), "num_atoms": torch.tensor([10.0])}
        target = {"energy": torch.tensor([2.0]), "num_atoms": torch.tensor([10.0])}

        raw = inner(pred, target)
        scaled = weighted(pred, target)
        assert torch.allclose(scaled, 4.0 * raw)

    def test_composite_loss_via_addition(self):
        """WeightedLoss + WeightedLoss should create a CompositeLoss."""
        from goal.ml.training.loss import (
            CompositeLoss,
            EnergyLoss,
            ForcesLoss,
            WeightedLoss,
        )

        a = WeightedLoss(EnergyLoss(), weight=1.0)
        b = WeightedLoss(ForcesLoss(), weight=1.0)
        composite = a + b

        assert isinstance(composite, CompositeLoss)
        assert len(composite.losses) == 2

    def test_composite_loss_forward(self):
        """CompositeLoss forward should return a dict with 'total' and individual keys."""
        from goal.ml.training.loss import (
            CompositeLoss,
            EnergyLoss,
            ForcesLoss,
            WeightedLoss,
        )

        composite = CompositeLoss(
            [
                WeightedLoss(EnergyLoss(), weight=4.0),
                WeightedLoss(ForcesLoss(), weight=100.0),
            ]
        )

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

    def test_zero_weight_term_is_logged_but_not_in_the_total(self):
        """A stage-disabled (weight 0) term must not turn an overflow into NaN (0 * inf)."""
        from goal.ml.training.loss import (
            CompositeLoss,
            EnergyLoss,
            ForcesLoss,
            WeightedLoss,
        )

        composite = CompositeLoss(
            [WeightedLoss(EnergyLoss(), weight=1.0), WeightedLoss(ForcesLoss(), weight=0.0)]
        )
        energy = torch.tensor([1.0], requires_grad=True)
        forces = torch.full((5, 3), float("inf"), requires_grad=True)
        pred = {"energy": energy, "num_atoms": torch.tensor([5.0]), "forces": forces}
        target = {
            "energy": torch.tensor([2.0]),
            "num_atoms": torch.tensor([5.0]),
            "forces": torch.zeros(5, 3),
        }

        result = composite(pred, target)
        result["total"].backward()
        assert torch.isfinite(result["total"])
        assert result["total"] == result["EnergyLoss"]
        assert "ForcesLoss" in result and not result["ForcesLoss"].requires_grad
        assert forces.grad is None and torch.isfinite(energy.grad).all()

    def test_loss_registry(self):
        """Built-in losses should be in the registry."""
        # Force import to trigger @register decorators
        import goal.ml.training.loss  # noqa: F401
        from goal.ml.registry import LOSS_REGISTRY

        assert "energy" in LOSS_REGISTRY
        assert "forces" in LOSS_REGISTRY
        assert "stress" in LOSS_REGISTRY


class TestConfigurableLossFn:
    """Verify that each loss class accepts and uses configurable loss_fn."""

    def test_energy_loss_mse_default(self):
        """EnergyLoss default should be MSE."""
        from goal.ml.training.loss import EnergyLoss

        loss = EnergyLoss()
        pred = {"energy": torch.tensor([1.0]), "num_atoms": torch.tensor([1.0])}
        target = {"energy": torch.tensor([2.0]), "num_atoms": torch.tensor([1.0])}
        result = loss(pred, target)
        expected = torch.nn.functional.mse_loss(torch.tensor([1.0]), torch.tensor([2.0]))
        assert torch.allclose(result, expected)

    def test_energy_loss_mae(self):
        """EnergyLoss with loss_fn='mae' should compute L1 loss."""
        from goal.ml.training.loss import EnergyLoss

        loss = EnergyLoss(loss_fn="mae")
        pred = {"energy": torch.tensor([1.0]), "num_atoms": torch.tensor([1.0])}
        target = {"energy": torch.tensor([3.0]), "num_atoms": torch.tensor([1.0])}
        result = loss(pred, target)
        expected = torch.nn.functional.l1_loss(torch.tensor([1.0]), torch.tensor([3.0]))
        assert torch.allclose(result, expected)

    def test_energy_loss_huber(self):
        """EnergyLoss with loss_fn='huber' should compute Huber loss."""
        from goal.ml.training.loss import EnergyLoss

        loss = EnergyLoss(loss_fn="huber")
        pred = {"energy": torch.tensor([1.0]), "num_atoms": torch.tensor([1.0])}
        target = {"energy": torch.tensor([10.0]), "num_atoms": torch.tensor([1.0])}
        result = loss(pred, target)
        expected = torch.nn.functional.huber_loss(torch.tensor([1.0]), torch.tensor([10.0]))
        assert torch.allclose(result, expected)

    def test_forces_loss_smooth_l1(self):
        """ForcesLoss with loss_fn='smooth_l1' should compute SmoothL1."""
        from goal.ml.training.loss import ForcesLoss

        loss = ForcesLoss(loss_fn="smooth_l1")
        pred_f = torch.randn(5, 3)
        target_f = torch.randn(5, 3)
        result = loss({"forces": pred_f}, {"forces": target_f})
        expected = torch.nn.functional.smooth_l1_loss(pred_f, target_f)
        assert torch.allclose(result, expected)

    def test_stress_loss_l1(self):
        """StressLoss with loss_fn='l1' should work (alias for mae)."""
        from goal.ml.training.loss import StressLoss

        loss = StressLoss(loss_fn="l1")
        pred_s = torch.randn(3, 3)
        target_s = torch.randn(3, 3)
        result = loss({"stress": pred_s}, {"stress": target_s})
        expected = torch.nn.functional.l1_loss(pred_s, target_s)
        assert torch.allclose(result, expected)

    def test_dipole_loss_configurable(self):
        """DipoleLoss should accept and use a custom loss_fn."""
        from goal.ml.training.loss import DipoleLoss

        loss = DipoleLoss(loss_fn="mae")
        pred_d = torch.tensor([[1.0, 2.0, 3.0]])
        target_d = torch.tensor([[4.0, 5.0, 6.0]])
        result = loss({"dipole": pred_d}, {"dipole": target_d})
        expected = torch.nn.functional.l1_loss(pred_d, target_d)
        assert torch.allclose(result, expected)

    def test_charge_loss_configurable(self):
        """ChargeLoss should accept a custom loss_fn."""
        from goal.ml.training.loss import ChargeLoss

        loss = ChargeLoss(loss_fn="mse")
        pred = {"total_charge": torch.tensor([0.5])}
        target = {"total_charge": torch.tensor([0.0])}
        result = loss(pred, target)
        expected = torch.nn.functional.mse_loss(torch.tensor([0.5]), torch.tensor([0.0]))
        assert torch.allclose(result, expected)

    def test_unknown_loss_fn_raises(self):
        """Unknown loss function name should raise ValueError."""
        from goal.ml.training.loss import EnergyLoss

        with pytest.raises(ValueError, match="Unknown loss function"):
            EnergyLoss(loss_fn="nonexistent")

    def test_resolve_loss_fn(self):
        """resolve_loss_fn should map names to callables."""
        import torch.nn.functional as F

        from goal.ml.training.loss import resolve_loss_fn

        assert resolve_loss_fn("mse") is F.mse_loss
        assert resolve_loss_fn("mae") is F.l1_loss
        assert resolve_loss_fn("l1") is F.l1_loss
        assert resolve_loss_fn("huber") is F.huber_loss
        assert resolve_loss_fn("smooth_l1") is F.smooth_l1_loss

    def test_different_fns_give_different_results(self):
        """MSE and MAE should give different values for the same inputs."""
        from goal.ml.training.loss import EnergyLoss

        pred = {"energy": torch.tensor([1.0]), "num_atoms": torch.tensor([1.0])}
        target = {"energy": torch.tensor([3.0]), "num_atoms": torch.tensor([1.0])}

        mse_loss = EnergyLoss(loss_fn="mse")(pred, target)
        mae_loss = EnergyLoss(loss_fn="mae")(pred, target)
        assert not torch.allclose(mse_loss, mae_loss)


class _FakeBatch(dict):
    """Minimal dict subclass that also exposes ``batch`` and ``weight`` as attributes.

    The real training loop passes a PyG ``AtomicGraph`` / ``Batch`` object to the
    loss, which supports both ``target["forces"]`` (dict subscript, because ``Data``
    subclasses ``dict``) and ``getattr(target, "batch", None)`` (attribute access).
    This helper replicates that dual interface for unit tests without requiring a
    full PyG graph.
    """

    def __init__(self, batch: torch.Tensor | None = None, **kwargs: torch.Tensor) -> None:
        super().__init__(**kwargs)
        self.batch = batch
        self.weight: torch.Tensor | None = None


class TestForcesLossNormalization:
    """Per-structure normalisation of the force loss (normalize_by_n_atoms)."""

    def test_flat_mean_no_batch_info(self) -> None:
        """When target has no batch attribute, flat mean is used regardless of flag."""
        from goal.ml.training.loss import ForcesLoss

        pred_f = torch.randn(5, 3)
        target_f = torch.randn(5, 3)
        loss_norm = ForcesLoss(loss_fn="mae", normalize_by_n_atoms=True)
        loss_flat = ForcesLoss(loss_fn="mae", normalize_by_n_atoms=False)
        expected = torch.nn.functional.l1_loss(pred_f, target_f)

        # Both fall back to flat mean when no batch attribute
        assert torch.allclose(loss_norm({"forces": pred_f}, {"forces": target_f}), expected)
        assert torch.allclose(loss_flat({"forces": pred_f}, {"forces": target_f}), expected)

    def test_equal_size_structures_give_same_result(self) -> None:
        """When all structures have the same number of atoms, per-structure
        normalisation and flat mean give identical results (both weight each
        atom equally when structures are the same size)."""
        from goal.ml.training.loss import ForcesLoss

        torch.manual_seed(0)
        n_atoms = 5
        n_structs = 3
        pred_f = torch.randn(n_atoms * n_structs, 3)
        target_f = torch.randn(n_atoms * n_structs, 3)
        batch_index = torch.repeat_interleave(torch.arange(n_structs), n_atoms)
        target = _FakeBatch(batch=batch_index, forces=target_f)

        loss_norm = ForcesLoss(loss_fn="mae", normalize_by_n_atoms=True)
        loss_flat = ForcesLoss(loss_fn="mae", normalize_by_n_atoms=False)

        val_norm = loss_norm({"forces": pred_f}, target)
        val_flat = loss_flat({"forces": pred_f}, target)

        assert torch.allclose(
            val_norm, val_flat, atol=1e-6
        ), "With equal-size structures, per-structure and flat-mean losses should agree."

    def test_large_structure_dominates_without_normalisation(self) -> None:
        """Without normalisation, a large structure dominates the gradient.
        With per-structure normalisation, both structures contribute equally.

        Build a batch with a 2-atom and a 10-atom molecule, random predictions
        and zero targets (error = prediction itself).  Verify the formulas:

            per-structure: mean(mae_struct_0, mae_struct_1)
            flat:          mean over all 36 components (large mol has 5× more weight)
        """
        from goal.ml.training.loss import ForcesLoss

        torch.manual_seed(7)
        f_pred = torch.randn(12, 3)  # 2 + 10 atoms
        f_target = torch.zeros(12, 3)
        batch_index = torch.cat(
            [torch.zeros(2, dtype=torch.long), torch.ones(10, dtype=torch.long)]
        )
        target = _FakeBatch(batch=batch_index, forces=f_target)
        pred = {"forces": f_pred}

        loss_norm = ForcesLoss(loss_fn="mae", normalize_by_n_atoms=True)
        loss_flat = ForcesLoss(loss_fn="mae", normalize_by_n_atoms=False)

        val_norm = loss_norm(pred, target)
        val_flat = loss_flat(pred, target)

        # Per-structure: mean(mae_struct_0, mae_struct_1) — equal weight per molecule
        mae_0 = f_pred[:2].abs().mean()
        mae_1 = f_pred[2:].abs().mean()
        expected_norm = (mae_0 + mae_1) / 2.0

        # Flat: mean over all 12 × 3 = 36 elements
        expected_flat = f_pred.abs().mean()

        assert torch.allclose(
            val_norm, expected_norm, atol=1e-6
        ), f"Per-structure loss = {val_norm.item():.6f}, expected {expected_norm.item():.6f}"
        assert torch.allclose(
            val_flat, expected_flat, atol=1e-6
        ), f"Flat loss = {val_flat.item():.6f}, expected {expected_flat.item():.6f}"
        assert not torch.allclose(
            val_norm, val_flat
        ), "Per-structure and flat losses unexpectedly agree for unequal-size structures."

    def test_normalize_false_gives_old_behavior(self) -> None:
        """normalize_by_n_atoms=False reproduces the original flat-mean formula."""
        from goal.ml.training.loss import ForcesLoss

        torch.manual_seed(3)
        pred_f = torch.randn(8, 3)
        target_f = torch.randn(8, 3)
        batch_index = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
        target = _FakeBatch(batch=batch_index, forces=target_f)

        loss = ForcesLoss(loss_fn="mae", normalize_by_n_atoms=False)
        result = loss({"forces": pred_f}, target)
        expected = torch.nn.functional.l1_loss(pred_f, target_f)
        assert torch.allclose(result, expected)

    def test_rmse_per_structure(self) -> None:
        """RMSE variant: per-structure sqrt(MSE_m), then mean over structures."""
        from goal.ml.training.loss import ForcesLoss

        torch.manual_seed(5)
        f_pred = torch.randn(6, 3)
        f_target = torch.zeros(6, 3)
        batch_index = torch.tensor([0, 0, 0, 1, 1, 1])
        target = _FakeBatch(batch=batch_index, forces=f_target)

        loss = ForcesLoss(loss_fn="rmse", normalize_by_n_atoms=True)
        result = loss({"forces": f_pred}, target)

        mse_0 = (f_pred[:3] ** 2).mean()
        mse_1 = (f_pred[3:] ** 2).mean()
        expected = (torch.sqrt(mse_0) + torch.sqrt(mse_1)) / 2.0
        assert torch.allclose(result, expected, atol=1e-6)


class TestEMA:
    """Verify the EMA wrapper."""

    def test_ema_update(self):
        """EMA shadow should move towards current parameters."""
        import torch.nn as nn

        from goal.ml.training.ema import EMAWrapper

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
        import torch.nn as nn

        from goal.ml.training.ema import EMAWrapper

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
