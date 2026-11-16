"""``training.safeguards`` in GOALModule and the BlowupMonitor callback."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from goal.ml.training.callbacks.monitor import BlowupMonitor
from goal.ml.training.loss import CompositeLoss, EnergyLoss, ForcesLoss, WeightedLoss
from goal.ml.training.module import GOALModule


def _module(safeguards: dict, stages: list | None = None) -> GOALModule:
    cfg = OmegaConf.create(
        {"training": {"ema": {"enabled": False}, "safeguards": safeguards, "stages": stages}}
    )
    loss = CompositeLoss(
        [
            WeightedLoss(EnergyLoss(), weight=1.0, label="energy"),
            WeightedLoss(ForcesLoss(), weight=100.0, label="forces"),
        ]
    )
    return GOALModule(backbone=nn.Linear(3, 1), head=None, loss=loss, config=cfg)


def test_unknown_safeguard_key_raises() -> None:
    with pytest.raises(ValueError, match="lr_warmup"):
        _module({"lr_warmup": 10})


def test_lr_warmup_scales_only_the_step() -> None:
    module = _module({"lr_warmup_steps": 4})
    seen: list[float] = []
    opt = SimpleNamespace(
        param_groups=[{"lr": 0.01}],
        step=lambda closure=None: seen.append(opt.param_groups[0]["lr"]),
    )
    for step in (0, 3, 10):
        module._trainer = SimpleNamespace(global_step=step)
        module.optimizer_step(0, 0, opt, None)
    assert seen == pytest.approx([0.0025, 0.01, 0.01])
    assert opt.param_groups[0]["lr"] == 0.01  # restored: schedulers never see the scale


def _weights_at(module: GOALModule, step: int, epoch: int) -> dict[str, float]:
    module._trainer = SimpleNamespace(
        global_step=step, num_training_batches=5, accumulate_grad_batches=1, optimizers=[]
    )
    module._maybe_advance_stage(epoch)
    return {k: wl.weight for k, wl in module._weighted_loss_by_label.items()}


def test_raised_weight_ramps_in_and_lowered_weight_drops_at_once() -> None:
    # the schedule that diverged: forces 1 -> 100 at the switch (5 steps per epoch)
    stages = [
        {"name": "warmup", "epochs": 1, "loss_weights": {"energy": 1.0, "forces": 1.0}},
        {"name": "main", "epochs": None, "loss_weights": {"energy": 1.0, "forces": 100.0}},
    ]
    module = _module({"weight_ramp_steps": 10}, stages)
    assert _weights_at(module, 3, 0)["forces"] == 1.0
    assert _weights_at(module, 5, 1)["forces"] == 1.0
    assert _weights_at(module, 10, 1)["forces"] == pytest.approx(50.5)
    assert _weights_at(module, 15, 1)["forces"] == 100.0
    assert _weights_at(module, 99, 19)["forces"] == 100.0  # resume far past the ramp

    stages[0]["loss_weights"] = {"energy": 100.0, "forces": 100.0}
    module = _module({"weight_ramp_steps": 10}, stages)
    assert _weights_at(module, 5, 1) == {"energy": 1.0, "forces": 100.0}


def test_nonfinite_step_is_skipped() -> None:
    module = _module({"skip_nonfinite": True})
    module._trainer = SimpleNamespace(global_step=7)
    opt = torch.optim.AdamW(module.parameters(), lr=0.1)
    before = [p.detach().clone() for p in module.parameters()]
    for p in module.parameters():
        p.grad = torch.ones_like(p)
    module.backbone.weight.grad[0, 0] = float("nan")
    with pytest.warns(UserWarning, match="step 7: skipped"):
        module.on_before_optimizer_step(opt)
    opt.step()
    assert all(torch.equal(a, b) for a, b in zip(before, module.parameters(), strict=True))
    assert not opt.state  # moment estimates untouched

    for p in module.parameters():
        p.grad = torch.ones_like(p)
    module.on_before_optimizer_step(opt)
    assert all(p.grad is not None for p in module.parameters())


def test_nonfinite_steps_in_a_row_stop_training() -> None:
    # blown-up (finite) weights make every batch non-finite: stop instead of idling
    module = _module({"skip_nonfinite": True})
    module._trainer = SimpleNamespace(global_step=0, should_stop=False)
    opt = torch.optim.AdamW(module.parameters(), lr=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(GOALModule.MAX_SKIPS_IN_ROW):
            assert not module._trainer.should_stop, i
            for p in module.parameters():
                p.grad = torch.full_like(p, float("inf"))
            module.on_before_optimizer_step(opt)
    assert module._trainer.should_stop


def test_blowup_monitor_logs_and_warns_once() -> None:
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    logged: dict[str, float] = {}
    pl_module = SimpleNamespace(
        backbone=model,
        named_parameters=model.named_parameters,
        log_dict=lambda m, **kw: logged.update(m),
    )
    trainer = SimpleNamespace(global_step=1)
    mon = BlowupMonitor(max_feature=100.0, max_update=10.0)
    mon.on_fit_start(trainer, pl_module)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    def step(scale: float) -> None:
        x = torch.full((1, 2), scale)
        model.layers[1](model.layers[0](x)).sum().backward()
        mon.on_before_optimizer_step(trainer, pl_module, opt)
        opt.step()
        opt.zero_grad()
        mon.on_train_batch_end(trainer, pl_module, None, None, 0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        step(1.0)  # healthy: SGD moves each weight by |grad| lr (~1 lr), features ~1
    assert set(logged) >= {"monitor/update_median", "monitor/feature/layers.0"}
    with pytest.warns(UserWarning, match="possible blow-up"):
        step(1.0e4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        step(1.0e4)  # still flagged: no repeat warning
