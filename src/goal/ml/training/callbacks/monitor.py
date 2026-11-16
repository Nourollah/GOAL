"""Early warning of a training blow-up: per-layer feature size and update size."""

from __future__ import annotations

import re
import typing

import lightning as L
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_warn


class BlowupMonitor(L.Callback):
    """Log per-layer feature size and optimiser update size; warn when either runs away.

    Feature size is the largest absolute output of every backbone submodule whose qualified
    name matches ``modules`` (a regex), on training steps.  Update size is,
    per parameter tensor, RMS(Δθ) / lr of the step just taken.  For Adam-type
    optimisers it stays well below 1 in healthy training: the median over
    tensors peaked at 0.21 on a SIMURGH GMD run, and it was above 2 on the step
    a divergence started, two steps before the features blew up.

    Logs ``monitor/update_median``, ``monitor/update_max`` and
    ``monitor/feature/<module>``, and warns once each time a threshold is
    crossed (again only after the run has recovered).

    Parameters
    ----------
    modules : str
        Regex searched in backbone submodule names.
    max_feature : float
        Warn when a watched output exceeds this in absolute value.
    max_update : float
        Warn when the median update / lr exceeds this.
    """

    def __init__(
        self,
        modules: str = r"layers\.\d+$",
        max_feature: float = 1.0e3,
        max_update: float = 1.0,
    ) -> None:
        self.pattern = re.compile(modules)
        self.max_feature = float(max_feature)
        self.max_update = float(max_update)
        self._handles: list[typing.Any] = []
        self._feat: dict[str, torch.Tensor] = {}
        self._before: list[tuple[str, torch.Tensor, torch.Tensor, float]] = []
        self._flagged = False

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.on_fit_end(trainer, pl_module)
        model = getattr(pl_module, "backbone", pl_module)
        for name, mod in model.named_modules():
            if name and self.pattern.search(name):
                self._handles.append(mod.register_forward_hook(self._hook(name)))

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _hook(self, name: str) -> typing.Callable[..., None]:
        def hook(mod: torch.nn.Module, inp: typing.Any, out: typing.Any) -> None:
            if mod.training and torch.is_tensor(out):
                self._feat[name] = out.detach().abs().max()

        return hook

    def on_before_optimizer_step(
        self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: typing.Any
    ) -> None:
        lr_of = {id(p): g["lr"] for g in optimizer.param_groups for p in g["params"]}
        self._before = [
            (n, p, p.detach().clone(), float(lr_of[id(p)]))
            for n, p in pl_module.named_parameters()
            if p.grad is not None and lr_of.get(id(p), 0.0) > 0
        ]

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: typing.Any,
        batch: typing.Any,
        batch_idx: int,
    ) -> None:
        metrics: dict[str, float] = {}
        bad: list[str] = []
        if self._before:
            ratios = torch.stack(
                [
                    (p.detach() - old).float().square().mean().sqrt() / lr
                    for _, p, old, lr in self._before
                ]
            ).tolist()
            worst = max(range(len(ratios)), key=lambda i: ratios[i])
            median = sorted(ratios)[len(ratios) // 2]
            metrics["monitor/update_median"] = median
            metrics["monitor/update_max"] = ratios[worst]
            if not median <= self.max_update:  # also catches NaN
                bad.append(
                    f"median update {median:.3g} lr > {self.max_update:g} "
                    f"(largest: {self._before[worst][0]} {ratios[worst]:.3g} lr)"
                )
            self._before = []
        if self._feat:
            names = list(self._feat)
            for name, v in zip(
                names, torch.stack(list(self._feat.values())).tolist(), strict=True
            ):
                metrics[f"monitor/feature/{name}"] = v
                if not v <= self.max_feature:
                    bad.append(f"|{name}| = {v:.3g} > {self.max_feature:g}")
            self._feat.clear()
        if metrics:
            pl_module.log_dict(metrics, on_step=True, on_epoch=False)
        if bad and not self._flagged:
            rank_zero_warn(
                f"[monitor] step {trainer.global_step}: possible blow-up: " + "; ".join(bad)
            )
        self._flagged = bool(bad)
