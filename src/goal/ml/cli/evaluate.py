"""Hydra-based evaluation entry point.

Evaluates a trained model checkpoint on the test split.
Compatible with Lightning 2.6+ and DDP/FSDP strategies.
"""

from __future__ import annotations

import typing

import hydra
import torch
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from goal.ml.data.datamodule import GOALDataModule
from goal.ml.registry import BACKBONE_REGISTRY, HEAD_REGISTRY, LOSS_REGISTRY
from goal.ml.training.loss import CompositeLoss, WeightedLoss
from goal.ml.training.module import GOALModule


def _instantiate_callbacks(cfg: DictConfig | None) -> list[Callback]:
    """Instantiate Lightning callbacks from Hydra config."""
    if not cfg:
        return []
    callbacks: list[Callback] = []
    for _, cb_conf in cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def _instantiate_loggers(cfg: DictConfig | None) -> list[Logger]:
    """Instantiate Lightning loggers from Hydra config."""
    if not cfg:
        return []
    loggers: list[Logger] = []
    for _, lg_conf in cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers


def _build_loss(cfg: DictConfig) -> CompositeLoss:
    """Construct the composite loss from config."""
    losses: list[WeightedLoss] = []
    for loss_cfg in cfg.training.losses:
        loss_cls: typing.Any = LOSS_REGISTRY.get(loss_cfg.name)
        weighted: WeightedLoss = WeightedLoss(loss_cls(), weight=loss_cfg.weight)
        losses.append(weighted)
    return CompositeLoss(losses)


def _build_head(cfg: DictConfig) -> typing.Any:
    """Build the task head from config."""
    head_cls: typing.Any = HEAD_REGISTRY.get(cfg.model.head.name)
    head_kwargs: dict[str, typing.Any] = {k: v for k, v in cfg.model.head.items() if k != "name"}
    return head_cls(**head_kwargs)


@hydra.main(version_base=None, config_path="../../../configs", config_name="eval")
def evaluate(cfg: DictConfig) -> None:
    """GOAL evaluation entry point.

    Loads a checkpoint and evaluates on the test set.
    Strategy (DDP, FSDP, etc.) is controlled via the trainer config group:
        python evaluate.py trainer=ddp ckpt_path=...
    """
    assert cfg.ckpt_path, "Must provide ckpt_path for evaluation"

    # Performance: TF32 for Ampere+ GPUs — falls back safely on older hardware
    perf: dict[str, typing.Any] = cfg.training.get("performance", {})
    has_tf32: bool = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    precision: str = perf.get(
        "float32_matmul_precision",
        "high" if has_tf32 else "highest",
    )
    torch.set_float32_matmul_precision(precision)

    # Build components via registry
    backbone_cls: typing.Any = BACKBONE_REGISTRY.get(cfg.model.backbone.name)
    backbone_kwargs: dict[str, typing.Any] = {
        k: v for k, v in cfg.model.backbone.items() if k != "name"
    }
    backbone: typing.Any = backbone_cls(**backbone_kwargs)

    head: typing.Any = _build_head(cfg)
    loss: CompositeLoss = _build_loss(cfg)

    module: GOALModule = GOALModule(backbone=backbone, head=head, loss=loss, config=cfg)
    datamodule: GOALDataModule = GOALDataModule(cfg)

    # Instantiate callbacks / loggers from Hydra config
    callbacks: list[Callback] = _instantiate_callbacks(cfg.get("callbacks"))
    loggers: list[Logger] = _instantiate_loggers(cfg.get("logger"))

    # Trainer from config group — strategy is determined by trainer yaml
    trainer: typing.Any = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks or None,
        logger=loggers or True,
    )

    trainer.test(module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    evaluate()
