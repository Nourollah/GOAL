"""Hydra-based fine-tuning entry point.

Fine-tunes a pre-trained foundation model (via adapter) on a
downstream dataset. Supports three loading modes:

1. **Pre-trained hub model**: ``backbone.pretrained=true backbone.variant=large``
2. **Local fine-tuned checkpoint**: ``backbone.local_checkpoint=/path/to/model.pt``
3. **Fresh backbone**: default — builds from config params

Optionally freezes the backbone for linear probing.
"""

from __future__ import annotations

import typing
from pathlib import Path

import hydra
import lightning as L
from lightning import Callback
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from goal.ml.data.datamodule import GOALDataModule
from goal.ml.registry import BACKBONE_REGISTRY, HEAD_REGISTRY, LOSS_REGISTRY
from goal.ml.training.callbacks.checkpoint import GOALCheckpoint
from goal.ml.training.callbacks.logging import RichLoggingCallback
from goal.ml.training.loss import CompositeLoss, WeightedLoss
from goal.ml.training.module import GOALModule
from goal.ml.training.strategies.factory import build_strategy


def _instantiate_callbacks(cfg: typing.Optional[DictConfig]) -> typing.List[Callback]:
    """Instantiate Lightning callbacks from Hydra config."""
    if not cfg:
        return []
    callbacks: typing.List[Callback] = []
    for _, cb_conf in cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def _instantiate_loggers(cfg: typing.Optional[DictConfig]) -> typing.List[Logger]:
    """Instantiate Lightning loggers from Hydra config."""
    if not cfg:
        return []
    loggers: typing.List[Logger] = []
    for _, lg_conf in cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers


def _build_loss(cfg: DictConfig) -> CompositeLoss:
    """Construct the composite loss from config.

    Each loss entry may specify ``fn`` as either:

    - A **string** (default ``"mse"``) — single loss function.
    - A **list** of ``{name, weight}`` dicts — multiple loss functions
      for the same property, each logged and weighted independently.
    """
    losses: typing.List[WeightedLoss] = []
    for loss_cfg in cfg.training.losses:
        loss_cls: typing.Any = LOSS_REGISTRY.get(loss_cfg.name)
        fn_spec: typing.Any = loss_cfg.get("fn", "mse")

        if isinstance(fn_spec, str):
            losses.append(WeightedLoss(
                loss_cls(loss_fn=fn_spec),
                weight=loss_cfg.weight,
                label=loss_cfg.name,
            ))
        else:
            for sub in fn_spec:
                sub_name: str = sub["name"] if isinstance(sub, dict) else sub.name
                sub_weight: float = float(
                    sub["weight"] if isinstance(sub, dict) else sub.weight
                )
                fn_label: str = sub_name.rsplit(".", 1)[-1]
                losses.append(WeightedLoss(
                    loss_cls(loss_fn=sub_name),
                    weight=sub_weight,
                    label=f"{loss_cfg.name}_{fn_label}",
                    group=loss_cfg.name,
                ))
    return CompositeLoss(losses)


def _build_head(cfg: DictConfig) -> typing.Any:
    """Build the task head from config."""
    head_cls: typing.Any = HEAD_REGISTRY.get(cfg.model.head.name)
    head_kwargs: typing.Dict[str, typing.Any] = {k: v for k, v in cfg.model.head.items() if k != "name"}
    return head_cls(**head_kwargs)


def _build_backbone(backbone_cfg: DictConfig) -> typing.Any:
    """Build backbone with support for pretrained, local, and fresh modes."""
    adapter_cls: typing.Any = BACKBONE_REGISTRY.get(backbone_cfg.name)

    # Mode 1: Load from local fine-tuned checkpoint
    local_ckpt: typing.Optional[str] = backbone_cfg.get("local_checkpoint")
    if local_ckpt:
        if not hasattr(adapter_cls, "from_local"):
            raise ValueError(
                f"Backbone '{backbone_cfg.name}' does not support from_local(). "
                "Use pretrained=true for hub models or provide backbone params."
            )
        return adapter_cls.from_local(local_ckpt)

    # Mode 2: Load pre-trained from hub
    if backbone_cfg.get("pretrained", False):
        return adapter_cls.from_pretrained(
            variant=backbone_cfg.get("variant", "large")
        )

    # Mode 3: Fresh backbone from config
    backbone_kwargs: typing.Dict[str, typing.Any] = {
        k: v for k, v in backbone_cfg.items()
        if k not in {"name", "pretrained", "variant", "local_checkpoint"}
    }
    return adapter_cls(**backbone_kwargs)


@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
def finetune(cfg: DictConfig) -> None:
    """GOAL fine-tuning entry point.

    Loads a pre-trained backbone (via adapter or checkpoint) and
    fine-tunes on the target dataset.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    checkpoint_dir: Path = Path(cfg.training.checkpoint_dir)

    backbone: typing.Any = _build_backbone(cfg.model.backbone)

    # Optionally freeze backbone (Lightning 2.6.1: freeze() returns self)
    if cfg.training.get("freeze_backbone", False):
        for param in backbone.parameters():
            param.requires_grad = False

    head: typing.Any = _build_head(cfg)
    loss: CompositeLoss = _build_loss(cfg)

    module: GOALModule = GOALModule(
        backbone=backbone,
        head=head,
        loss=loss,
        config=cfg,
        compile_model=cfg.training.get("compile_model", False),
    )
    datamodule: GOALDataModule = GOALDataModule(cfg)

    last_ckpt: Path = checkpoint_dir / "last.ckpt"
    resume_path: typing.Optional[str] = str(last_ckpt) if last_ckpt.exists() else None

    callbacks: typing.List[Callback] = [
        GOALCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="epoch={epoch:04d}-val_loss={val/total:.4f}",
            monitor="val/total",
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
            save_on_exception=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichLoggingCallback(),
    ]

    # Instantiate callbacks and loggers from Hydra config (if present)
    hydra_callbacks: typing.List[Callback] = _instantiate_callbacks(cfg.get("callbacks"))
    loggers: typing.List[Logger] = _instantiate_loggers(cfg.get("logger"))
    all_callbacks: typing.List[Callback] = callbacks + (hydra_callbacks or [])

    # Strategy: if cfg.strategy exists, use the strategy factory;
    # otherwise fall through to hydra.utils.instantiate (backward compat).
    strategy_override: typing.Dict[str, typing.Any] = {}
    if cfg.get("strategy") is not None:
        strategy_override["strategy"] = build_strategy(cfg)

    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=all_callbacks,
        logger=loggers or True,
        **strategy_override,
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_path)


if __name__ == "__main__":
    finetune()
