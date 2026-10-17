"""Hydra-based fine-tuning entry point.

Fine-tunes a pre-trained model on a downstream dataset.  The default
config, ``finetune/simurgh_lora``, adapts a trained SIMURGH checkpoint with
LoRA::

    goal-finetune model.backbone.checkpoint_path=logs/<task>/runs/<run>/checkpoints \
        data.train_dir=data/dicarboxylic/train

Any config whose backbone is an adapter works too.  The backbone is built in
one of three modes:

1. **Pre-trained hub model**: ``backbone.pretrained=true backbone.variant=large``
2. **Local fine-tuned checkpoint**: ``backbone.local_checkpoint=/path/to/model.pt``
3. **Fresh backbone**: default — builds from config params (this is how
   ``simurgh_lora`` loads its checkpoint)

Monolithic backbones (``head: null``) are supported.  Optionally freezes
the backbone for linear probing (``training.freeze_backbone``).
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

from goal.ml.cli import CONFIGS_ML_DIR
from goal.ml.cli.builders import (
    build_head,
    build_loss,
    instantiate_callbacks,
    instantiate_loggers,
)
from goal.ml.data.datamodule import GOALDataModule
from goal.ml.registry import BACKBONE_REGISTRY
from goal.ml.training.callbacks.checkpoint import GOALCheckpoint
from goal.ml.training.callbacks.logging import RichLoggingCallback
from goal.ml.training.loss import CompositeLoss
from goal.ml.training.module import GOALModule
from goal.ml.training.strategies.factory import build_strategy


def _build_backbone(backbone_cfg: DictConfig) -> typing.Any:
    """Build backbone with support for pretrained, local, and fresh modes."""
    adapter_cls: typing.Any = BACKBONE_REGISTRY.get(backbone_cfg.name)

    # Mode 1: Load from local fine-tuned checkpoint
    local_ckpt: str | None = backbone_cfg.get("local_checkpoint")
    if local_ckpt:
        if not hasattr(adapter_cls, "from_local"):
            raise ValueError(
                f"Backbone '{backbone_cfg.name}' does not support from_local(). "
                "Use pretrained=true for hub models or provide backbone params."
            )
        return adapter_cls.from_local(local_ckpt)

    # Mode 2: Load pre-trained from hub
    if backbone_cfg.get("pretrained", False):
        return adapter_cls.from_pretrained(variant=backbone_cfg.get("variant", "large"))

    # Mode 3: Fresh backbone from config
    backbone_kwargs: dict[str, typing.Any] = {
        k: v
        for k, v in backbone_cfg.items()
        if k not in {"name", "pretrained", "variant", "local_checkpoint"}
    }
    return adapter_cls(**backbone_kwargs)


@hydra.main(version_base=None, config_path=CONFIGS_ML_DIR, config_name="finetune/simurgh_lora")
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

    head: typing.Any = build_head(cfg, backbone)  # None for monolithic backbones
    loss: CompositeLoss = build_loss(cfg)

    module: GOALModule = GOALModule(
        backbone=backbone,
        head=head,
        loss=loss,
        config=cfg,
        compile_model=cfg.training.get("compile_model", False),
    )
    datamodule: GOALDataModule = GOALDataModule(cfg)

    last_ckpt: Path = checkpoint_dir / "last.ckpt"
    resume_path: str | None = str(last_ckpt) if last_ckpt.exists() else None

    callbacks: list[Callback] = [
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
    hydra_callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))
    all_callbacks: list[Callback] = callbacks + (hydra_callbacks or [])

    # Strategy: if cfg.strategy exists, use the strategy factory;
    # otherwise fall through to hydra.utils.instantiate (backward compat).
    strategy_override: dict[str, typing.Any] = {}
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
