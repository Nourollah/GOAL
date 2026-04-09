"""Hydra-based hyperparameter tuning entry point.

Supports three modes, selected automatically from the config:

1. **Basic** (``hparams_search.method: tuner``) — Lightning's built-in
   ``Tuner`` for learning rate and batch size auto-discovery.
2. **Advanced** (``hparams_search.method: ray``) — Ray Tune with ASHA,
   PBT, Optuna, or HyperOpt.
3. **W&B Sweeps** (``hparams_search.method: wandb``) — cloud-managed
   Bayesian, grid, or random sweeps via Weights & Biases.

Usage::

    # Basic: find optimal LR
    goal-tune hparams_search=basic

    # Advanced: Ray Tune with ASHA + Optuna
    goal-tune hparams_search=ray_tune

    # W&B Sweeps: Bayesian optimisation
    goal-tune hparams_search=wandb_sweep
"""

from __future__ import annotations

import typing
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from goal.ml.data.datamodule import GOALDataModule
from goal.ml.registry import BACKBONE_REGISTRY, HEAD_REGISTRY, LOSS_REGISTRY
from goal.ml.training.loss import CompositeLoss, WeightedLoss
from goal.ml.training.module import GOALModule


def _build_loss(cfg: DictConfig) -> CompositeLoss:
    """Construct the composite loss from config."""
    losses: list[WeightedLoss] = []
    for loss_cfg in cfg.training.losses:
        loss_cls: typing.Any = LOSS_REGISTRY.get(loss_cfg.name)
        fn_spec: typing.Any = loss_cfg.get("fn", "mse")

        if isinstance(fn_spec, str):
            losses.append(
                WeightedLoss(
                    loss_cls(loss_fn=fn_spec),
                    weight=loss_cfg.weight,
                    label=loss_cfg.name,
                )
            )
        else:
            for sub in fn_spec:
                sub_name: str = sub["name"] if isinstance(sub, dict) else sub.name
                sub_weight: float = float(sub["weight"] if isinstance(sub, dict) else sub.weight)
                fn_label: str = sub_name.rsplit(".", 1)[-1]
                losses.append(
                    WeightedLoss(
                        loss_cls(loss_fn=sub_name),
                        weight=sub_weight,
                        label=f"{loss_cfg.name}_{fn_label}",
                        group=loss_cfg.name,
                    )
                )
    return CompositeLoss(losses)


def _build_components(cfg: DictConfig) -> tuple[typing.Any, typing.Any, CompositeLoss]:
    """Build backbone, head, and loss from config."""
    backbone_cls: typing.Any = BACKBONE_REGISTRY.get(cfg.model.backbone.name)
    backbone_kwargs: dict[str, typing.Any] = {
        k: v for k, v in cfg.model.backbone.items() if k != "name"
    }
    backbone: typing.Any = backbone_cls(**backbone_kwargs)

    head_cls: typing.Any = HEAD_REGISTRY.get(cfg.model.head.name)
    head_kwargs: dict[str, typing.Any] = {k: v for k, v in cfg.model.head.items() if k != "name"}
    head: typing.Any = head_cls(**head_kwargs)

    loss: CompositeLoss = _build_loss(cfg)
    return backbone, head, loss


def _ray_train_fn(config: dict[str, typing.Any], base_cfg: DictConfig) -> None:
    """Single Ray Tune trial — called by Ray with a config dict.

    Merges Ray's sampled hyperparameters into the Hydra config,
    builds components, and trains for one trial.
    """
    from ray.train import report

    # Merge sampled params into base config
    merged: DictConfig = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    for key, value in config.items():
        OmegaConf.update(merged, key, value)

    if merged.get("seed"):
        L.seed_everything(merged.seed, workers=True)

    backbone, head, loss = _build_components(merged)
    module: GOALModule = GOALModule(
        backbone=backbone,
        head=head,
        loss=loss,
        config=merged,
    )
    datamodule: GOALDataModule = GOALDataModule(merged)

    trainer: L.Trainer = L.Trainer(
        max_epochs=merged.hparams_search.get("max_epochs", merged.training.get("max_epochs", 500)),
        accelerator="auto",
        devices="auto",
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(module, datamodule=datamodule)

    # Report final validation metric to Ray
    val_metrics: dict[str, float] = {
        k: v.item() if hasattr(v, "item") else v
        for k, v in trainer.callback_metrics.items()
        if k.startswith("val/")
    }
    report(val_metrics)


@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
def tune(cfg: DictConfig) -> None:
    """GOAL hyperparameter tuning entry point.

    Dispatches to Lightning Tuner or Ray Tune based on
    ``hparams_search.method``.
    """
    hp_cfg: DictConfig | None = cfg.get("hparams_search")
    if hp_cfg is None:
        raise ValueError(
            "No hparams_search config found. Use:\n"
            "  goal-tune hparams_search=basic     # Lightning Tuner\n"
            "  goal-tune hparams_search=ray_tune   # Ray Tune"
        )

    method: str = hp_cfg.get("method", "tuner")

    if method == "tuner":
        from goal.ml.training.tuning import run_lightning_tuner

        if cfg.get("seed"):
            L.seed_everything(cfg.seed, workers=True)

        backbone, head, loss = _build_components(cfg)
        module: GOALModule = GOALModule(
            backbone=backbone,
            head=head,
            loss=loss,
            config=cfg,
        )
        datamodule: GOALDataModule = GOALDataModule(cfg)
        trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

        results: dict[str, typing.Any] = run_lightning_tuner(
            module,
            datamodule,
            trainer,
            cfg,
        )
        print(f"\n[Tuner results] {results}")

    elif method == "ray":
        from goal.ml.training.tuning import run_ray_tune

        result_grid = run_ray_tune(_ray_train_fn, cfg)
        best = result_grid.get_best_result()
        print(f"\n[Ray Tune] Best config: {best.config}")
        print(f"[Ray Tune] Best metric: {best.metrics}")

    elif method == "wandb":
        from goal.ml.training.tuning import run_wandb_sweep

        def _wandb_train_fn() -> None:
            """Single W&B sweep trial — called by wandb.agent."""
            import wandb

            with wandb.init() as run:
                # Merge W&B-sampled params into base config
                merged = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                for key, value in run.config.items():
                    OmegaConf.update(merged, key, value)

                if merged.get("seed"):
                    L.seed_everything(merged.seed, workers=True)

                backbone, head, loss = _build_components(merged)
                module = GOALModule(
                    backbone=backbone,
                    head=head,
                    loss=loss,
                    config=merged,
                )
                datamodule = GOALDataModule(merged)

                from lightning.pytorch.loggers import WandbLogger

                wandb_logger = WandbLogger(experiment=run)
                trainer = L.Trainer(
                    max_epochs=int(merged.training.get("max_epochs", 500)),
                    accelerator="auto",
                    devices="auto",
                    logger=wandb_logger,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                )
                trainer.fit(module, datamodule=datamodule)

        sweep_id = run_wandb_sweep(_wandb_train_fn, cfg)
        print(f"\n[W&B Sweep] Sweep ID: {sweep_id}")

    else:
        raise ValueError(f"Unknown tuning method '{method}'. Use 'tuner', 'ray', or 'wandb'.")


if __name__ == "__main__":
    tune()
