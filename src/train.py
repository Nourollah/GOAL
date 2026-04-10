from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# Also add src/ to PYTHONPATH so that 'goal' is importable directly
import sys

sys.path.insert(0, str(root / "src"))

from goal.ml.data.datamodule import GOALDataModule
from goal.ml.registry import (
    BACKBONE_REGISTRY,
    HEAD_REGISTRY,
    LOSS_REGISTRY,
    MODEL_REGISTRY,
)
from goal.ml.training.loss import CompositeLoss, WeightedLoss
from goal.ml.training.module import GOALModule
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _build_goal_module(cfg: DictConfig) -> GOALModule:
    """Build a GOALModule from the Hydra config using the registry system.

    Supports two modes:

    **Modular** — ``model.backbone`` + ``model.head`` both present.
        Backbone from ``MODEL_REGISTRY`` (falling back to ``BACKBONE_REGISTRY``),
        head from ``HEAD_REGISTRY``.

    **Monolithic** — ``model.head`` is ``null`` or absent.
        The backbone is expected to satisfy the ``MonolithicModel`` protocol
        and return a property dictionary directly.
    """
    # Backbone — try MODEL_REGISTRY first, fall back to BACKBONE_REGISTRY
    backbone_name: str = cfg.model.backbone.name
    if backbone_name in MODEL_REGISTRY:
        backbone_cls = MODEL_REGISTRY.get(backbone_name)
    else:
        backbone_cls = BACKBONE_REGISTRY.get(backbone_name)
    backbone_kwargs = {k: v for k, v in cfg.model.backbone.items() if k != "name"}
    backbone = backbone_cls(**backbone_kwargs)

    # Head — None for monolithic models
    head = None
    if cfg.model.get("head") is not None:
        head_cls = HEAD_REGISTRY.get(cfg.model.head.name)
        head_kwargs = {k: v for k, v in cfg.model.head.items() if k != "name"}
        head = head_cls(**head_kwargs)

    # Loss
    losses = []
    for loss_cfg in cfg.training.losses:
        loss_cls = LOSS_REGISTRY.get(loss_cfg.name)
        # Forward extra kwargs (e.g. property_name, fn, per_atom) to the loss
        loss_kwargs: dict[str, Any] = {
            k: v for k, v in loss_cfg.items() if k not in ("name", "weight", "fn")
        }
        fn_spec = loss_cfg.get("fn", "mse")
        if isinstance(fn_spec, str):
            loss_kwargs["loss_fn"] = fn_spec
        losses.append(
            WeightedLoss(
                loss_cls(**loss_kwargs),
                weight=loss_cfg.weight,
                label=loss_cfg.name,
            )
        )
    loss = CompositeLoss(losses)

    return GOALModule(backbone=backbone, head=head, loss=loss, config=cfg)


def _find_latest_checkpoint(cfg: DictConfig) -> str | None:
    """Scan previous run directories for the most recent ``last.ckpt``.

    Searches ``{log_dir}/{task_name}/runs/`` for directories whose name ends
    with ``_{dataset_type}_{backbone_name}`` and contains a
    ``checkpoints/last.ckpt`` file.  Returns the path to the most recently
    modified checkpoint, or ``None`` if nothing is found.

    This deliberately skips the *current* run directory (which Hydra just
    created and is still empty) so we only resume from genuinely earlier runs.
    """
    log_dir: Path = Path(cfg.paths.log_dir)
    task_name: str = cfg.task_name
    dataset_type: str = cfg.data.dataset_type
    backbone_name: str = cfg.model.backbone.name
    current_output_dir: Path = Path(cfg.paths.output_dir)

    runs_dir: Path = log_dir / task_name / "runs"
    if not runs_dir.exists():
        return None

    suffix: str = f"_{dataset_type}_{backbone_name}"
    candidates: list[tuple[float, Path]] = []

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.endswith(suffix):
            continue
        # Skip the directory that Hydra just created for *this* invocation.
        if run_dir.resolve() == current_output_dir.resolve():
            continue
        ckpt: Path = run_dir / "checkpoints" / "last.ckpt"
        if ckpt.exists():
            candidates.append((ckpt.stat().st_mtime, ckpt))

    if not candidates:
        return None

    # Most recently modified checkpoint wins.
    candidates.sort(key=lambda x: x[0], reverse=True)
    return str(candidates[0][1])


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Building GOAL DataModule...")
    datamodule = GOALDataModule(cfg)

    log.info("Building GOAL Module via registry...")
    model = _build_goal_module(cfg)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")

        # Resolve checkpoint path: explicit > auto-resume > None (fresh start)
        ckpt_path: str | None = cfg.get("ckpt_path")
        if ckpt_path is None and cfg.get("auto_resume", False):
            ckpt_path = _find_latest_checkpoint(cfg)
            if ckpt_path is not None:
                log.info(f"Auto-resume: resuming from {ckpt_path}")
            else:
                log.info("Auto-resume: no previous checkpoint found — starting fresh.")

        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
