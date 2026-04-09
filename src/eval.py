from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# Also add src/ to PYTHONPATH so that 'gmd' is importable directly
import sys
sys.path.insert(0, str(root / "src"))

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

from gmd.data.datamodule import GMDDataModule
from gmd.registry import BACKBONE_REGISTRY, HEAD_REGISTRY, LOSS_REGISTRY
from gmd.training.loss import CompositeLoss, WeightedLoss
from gmd.training.module import GMDModule

log = RankedLogger(__name__, rank_zero_only=True)


def _build_gmd_module(cfg: DictConfig) -> GMDModule:
    """Build a GMDModule from the Hydra config using the registry system."""
    backbone_cls = BACKBONE_REGISTRY.get(cfg.model.backbone.name)
    backbone_kwargs = {k: v for k, v in cfg.model.backbone.items() if k != "name"}
    backbone = backbone_cls(**backbone_kwargs)

    head_cls = HEAD_REGISTRY.get(cfg.model.head.name)
    head_kwargs = {k: v for k, v in cfg.model.head.items() if k != "name"}
    head = head_cls(**head_kwargs)

    losses = []
    for loss_cfg in cfg.training.losses:
        loss_cls = LOSS_REGISTRY.get(loss_cfg.name)
        losses.append(WeightedLoss(loss_cls(), weight=loss_cfg.weight))
    loss = CompositeLoss(losses)

    return GMDModule(backbone=backbone, head=head, loss=loss, config=cfg)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info("Building GMD DataModule...")
    datamodule = GMDDataModule(cfg)

    log.info("Building GMD Module via registry...")
    model = _build_gmd_module(cfg)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
