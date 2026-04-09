"""Hydra-based training entry point with automatic SLURM resumption.

This is the main training command for GOAL. It:
1. Builds components via the registry system
2. Automatically resumes from ``last.ckpt`` if it exists
3. Writes a ``TRAINING_COMPLETE`` sentinel to prevent SLURM requeue loops
4. Supports DDP, FSDP, ModelParallel strategies via trainer configs
5. Compatible with Lightning 2.6+ features (EMAWeightAveraging, torch.compile)
6. Performance-tuned: TF32 matmul, cuDNN benchmark, gradient accumulation
"""

from __future__ import annotations

import typing
from pathlib import Path

import hydra
import lightning as L
import torch
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


def _gpu_supports_tf32() -> bool:
    """Check whether the current CUDA device supports TF32 (Ampere+, sm_80+)."""
    if not torch.cuda.is_available():
        return False
    capability: tuple[int, int] = torch.cuda.get_device_capability()
    return capability[0] >= 8  # Ampere = 8.0, Hopper = 9.0


def _setup_performance(cfg: DictConfig) -> None:
    """Apply global PyTorch performance settings before training.

    Settings are applied conditionally based on GPU availability and
    hardware capability. TF32 and cuDNN options are silently skipped
    when running on CPU or pre-Ampere GPUs.

    - **TF32 matmul precision**: Uses TF32 tensor cores for float32 ops,
      giving ~3× speedup with negligible precision loss. Requires Ampere+
      (A100, H100, RTX 30xx/40xx).  Falls back to ``"highest"`` (full fp32)
      on older hardware.
    - **cuDNN benchmark**: Auto-tunes convolution algorithms for the
      given input sizes.  Adds startup overhead but faster thereafter.
      Only meaningful when CUDA is available.
    - **cuDNN deterministic**: Forces deterministic cuDNN algorithms.
      Only meaningful when CUDA is available.
    """
    perf: dict[str, typing.Any] = cfg.training.get("performance", {})
    has_cuda: bool = torch.cuda.is_available()

    # TF32 for matmul — only effective on Ampere+ GPUs (sm_80+)
    if _gpu_supports_tf32():
        matmul_precision: str = perf.get("float32_matmul_precision", "high")
    else:
        matmul_precision = perf.get("float32_matmul_precision", "highest")
    torch.set_float32_matmul_precision(matmul_precision)

    # cuDNN benchmark — only effective when CUDA is present
    if has_cuda:
        torch.backends.cudnn.benchmark = perf.get("cudnn_benchmark", True)
        torch.backends.cudnn.deterministic = perf.get("cudnn_deterministic", False)


def _build_loss(cfg: DictConfig) -> CompositeLoss:
    """Construct the composite loss from config.

    Each loss entry may specify ``fn`` as either:

    - A **string** (default ``"mse"``) — single loss function.
    - A **list** of ``{name, weight}`` dicts — multiple loss functions
      for the same property, each logged and weighted independently.

    Examples::

        # Single fn (backward compatible)
        losses:
          - name: energy
            weight: 4.0
            fn: mse

        # Composite fn per property
        losses:
          - name: forces
            fn:
              - name: mse
                weight: 4.0
              - name: rmse
                weight: 8.0
    """
    losses: list[WeightedLoss] = []
    for loss_cfg in cfg.training.losses:
        loss_cls: typing.Any = LOSS_REGISTRY.get(loss_cfg.name)
        fn_spec: typing.Any = loss_cfg.get("fn", "mse")

        if isinstance(fn_spec, str):
            # Single loss function — backward compatible
            losses.append(
                WeightedLoss(
                    loss_cls(loss_fn=fn_spec),
                    weight=loss_cfg.weight,
                    label=loss_cfg.name,
                )
            )
        else:
            # Composite: list of {name, weight} sub-fns
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


def _build_head(cfg: DictConfig) -> typing.Any:
    """Build the task head from config."""
    head_cls: typing.Any = HEAD_REGISTRY.get(cfg.model.head.name)
    head_kwargs: dict[str, typing.Any] = {k: v for k, v in cfg.model.head.items() if k != "name"}
    return head_cls(**head_kwargs)


@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
def train(cfg: DictConfig) -> None:
    """GOAL training entry point.

    Driven by Hydra configuration. Strategy (DDP, FSDP, ModelParallel)
    is controlled via trainer config group:
        python train.py trainer=ddp
        python train.py trainer=fsdp
        python train.py trainer=model_parallel
    """
    # Seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Apply global PyTorch performance settings
    _setup_performance(cfg)

    # Check for completion sentinel — stops SLURM requeue loop
    checkpoint_dir: Path = Path(cfg.training.checkpoint_dir)
    if (checkpoint_dir / "TRAINING_COMPLETE").exists():
        print("Training already complete. Exiting.")
        return

    # Build components via registry
    backbone_cls: typing.Any = BACKBONE_REGISTRY.get(cfg.model.backbone.name)
    backbone_kwargs: dict[str, typing.Any] = {
        k: v for k, v in cfg.model.backbone.items() if k != "name"
    }
    backbone: typing.Any = backbone_cls(**backbone_kwargs)

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

    # Automatic checkpoint resumption — last.ckpt if exists, else None
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
            save_on_exception=True,  # Lightning 2.5.3+
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichLoggingCallback(),
    ]

    # Optionally add SLURM plugin
    plugins: list[typing.Any] = []
    if cfg.training.get("slurm_mode", False):
        from lightning.pytorch.plugins.environments import SLURMEnvironment

        plugins.append(SLURMEnvironment(auto_requeue=True))

    # Instantiate callbacks and loggers from Hydra config (if present)
    hydra_callbacks: list[Callback] = _instantiate_callbacks(cfg.get("callbacks"))
    loggers: list[Logger] = _instantiate_loggers(cfg.get("logger"))

    # Merge callbacks: GOAL-specific + Hydra-configured
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
        plugins=plugins or None,
        **strategy_override,
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_path)

    # Mark completion so SLURM doesn't requeue after success
    if trainer.is_global_zero:
        (checkpoint_dir / "TRAINING_COMPLETE").touch()


if __name__ == "__main__":
    train()
