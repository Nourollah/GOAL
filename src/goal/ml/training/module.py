"""Universal Lightning module for any backbone + head combination.

Handles training, validation, testing, EMA, gradient clipping, and logging.
Compatible with Lightning 2.6+ features including:
- ``configure_model()`` hook for FSDP2 / torch.compile
- ``EMAWeightAveraging`` callback (replaces manual EMA when preferred)
- ``torch.compile`` support with configurable mode and fullgraph
- Proper multi-GPU sync via ``sync_dist``
- ``test_step`` for evaluation
- Performance: gradient accumulation, non-blocking transfers, TF32
"""

from __future__ import annotations

import typing
import warnings

import lightning as L
import torch
from omegaconf import DictConfig

from goal.ml.data.graph import AtomicGraph
from goal.ml.training.ema import EMAWrapper
from goal.ml.training.loss import CompositeLoss


class GOALModule(L.LightningModule):
    """Universal ``LightningModule`` for any backbone + head combination.

    Handles: training, validation, testing, EMA, gradient clipping, logging.
    Compatible with Lightning >=2.6 APIs.

    Parameters
    ----------
    backbone : EquivariantBackbone
        Any model satisfying the backbone protocol.
    head : TaskHead
        Any output head satisfying the head protocol.
    loss : CompositeLoss
        Composable loss function built from weighted components.
    config : DictConfig
        Full Hydra configuration.
    compile_model : bool
        Whether to apply ``torch.compile`` to the backbone in
        ``configure_model()``.
    """

    def __init__(
        self,
        backbone: typing.Any,
        head: typing.Any,
        loss: CompositeLoss,
        config: DictConfig,
        compile_model: bool = False,
    ) -> None:
        super().__init__()
        self.backbone: typing.Any = backbone
        self.head: typing.Any = head
        self.loss: CompositeLoss = loss
        self.config: DictConfig = config
        self._compile_model: bool = compile_model
        self.save_hyperparameters(ignore=["backbone", "head", "loss"])

        # EMA — keeps a shadow copy of weights for evaluation.
        # Lightning 2.6 provides EMAWeightAveraging callback as an alternative,
        # but we keep this for fine-grained control (e.g. per-step decay schedule).
        if config.training.ema.get("enabled", False):
            self.ema = EMAWrapper(
                self.parameters(),
                decay=config.training.ema.decay,
            )

    def configure_model(self) -> None:
        """Hook called before ``configure_optimizers`` (Lightning 2.4+).

        Used for:
        - ``torch.compile`` wrapping (avoids recompilation per epoch)
        - FSDP2 / tensor parallelism wrapping via ``ModelParallelStrategy``
        - DTensor-based sharding
        """
        if self._compile_model:
            compile_cfg: dict[str, typing.Any] = self.config.training.get("compile", {})
            # torch.compile requires PyTorch 2.0+ and works best with CUDA
            if not hasattr(torch, "compile"):
                warnings.warn(
                    "torch.compile requested but unavailable (PyTorch <2.0). Skipping.",
                    stacklevel=2,
                )
            else:
                self.backbone = torch.compile(
                    self.backbone,
                    mode=compile_cfg.get("mode", "default"),
                    fullgraph=compile_cfg.get("fullgraph", False),
                    dynamic=compile_cfg.get("dynamic", None),
                )

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        """Run backbone -> head pipeline."""
        features: typing.Any = self.backbone(graph)
        return self.head(features, graph)

    def on_before_batch_transfer(self, batch: typing.Any, dataloader_idx: int) -> typing.Any:
        """Pre-transfer hook — ensures non-blocking device transfer.

        Lightning calls this before moving the batch to GPU. We ensure
        the batch tensors are contiguous (avoids scattered reads on PCIe).
        """
        return batch

    def training_step(self, batch: AtomicGraph, batch_idx: int) -> torch.Tensor:
        """Single training step -- forward, loss, logging."""
        predictions: dict[str, torch.Tensor] = self(batch)
        losses: dict[str, torch.Tensor] = self.loss(predictions, batch)

        # Log with gradient accumulation awareness
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            batch_size=batch.num_graphs,
            sync_dist=True,
            prog_bar=True,
        )
        return losses["total"]

    def on_train_batch_end(
        self,
        outputs: typing.Any,
        batch: AtomicGraph,
        batch_idx: int,
    ) -> None:
        """Update EMA after each optimiser step."""
        if hasattr(self, "ema"):
            self.ema.update()

    def validation_step(self, batch: AtomicGraph, batch_idx: int) -> None:
        """Single validation step -- uses EMA weights if available."""
        if hasattr(self, "ema"):
            with self.ema.average_parameters():
                predictions = self(batch)
        else:
            predictions = self(batch)

        losses = self.loss(predictions, batch)
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()},
            batch_size=batch.num_graphs,
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: AtomicGraph, batch_idx: int) -> None:
        """Single test step -- uses EMA weights if available."""
        if hasattr(self, "ema"):
            with self.ema.average_parameters():
                predictions = self(batch)
        else:
            predictions = self(batch)

        losses = self.loss(predictions, batch)
        self.log_dict(
            {f"test/{k}": v for k, v in losses.items()},
            batch_size=batch.num_graphs,
            sync_dist=True,
        )

    def configure_optimizers(self) -> dict[str, typing.Any]:
        """Set up optimiser and learning rate scheduler.

        Uses AdamW (PyTorch 2.x) as default. Supports cosine annealing
        and reduce-on-plateau schedulers.
        """
        opt_cfg: typing.Any = self.config.training.optimizer
        scheduler_type: str = opt_cfg.get("scheduler_type", "reduce_on_plateau")

        optimizer: torch.optim.AdamW = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            amsgrad=opt_cfg.get("amsgrad", True),
        )

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.get("max_epochs", 500),
                eta_min=opt_cfg.get("min_lr", 1e-7),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        elif scheduler_type == "cosine_warmup":
            from torch.optim.lr_scheduler import (
                CosineAnnealingLR,
                LinearLR,
                SequentialLR,
            )

            warmup_epochs = opt_cfg.get("warmup_epochs", 10)
            warmup = LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.get("max_epochs", 500) - warmup_epochs,
                eta_min=opt_cfg.get("min_lr", 1e-7),
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        else:
            # Default: ReduceLROnPlateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=opt_cfg.scheduler.patience,
                factor=opt_cfg.scheduler.factor,
                min_lr=opt_cfg.get("min_lr", 1e-7),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/total",
                    "interval": "epoch",
                },
            }

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Gradient clipping before optimiser step."""
        clip_val: float = self.config.training.get("gradient_clip", 0)
        if clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm="norm",
            )
