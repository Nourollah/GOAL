"""Fabric-based custom training loop for full control over the training process.

Wraps `lightning.Fabric <https://lightning.ai/docs/fabric/>`_ to give users
complete ownership of the training loop while still benefitting from:

- **Multi-GPU** — DDP, FSDP, FSDP2, DeepSpeed (same strategies as the Lightning pipeline)
- **Mixed precision** — bf16, fp16, fp64
- **torch.compile** — compile backbone before ``fabric.setup()``
- **Gradient clipping, accumulation** — under your control
- **EMA** — optional, via ``goal.ml.training.ema.EMAWrapper``

This module does NOT depend on Hydra, ``GOALModule``, or any Lightning
``Trainer`` logic.  It reuses the existing GOAL building blocks (backbone,
head, loss, data) and lets you write a plain Python ``for`` loop.

Two usage modes:

1. **FabricTrainer class** — batteries-included custom loop with sensible
   defaults, configurable via constructor kwargs.
2. **Standalone helpers** — ``setup_fabric()`` to get a configured ``Fabric``
   instance, then write your own loop from scratch.

Example (FabricTrainer)::

    from goal.ml.utils.fabric_trainer import FabricTrainer

    ft = FabricTrainer(
        model=my_model,
        loss_fn=my_loss,
        optimizer=my_optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        accelerator="auto",
        strategy="ddp",
        precision="bf16-mixed",
    )
    history = ft.fit(epochs=50)

Example (standalone)::

    from goal.ml.utils.fabric_trainer import setup_fabric

    fabric = setup_fabric(accelerator="auto", strategy="ddp", precision="bf16-mixed")
    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = my_custom_step(model, batch)
            fabric.backward(loss)
            optimizer.step()
"""

from __future__ import annotations

import copy
import time
import typing
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from goal.ml.utils.mini_trainer import TrainingHistory


# ---------------------------------------------------------------------------
# Fabric setup helper
# ---------------------------------------------------------------------------


def setup_fabric(
    accelerator: str = "auto",
    strategy: str = "auto",
    devices: typing.Union[int, str] = "auto",
    num_nodes: int = 1,
    precision: typing.Optional[str] = None,
) -> "lightning.Fabric":
    """Create and launch a configured ``lightning.Fabric`` instance.

    Thin convenience wrapper so callers don't need to remember the
    import path or the ``.launch()`` call.

    Args:
        accelerator: ``'auto'``, ``'cpu'``, ``'gpu'``, ``'mps'``.
        strategy: ``'auto'``, ``'ddp'``, ``'fsdp'``, ``'deepspeed'``, etc.
        devices: Number of devices or ``'auto'``.
        num_nodes: Number of nodes for multi-node training.
        precision: ``'32-true'``, ``'bf16-mixed'``, ``'16-mixed'``, ``'64-true'``, etc.
            ``None`` uses PyTorch default (32-bit).

    Returns:
        A launched ``Fabric`` instance ready for ``setup()`` calls.

    Example::

        fabric = setup_fabric(strategy="ddp", precision="bf16-mixed")
        model, optimizer = fabric.setup(model, optimizer)
    """
    import lightning

    kwargs: typing.Dict[str, typing.Any] = {
        "accelerator": accelerator,
        "strategy": strategy,
        "devices": devices,
        "num_nodes": num_nodes,
    }
    if precision is not None:
        kwargs["precision"] = precision

    fabric: lightning.Fabric = lightning.Fabric(**kwargs)
    fabric.launch()
    return fabric


# ---------------------------------------------------------------------------
# Step function protocol (same contract as MiniTrainer)
# ---------------------------------------------------------------------------


class FabricStepFn(typing.Protocol):
    """Protocol for custom step functions used with ``FabricTrainer``.

    Receives a batch and the model, returns a dict with at least
    a ``'loss'`` or ``'total'`` key (scalar tensor supporting ``.backward()``).

    Unlike ``MiniTrainer.StepFn``, there is no ``device`` parameter —
    Fabric handles device placement automatically.
    """

    def __call__(
        self,
        batch: typing.Any,
        model: nn.Module,
        loss_fn: nn.Module,
    ) -> typing.Dict[str, torch.Tensor]: ...


# ---------------------------------------------------------------------------
# Built-in step functions
# ---------------------------------------------------------------------------


def default_fabric_step(
    batch: typing.Any,
    model: nn.Module,
    loss_fn: nn.Module,
) -> typing.Dict[str, torch.Tensor]:
    """Default step for ``(x, y)`` tuple batches.

    Same semantics as ``mini_trainer.default_step`` but without
    explicit device placement (Fabric manages that).

    Args:
        batch: A tuple ``(x, y)`` from the data loader.
        model: The downstream model.
        loss_fn: Loss function callable.

    Returns:
        Dict with ``'loss'`` key.
    """
    x: torch.Tensor
    y: torch.Tensor
    x, y = batch
    pred: torch.Tensor = model(x)
    loss: torch.Tensor = loss_fn(pred.squeeze(-1), y)
    return {"loss": loss}


def graph_fabric_step(
    batch: typing.Any,
    model: nn.Module,
    loss_fn: nn.Module,
) -> typing.Dict[str, torch.Tensor]:
    """Step for ``AtomicGraph`` batches with a backbone + head model.

    Expects the model to return ``Dict[str, Tensor]`` and the loss
    to be a ``CompositeLoss`` (returns dict with ``'total'`` key).

    Args:
        batch: An ``AtomicGraph`` / PyG ``Batch``.
        model: A callable returning ``Dict[str, Tensor]``.
        loss_fn: A ``CompositeLoss`` or similar dict-aware loss.

    Returns:
        Dict with ``'total'`` and per-component breakdown.
    """
    predictions: typing.Dict[str, torch.Tensor] = model(batch)
    return loss_fn(predictions, batch)


# ---------------------------------------------------------------------------
# FabricTrainer
# ---------------------------------------------------------------------------


class FabricTrainer:
    """Custom training loop built on Lightning Fabric.

    Gives you complete control over the training loop while Fabric
    handles device placement, distributed communication, mixed
    precision, and strategy orchestration.

    This is the middle ground between ``MiniTrainer`` (single device,
    pure PyTorch) and ``GOALModule`` + Lightning ``Trainer`` (fully
    managed, limited customisation).

    Use when you need:
    - Multi-GPU / multi-node training with a **custom** loop
    - Custom gradient accumulation schedules
    - Manual optimisation (multiple optimisers, GAN-style training)
    - Custom logging logic per-step
    - Interleaving training with non-standard operations

    Parameters
    ----------
    model : nn.Module
        The model to train.
    loss_fn : nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Pre-configured optimiser.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader or None
        Validation data loader.
    accelerator : str
        ``'auto'``, ``'cpu'``, ``'gpu'``, ``'mps'``.
    strategy : str
        ``'auto'``, ``'ddp'``, ``'fsdp'``, ``'deepspeed'``, etc.
    devices : int or str
        Number of devices or ``'auto'``.
    num_nodes : int
        Number of nodes.
    precision : str or None
        Mixed-precision mode.
    scheduler : LRScheduler or None
        Learning rate scheduler (stepped per epoch).
    step_fn : FabricStepFn or None
        Custom step function. Defaults to ``default_fabric_step``.
    grad_clip : float or None
        Max gradient norm for clipping.
    grad_accumulation_steps : int
        Accumulate gradients over this many steps before updating.
    enable_progress : bool
        Show tqdm progress bars.

    Example::

        ft = FabricTrainer(
            model=backbone_head_model,
            loss_fn=composite_loss,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            strategy="ddp",
            precision="bf16-mixed",
            step_fn=graph_fabric_step,
        )
        history = ft.fit(epochs=100, early_stopping_patience=20)
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: typing.Any,
        val_loader: typing.Optional[typing.Any] = None,
        accelerator: str = "auto",
        strategy: str = "auto",
        devices: typing.Union[int, str] = "auto",
        num_nodes: int = 1,
        precision: typing.Optional[str] = None,
        scheduler: typing.Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        step_fn: typing.Optional[FabricStepFn] = None,
        grad_clip: typing.Optional[float] = None,
        grad_accumulation_steps: int = 1,
        enable_progress: bool = True,
    ) -> None:
        self.loss_fn: nn.Module = loss_fn
        self.scheduler: typing.Optional[torch.optim.lr_scheduler.LRScheduler] = scheduler
        self.step_fn: FabricStepFn = step_fn or default_fabric_step
        self.grad_clip: typing.Optional[float] = grad_clip
        self.grad_accumulation_steps: int = max(1, grad_accumulation_steps)
        self.enable_progress: bool = enable_progress

        # Set up Fabric
        self.fabric: typing.Any = setup_fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
        )

        # Fabric-wrap model and optimiser
        self.model: nn.Module
        self.optimizer: torch.optim.Optimizer
        self.model, self.optimizer = self.fabric.setup(model, optimizer)

        # Fabric-wrap data loaders
        self.train_loader: typing.Any = self.fabric.setup_dataloaders(train_loader)
        self.val_loader: typing.Optional[typing.Any] = None
        if val_loader is not None:
            self.val_loader = self.fabric.setup_dataloaders(val_loader)

        # Best model state
        self._best_val_loss: float = float("inf")
        self._best_state: typing.Optional[typing.Dict[str, typing.Any]] = None

    # ------------------------------------------------------------------
    # Progress bar helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_progress_bar(
        iterable: typing.Any,
        total: int,
        desc: str,
        enabled: bool,
    ) -> typing.Any:
        """Wrap an iterable with tqdm if available and enabled."""
        if not enabled:
            return iterable
        try:
            from tqdm.auto import tqdm
            return tqdm(iterable, total=total, desc=desc, leave=False)
        except ImportError:
            return iterable

    # ------------------------------------------------------------------
    # Train / val epochs
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        """Run one training epoch with optional gradient accumulation.

        Returns:
            Mean training loss over all batches.
        """
        self.model.train()
        total_loss: float = 0.0
        num_batches: int = 0

        progress: typing.Any = self._get_progress_bar(
            self.train_loader, total=len(self.train_loader),
            desc="train", enabled=self.enable_progress,
        )

        for batch_idx, batch in enumerate(progress):
            is_accumulating: bool = (
                (batch_idx + 1) % self.grad_accumulation_steps != 0
            )

            # Fabric context manager for gradient accumulation —
            # skips all-reduce sync on accumulation steps (performance).
            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                result: typing.Dict[str, torch.Tensor] = self.step_fn(
                    batch, self.model, self.loss_fn,
                )
                loss: torch.Tensor = result.get("loss", result.get("total"))
                self.fabric.backward(loss / self.grad_accumulation_steps)

            if not is_accumulating or (batch_idx + 1) == len(self.train_loader):
                if self.grad_clip is not None:
                    self.fabric.clip_gradients(
                        self.model, self.optimizer,
                        max_norm=self.grad_clip, norm_type=2.0,
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _val_epoch(self) -> float:
        """Run one validation epoch.

        Returns:
            Mean validation loss over all batches.
        """
        self.model.eval()
        total_loss: float = 0.0
        num_batches: int = 0

        progress: typing.Any = self._get_progress_bar(
            self.val_loader, total=len(self.val_loader),
            desc="val", enabled=self.enable_progress,
        )

        for batch in progress:
            result: typing.Dict[str, torch.Tensor] = self.step_fn(
                batch, self.model, self.loss_fn,
            )
            loss: torch.Tensor = result.get("loss", result.get("total"))
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        epochs: int = 10,
        early_stopping_patience: typing.Optional[int] = None,
        checkpoint_best: bool = True,
        verbose: bool = True,
    ) -> TrainingHistory:
        """Run the custom training loop.

        Args:
            epochs: Number of epochs.
            early_stopping_patience: Stop if no val improvement for
                this many epochs. ``None`` disables.
            checkpoint_best: Keep in-memory copy of best model state.
            verbose: Print per-epoch summary.

        Returns:
            ``TrainingHistory`` with loss curves and timing.
        """
        history: TrainingHistory = TrainingHistory()
        patience_counter: int = 0

        for epoch in range(epochs):
            t0: float = time.perf_counter()

            # --- Train ---
            train_loss: float = self._train_epoch()
            history.train_loss.append(train_loss)

            # --- Validate ---
            val_loss_str: str = "—"
            if self.val_loader is not None:
                val_loss: float = self._val_epoch()
                history.val_loss.append(val_loss)
                val_loss_str = f"{val_loss:.6f}"

                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    patience_counter = 0
                    if checkpoint_best:
                        # Get raw model state (unwrap Fabric wrapper)
                        raw_model: nn.Module = (
                            self.model.module
                            if hasattr(self.model, "module")
                            else self.model
                        )
                        self._best_state = copy.deepcopy(raw_model.state_dict())
                else:
                    patience_counter += 1
            else:
                val_loss = float("nan")

            # --- Scheduler ---
            current_lr: float = self.optimizer.param_groups[0]["lr"]
            history.lr.append(current_lr)
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    metric: float = val_loss if self.val_loader is not None else train_loss
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            # --- Timing ---
            elapsed: float = time.perf_counter() - t0
            history.epoch_times.append(elapsed)

            # --- Logging (rank 0 only) ---
            if verbose and self.fabric.is_global_zero:
                parts: typing.List[str] = [
                    f"Epoch {epoch + 1:>{len(str(epochs))}}/{epochs}",
                    f"train_loss={train_loss:.6f}",
                    f"val_loss={val_loss_str}",
                    f"lr={current_lr:.2e}",
                    f"({elapsed:.1f}s)",
                ]
                print("  ".join(parts))

            # --- Early stopping ---
            if (
                early_stopping_patience is not None
                and self.val_loader is not None
                and patience_counter >= early_stopping_patience
            ):
                if verbose and self.fabric.is_global_zero:
                    print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(no improvement for {early_stopping_patience} epochs)"
                    )
                break

        return history

    def load_best(self) -> None:
        """Restore model weights from the best validation checkpoint.

        Raises:
            RuntimeError: If no best checkpoint was saved.
        """
        if self._best_state is None:
            raise RuntimeError(
                "No best checkpoint available. "
                "Run fit() with val_loader and checkpoint_best=True first."
            )
        raw_model: nn.Module = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        raw_model.load_state_dict(self._best_state)

    def save_checkpoint(self, path: str) -> None:
        """Save model and optimiser state to disk via Fabric.

        Handles distributed checkpointing automatically (only rank 0
        writes for DDP; sharded writes for FSDP).

        Args:
            path: File path for the checkpoint.
        """
        state: typing.Dict[str, typing.Any] = {
            "model": self.model,
            "optimizer": self.optimizer,
        }
        self.fabric.save(path, state)

    def load_checkpoint(self, path: str) -> None:
        """Load model and optimiser state from a Fabric checkpoint.

        Args:
            path: File path to the checkpoint.
        """
        state: typing.Dict[str, typing.Any] = {
            "model": self.model,
            "optimizer": self.optimizer,
        }
        self.fabric.load(path, state)
