"""Standalone mini trainer for quick experimentation in Jupyter notebooks.

A lightweight training loop decoupled from the Lightning pipeline,
designed for rapid prototyping with extracted features. Typical use-case:
freeze a foundation model (MACE, FairChem, etc.), extract representations,
then train your own downstream head on those features — all within a
notebook cell.

Does NOT touch the main training code, Hydra configs, or Lightning
infrastructure. Operates purely on PyTorch primitives so it can be
wielded freely in interactive sessions.

Example::

    from goal.ml.utils.mini_trainer import MiniTrainer

    trainer = MiniTrainer(
        model=my_head,
        loss_fn=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(my_head.parameters(), lr=1e-3),
    )
    history = trainer.fit(train_loader, val_loader=val_loader, epochs=50)
"""

from __future__ import annotations

import copy
import time
import typing
from dataclasses import dataclass, field

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# History dataclass — stores per-epoch metrics
# ---------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """Container for per-epoch training and validation metrics.

    Stores loss curves and any extra metrics logged during training.
    Provides convenience methods for plotting and inspection.

    Attributes:
        train_loss: Per-epoch mean training loss.
        val_loss: Per-epoch mean validation loss (empty if no val loader).
        lr: Per-epoch learning rate.
        epoch_times: Wall-clock seconds per epoch.
        extra: Dict of additional per-epoch metric lists.
    """

    train_loss: typing.List[float] = field(default_factory=list)
    val_loss: typing.List[float] = field(default_factory=list)
    lr: typing.List[float] = field(default_factory=list)
    epoch_times: typing.List[float] = field(default_factory=list)
    extra: typing.Dict[str, typing.List[float]] = field(default_factory=dict)

    @property
    def best_val_loss(self) -> typing.Optional[float]:
        """Return the lowest validation loss, or ``None`` if not tracked."""
        if not self.val_loss:
            return None
        return min(self.val_loss)

    @property
    def best_epoch(self) -> typing.Optional[int]:
        """Return the epoch index (0-based) with the lowest validation loss."""
        if not self.val_loss:
            return None
        return int(torch.tensor(self.val_loss).argmin().item())

    def plot(self, figsize: typing.Tuple[int, int] = (10, 4)) -> None:
        """Plot training and validation loss curves.

        Requires ``matplotlib``. Safe to call in Jupyter notebooks.

        Args:
            figsize: Figure size as ``(width, height)`` in inches.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        epochs: typing.List[int] = list(range(1, len(self.train_loss) + 1))
        ax.plot(epochs, self.train_loss, label="train")
        if self.val_loss:
            ax.plot(epochs, self.val_loss, label="val")
            best: typing.Optional[int] = self.best_epoch
            if best is not None:
                ax.axvline(
                    x=best + 1, color="gray", linestyle="--", alpha=0.5,
                    label=f"best val (epoch {best + 1})",
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("MiniTrainer — Loss Curve")
        ax.legend()
        ax.set_yscale("log")
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Step function protocol
# ---------------------------------------------------------------------------


class StepFn(typing.Protocol):
    """Protocol for custom step functions passed to ``MiniTrainer``.

    A step function receives a single batch from the data loader and must
    return a dict containing at least ``'loss'`` (a scalar tensor that
    supports ``.backward()``).

    This allows full flexibility over how the model, loss, and targets
    interact without sub-classing or modifying the trainer.
    """

    def __call__(
        self,
        batch: typing.Any,
        model: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> typing.Dict[str, torch.Tensor]: ...


# ---------------------------------------------------------------------------
# Default step functions
# ---------------------------------------------------------------------------


def default_step(
    batch: typing.Any,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
) -> typing.Dict[str, torch.Tensor]:
    """Default training step for ``(features, targets)`` tuple batches.

    Expects each batch to unpack as ``(x, y)`` where ``x`` is the input
    tensor and ``y`` is the target tensor.  Suitable for pre-extracted
    feature datasets stored as ``TensorDataset``.

    Args:
        batch: A tuple ``(x, y)`` from the data loader.
        model: The downstream model (e.g. an MLP head).
        loss_fn: Loss function callable.
        device: Target device for tensors.

    Returns:
        Dict with ``'loss'`` key containing the scalar loss tensor.
    """
    x: torch.Tensor
    y: torch.Tensor
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    pred: torch.Tensor = model(x)
    loss: torch.Tensor = loss_fn(pred.squeeze(-1), y)
    return {"loss": loss}


def graph_step(
    batch: typing.Any,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
) -> typing.Dict[str, torch.Tensor]:
    """Training step for ``AtomicGraph`` batches with a full backbone + head.

    Expects the batch to be a PyG ``Data``/``Batch`` object with ``energy``
    and/or ``forces`` targets. The model should return a dict with matching
    keys (like ``GOALModule.forward``).

    Args:
        batch: An ``AtomicGraph`` / PyG ``Batch`` from the data loader.
        model: A callable returning ``Dict[str, Tensor]``.
        loss_fn: A ``CompositeLoss`` or similar dict-aware loss.
        device: Target device.

    Returns:
        Dict with ``'loss'`` key and per-component breakdown.
    """
    batch = batch.to(device, non_blocking=True)
    predictions: typing.Dict[str, torch.Tensor] = model(batch)
    losses: typing.Dict[str, torch.Tensor] = loss_fn(predictions, batch)
    return losses  # Must contain 'total' key from CompositeLoss


# ---------------------------------------------------------------------------
# MiniTrainer
# ---------------------------------------------------------------------------


class MiniTrainer:
    """Lightweight training loop for interactive / notebook use.

    Completely standalone — does not depend on Lightning, Hydra, or
    any of the main GOAL training infrastructure. Operates on raw
    PyTorch primitives: a model, a loss function, an optimiser, and
    a data loader.

    Designed for the common workflow:
    1. Extract features from a frozen backbone (MACE, etc.)
    2. Cache them as a ``TensorDataset``
    3. Train a lightweight downstream head with ``MiniTrainer``

    Parameters
    ----------
    model : nn.Module
        The model to train (e.g. an MLP head).
    loss_fn : nn.Module
        Loss function. For simple regression use ``torch.nn.MSELoss()``.
        For the full GOAL loss system use ``CompositeLoss``.
    optimizer : torch.optim.Optimizer
        Pre-configured optimiser for ``model.parameters()``.
    scheduler : torch.optim.lr_scheduler.LRScheduler or None
        Optional learning rate scheduler. Stepped once per epoch.
    device : str or torch.device
        Device to train on. Auto-detected if ``'auto'``.
    step_fn : StepFn or None
        Custom step function. If ``None``, uses ``default_step``
        (expects ``(x, y)`` tuple batches). Use ``graph_step`` for
        ``AtomicGraph`` batches with ``CompositeLoss``.
    grad_clip : float or None
        Max gradient norm for clipping. ``None`` disables clipping.
    enable_progress : bool
        Show a ``tqdm`` progress bar per epoch. Defaults to ``True``.

    Example::

        trainer = MiniTrainer(
            model=head,
            loss_fn=nn.MSELoss(),
            optimizer=torch.optim.Adam(head.parameters(), lr=1e-3),
        )
        history = trainer.fit(train_loader, val_loader=val_loader, epochs=50)
        history.plot()
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: typing.Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: typing.Union[str, torch.device] = "auto",
        step_fn: typing.Optional[StepFn] = None,
        grad_clip: typing.Optional[float] = None,
        enable_progress: bool = True,
    ) -> None:
        self.model: nn.Module = model
        self.loss_fn: nn.Module = loss_fn
        self.optimizer: torch.optim.Optimizer = optimizer
        self.scheduler: typing.Optional[torch.optim.lr_scheduler.LRScheduler] = scheduler
        self.device: torch.device = self._resolve_device(device)
        self.step_fn: StepFn = step_fn or default_step
        self.grad_clip: typing.Optional[float] = grad_clip
        self.enable_progress: bool = enable_progress

        # Move model and loss to device
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        # State
        self._best_val_loss: float = float("inf")
        self._best_state: typing.Optional[typing.Dict[str, typing.Any]] = None

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: typing.Union[str, torch.device]) -> torch.device:
        """Resolve ``'auto'`` to the best available device.

        Args:
            device: ``'auto'``, ``'cpu'``, ``'cuda'``, ``'mps'``, etc.

        Returns:
            Resolved ``torch.device``.
        """
        if isinstance(device, torch.device):
            return device
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

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
        """Wrap an iterable with tqdm if available and enabled.

        Falls back to the bare iterable if tqdm is not installed.

        Args:
            iterable: The iterable to wrap.
            total: Total number of items.
            desc: Description for the progress bar.
            enabled: Whether to show the progress bar.

        Returns:
            Wrapped iterable (with or without progress bar).
        """
        if not enabled:
            return iterable
        try:
            from tqdm.auto import tqdm
            return tqdm(iterable, total=total, desc=desc, leave=False)
        except ImportError:
            return iterable

    # ------------------------------------------------------------------
    # Training / validation loops
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: typing.Any,
    ) -> float:
        """Run one training epoch.

        Args:
            loader: Training data loader.

        Returns:
            Mean training loss over all batches.
        """
        self.model.train()
        total_loss: float = 0.0
        num_batches: int = 0

        progress: typing.Any = self._get_progress_bar(
            loader, total=len(loader), desc="train", enabled=self.enable_progress,
        )

        for batch in progress:
            self.optimizer.zero_grad()
            result: typing.Dict[str, torch.Tensor] = self.step_fn(
                batch, self.model, self.loss_fn, self.device,
            )
            # Support both 'loss' and 'total' keys (CompositeLoss uses 'total')
            loss: torch.Tensor = result.get("loss", result.get("total"))
            loss.backward()

            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _val_epoch(
        self,
        loader: typing.Any,
    ) -> float:
        """Run one validation epoch.

        Args:
            loader: Validation data loader.

        Returns:
            Mean validation loss over all batches.
        """
        self.model.eval()
        total_loss: float = 0.0
        num_batches: int = 0

        progress: typing.Any = self._get_progress_bar(
            loader, total=len(loader), desc="val", enabled=self.enable_progress,
        )

        for batch in progress:
            result: typing.Dict[str, torch.Tensor] = self.step_fn(
                batch, self.model, self.loss_fn, self.device,
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
        train_loader: typing.Any,
        val_loader: typing.Optional[typing.Any] = None,
        epochs: int = 10,
        early_stopping_patience: typing.Optional[int] = None,
        checkpoint_best: bool = True,
        verbose: bool = True,
    ) -> TrainingHistory:
        """Run the training loop for a specified number of epochs.

        Args:
            train_loader: PyTorch ``DataLoader`` for training data.
            val_loader: Optional ``DataLoader`` for validation. If provided,
                validation loss is tracked and used for early stopping /
                best-model checkpointing.
            epochs: Number of epochs to train.
            early_stopping_patience: Stop if validation loss does not
                improve for this many consecutive epochs. ``None`` disables
                early stopping.
            checkpoint_best: If ``True`` and ``val_loader`` is given,
                keep an in-memory copy of the best model state dict
                (by validation loss). Call ``load_best()`` afterwards.
            verbose: Print per-epoch summary to stdout.

        Returns:
            ``TrainingHistory`` containing loss curves and timing info.
        """
        history: TrainingHistory = TrainingHistory()
        patience_counter: int = 0

        for epoch in range(epochs):
            t0: float = time.perf_counter()

            # --- Train ---
            train_loss: float = self._train_epoch(train_loader)
            history.train_loss.append(train_loss)

            # --- Validate ---
            val_loss_str: str = "—"
            if val_loader is not None:
                val_loss: float = self._val_epoch(val_loader)
                history.val_loss.append(val_loss)
                val_loss_str = f"{val_loss:.6f}"

                # Best model checkpointing
                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    patience_counter = 0
                    if checkpoint_best:
                        self._best_state = copy.deepcopy(self.model.state_dict())
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
                    metric: float = val_loss if val_loader is not None else train_loss
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            # --- Timing ---
            elapsed: float = time.perf_counter() - t0
            history.epoch_times.append(elapsed)

            # --- Logging ---
            if verbose:
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
                and val_loader is not None
                and patience_counter >= early_stopping_patience
            ):
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(no improvement for {early_stopping_patience} epochs)"
                    )
                break

        return history

    def load_best(self) -> None:
        """Restore the model weights from the best validation checkpoint.

        Only available if ``checkpoint_best=True`` was passed to ``fit()``
        and a validation loader was provided.

        Raises:
            RuntimeError: If no best checkpoint was saved.
        """
        if self._best_state is None:
            raise RuntimeError(
                "No best checkpoint available. "
                "Run fit() with val_loader and checkpoint_best=True first."
            )
        self.model.load_state_dict(self._best_state)

    @torch.no_grad()
    def predict(
        self,
        loader: typing.Any,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Run inference on a data loader and collect predictions + targets.

        Expects ``(x, y)`` tuple batches (compatible with ``default_step``).

        Args:
            loader: Data loader yielding ``(x, y)`` tuples.

        Returns:
            Tuple of ``(predictions, targets)`` tensors concatenated
            across all batches.
        """
        self.model.eval()
        all_preds: typing.List[torch.Tensor] = []
        all_targets: typing.List[torch.Tensor] = []

        for batch in loader:
            x: torch.Tensor
            y: torch.Tensor
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            pred: torch.Tensor = self.model(x).squeeze(-1)
            all_preds.append(pred.cpu())
            all_targets.append(y)

        return torch.cat(all_preds), torch.cat(all_targets)
