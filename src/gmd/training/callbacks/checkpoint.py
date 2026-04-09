"""Smart checkpointing callback with SLURM-aware completion sentinel."""

from __future__ import annotations

import typing
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint


class GMDCheckpoint(ModelCheckpoint):
    """Extended ``ModelCheckpoint`` that writes a completion sentinel.

    When training finishes normally, a ``TRAINING_COMPLETE`` file is
    written to the checkpoint directory. SLURM requeue scripts can
    check for this file to avoid restarting completed jobs.

    Parameters
    ----------
    sentinel_name : str
        Name of the completion marker file.
    **kwargs
        All arguments forwarded to ``ModelCheckpoint``.
    """

    def __init__(self, sentinel_name: str = "TRAINING_COMPLETE", **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.sentinel_name: str = sentinel_name

    def on_train_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Write completion sentinel after successful training."""
        super().on_train_end(trainer, pl_module)
        if trainer.is_global_zero and self.dirpath:
            sentinel: Path = Path(self.dirpath) / self.sentinel_name
            sentinel.touch()
