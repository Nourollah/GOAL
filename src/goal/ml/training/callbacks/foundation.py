"""Callback: re-estimate foundation-model atomic energies from the training set.

Foundation models carry E0s fit to *their* training distribution (e.g.
MPTrj).  Fine-tuning on a different dataset without re-baselining hands the
interaction network an O(eV) offset to absorb — the most common cause of
fine-tuning failure.  This callback fixes that generically, without any
change to ``GOALModule``: at ``on_fit_start`` it asks the backbone to
re-estimate E0s from ``trainer.datamodule.data_train`` when the backbone
advertises ``reestimate_e0s = True`` and implements
``reestimate_atomic_energies``.

Enable it via the Hydra ``callbacks:`` block of a fine-tuning config::

    callbacks:
      foundation_e0:
        _target_: goal.ml.training.callbacks.foundation.FoundationE0Callback
"""

from __future__ import annotations

import logging

import lightning as L
from lightning import Callback

log = logging.getLogger(__name__)


class FoundationE0Callback(Callback):
    """Re-baseline a foundation backbone's atomic energies before training."""

    def __init__(self) -> None:
        super().__init__()
        self._done: bool = False

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Re-estimate the backbone's atomic energies from the training set, once.

        Calls ``backbone.reestimate_atomic_energies(trainer.datamodule.data_train)``
        when ``pl_module.backbone`` exists and sets ``reestimate_e0s = True``.
        Logs a warning and skips if that method is missing or no training
        dataset is available.  After a successful re-estimation later calls
        do nothing.
        """
        if self._done:
            return
        backbone = getattr(pl_module, "backbone", None)
        if backbone is None:
            return
        if not getattr(backbone, "reestimate_e0s", False):
            return
        reestimate = getattr(backbone, "reestimate_atomic_energies", None)
        if not callable(reestimate):
            log.warning(
                "[finetune] backbone has reestimate_e0s=True but no "
                "reestimate_atomic_energies() method — skipping E0 re-estimation."
            )
            return

        datamodule = getattr(trainer, "datamodule", None)
        train_ds = getattr(datamodule, "data_train", None) if datamodule is not None else None
        if train_ds is None:
            log.warning(
                "[finetune] no training dataset available at on_fit_start — "
                "skipping E0 re-estimation."
            )
            return

        reestimate(train_ds)
        self._done = True
