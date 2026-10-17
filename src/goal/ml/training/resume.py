"""Resuming — and *extending* — a GOAL training run.

Two different situations share one mechanism:

**Resume** — a run was interrupted (crash, preemption, SLURM timeout)
before it reached ``trainer.max_epochs``.  Relaunching picks up
``last.ckpt`` and carries on to the original budget.  This is what the
``TRAINING_COMPLETE`` sentinel plus SLURM auto-requeue has always done.

**Extension** — a run *finished*: it reached its budget, wrote the
sentinel, and stopped.  Relaunching with a larger ``trainer.max_epochs``
should continue from the final weights rather than refuse to start.

The two are distinguished by the sentinel, but they need the same
machinery, and both hit the same trap: Lightning restores learning-rate
schedulers wholesale.  ``LRScheduler.state_dict()`` carries ``T_max``,
``base_lrs`` and ``last_epoch``, so
:meth:`~lightning.pytorch.trainer.connectors.checkpoint_connector.
_CheckpointConnector.restore_lr_schedulers` overwrites whatever
``configure_optimizers`` just built.  A raised ``max_epochs`` never
reaches the scheduler, and because ``CosineAnnealingLR`` is periodic
with period ``2 * T_max`` the LR then climbs *back up* toward its
initial value over the added epochs — silently undoing the run.

:class:`LRScheduleRetarget` closes that gap after the restore, under a
policy chosen by ``resume.lr_schedule``:

``stretch``
    Re-aim the existing schedule at the new budget.  The LR lands where
    a run configured with the new ``max_epochs`` from the start would be
    at this epoch, and keeps annealing down from there.
``restart``
    A fresh cosine cycle over the *added* epochs only, from the
    configured base LR (SGDR-style warm restart).
``keep``
    Leave the restored schedule alone — the pre-existing behaviour,
    kept for reproducing an older resume.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning import Callback
from omegaconf import DictConfig

log = logging.getLogger(__name__)

SENTINEL_NAME: str = "TRAINING_COMPLETE"
LAST_CKPT_NAME: str = "last.ckpt"

#: Accepted values for ``resume.lr_schedule``.
LR_POLICIES: frozenset[str] = frozenset({"stretch", "restart", "keep"})


# ----------------------------------------------------------------------
# Reading progress out of a checkpoint
# ----------------------------------------------------------------------


def read_epochs_completed(ckpt_path: str | Path) -> int:
    """Number of epochs finished in the run that wrote ``ckpt_path``.

    Lightning's fit loop tracks epochs with a four-stage counter
    (``ready``/``started``/``processed``/``completed``).  ``processed``
    is incremented in ``_FitLoop.on_advance_end``, which runs *after*
    validation — and :class:`~goal.ml.training.callbacks.
    checkpoint_manager.GOALCheckpointManager` writes ``last.ckpt`` from
    ``on_validation_epoch_end``.  So a checkpoint saved at the end of
    epoch *n* records ``processed = n`` while ``ready = n + 1``.

    Lightning itself compensates on restore (the
    ``RESTARTED_MID_EPOCH`` branch of ``_FitLoop.reset`` increments
    ``processed`` when the epoch's last batch was already seen), so the
    epoch is not re-run.  This mirrors that compensation to report the
    count Lightning will act on — the number that ``_FitLoop.done``
    compares against ``max_epochs``.

    Falls back to the checkpoint's top-level ``epoch`` key when the loop
    state is missing or shaped unexpectedly.

    Loaded with ``mmap=True``: only the loop state is wanted, and under
    DDP every rank runs this at startup.  Materialising a 142 M-parameter
    state dict on each of them — before Lightning loads it again for the
    actual restore — is a gigabyte per rank for an integer.
    """
    try:
        ckpt: dict[str, Any] = torch.load(
            str(ckpt_path), map_location="cpu", weights_only=False, mmap=True
        )
    except (RuntimeError, ValueError):
        # Legacy (non-zipfile) serialisation cannot be mmap'd.
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return epochs_completed_from_state(ckpt)


def epochs_completed_from_state(ckpt: dict[str, Any]) -> int:
    """``read_epochs_completed`` for an already-loaded checkpoint dict."""
    try:
        progress = ckpt["loops"]["fit_loop"]["epoch_progress"]["current"]
        processed = int(progress["processed"])
        ready = int(progress["ready"])
    except (KeyError, TypeError, ValueError):
        # Pre-2.0 checkpoint, or one written by something other than a
        # Lightning fit loop: ``epoch`` is the 0-based index of the
        # epoch in flight, so one more than that has been reached.
        return int(ckpt.get("epoch", 0)) + 1
    # ``ready > processed`` means the epoch counter had been bumped for an
    # epoch whose batches are done but whose ``on_advance_end`` had not run
    # when the checkpoint was taken — i.e. a save from validation end.
    return processed + 1 if ready > processed else processed


# ----------------------------------------------------------------------
# Resolving what to resume from
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResumeDecision:
    """What :func:`resolve_resume` concluded about this launch.

    Attributes
    ----------
    ckpt_path:
        Passed straight to ``trainer.fit(ckpt_path=...)``; ``None``
        starts fresh.
    source_dir:
        Directory the checkpoint came from.  Pool state
        (``checkpoint_state.json``) and the sentinel live here, which is
        not necessarily where the continued run writes its own
        checkpoints.
    epochs_completed:
        Epochs already finished; 0 for a fresh run.
    is_extension:
        The source run had finished (sentinel present) and is being
        continued past its original budget.
    stop_reason:
        Non-``None`` means: do not train, print this and exit.
    """

    ckpt_path: str | None = None
    source_dir: Path | None = None
    epochs_completed: int = 0
    is_extension: bool = False
    stop_reason: str | None = None

    @property
    def should_train(self) -> bool:
        return self.stop_reason is None

    @property
    def is_resume(self) -> bool:
        return self.ckpt_path is not None


def _resolve_requested(raw: str) -> tuple[Path, Path]:
    """Map a user-supplied ``resume.from`` to ``(ckpt_file, source_dir)``.

    Accepts a ``.ckpt`` file, a checkpoint directory, or a run directory
    holding a ``checkpoints/`` subdirectory — the three things a user
    plausibly copies out of a log tree.
    """
    requested = Path(raw).expanduser()

    if requested.is_file():
        return requested, requested.parent

    if not requested.is_dir():
        raise FileNotFoundError(f"resume.from does not exist: {requested}")

    for candidate in (requested / LAST_CKPT_NAME, requested / "checkpoints" / LAST_CKPT_NAME):
        if candidate.is_file():
            return candidate, candidate.parent

    raise FileNotFoundError(
        f"resume.from={requested} contains no {LAST_CKPT_NAME} "
        f"(looked in {requested} and {requested / 'checkpoints'}).  "
        f"Point resume.from at a checkpoint directory or a .ckpt file."
    )


def resolve_resume(cfg: DictConfig) -> ResumeDecision:
    """Decide whether this launch resumes, extends, or starts fresh.

    Resolution order for the checkpoint:

    1. ``resume.from`` — an explicit run/checkpoint directory or file.
    2. ``ckpt_path`` — the legacy top-level key, same meaning.
    3. ``training.checkpoint_dir/last.ckpt`` — this run's own directory,
       which is how an interrupted run picks itself back up.

    Refuses to start (``stop_reason``) in exactly two cases: the budget
    is already spent, or there is nothing to resume and the directory is
    flagged complete.  The first is what keeps a SLURM requeue from
    looping — a requeued job re-reads the same config, so its
    ``max_epochs`` is unchanged and still spent.
    """
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    resume_cfg: Any = cfg.get("resume") or {}
    requested: str | None = resume_cfg.get("from") or cfg.get("ckpt_path")

    if requested:
        ckpt_file, source_dir = _resolve_requested(str(requested))
    else:
        candidate = checkpoint_dir / LAST_CKPT_NAME
        if not candidate.is_file():
            # Nothing to resume.  A sentinel here means a finished run
            # whose checkpoints were moved or deleted — there is no
            # state to continue from, so honour it and stop.
            if (checkpoint_dir / SENTINEL_NAME).exists():
                return ResumeDecision(
                    stop_reason=(
                        f"Training already complete ({checkpoint_dir / SENTINEL_NAME} "
                        f"exists) and no {LAST_CKPT_NAME} remains to continue from."
                    )
                )
            return ResumeDecision()
        ckpt_file, source_dir = candidate, checkpoint_dir

    epochs_completed = read_epochs_completed(ckpt_file)
    finished = (source_dir / SENTINEL_NAME).exists()
    max_epochs = int(cfg.trainer.max_epochs)

    if epochs_completed >= max_epochs:
        return ResumeDecision(
            stop_reason=(
                f"{ckpt_file} has already completed {epochs_completed} epochs and "
                f"trainer.max_epochs={max_epochs} — nothing left to train.\n"
                f"To continue this run, raise the budget above {epochs_completed}, e.g.\n"
                f"    goal-train --config-name {cfg.get('task_name', '<config>')} "
                f"trainer.max_epochs={max_epochs * 2} "
                f"resume.from={source_dir}"
            )
        )

    return ResumeDecision(
        ckpt_path=str(ckpt_file),
        source_dir=source_dir,
        epochs_completed=epochs_completed,
        is_extension=finished,
    )


# ----------------------------------------------------------------------
# Re-aiming the restored LR schedule
# ----------------------------------------------------------------------


def _cosine_closed_form(scheduler: Any) -> list[float]:
    """LR at ``scheduler.last_epoch`` from the cosine's closed form.

    ``CosineAnnealingLR.get_lr`` is *recursive* — it scales the group's
    current LR by a ratio of cosines rather than evaluating the curve.
    That makes a mid-flight ``T_max`` change useless on its own: a run
    that annealed to ``eta_min`` stays pinned at ``eta_min``, because
    every subsequent step just scales that floor.  Re-seeding the groups
    with the closed-form value puts the recursion back on the intended
    curve.
    """
    eta_min: float = scheduler.eta_min
    t_max: int = scheduler.T_max
    epoch: int = scheduler.last_epoch
    return [
        eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * epoch / t_max)) / 2
        for base_lr in scheduler.base_lrs
    ]


def _apply_lrs(scheduler: Any, lrs: list[float]) -> None:
    for group, lr in zip(scheduler.optimizer.param_groups, lrs):
        group["lr"] = lr
    scheduler._last_lr = list(lrs)


def _retarget_cosine(
    scheduler: Any,
    epoch: int,
    new_t_max: int,
    policy: str,
    base_lr: float | None,
) -> str:
    """Point one ``CosineAnnealingLR`` at a new horizon.  Returns a summary."""
    old_t_max: int = scheduler.T_max

    if policy == "restart":
        remaining = max(1, new_t_max - epoch)
        if base_lr is not None:
            scheduler.base_lrs = [float(base_lr)] * len(scheduler.base_lrs)
        scheduler.T_max = remaining
        scheduler.last_epoch = 0
        scheduler._step_count = 1
        _apply_lrs(scheduler, list(scheduler.base_lrs))
        return (
            f"restart: fresh cosine over {remaining} epochs "
            f"from lr={scheduler.base_lrs[0]:.3e} (was T_max={old_t_max})"
        )

    # "stretch"
    scheduler.T_max = max(1, new_t_max)
    scheduler.last_epoch = epoch
    scheduler._step_count = epoch + 1
    lrs = _cosine_closed_form(scheduler)
    _apply_lrs(scheduler, lrs)
    return (
        f"stretch: T_max {old_t_max} -> {scheduler.T_max}, "
        f"lr re-seeded to {lrs[0]:.3e} at epoch {epoch}"
    )


def retarget_scheduler(
    scheduler: Any,
    epoch: int,
    max_epochs: int,
    policy: str,
    base_lr: float | None = None,
) -> str | None:
    """Re-aim a restored scheduler at ``max_epochs``.

    Returns a human-readable summary of what changed, or ``None`` when
    the scheduler needs no adjustment (already correct, or not
    epoch-budget dependent — ``ReduceLROnPlateau`` tracks a metric, so
    extending the budget costs it nothing).
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR

    if isinstance(scheduler, CosineAnnealingLR):
        if policy == "stretch" and scheduler.T_max == max_epochs:
            return None
        return _retarget_cosine(scheduler, epoch, max_epochs, policy, base_lr)

    if isinstance(scheduler, SequentialLR):
        # ``cosine_warmup``: LinearLR warmup, then a cosine, switching at
        # the milestone.  Only the cosine leg has a horizon to re-aim;
        # the warmup is defined in absolute epochs and is either already
        # consumed or unaffected.
        nested = list(scheduler._schedulers)
        milestone = int(scheduler._milestones[0]) if scheduler._milestones else 0
        cosine = nested[-1]
        if not isinstance(cosine, CosineAnnealingLR):
            log.warning(
                "[resume/lr] SequentialLR's final leg is %s, not CosineAnnealingLR — "
                "leaving the schedule as restored.",
                type(cosine).__name__,
            )
            return None

        new_t_max = max(1, max_epochs - milestone)
        if epoch < milestone:
            # Still warming up; the cosine has not started, so only its
            # horizon needs correcting.
            if cosine.T_max == new_t_max:
                return None
            old = cosine.T_max
            cosine.T_max = new_t_max
            return f"stretch (warmup in progress): cosine T_max {old} -> {new_t_max}"

        if policy == "stretch" and cosine.T_max == new_t_max:
            return None
        summary = _retarget_cosine(cosine, epoch - milestone, new_t_max, policy, base_lr)
        scheduler.last_epoch = epoch
        scheduler._last_lr = list(cosine._last_lr)
        return summary

    return None


class LRScheduleRetarget(Callback):
    """Re-aim restored LR schedules at the current ``trainer.max_epochs``.

    Runs from ``on_train_start``, which Lightning calls *after*
    ``restore_training_state`` has loaded the scheduler state dicts —
    the only point at which the restored values can be corrected.

    Parameters
    ----------
    policy:
        ``stretch``, ``restart`` or ``keep`` (see the module docstring).
    base_lr:
        Configured optimiser LR, used as the peak of a ``restart``
        cycle.  ``stretch`` keeps the checkpoint's ``base_lrs``.
    epochs_completed:
        Epochs the checkpoint finished, from
        :func:`read_epochs_completed`.  Passed in rather than read from
        ``trainer.current_epoch``, which at ``on_train_start`` still
        lags by one: ``_FitLoop.on_run_start`` sets ``completed`` from
        ``processed``, and ``processed`` is only bumped for the restored
        epoch once the loop starts advancing.  Seeding the cosine one
        epoch early would shift the whole remaining curve.
    """

    def __init__(
        self,
        policy: str = "stretch",
        base_lr: float | None = None,
        epochs_completed: int | None = None,
    ) -> None:
        super().__init__()
        if policy not in LR_POLICIES:
            raise ValueError(
                f"resume.lr_schedule must be one of {sorted(LR_POLICIES)}, got '{policy}'"
            )
        self.policy: str = policy
        self.base_lr: float | None = base_lr
        self.epochs_completed: int | None = epochs_completed
        self.summaries: list[str] = []

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Re-aim every restored LR scheduler at ``trainer.max_epochs``.

        Does nothing (beyond a log line) for ``policy="keep"``, and nothing
        when ``trainer.max_epochs`` is not a positive int.  Otherwise each
        scheduler in ``trainer.lr_scheduler_configs`` is passed to
        :func:`retarget_scheduler` at epoch ``epochs_completed`` (falling
        back to ``trainer.current_epoch``) and modified in place; every
        non-``None`` summary is appended to ``self.summaries``, logged and
        printed.
        """
        if self.policy == "keep":
            log.info("[resume/lr] policy='keep' — restored schedule left untouched.")
            return

        max_epochs = trainer.max_epochs
        if not isinstance(max_epochs, int) or max_epochs <= 0:
            return

        epoch = (
            int(self.epochs_completed)
            if self.epochs_completed is not None
            else int(trainer.current_epoch)
        )
        for config in trainer.lr_scheduler_configs:
            summary = retarget_scheduler(
                config.scheduler,
                epoch=epoch,
                max_epochs=max_epochs,
                policy=self.policy,
                base_lr=self.base_lr,
            )
            if summary is not None:
                self.summaries.append(summary)
                log.info("[resume/lr] %s", summary)
                print(f"[resume/lr] {summary}")
