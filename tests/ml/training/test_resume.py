"""Tests for resuming — and extending — a training run.

The end-to-end cases drive a real ``lightning.Trainer`` with a toy
module, because the behaviour under test lives in the interaction
between Lightning's checkpoint restore and GOAL's callbacks, not in any
one function.  They are CPU-only and finish in a couple of seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from goal.ml.training.callbacks.checkpoint_manager import GOALCheckpointManager
from goal.ml.training.resume import (
    LAST_CKPT_NAME,
    SENTINEL_NAME,
    LRScheduleRetarget,
    epochs_completed_from_state,
    read_epochs_completed,
    resolve_resume,
    retarget_scheduler,
)

# ---------------------------------------------------------------------------
# Toy training setup
# ---------------------------------------------------------------------------


class _ToyModule(L.LightningModule):
    """Four-parameter regressor that records the LR it trains each epoch."""

    def __init__(self, max_epochs: int, scheduler: str = "cosine", lr: float = 0.01) -> None:
        super().__init__()
        self.net = torch.nn.Linear(4, 1)
        self.max_epochs = max_epochs
        self.scheduler = scheduler
        self.lr = lr
        self.epoch_lrs: list[tuple[int, float]] = []

    def training_step(self, batch: Any, _: int) -> torch.Tensor:
        return ((self.net(batch[0]) - batch[1]) ** 2).mean()

    def on_train_epoch_start(self) -> None:
        self.epoch_lrs.append((self.current_epoch, self.optimizers().param_groups[0]["lr"]))

    def validation_step(self, batch: Any, _: int) -> None:
        # Monotonically improving, so the top-k pool keeps churning.
        self.log("val_forces/forces_mae", torch.tensor(1.0 / (self.current_epoch + 1)))

    def configure_optimizers(self) -> dict[str, Any]:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.scheduler == "cosine":
            sched: Any = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.max_epochs, eta_min=1e-7
            )
        else:
            warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=2)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.max_epochs - 2, eta_min=1e-7
            )
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup, cosine], milestones=[2]
            )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def _loaders() -> tuple[DataLoader, DataLoader]:
    dataset = TensorDataset(torch.randn(16, 4), torch.randn(16, 1))
    return DataLoader(dataset, batch_size=4), DataLoader(dataset, batch_size=4)


def _fit(
    dirpath: Path,
    max_epochs: int,
    ckpt_path: str | None = None,
    scheduler: str = "cosine",
    lr_policy: str = "stretch",
    pools_from: Path | None = None,
) -> tuple[_ToyModule, L.Trainer, GOALCheckpointManager]:
    """Run one fit, optionally resuming, with the real checkpoint manager."""
    torch.manual_seed(0)
    module = _ToyModule(max_epochs, scheduler=scheduler)
    manager = GOALCheckpointManager(
        dirpath=str(dirpath),
        top_k={"enabled": True, "k": 2, "metric": "val_forces/forces_mae", "mode": "min"},
        interval={"enabled": False},
        save_archive=False,
    )
    callbacks: list[L.Callback] = [manager]
    if ckpt_path is not None:
        GOALCheckpointManager.restore_pools_from_dir(
            manager, str(pools_from or Path(ckpt_path).parent)
        )
        callbacks.append(
            LRScheduleRetarget(
                policy=lr_policy,
                base_lr=module.lr,
                epochs_completed=read_epochs_completed(ckpt_path),
            )
        )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
    )
    train_loader, val_loader = _loaders()
    trainer.fit(module, train_loader, val_loader, ckpt_path=ckpt_path)
    return module, trainer, manager


@pytest.fixture
def finished_run(tmp_path: Path) -> Path:
    """A run that completed 3 of 3 epochs and wrote its sentinel."""
    dirpath = tmp_path / "run_a"
    _fit(dirpath, max_epochs=3)
    (dirpath / SENTINEL_NAME).touch()
    return dirpath


# ---------------------------------------------------------------------------
# Reading progress out of a checkpoint
# ---------------------------------------------------------------------------


def test_epochs_completed_counts_the_epoch_saved_from_validation_end() -> None:
    """``last.ckpt`` is written before ``processed`` is incremented.

    GOALCheckpointManager saves from ``on_validation_epoch_end``, which
    runs inside the training epoch — so the finishing epoch shows up as
    ``ready`` but not yet ``processed``.
    """
    state = {
        "epoch": 2,
        "loops": {
            "fit_loop": {"epoch_progress": {"current": {"ready": 3, "started": 3, "processed": 2}}}
        },
    }
    assert epochs_completed_from_state(state) == 3


def test_epochs_completed_trusts_processed_when_the_epoch_fully_closed() -> None:
    state = {
        "epoch": 3,
        "loops": {
            "fit_loop": {"epoch_progress": {"current": {"ready": 3, "started": 3, "processed": 3}}}
        },
    }
    assert epochs_completed_from_state(state) == 3


def test_epochs_completed_falls_back_to_the_epoch_key() -> None:
    assert epochs_completed_from_state({"epoch": 7}) == 8


def test_epochs_completed_reads_a_real_checkpoint(finished_run: Path) -> None:
    assert read_epochs_completed(finished_run / LAST_CKPT_NAME) == 3


# ---------------------------------------------------------------------------
# resolve_resume
# ---------------------------------------------------------------------------


def _cfg(checkpoint_dir: Path, max_epochs: int = 10, **resume: Any) -> Any:
    return OmegaConf.create(
        {
            "task_name": "toy",
            "trainer": {"max_epochs": max_epochs},
            "training": {"checkpoint_dir": str(checkpoint_dir)},
            "resume": {"from": None, "lr_schedule": "stretch", **resume},
        }
    )


def test_fresh_run_when_nothing_to_resume(tmp_path: Path) -> None:
    decision = resolve_resume(_cfg(tmp_path / "empty"))
    assert decision.should_train
    assert not decision.is_resume
    assert decision.epochs_completed == 0


def test_auto_resume_picks_up_this_runs_own_checkpoint(tmp_path: Path) -> None:
    dirpath = tmp_path / "run"
    _fit(dirpath, max_epochs=2)
    decision = resolve_resume(_cfg(dirpath, max_epochs=5))
    assert decision.is_resume
    assert decision.source_dir == dirpath
    assert decision.epochs_completed == 2
    assert not decision.is_extension  # no sentinel — this run was interrupted


def test_finished_run_is_extended_when_the_budget_is_raised(finished_run: Path) -> None:
    decision = resolve_resume(_cfg(finished_run, max_epochs=6))
    assert decision.should_train
    assert decision.is_extension
    assert decision.epochs_completed == 3


def test_finished_run_refuses_to_start_when_the_budget_is_unchanged(finished_run: Path) -> None:
    """What stops SLURM requeueing a completed job: same config, same budget."""
    decision = resolve_resume(_cfg(finished_run, max_epochs=3))
    assert not decision.should_train
    assert "already completed 3 epochs" in decision.stop_reason


def test_refuses_when_the_budget_is_lowered_below_what_is_done(finished_run: Path) -> None:
    decision = resolve_resume(_cfg(finished_run, max_epochs=2))
    assert not decision.should_train


def test_resume_from_accepts_a_checkpoint_dir_a_run_dir_and_a_file(
    tmp_path: Path, finished_run: Path
) -> None:
    fresh = tmp_path / "new_run"
    for requested in (
        finished_run,  # checkpoint directory
        finished_run / LAST_CKPT_NAME,  # explicit file
    ):
        decision = resolve_resume(_cfg(fresh, max_epochs=6, **{"from": str(requested)}))
        assert decision.is_resume
        assert decision.source_dir == finished_run
        assert decision.epochs_completed == 3

    # A run directory holding checkpoints/ resolves one level down.
    run_dir = tmp_path / "logs" / "run_b"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / LAST_CKPT_NAME).write_bytes(
        (finished_run / LAST_CKPT_NAME).read_bytes()
    )
    decision = resolve_resume(_cfg(fresh, max_epochs=6, **{"from": str(run_dir)}))
    assert decision.source_dir == run_dir / "checkpoints"


def test_resume_from_a_missing_path_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_resume(_cfg(tmp_path, **{"from": str(tmp_path / "nope")}))


def test_resume_from_a_dir_without_a_checkpoint_is_an_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="contains no last.ckpt"):
        resolve_resume(_cfg(tmp_path, **{"from": str(empty)}))


def test_sentinel_without_a_checkpoint_still_stops(tmp_path: Path) -> None:
    dirpath = tmp_path / "gone"
    dirpath.mkdir()
    (dirpath / SENTINEL_NAME).touch()
    decision = resolve_resume(_cfg(dirpath))
    assert not decision.should_train


def test_legacy_ckpt_path_key_is_honoured(tmp_path: Path, finished_run: Path) -> None:
    cfg = OmegaConf.create(
        {
            "trainer": {"max_epochs": 6},
            "training": {"checkpoint_dir": str(tmp_path / "new")},
            "ckpt_path": str(finished_run / LAST_CKPT_NAME),
        }
    )
    decision = resolve_resume(cfg)
    assert decision.is_resume
    assert decision.epochs_completed == 3


# ---------------------------------------------------------------------------
# Extending a finished run, end to end
# ---------------------------------------------------------------------------


def test_extension_continues_the_epoch_and_step_counters(finished_run: Path) -> None:
    module, trainer, _ = _fit(
        finished_run, max_epochs=6, ckpt_path=str(finished_run / LAST_CKPT_NAME)
    )
    assert [epoch for epoch, _ in module.epoch_lrs] == [3, 4, 5]
    assert trainer.global_step == 24  # 6 epochs x 4 batches, no epoch re-run


@pytest.mark.parametrize("scheduler", ["cosine", "cosine_warmup"])
def test_stretch_keeps_the_learning_rate_annealing_down(
    finished_run: Path, tmp_path: Path, scheduler: str
) -> None:
    """The bug this feature exists to prevent.

    A cosine is periodic with period ``2 * T_max``, so replaying a
    restored schedule past its horizon walks back *up* the curve.
    """
    dirpath = tmp_path / f"run_{scheduler}"
    _fit(dirpath, max_epochs=3, scheduler=scheduler)
    module, _, _ = _fit(
        dirpath,
        max_epochs=6,
        ckpt_path=str(dirpath / LAST_CKPT_NAME),
        scheduler=scheduler,
        lr_policy="stretch",
    )
    lrs = [lr for _, lr in module.epoch_lrs]
    assert lrs == sorted(lrs, reverse=True), f"LR should keep descending, got {lrs}"


def test_stretch_matches_a_run_configured_with_the_new_budget_from_the_start(
    tmp_path: Path,
) -> None:
    """Extension is not merely monotonic — it lands on the intended curve."""
    reference, _, _ = _fit(tmp_path / "reference", max_epochs=6)
    short = tmp_path / "short"
    _fit(short, max_epochs=3)
    extended, _, _ = _fit(short, max_epochs=6, ckpt_path=str(short / LAST_CKPT_NAME))

    expected = dict(reference.epoch_lrs)
    for epoch, lr in extended.epoch_lrs:
        assert lr == pytest.approx(expected[epoch], rel=1e-6)


def test_keep_reproduces_the_unpatched_behaviour(tmp_path: Path) -> None:
    """``keep`` is documented as the old behaviour — so it must still climb."""
    dirpath = tmp_path / "run"
    _fit(dirpath, max_epochs=3)
    module, _, _ = _fit(
        dirpath, max_epochs=6, ckpt_path=str(dirpath / LAST_CKPT_NAME), lr_policy="keep"
    )
    lrs = [lr for _, lr in module.epoch_lrs]
    assert lrs[-1] > lrs[0], "without retargeting the cosine walks back up its curve"


def test_restart_opens_a_fresh_cycle_at_the_configured_lr(tmp_path: Path) -> None:
    dirpath = tmp_path / "run"
    _fit(dirpath, max_epochs=3)
    module, _, _ = _fit(
        dirpath, max_epochs=6, ckpt_path=str(dirpath / LAST_CKPT_NAME), lr_policy="restart"
    )
    lrs = [lr for _, lr in module.epoch_lrs]
    assert lrs[0] == pytest.approx(0.01)
    assert lrs == sorted(lrs, reverse=True)


def test_an_unchanged_budget_leaves_the_schedule_untouched() -> None:
    """A plain crash-resume must behave exactly as it did before."""
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-7)
    assert retarget_scheduler(sched, epoch=4, max_epochs=10, policy="stretch") is None


def test_plateau_schedulers_need_no_retargeting() -> None:
    """ReduceLROnPlateau tracks a metric, so the budget is irrelevant to it."""
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
    assert retarget_scheduler(sched, epoch=4, max_epochs=99, policy="stretch") is None


def test_unknown_lr_policy_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="resume.lr_schedule"):
        LRScheduleRetarget(policy="anneal-harder")


# ---------------------------------------------------------------------------
# Continuing into a different directory
# ---------------------------------------------------------------------------


def test_extension_into_a_new_dir_never_deletes_the_source_runs_checkpoints(
    finished_run: Path, tmp_path: Path
) -> None:
    """Pools are inherited by absolute path; eviction must stay local.

    ``checkpoint_manager.dirpath`` follows Hydra's timestamped output
    directory, so a continued run almost always writes somewhere else
    than the run it continues.
    """
    before = {p.name for p in finished_run.glob("*.ckpt")}
    assert before, "fixture should have written checkpoints"

    new_dir = tmp_path / "run_b"
    _, _, manager = _fit(
        new_dir,
        max_epochs=8,
        ckpt_path=str(finished_run / LAST_CKPT_NAME),
        pools_from=finished_run,
    )

    after = {p.name for p in finished_run.glob("*.ckpt")}
    assert after == before, "the source run's checkpoints must survive"
    assert manager.dirpath == str(new_dir)
    assert manager.source_dirpath == str(finished_run)
    assert list(new_dir.glob("*.ckpt")), "the continued run writes its own checkpoints"


def test_extension_is_recorded_in_metadata(finished_run: Path, tmp_path: Path) -> None:
    import json

    new_dir = tmp_path / "run_b"
    manager = GOALCheckpointManager(dirpath=str(new_dir), save_archive=False)
    manager.record_extension(from_epoch=3, to_epoch=8, source_ckpt="last.ckpt")
    assert manager.extension is not None

    new_dir.mkdir(parents=True)
    (new_dir / "metadata.json").write_text(json.dumps({"model_class": "Toy"}))
    manager._resolved_dirpath = str(new_dir)
    module = _ToyModule(8)
    module.config = OmegaConf.create({"trainer": {"max_epochs": 8}})
    manager._save_extension_record(new_dir, module)

    metadata = json.loads((new_dir / "metadata.json").read_text())
    assert metadata["model_class"] == "Toy", "existing provenance is preserved"
    assert metadata["extensions"] == [
        {
            "from_epoch": 3,
            "to_epoch": 8,
            "source_checkpoint": "last.ckpt",
            "timestamp": manager.extension["timestamp"],
        }
    ]
    assert list(new_dir.glob("config_extended_*.yaml"))


# ---------------------------------------------------------------------------
# EMA persistence
# ---------------------------------------------------------------------------


def test_ema_shadow_survives_a_checkpoint_round_trip() -> None:
    """Without the hooks the shadow is absent from ``state_dict`` entirely."""
    from goal.ml.training.ema import EMAWrapper
    from goal.ml.training.module import GOALModule

    module = _ToyModule(1)
    module.ema = EMAWrapper(module.parameters(), decay=0.5)
    with torch.no_grad():
        for param in module.parameters():
            param.add_(1.0)
    module.ema.update()
    shadow = [tensor.clone() for tensor in module.ema.state_dict()["shadow"]]

    checkpoint: dict[str, Any] = {}
    GOALModule.on_save_checkpoint(module, checkpoint)
    assert "goal_ema" in checkpoint

    restored = _ToyModule(1)
    restored.ema = EMAWrapper(restored.parameters(), decay=0.5)
    GOALModule.on_load_checkpoint(restored, checkpoint)
    for saved, loaded in zip(shadow, restored.ema.state_dict()["shadow"]):
        assert torch.allclose(saved, loaded)


def test_ema_load_tolerates_a_checkpoint_without_ema_state() -> None:
    from goal.ml.training.ema import EMAWrapper
    from goal.ml.training.module import GOALModule

    module = _ToyModule(1)
    module.ema = EMAWrapper(module.parameters(), decay=0.5)
    GOALModule.on_load_checkpoint(module, {"state_dict": {}})  # no goal_ema key


def test_ema_load_discards_a_mismatched_shadow() -> None:
    from goal.ml.training.ema import EMAWrapper
    from goal.ml.training.module import GOALModule

    module = _ToyModule(1)
    module.ema = EMAWrapper(module.parameters(), decay=0.5)
    with pytest.warns(UserWarning, match="EMA state"):
        GOALModule.on_load_checkpoint(module, {"goal_ema": {"shadow": [torch.zeros(1)]}})


def test_ema_hooks_are_inert_when_ema_is_disabled() -> None:
    from goal.ml.training.module import GOALModule

    module = _ToyModule(1)  # no .ema attribute
    checkpoint: dict[str, Any] = {}
    GOALModule.on_save_checkpoint(module, checkpoint)
    assert "goal_ema" not in checkpoint
    GOALModule.on_load_checkpoint(module, {"goal_ema": {"shadow": []}})
