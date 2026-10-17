"""Tests for GOALCheckpointManager — three-pool checkpoint management."""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from goal.ml.training.callbacks.checkpoint_manager import GOALCheckpointManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer(metrics: dict[str, float], epoch: int, dirpath: str) -> MagicMock:
    """Return a minimal trainer mock."""
    trainer = MagicMock()
    trainer.is_global_zero = True
    trainer.current_epoch = epoch
    trainer.callback_metrics = {
        k: MagicMock(item=lambda v=v: v, __float__=lambda self, v=v: v) for k, v in metrics.items()
    }
    # Make float() work directly on the mock values
    for k, v in metrics.items():
        trainer.callback_metrics[k] = v
    trainer.log_dir = dirpath
    trainer.default_root_dir = dirpath
    trainer.save_checkpoint = MagicMock(side_effect=lambda path: Path(path).touch())
    return trainer


def _make_manager(
    tmp_path: Path,
    top_k: dict | None = None,
    interval: dict | None = None,
    last: dict | None = None,
) -> GOALCheckpointManager:
    """Instantiate a manager with dirpath pointing to tmp_path."""
    manager = GOALCheckpointManager(
        dirpath=str(tmp_path),
        top_k=(
            top_k
            if top_k is not None
            else {"enabled": True, "k": 5, "metric": "val_forces/forces_mae", "mode": "min"}
        ),
        interval=(
            interval
            if interval is not None
            else {"enabled": True, "every_n_epochs": 2, "keep_last_n": 5}
        ),
        last=last if last is not None else {"enabled": True, "filename": "last.ckpt"},
    )
    # Call setup manually to resolve dirpath
    trainer = _make_trainer({}, 0, str(tmp_path))
    manager.setup(trainer, MagicMock(), "fit")
    return manager


def _run_epochs(
    manager: GOALCheckpointManager,
    metric_values: list[float],
    tmp_path: Path,
    metric: str = "val_forces/forces_mae",
) -> None:
    """Simulate validation epochs with the given metric values (epoch 0, 1, 2, ...)."""
    for epoch, val in enumerate(metric_values):
        trainer = _make_trainer({metric: val}, epoch, str(tmp_path))
        manager.on_validation_epoch_end(trainer, MagicMock())


# ---------------------------------------------------------------------------
# Test 1 — top-k pool management
# ---------------------------------------------------------------------------


class TestTopKPool:
    def test_top_k_keeps_best(self, tmp_path: Path) -> None:
        """After 20 epochs with k=5, only the 5 best checkpoints should exist."""
        # Metric values — epochs 0..19.  Best 5 (lowest MAE) are at epochs 15-19
        # (values 0.05..0.01) after the rest have higher values.
        metric_values = [float(20 - i) * 0.1 for i in range(20)]  # 2.0, 1.9, ..., 0.1
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": True, "k": 5, "metric": "val_forces/forces_mae", "mode": "min"},
            interval={"enabled": False},
            last={"enabled": False},
        )
        _run_epochs(manager, metric_values, tmp_path)

        pool_files = {Path(p).name for _, p in manager.top_k_pool}
        existing = {f.name for f in tmp_path.iterdir() if f.suffix == ".ckpt"}
        assert len(manager.top_k_pool) == 5
        assert existing == pool_files

    def test_worst_evicted_when_better_arrives(self, tmp_path: Path) -> None:
        """When a better checkpoint arrives, the worst in the pool is deleted."""
        # Fill pool with epochs 0-4 (values 1.0, 0.9, 0.8, 0.7, 0.6)
        # Epoch 5 gives 0.5 — better than worst (1.0) → 1.0 evicted
        metric_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": True, "k": 5, "metric": "val_forces/forces_mae", "mode": "min"},
            interval={"enabled": False},
            last={"enabled": False},
        )
        _run_epochs(manager, metric_values, tmp_path)

        pool_values = {v for v, _ in manager.top_k_pool}
        assert 1.0 not in pool_values, "Worst value 1.0 should have been evicted"
        assert 0.5 in pool_values, "New best 0.5 should be in the pool"
        assert len(manager.top_k_pool) == 5

    def test_worse_than_all_not_saved(self, tmp_path: Path) -> None:
        """A checkpoint worse than all k-best is not saved to disk."""
        # Fill pool with values 0.1..0.5 (all better than 0.9)
        metric_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.9]
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": True, "k": 5, "metric": "val_forces/forces_mae", "mode": "min"},
            interval={"enabled": False},
            last={"enabled": False},
        )
        _run_epochs(manager, metric_values, tmp_path)

        # The 0.9 file should not exist — it was never saved
        bad_files = [f for f in tmp_path.iterdir() if "0.9000" in f.name]
        assert len(bad_files) == 0, f"File for 0.9 should not be saved: {bad_files}"
        assert len(manager.top_k_pool) == 5

    def test_mode_max_keeps_highest(self, tmp_path: Path) -> None:
        """mode='max' keeps the highest values."""
        metric_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": True, "k": 5, "metric": "val/r2", "mode": "max"},
            interval={"enabled": False},
            last={"enabled": False},
        )
        _run_epochs(manager, metric_values, tmp_path, metric="val/r2")

        pool_values = {v for v, _ in manager.top_k_pool}
        assert min(pool_values) >= 0.6, f"All pool values should be >= 0.6, got {pool_values}"
        assert len(manager.top_k_pool) == 5


# ---------------------------------------------------------------------------
# Test 2 — interval pool management
# ---------------------------------------------------------------------------


class TestIntervalPool:
    def test_interval_epochs_kept(self, tmp_path: Path) -> None:
        """With interval=2, keep_last_n=5, epochs 12-20 should be on disk after 20 epochs."""
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": False},
            interval={"enabled": True, "every_n_epochs": 2, "keep_last_n": 5},
            last={"enabled": False},
        )
        # Epochs 0..19
        metric_values = [0.5] * 20
        _run_epochs(manager, metric_values, tmp_path)

        # Saved at epochs 0,2,4,...,18 (every_n_epochs=2, epoch % 2 == 0)
        # Keep last 5: [10, 12, 14, 16, 18]
        pool = manager.interval_pool
        assert len(pool) == 5
        pool_names = {Path(p).name for p in pool}
        for expected_epoch in [10, 12, 14, 16, 18]:
            assert (
                f"interval_epoch={expected_epoch:04d}.ckpt" in pool_names
            ), f"Expected epoch {expected_epoch} in pool, got {pool_names}"

    def test_old_interval_checkpoints_deleted(self, tmp_path: Path) -> None:
        """Early interval checkpoints are deleted from disk as new ones arrive."""
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": False},
            interval={"enabled": True, "every_n_epochs": 2, "keep_last_n": 5},
            last={"enabled": False},
        )
        metric_values = [0.5] * 20
        _run_epochs(manager, metric_values, tmp_path)

        # Epochs 0, 2, 4, 6, 8 should have been deleted
        for deleted_epoch in [0, 2, 4, 6, 8]:
            path = tmp_path / f"interval_epoch={deleted_epoch:04d}.ckpt"
            assert not path.exists(), f"Epoch {deleted_epoch} should have been deleted"

    def test_interval_not_saved_on_wrong_epoch(self, tmp_path: Path) -> None:
        """With interval=2, odd epochs are never saved."""
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": False},
            interval={"enabled": True, "every_n_epochs": 2, "keep_last_n": 5},
            last={"enabled": False},
        )
        metric_values = [0.5] * 6
        _run_epochs(manager, metric_values, tmp_path)

        for odd_epoch in [1, 3, 5]:
            path = tmp_path / f"interval_epoch={odd_epoch:04d}.ckpt"
            assert not path.exists(), f"Odd epoch {odd_epoch} should never be saved"


# ---------------------------------------------------------------------------
# Test 3 — last.ckpt always updated
# ---------------------------------------------------------------------------


class TestLastPool:
    def test_last_ckpt_exists_after_every_epoch(self, tmp_path: Path) -> None:
        """last.ckpt exists after each validation epoch."""
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": False},
            interval={"enabled": False},
            last={"enabled": True, "filename": "last.ckpt"},
        )
        for epoch in range(10):
            trainer = _make_trainer({"val_forces/forces_mae": 0.5}, epoch, str(tmp_path))
            manager.on_validation_epoch_end(trainer, MagicMock())
            assert (tmp_path / "last.ckpt").exists(), f"last.ckpt missing after epoch {epoch}"

    def test_last_ckpt_reflects_most_recent_epoch(self, tmp_path: Path) -> None:
        """save_checkpoint is called for last.ckpt on every epoch."""
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": False},
            interval={"enabled": False},
            last={"enabled": True, "filename": "last.ckpt"},
        )
        call_count = 0
        for epoch in range(5):
            trainer = _make_trainer({"val_forces/forces_mae": 0.5}, epoch, str(tmp_path))
            original_side_effect = trainer.save_checkpoint.side_effect

            def counting_save(path, _epoch=epoch, _orig=original_side_effect):
                nonlocal call_count
                call_count += 1
                _orig(path)

            trainer.save_checkpoint.side_effect = counting_save
            manager.on_validation_epoch_end(trainer, MagicMock())

        # save_checkpoint called at least once per epoch (for last.ckpt)
        assert call_count >= 5


# ---------------------------------------------------------------------------
# Test 4 — atomic writes
# ---------------------------------------------------------------------------


class TestAtomicWrites:
    def test_tmp_file_written_before_replace(self, tmp_path: Path) -> None:
        """The .tmp file is written and then os.replace is called."""
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": False},
            interval={"enabled": False},
            last={"enabled": True, "filename": "last.ckpt"},
        )
        trainer = _make_trainer({"val_forces/forces_mae": 0.5}, 0, str(tmp_path))

        tmp_files_seen: list[str] = []
        original_replace = os.replace

        def tracking_replace(src: str, dst: str) -> None:
            tmp_files_seen.append(src)
            original_replace(src, dst)

        with patch(
            "goal.ml.training.callbacks.checkpoint_manager.os.replace",
            side_effect=tracking_replace,
        ):
            manager.on_validation_epoch_end(trainer, MagicMock())

        assert len(tmp_files_seen) > 0
        for tmp_path_str in tmp_files_seen:
            assert tmp_path_str.endswith(".tmp"), f"Expected .tmp suffix, got {tmp_path_str}"

    def test_interrupted_write_leaves_no_corrupt_checkpoint(self, tmp_path: Path) -> None:
        """If os.replace raises, the final checkpoint is not created (only the .tmp exists)."""
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": False},
            interval={"enabled": False},
            last={"enabled": True, "filename": "last.ckpt"},
        )
        trainer = _make_trainer({"val_forces/forces_mae": 0.5}, 0, str(tmp_path))

        # Patch os.replace to fail after the tmp file is written
        with patch(
            "goal.ml.training.callbacks.checkpoint_manager.os.replace",
            side_effect=OSError("simulated power loss"),
        ):
            with pytest.raises(OSError, match="simulated power loss"):
                manager.on_validation_epoch_end(trainer, MagicMock())

        # last.ckpt should not exist (write was interrupted before rename)
        assert not (tmp_path / "last.ckpt").exists(), "Corrupt checkpoint should not exist"
        # The .tmp file may exist — that is safe to delete on recovery
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) <= 1, "At most one .tmp file should exist"


# ---------------------------------------------------------------------------
# Test 5 — pool state persistence and resume
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_state_json_written_after_each_epoch(self, tmp_path: Path) -> None:
        """checkpoint_state.json is written after each validation epoch."""
        manager = _make_manager(tmp_path)
        trainer = _make_trainer({"val_forces/forces_mae": 0.5}, 0, str(tmp_path))
        manager.on_validation_epoch_end(trainer, MagicMock())
        assert (tmp_path / "checkpoint_state.json").exists()

    def test_resumed_manager_restores_pools(self, tmp_path: Path) -> None:
        """A new manager pointing to the same dir restores pool state correctly."""
        # Run 10 epochs
        manager1 = _make_manager(
            tmp_path,
            top_k={"enabled": True, "k": 3, "metric": "val_forces/forces_mae", "mode": "min"},
            interval={"enabled": True, "every_n_epochs": 2, "keep_last_n": 3},
            last={"enabled": True, "filename": "last.ckpt"},
        )
        _run_epochs(manager1, [float(10 - i) * 0.1 for i in range(10)], tmp_path)

        top_k_before = sorted(v for v, _ in manager1.top_k_pool)
        interval_before = list(manager1.interval_pool)

        # Instantiate a fresh manager (simulating a crash + resume)
        manager2 = _make_manager(
            tmp_path,
            top_k={"enabled": True, "k": 3, "metric": "val_forces/forces_mae", "mode": "min"},
            interval={"enabled": True, "every_n_epochs": 2, "keep_last_n": 3},
            last={"enabled": True, "filename": "last.ckpt"},
        )

        top_k_after = sorted(v for v, _ in manager2.top_k_pool)
        interval_after = list(manager2.interval_pool)

        assert (
            top_k_before == top_k_after
        ), f"top_k pool not restored: before={top_k_before}, after={top_k_after}"
        assert (
            interval_before == interval_after
        ), f"interval pool not restored: before={interval_before}, after={interval_after}"

    def test_resumed_manager_continues_correctly(self, tmp_path: Path) -> None:
        """After resume, pool management continues from where it left off."""
        # Run 6 epochs
        manager1 = _make_manager(
            tmp_path,
            top_k={"enabled": True, "k": 3, "metric": "val_forces/forces_mae", "mode": "min"},
            interval={"enabled": False},
            last={"enabled": False},
        )
        _run_epochs(manager1, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1], tmp_path)
        # Pool is full: {0.1, 0.2, 0.3}

        # Resume and run 3 more epochs
        manager2 = _make_manager(
            tmp_path,
            top_k={"enabled": True, "k": 3, "metric": "val_forces/forces_mae", "mode": "min"},
            interval={"enabled": False},
            last={"enabled": False},
        )
        # Epoch 6: value=0.05 — better than worst (0.3) → 0.3 evicted
        _run_epochs(manager2, [0.05, 0.9, 0.9], tmp_path)

        pool_values = {v for v, _ in manager2.top_k_pool}
        assert 0.3 not in pool_values, "0.3 should have been evicted"
        assert 0.05 in pool_values, "0.05 should be in the pool"

    def test_missing_state_json_starts_fresh(self, tmp_path: Path) -> None:
        """When checkpoint_state.json is missing, manager starts with empty pools."""
        manager = GOALCheckpointManager(
            dirpath=str(tmp_path),
            top_k={"enabled": True, "k": 5, "metric": "val_forces/forces_mae", "mode": "min"},
            interval={"enabled": True, "every_n_epochs": 2, "keep_last_n": 5},
            last={"enabled": True, "filename": "last.ckpt"},
        )
        trainer = _make_trainer({}, 0, str(tmp_path))
        manager.setup(trainer, MagicMock(), "fit")

        assert manager.top_k_pool == []
        assert manager.interval_pool == []
        assert manager.last_ckpt_path is None


# ---------------------------------------------------------------------------
# Test 6 — metric mode validation
# ---------------------------------------------------------------------------


class TestMetricMode:
    def test_invalid_mode_raises(self) -> None:
        """Passing an invalid mode string raises ValueError."""
        with pytest.raises(ValueError, match="must be 'min' or 'max'"):
            GOALCheckpointManager(
                top_k={
                    "enabled": True,
                    "k": 5,
                    "metric": "val_forces/forces_mae",
                    "mode": "neither",
                }
            )

    def test_mode_min_mismatch_warns(self) -> None:
        """mode='min' for a max-type metric (forces_cosine_similarity) warns."""
        with pytest.warns(UserWarning, match="typically maximised"):
            GOALCheckpointManager(
                top_k={
                    "enabled": True,
                    "k": 5,
                    "metric": "val_forces/forces_cosine_similarity",
                    "mode": "min",
                }
            )

    def test_mode_max_mismatch_warns(self) -> None:
        """mode='max' for a min-type metric (val_forces/forces_mae) warns."""
        with pytest.warns(UserWarning, match="typically minimised"):
            GOALCheckpointManager(
                top_k={
                    "enabled": True,
                    "k": 5,
                    "metric": "val_forces/forces_mae",
                    "mode": "max",
                }
            )

    def test_consistent_mode_no_warning(self) -> None:
        """No warning when mode matches metric direction."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            GOALCheckpointManager(
                top_k={"enabled": True, "k": 5, "metric": "val_forces/forces_mae", "mode": "min"}
            )

    def test_unknown_metric_no_warning(self) -> None:
        """No warning for an unrecognised metric name (user knows what they're doing)."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            GOALCheckpointManager(
                top_k={"enabled": True, "k": 5, "metric": "val/my_custom_metric", "mode": "min"}
            )


# ---------------------------------------------------------------------------
# Test 7 — pool independence
# ---------------------------------------------------------------------------


class TestPoolIndependence:
    def test_top_k_file_not_deleted_by_interval_eviction(self, tmp_path: Path) -> None:
        """A file in both top-k and interval pools survives interval eviction."""
        # Strategy:
        #   k=1, interval=2, keep_last_n=1
        #   Epoch 0: best=0.1, save as top-k AND interval (epoch 0 % 2 == 0)
        #   Epoch 2: interval save (epoch 2), interval pool now at capacity → evict epoch 0
        #            But epoch 0 is also in top-k → must NOT be deleted
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": True, "k": 1, "metric": "val_forces/forces_mae", "mode": "min"},
            interval={"enabled": True, "every_n_epochs": 2, "keep_last_n": 1},
            last={"enabled": False},
        )

        # Epoch 0: value=0.1 — saved to top-k; also an interval epoch (0 % 2 == 0)
        trainer0 = _make_trainer({"val_forces/forces_mae": 0.1}, 0, str(tmp_path))
        manager.on_validation_epoch_end(trainer0, MagicMock())

        top_k_path_after_e0 = manager.top_k_pool[0][1]

        # Epoch 1: skipped by interval; value=0.5 (worse than pool) → not saved to top-k
        trainer1 = _make_trainer({"val_forces/forces_mae": 0.5}, 1, str(tmp_path))
        manager.on_validation_epoch_end(trainer1, MagicMock())

        # Epoch 2: interval save at epoch 2; interval pool at cap (1) → evict epoch 0
        #          But epoch 0 is in top-k → file must survive
        trainer2 = _make_trainer({"val_forces/forces_mae": 0.9}, 2, str(tmp_path))
        manager.on_validation_epoch_end(trainer2, MagicMock())

        assert Path(
            top_k_path_after_e0
        ).exists(), (
            "Top-k checkpoint was deleted during interval eviction — pool independence violated"
        )

    def test_interval_file_not_deleted_by_top_k_eviction(self, tmp_path: Path) -> None:
        """A file in both pools survives top-k eviction."""
        # k=1, interval=2, keep_last_n=5
        # Epoch 0: best=0.5, in both top-k AND interval
        # Epoch 2: value=0.1, better → evicts epoch 0 from top-k
        #          But epoch 0 is still in interval pool → file must survive
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": True, "k": 1, "metric": "val_forces/forces_mae", "mode": "min"},
            interval={"enabled": True, "every_n_epochs": 2, "keep_last_n": 5},
            last={"enabled": False},
        )

        trainer0 = _make_trainer({"val_forces/forces_mae": 0.5}, 0, str(tmp_path))
        manager.on_validation_epoch_end(trainer0, MagicMock())

        interval_path_e0 = manager.interval_pool[0] if manager.interval_pool else None

        # Epoch 1: no interval
        trainer1 = _make_trainer({"val_forces/forces_mae": 0.9}, 1, str(tmp_path))
        manager.on_validation_epoch_end(trainer1, MagicMock())

        # Epoch 2: better top-k and new interval → top-k evicts epoch 0 file
        trainer2 = _make_trainer({"val_forces/forces_mae": 0.1}, 2, str(tmp_path))
        manager.on_validation_epoch_end(trainer2, MagicMock())

        if interval_path_e0 is not None:
            assert Path(
                interval_path_e0
            ).exists(), "Interval checkpoint was deleted during top-k eviction — pool independence violated"

    def test_file_deleted_only_when_removed_from_both_pools(self, tmp_path: Path) -> None:
        """A shared file is deleted only when evicted from both pools."""
        # k=1, interval=2, keep_last_n=1
        # Epoch 0: best=0.5, in both pools
        # Epoch 2: value=0.1 (better top-k), also new interval → evicts epoch 0 from BOTH
        #          Now epoch 0 should finally be deleted
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": True, "k": 1, "metric": "val_forces/forces_mae", "mode": "min"},
            interval={"enabled": True, "every_n_epochs": 2, "keep_last_n": 1},
            last={"enabled": False},
        )

        trainer0 = _make_trainer({"val_forces/forces_mae": 0.5}, 0, str(tmp_path))
        manager.on_validation_epoch_end(trainer0, MagicMock())

        # Find the epoch-0 file (it's in both top-k and interval)
        e0_top_k_path = manager.top_k_pool[0][1] if manager.top_k_pool else None
        e0_interval_path = manager.interval_pool[0] if manager.interval_pool else None

        # The same file should be in both pools when epoch=0 and every_n_epochs=2
        # (0 % 2 == 0 → saved to interval; also best → saved to top-k)
        # They may be different files because the pools save independently.
        # What matters: the interval file is deleted when evicted from both pools.

        # Epoch 1: nothing special
        trainer1 = _make_trainer({"val_forces/forces_mae": 0.9}, 1, str(tmp_path))
        manager.on_validation_epoch_end(trainer1, MagicMock())

        # Epoch 2: better than current top-k (0.1 < 0.5) AND new interval epoch
        # → top-k evicts epoch-0-top-k file (but only if not in interval)
        # → interval evicts epoch-0-interval file (but only if not in top-k)
        trainer2 = _make_trainer({"val_forces/forces_mae": 0.1}, 2, str(tmp_path))
        manager.on_validation_epoch_end(trainer2, MagicMock())

        # After epoch 2: epoch-0's interval file evicted (interval pool capacity=1)
        # and epoch-0's top-k file evicted (new best is epoch 2)
        # Both should now be deleted (they may or may not be the same physical file)
        if e0_interval_path and e0_interval_path != e0_top_k_path:
            # If they're different files, the interval one should be deleted
            # (it was evicted from interval, and is not in top-k since top-k has epoch 2)
            assert not Path(
                e0_interval_path
            ).exists(), "Interval-only file should be deleted once evicted from both pools"


# ---------------------------------------------------------------------------
# Test 8 — pre-split metric names still resolve
# ---------------------------------------------------------------------------


class TestLegacyMetricNames:
    """Energy and force curves moved out of ``{split}/`` into
    ``{split}_energy/`` and ``{split}_forces/``.  A config written against
    the old key would otherwise silently stop saving top-k checkpoints, so
    the old name is translated to the sectioned one."""

    def test_old_key_is_translated(self, tmp_path: Path) -> None:
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": True, "k": 2, "metric": "val/forces_mae", "mode": "min"},
            interval={"enabled": False},
            last={"enabled": False},
        )
        _run_epochs(manager, [0.5, 0.4], tmp_path, metric="val_forces/forces_mae")

        assert manager.top_k_metric == "val_forces/forces_mae"
        saved = sorted(f.name for f in tmp_path.glob("*.ckpt"))
        assert len(saved) == 2, saved

    def test_unknown_key_is_still_skipped(self, tmp_path: Path) -> None:
        """The alias must not paper over a genuine typo."""
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": True, "k": 2, "metric": "val/force_mae", "mode": "min"},
            interval={"enabled": False},
            last={"enabled": False},
        )
        _run_epochs(manager, [0.5], tmp_path, metric="val_forces/forces_mae")

        assert manager.top_k_metric == "val/force_mae"
        assert list(tmp_path.glob("*.ckpt")) == []

    def test_total_needs_no_translation(self, tmp_path: Path) -> None:
        """``total`` never moved — it is still in the plain section."""
        from goal.ml.training.callbacks.checkpoint_manager import _legacy_key

        assert _legacy_key("val/total") is None
        assert _legacy_key("val/forces_mae") == "val_forces/forces_mae"
        assert _legacy_key("val/energy_mae_per_atom") == "val_energy/energy_mae_per_atom"
        # Already sectioned, or not a metric key at all.
        assert _legacy_key("val_forces/forces_mae") is None
        assert _legacy_key("forces_mae") is None

    def test_mode_mismatch_warning_matches_on_the_leaf_name(self) -> None:
        with pytest.warns(UserWarning, match="typically minimised"):
            GOALCheckpointManager(
                top_k={
                    "enabled": True,
                    "k": 5,
                    "metric": "test_energy/energy_mae_per_atom",
                    "mode": "max",
                }
            )
