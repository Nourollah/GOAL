"""Three-pool checkpoint manager for GOAL training.

Pools
-----
1. top_k  — keep the K best checkpoints ranked by a validation metric.
2. interval — keep the last N checkpoints saved every M epochs.
3. last  — always overwrite last.ckpt (for crash recovery / resume).

All checkpoint writes are atomic (write to .tmp then os.replace).
Pool membership is persisted to checkpoint_state.json so resumed
training can continue managing deletions correctly.

Self-contained checkpoints
--------------------------
At ``on_train_start`` the manager freezes everything a checkpoint needs
to be loadable forever, regardless of later source changes:

* ``{dirpath}/frozen_source/`` — every ``goal`` .py file the model
  imports (walked automatically, see :mod:`goal.ml.training.archive`)
* ``{dirpath}/config.yaml``    — the fully resolved Hydra config
* ``{dirpath}/metadata.json``  — version, git commit, elements, scale, …

Every ``.ckpt`` write also gets a small ``.json`` sidecar recording the
epoch, metrics, and pool it came from.  At ``on_train_end`` (and
optionally at every interval checkpoint) the best checkpoint is packed
into a single shippable ``.simurgh`` archive.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning as L
from lightning import Callback

from goal.ml.training.archive import (
    ARCHIVE_SUFFIX,
    FROZEN_SOURCE_DIRNAME,
    freeze_model_source,
    gather_model_metadata,
    pack_simurgh_archive,
)
from goal.ml.training.metrics import metric_section

log = logging.getLogger(__name__)

# Metric *leaf* names (the part after the last "/") whose natural
# direction is "lower is better".  Matching on the leaf keeps this
# working across logger sections, so ``val/total``,
# ``val_forces/forces_mae`` and ``test_energy/energy_mae_per_atom`` are
# all recognised without listing every section separately.
_KNOWN_MIN_METRICS: frozenset[str] = frozenset(
    {
        "total",
        "energy",
        "forces",
        "energy_mae",
        "energy_mae_per_atom",
        "energy_rmse_per_atom",
        "forces_mae",
        "forces_rmse",
        "forces_magnitude_mae",
        "energy_mse",
        "forces_mse",
        "newton_violation",
        "loss",
    }
)

# Metric leaf names whose natural direction is "higher is better"
_KNOWN_MAX_METRICS: frozenset[str] = frozenset(
    {
        "forces_cosine_similarity",
        "accuracy",
        "r2",
    }
)


def _legacy_key(metric: str) -> str | None:
    """Sectioned equivalent of a pre-split metric key, or ``None``.

    Energy and force curves used to share the ``{split}/`` section with
    the loss total; they now live in ``{split}_energy/`` and
    ``{split}_forces/``.  A config written against the old names would
    otherwise silently stop saving top-k checkpoints, so the old key is
    translated once and a warning points at the config:
    ``"val/forces_mae"`` -> ``"val_forces/forces_mae"``.
    """
    split, sep, name = metric.partition("/")
    # Only the three plain splits can be legacy keys — anything else
    # (``val_forces/…``) is already sectioned and must be left alone.
    if split not in ("train", "val", "test") or not sep or "/" in name:
        return None
    section: str | None = metric_section(name)
    return f"{split}_{section}/{name}" if section else None


def _is_better(new: float, old: float, mode: str) -> bool:
    if mode == "min":
        return new < old
    return new > old


def _worst_index(pool: list[tuple[float, str]], mode: str) -> int:
    """Return the index of the worst entry in the top-k pool."""
    if mode == "min":
        # Worst = highest value
        return max(range(len(pool)), key=lambda i: pool[i][0])
    # Worst = lowest value
    return min(range(len(pool)), key=lambda i: pool[i][0])


class GOALCheckpointManager(Callback):
    """Three-pool checkpoint manager.

    Parameters
    ----------
    dirpath:
        Directory where checkpoints are saved.  Defaults to the
        Lightning trainer's log directory if ``None``.
    top_k:
        Config dict for pool 1.  Keys: ``enabled``, ``k``, ``metric``, ``mode``.
    interval:
        Config dict for pool 2.  Keys: ``enabled``, ``every_n_epochs``, ``keep_last_n``.
    last:
        Config dict for pool 3.  Keys: ``enabled``, ``filename``.
    save_archive:
        Pack the best top-k checkpoint into ``{dirpath}/best_{metric}.simurgh``
        at the end of training.
    archive_interval:
        Also pack a ``.simurgh`` archive for every interval checkpoint
        (evicted together with its checkpoint).
    """

    def __init__(
        self,
        dirpath: str | None = None,
        top_k: dict[str, Any] | None = None,
        interval: dict[str, Any] | None = None,
        last: dict[str, Any] | None = None,
        save_archive: bool = True,
        archive_interval: bool = False,
    ) -> None:
        super().__init__()

        self._dirpath: str | None = dirpath
        self.save_archive: bool = bool(save_archive)
        self.archive_interval: bool = bool(archive_interval)

        # --- Pool 1 config ---
        _tk = top_k or {}
        self.top_k_enabled: bool = bool(_tk.get("enabled", True))
        self.top_k_k: int = int(_tk.get("k", 5))
        self.top_k_metric: str = str(_tk.get("metric", "val_forces/forces_mae"))
        self.top_k_mode: str = str(_tk.get("mode", "min"))
        if self.top_k_mode not in ("min", "max"):
            raise ValueError(f"top_k.mode must be 'min' or 'max', got '{self.top_k_mode}'")

        # Warn if mode looks inconsistent with the metric name
        if self.top_k_enabled:
            self._warn_mode_mismatch()

        # --- Pool 2 config ---
        _iv = interval or {}
        self.interval_enabled: bool = bool(_iv.get("enabled", True))
        self.interval_every_n: int = int(_iv.get("every_n_epochs", 2))
        self.interval_keep_n: int = int(_iv.get("keep_last_n", 5))

        # --- Pool 3 config ---
        _la = last or {}
        self.last_enabled: bool = bool(_la.get("enabled", True))
        self.last_filename: str = str(_la.get("filename", "last.ckpt"))

        # --- Runtime state ---
        # Pool 1: list of (metric_value, filepath)
        self._top_k_pool: list[tuple[float, str]] = []
        # Pool 2: deque of filepaths (oldest first)
        self._interval_pool: deque[str] = deque(maxlen=None)
        # Pool 3: path to last.ckpt
        self._last_ckpt_path: str | None = None

        # resolved dirpath (set in setup)
        self._resolved_dirpath: str | None = None
        # Directory a resumed run imported its pools from, when that is
        # not where this run writes.  Files there belong to the earlier
        # run and are never deleted — see ``_owns``.
        self._source_dirpath: str | None = None
        # Extension provenance, recorded by train.py when a finished run
        # is continued past its original budget.
        self._extension: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Ownership
    # ------------------------------------------------------------------

    def _owns(self, path: str) -> bool:
        """Whether ``path`` is a checkpoint this run is allowed to delete.

        A resumed run inherits pool entries from the run it continues
        (:meth:`restore_pools_from_dir`).  When it writes to a *different*
        directory — the normal case, since ``checkpoint_manager.dirpath``
        follows Hydra's timestamped output dir — those inherited entries
        still name files under the earlier run.  Evicting one from the
        pool must not delete the earlier run's checkpoint.
        """
        if self._resolved_dirpath is None:
            return False
        try:
            return Path(path).resolve().parent == Path(self._resolved_dirpath).resolve()
        except OSError:  # unresolvable path — refuse rather than guess
            return False

    def _evict(self, path: str, pool: str) -> None:
        """Drop ``path`` from a pool, deleting it only if this run owns it."""
        if self._owns(path):
            _delete_checkpoint(path)
            log.info("[ckpt/%s] Evicted %s", pool, Path(path).name)
        else:
            log.info(
                "[ckpt/%s] Evicted from pool but kept on disk (owned by the resumed run): %s",
                pool,
                path,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Resolve dirpath and attempt to restore pool state from disk."""
        if stage != "fit":
            return

        if self._dirpath is not None:
            self._resolved_dirpath = self._dirpath
        else:
            # Fall back to Lightning's default log directory
            log_dir = trainer.log_dir or trainer.default_root_dir
            self._resolved_dirpath = str(Path(log_dir) / "checkpoints")

        os.makedirs(self._resolved_dirpath, exist_ok=True)

        # Try to restore pool state from a previous run
        state_path = Path(self._resolved_dirpath) / "checkpoint_state.json"
        if state_path.exists():
            self._load_state(state_path)

    # ------------------------------------------------------------------
    # Source freezing — fires once, before the first training step
    # ------------------------------------------------------------------

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Freeze model source, resolved config, and metadata into dirpath.

        Runs before any weights are saved, so every checkpoint written to
        this directory is loadable from the frozen snapshot regardless of
        later changes to the codebase.  On resume the existing snapshot is
        kept — it matches the code the run *started* with.
        """
        if not trainer.is_global_zero or self._resolved_dirpath is None:
            return

        dirpath = Path(self._resolved_dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        frozen_dir = dirpath / FROZEN_SOURCE_DIRNAME

        if frozen_dir.exists():
            log.info(
                "[ckpt/freeze] %s already exists — keeping the snapshot from "
                "the original training start (resume detected).",
                frozen_dir,
            )
        else:
            try:
                frozen_files = freeze_model_source(pl_module, frozen_dir)
                self._save_resolved_config(dirpath, pl_module)
                self._save_metadata(dirpath, pl_module)
            except Exception:  # noqa: BLE001
                log.exception(
                    "[ckpt/freeze] FAILED to freeze model source into %s. "
                    "Training continues, but checkpoints in this directory will "
                    "NOT be self-contained.",
                    dirpath,
                )
                return

            log.info(
                "Model source frozen to %s/ (%d files)\n"
                "Config saved to %s\n"
                "All checkpoints in this directory are self-contained.",
                frozen_dir,
                len(frozen_files),
                dirpath / "config.yaml",
            )

        if self._extension is not None:
            self._save_extension_record(dirpath, pl_module)

    # ------------------------------------------------------------------
    # Extension provenance
    # ------------------------------------------------------------------

    def record_extension(self, from_epoch: int, to_epoch: int, source_ckpt: str) -> None:
        """Note that this run continues a finished one past its budget.

        Called by ``train.py`` before ``trainer.fit``; written to disk at
        ``on_train_start`` once the checkpoint directory exists.
        """
        self._extension = {
            "from_epoch": int(from_epoch),
            "to_epoch": int(to_epoch),
            "source_checkpoint": str(source_ckpt),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def _save_extension_record(self, dirpath: Path, pl_module: L.LightningModule) -> None:
        """Append the extension to ``metadata.json`` and snapshot its config.

        ``config.yaml`` is left untouched: every sidecar in the directory
        points at it as the config the *weights* were built from, and the
        architecture has not changed.  The extended budget goes into a
        dated sibling so the two are distinguishable after the fact.
        """
        assert self._extension is not None
        metadata_path = dirpath / "metadata.json"
        try:
            metadata: dict[str, Any] = (
                json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            )
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(
                "[ckpt/extend] Could not read %s (%s) — starting a new record.",
                metadata_path,
                exc,
            )
            metadata = {}

        extensions: list[Any] = list(metadata.get("extensions") or [])
        extensions.append(self._extension)
        metadata["extensions"] = extensions

        try:
            tmp = str(metadata_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(metadata, f, indent=2)
            os.replace(tmp, str(metadata_path))
        except OSError as exc:
            log.warning("[ckpt/extend] Could not write %s: %s", metadata_path, exc)
            return

        stamp = self._extension["timestamp"].replace(":", "-")
        self._save_resolved_config(dirpath, pl_module, filename=f"config_extended_{stamp}.yaml")
        log.info(
            "[ckpt/extend] Continuing from epoch %d to %d (source: %s)",
            self._extension["from_epoch"],
            self._extension["to_epoch"],
            self._extension["source_checkpoint"],
        )

    def _save_resolved_config(
        self,
        dirpath: Path,
        pl_module: L.LightningModule,
        filename: str = "config.yaml",
    ) -> None:
        """Write the fully resolved Hydra config to ``{dirpath}/{filename}``."""
        cfg = getattr(pl_module, "config", None)
        if cfg is None:
            log.warning("[ckpt/freeze] pl_module has no .config — config.yaml not written.")
            return
        from omegaconf import OmegaConf

        try:
            text = OmegaConf.to_yaml(cfg, resolve=True)
        except Exception:  # noqa: BLE001 — unresolvable interpolations
            log.warning(
                "[ckpt/freeze] Config has unresolvable interpolations — "
                "saving config.yaml unresolved."
            )
            text = OmegaConf.to_yaml(cfg, resolve=False)
        tmp = dirpath / f"{filename}.tmp"
        tmp.write_text(text)
        os.replace(tmp, dirpath / filename)

    def _save_metadata(self, dirpath: Path, pl_module: L.LightningModule) -> None:
        """Write provenance metadata to ``{dirpath}/metadata.json``."""
        model = getattr(pl_module, "backbone", None) or pl_module
        metadata = gather_model_metadata(model)
        tmp = dirpath / "metadata.json.tmp"
        with open(tmp, "w") as f:
            json.dump(metadata, f, indent=2)
        os.replace(tmp, dirpath / "metadata.json")

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Pack the best top-k checkpoint into a ``.simurgh`` archive."""
        if not trainer.is_global_zero or self._resolved_dirpath is None:
            return
        if not self.save_archive:
            return
        if not self._top_k_pool:
            log.info("[archive] No top-k checkpoints — skipping end-of-training archive.")
            return

        pick = max if self.top_k_mode == "max" else min
        _, best_path = pick(self._top_k_pool, key=lambda t: t[0])
        metric_slug = self.top_k_metric.replace("/", "_")
        output = Path(self._resolved_dirpath) / f"best_{metric_slug}{ARCHIVE_SUFFIX}"
        try:
            pack_simurgh_archive(best_path, output)
            log.info("[archive] Best checkpoint packed to %s", output)
        except Exception:  # noqa: BLE001
            log.exception("[archive] Failed to pack end-of-training archive %s", output)

    # ------------------------------------------------------------------
    # Main hook
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Update the three checkpoint pools after each validation epoch.

        Runs on global rank zero only, and only once :meth:`setup` has
        resolved the checkpoint directory.  Using ``trainer.callback_metrics``,
        each enabled pool saves (and evicts) checkpoints: top-k by
        ``top_k_metric`` (pool 1), every ``interval_every_n`` epochs
        (pool 2), and the rolling last checkpoint (pool 3).  The pool state
        is then written to ``checkpoint_state.json`` in that directory.
        """
        if not trainer.is_global_zero:
            return
        if self._resolved_dirpath is None:
            return

        epoch: int = trainer.current_epoch
        metrics: dict[str, Any] = trainer.callback_metrics

        # Pool 1 — top-k by metric
        if self.top_k_enabled:
            self._update_top_k(trainer, epoch, metrics)

        # Pool 2 — interval
        if self.interval_enabled and (epoch % self.interval_every_n == 0):
            self._update_interval(trainer, epoch, metrics)

        # Pool 3 — always save last
        if self.last_enabled:
            self._update_last(trainer, epoch, metrics)

        # Persist pool state after every epoch
        self._save_state()

    # ------------------------------------------------------------------
    # Pool 1
    # ------------------------------------------------------------------

    def _update_top_k(
        self,
        trainer: L.Trainer,
        epoch: int,
        metrics: dict[str, Any],
    ) -> None:
        if self.top_k_metric not in metrics:
            alias: str | None = _legacy_key(self.top_k_metric)
            if alias is not None and alias in metrics:
                log.warning(
                    "GOALCheckpointManager: top_k metric '%s' is the pre-split name; "
                    "using '%s' instead.  Update the config to the new key.",
                    self.top_k_metric,
                    alias,
                )
                self.top_k_metric = alias
            else:
                log.warning(
                    "GOALCheckpointManager: top_k metric '%s' not found in callback_metrics "
                    "(available: %s). Skipping top-k save.",
                    self.top_k_metric,
                    list(metrics.keys()),
                )
                return

        value: float = float(metrics[self.top_k_metric])
        metric_slug = self.top_k_metric.replace("/", "_")
        filename = f"best_{metric_slug}={value:.4f}_epoch={epoch:04d}.ckpt"
        filepath = str(Path(self._resolved_dirpath) / filename)

        if len(self._top_k_pool) < self.top_k_k:
            # Pool not full yet — always save
            self._save_atomic(trainer, filepath)
            self._write_sidecar(filepath, epoch, metrics, pool="top_k")
            self._top_k_pool.append((value, filepath))
            log.info(
                "[ckpt/top_k] Saved %s (pool size %d/%d)",
                filename,
                len(self._top_k_pool),
                self.top_k_k,
            )
        else:
            worst_idx = _worst_index(self._top_k_pool, self.top_k_mode)
            worst_value, worst_path = self._top_k_pool[worst_idx]

            if _is_better(value, worst_value, self.top_k_mode):
                # New checkpoint is better than the worst — save and evict
                self._save_atomic(trainer, filepath)
                self._write_sidecar(filepath, epoch, metrics, pool="top_k")
                # Only delete if the file is not also in the interval pool
                if worst_path not in self._interval_pool:
                    self._evict(worst_path, pool="top_k")
                else:
                    log.info(
                        "[ckpt/top_k] Evicted from top-k but kept (in interval pool): %s",
                        Path(worst_path).name,
                    )
                self._top_k_pool[worst_idx] = (value, filepath)
                log.info("[ckpt/top_k] Saved %s (replaced worst %.4f)", filename, worst_value)
            else:
                log.debug(
                    "[ckpt/top_k] Skipped epoch %d (value %.4f not better than worst %.4f)",
                    epoch,
                    value,
                    worst_value,
                )

    # ------------------------------------------------------------------
    # Pool 2
    # ------------------------------------------------------------------

    def _update_interval(self, trainer: L.Trainer, epoch: int, metrics: dict[str, Any]) -> None:
        filename = f"interval_epoch={epoch:04d}.ckpt"
        filepath = str(Path(self._resolved_dirpath) / filename)

        # If pool is at capacity, evict the oldest
        if len(self._interval_pool) >= self.interval_keep_n:
            oldest_path = self._interval_pool[0]
            # Only delete if not in top-k pool
            top_k_paths = {p for _, p in self._top_k_pool}
            if oldest_path not in top_k_paths:
                self._evict(oldest_path, pool="interval")
            else:
                log.info(
                    "[ckpt/interval] Evicted from interval but kept (in top-k pool): %s",
                    Path(oldest_path).name,
                )
            # The interval archive tracks its checkpoint's lifetime either way
            if self._owns(oldest_path):
                _delete_file(str(Path(oldest_path).with_suffix(ARCHIVE_SUFFIX)))
            self._interval_pool.popleft()

        self._save_atomic(trainer, filepath)
        self._write_sidecar(filepath, epoch, metrics, pool="interval")
        self._interval_pool.append(filepath)
        log.info(
            "[ckpt/interval] Saved %s (pool size %d/%d)",
            filename,
            len(self._interval_pool),
            self.interval_keep_n,
        )

        if self.archive_interval:
            try:
                pack_simurgh_archive(filepath, Path(filepath).with_suffix(ARCHIVE_SUFFIX))
            except Exception:  # noqa: BLE001
                log.exception("[archive] Failed to pack interval archive for %s", filename)

    # ------------------------------------------------------------------
    # Pool 3
    # ------------------------------------------------------------------

    def _update_last(self, trainer: L.Trainer, epoch: int, metrics: dict[str, Any]) -> None:
        filepath = str(Path(self._resolved_dirpath) / self.last_filename)
        self._save_atomic(trainer, filepath)
        self._write_sidecar(filepath, epoch, metrics, pool="last")
        self._last_ckpt_path = filepath

    # ------------------------------------------------------------------
    # Atomic write
    # ------------------------------------------------------------------

    def _save_atomic(self, trainer: L.Trainer, filepath: str) -> None:
        """Write checkpoint atomically: write to .tmp then os.replace."""
        tmp_path = filepath + ".tmp"
        trainer.save_checkpoint(tmp_path)
        os.replace(tmp_path, filepath)

    # ------------------------------------------------------------------
    # Per-checkpoint sidecar
    # ------------------------------------------------------------------

    def _write_sidecar(
        self,
        filepath: str,
        epoch: int,
        metrics: dict[str, Any],
        pool: str,
    ) -> None:
        """Write ``<ckpt>.json`` next to ``<ckpt>.ckpt``.

        Records the epoch, every scalar metric, the pool the checkpoint
        belongs to, and paths (relative to the sidecar itself, so the
        directory stays relocatable) to the shared config / frozen source
        / metadata written at training start.
        """
        sidecar_path = Path(filepath).with_suffix(".json")
        dirpath = sidecar_path.parent

        scalar_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                scalar_metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

        payload: dict[str, Any] = {
            "epoch": int(epoch),
            "pool": pool,
            "metrics": scalar_metrics,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config_path": os.path.relpath(dirpath / "config.yaml", start=dirpath),
            "frozen_source_path": os.path.relpath(dirpath / FROZEN_SOURCE_DIRNAME, start=dirpath)
            + "/",
            "metadata_path": os.path.relpath(dirpath / "metadata.json", start=dirpath),
        }

        tmp = str(sidecar_path) + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, str(sidecar_path))
        except OSError as exc:
            log.warning("[ckpt/sidecar] Could not write %s: %s", sidecar_path, exc)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        if self._resolved_dirpath is None:
            return
        state = {
            "top_k_pool": self._top_k_pool,
            "interval_pool": list(self._interval_pool),
            "last_ckpt_path": self._last_ckpt_path,
            "config": {
                "top_k_k": self.top_k_k,
                "top_k_metric": self.top_k_metric,
                "top_k_mode": self.top_k_mode,
                "interval_every_n": self.interval_every_n,
                "interval_keep_n": self.interval_keep_n,
                "last_filename": self.last_filename,
            },
        }
        state_path = Path(self._resolved_dirpath) / "checkpoint_state.json"
        tmp_path = str(state_path) + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, str(state_path))

    def _load_state(self, state_path: Path) -> None:
        """Restore pool state from a previous run's checkpoint_state.json."""
        try:
            with open(state_path) as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(
                "GOALCheckpointManager: could not load checkpoint_state.json (%s). "
                "Starting with empty pools.",
                exc,
            )
            return

        raw_top_k = state.get("top_k_pool", [])
        # Each entry is [value, path] from JSON
        restored_top_k: list[tuple[float, str]] = []
        for entry in raw_top_k:
            val, path = float(entry[0]), str(entry[1])
            if Path(path).exists():
                restored_top_k.append((val, path))
            else:
                log.warning("[ckpt/resume] top_k file missing on disk, dropping: %s", path)
        self._top_k_pool = restored_top_k

        restored_interval: list[str] = []
        for path in state.get("interval_pool", []):
            if Path(path).exists():
                restored_interval.append(str(path))
            else:
                log.warning("[ckpt/resume] interval file missing on disk, dropping: %s", path)
        self._interval_pool = deque(restored_interval)

        last = state.get("last_ckpt_path")
        if last and Path(last).exists():
            self._last_ckpt_path = last
        elif last:
            log.warning("[ckpt/resume] last.ckpt path in state does not exist: %s", last)

        log.info(
            "[ckpt/resume] Restored pools: top_k=%d, interval=%d, last=%s",
            len(self._top_k_pool),
            len(self._interval_pool),
            self._last_ckpt_path,
        )

    # ------------------------------------------------------------------
    # Class-method for train.py to restore state into a fresh instance
    # ------------------------------------------------------------------

    @classmethod
    def restore_pools_from_dir(cls, manager: GOALCheckpointManager, dirpath: str) -> None:
        """Load checkpoint_state.json from ``dirpath`` into ``manager``.

        Called by train.py after constructing the manager but before
        ``trainer.fit()``, so pool state survives a crash/resume cycle.
        """
        state_path = Path(dirpath) / "checkpoint_state.json"
        if state_path.exists():
            # Deliberately *not* ``_resolved_dirpath``: that is where this
            # run writes, and ``setup()`` sets it from the config.  The
            # restored pools may name files under ``dirpath`` instead, and
            # ``_owns`` uses the difference to keep them from being deleted.
            manager._source_dirpath = dirpath
            manager._load_state(state_path)
        else:
            log.info("[ckpt/resume] No checkpoint_state.json in %s — starting fresh.", dirpath)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _warn_mode_mismatch(self) -> None:
        metric = self.top_k_metric
        leaf = metric.rsplit("/", 1)[-1]
        mode = self.top_k_mode
        if leaf in _KNOWN_MIN_METRICS and mode != "min":
            warnings.warn(
                f"GOALCheckpointManager: metric '{metric}' is typically minimised "
                f"but mode='{mode}' was given.  Set mode='min' unless you intend this.",
                UserWarning,
                stacklevel=3,
            )
        elif leaf in _KNOWN_MAX_METRICS and mode != "max":
            warnings.warn(
                f"GOALCheckpointManager: metric '{metric}' is typically maximised "
                f"but mode='{mode}' was given.  Set mode='max' unless you intend this.",
                UserWarning,
                stacklevel=3,
            )

    @property
    def dirpath(self) -> str | None:
        return self._resolved_dirpath

    @property
    def source_dirpath(self) -> str | None:
        """Directory this run inherited its pools from, if it resumed one."""
        return self._source_dirpath

    @property
    def extension(self) -> dict[str, Any] | None:
        """Extension provenance recorded for this run, if any."""
        return dict(self._extension) if self._extension is not None else None

    @property
    def top_k_pool(self) -> list[tuple[float, str]]:
        return list(self._top_k_pool)

    @property
    def interval_pool(self) -> list[str]:
        return list(self._interval_pool)

    @property
    def last_ckpt_path(self) -> str | None:
        return self._last_ckpt_path


# ------------------------------------------------------------------
# File helpers
# ------------------------------------------------------------------


def _delete_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        log.warning("GOALCheckpointManager: could not delete %s: %s", path, exc)


def _delete_checkpoint(path: str) -> None:
    """Delete a checkpoint together with its .json sidecar."""
    _delete_file(path)
    _delete_file(str(Path(path).with_suffix(".json")))
