"""Hydra-based evaluation entry point.

Evaluates a trained checkpoint on a test split::

    goal-eval ckpt_path=logs/<task>/runs/<run>/checkpoints          # managed directory
    goal-eval ckpt_path=models/simurgh_legacy_gmd26.simurgh                # archive
    goal-eval ckpt_path=.../last.ckpt +data.test_paths=[data/test.xyz]

The **model** is always rebuilt from the checkpoint itself — the same
loader as :class:`~goal.ml.utils.calculator.GOALCalculator`: frozen source
for archives and checkpoint directories, the saved config (including the
dataset-derived ``elements`` / ``scale`` / ``avg_num_neighbors``), and the
EMA weights when the run kept them.  The config passed to this command
only says what to evaluate on:

* ``data`` is merged onto the checkpoint's own ``data:`` section, so the
  default (empty) evaluates on exactly the test split the run used;
* ``trainer`` sets device / precision for the test loop.

``--config-name <experiment>`` still works: that experiment's ``data:``
section then replaces the checkpoint's.

``predictions_file=<path>.npz`` also writes every per-structure prediction
in GOAL's evaluation format (see :data:`~goal.ml.training.metrics.EVALUATION_KEYS`)
and reports the dataset-level metrics of
:func:`~goal.ml.training.metrics.evaluation_metrics` (centred and offset
energy MAE, per-molecule signed error, force tails) — one extra pass over
the test split.
"""

from __future__ import annotations

import json
import typing
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from goal.ml.cli import CONFIGS_ML_DIR
from goal.ml.cli.builders import instantiate_loggers
from goal.ml.data.datamodule import GOALDataModule
from goal.ml.training.metrics import evaluation_metrics

#: Keys of a plain ``trainer:`` block (no ``_target_``) forwarded to ``L.Trainer``.
_TRAINER_KEYS: tuple[str, ...] = (
    "accelerator",
    "devices",
    "num_nodes",
    "precision",
    "strategy",
    "logger",
    "enable_progress_bar",
    "limit_test_batches",
    "inference_mode",
)


def build_eval_config(cfg: DictConfig, ckpt_cfg: DictConfig) -> DictConfig:
    """The checkpoint's config with ``cfg.data`` merged over its ``data:``."""
    merged: DictConfig = OmegaConf.create(OmegaConf.to_container(ckpt_cfg, resolve=False))
    override: typing.Any = cfg.get("data", None)
    if override:
        with open_dict(merged):
            merged.data = OmegaConf.merge(merged.get("data", {}), override)
    return merged


def build_eval_trainer(cfg: DictConfig, ckpt_cfg: DictConfig) -> L.Trainer:
    """Test-loop trainer from ``cfg.trainer``.

    A ``trainer:`` block with ``_target_`` (an experiment config) is
    instantiated as-is; a plain one (``eval.yaml``) builds ``L.Trainer``
    from the keys above.  ``precision: null`` uses the checkpoint's own
    ``trainer.precision`` so a float64 model is tested in float64.
    """
    trainer_cfg: typing.Any = cfg.get("trainer", None) or {}
    loggers: list[typing.Any] = instantiate_loggers(cfg.get("logger"))
    if "_target_" in trainer_cfg:
        return hydra.utils.instantiate(trainer_cfg, callbacks=None, logger=loggers or False)

    kwargs: dict[str, typing.Any] = {
        k: trainer_cfg[k] for k in _TRAINER_KEYS if trainer_cfg.get(k, None) is not None
    }
    if kwargs.get("precision") is None:
        saved: typing.Any = (ckpt_cfg.get("trainer", None) or {}).get("precision", None)
        if saved is not None:
            kwargs["precision"] = saved
    kwargs.setdefault("logger", loggers or False)
    kwargs.setdefault("accelerator", "auto")
    kwargs.setdefault("devices", 1)
    # Forces are -dE/dpos: the test loop must keep autograd available.
    kwargs.setdefault("inference_mode", False)
    return L.Trainer(**kwargs)


def collect_predictions(module: typing.Any, dataloader: typing.Any) -> dict[str, np.ndarray]:
    """Per-structure predictions and labels of every batch, in the evaluation format."""
    device = next(module.parameters()).device
    parts: dict[str, list[np.ndarray]] = {}

    def _add(key: str, value: torch.Tensor) -> None:
        parts.setdefault(key, []).append(value.detach().cpu().double().numpy())

    module.eval()
    for batch in dataloader:
        batch = batch.to(device)
        with torch.enable_grad():  # forces are -dE/dpos
            out = module(batch)
        _add("energy", out["energy"].reshape(-1))
        _add("energy_ref", batch.energy.reshape(-1))
        _add("num_atoms", torch.bincount(batch.batch, minlength=batch.num_graphs))
        _add("atomic_numbers", batch.atomic_numbers)
        if "forces" in out and batch.get("forces", None) is not None:
            _add("forces", out["forces"])
            _add("forces_ref", batch.forces)
    data = {k: np.concatenate(v) for k, v in parts.items()}
    data["num_atoms"] = data["num_atoms"].astype(np.int64)
    data["atomic_numbers"] = data["atomic_numbers"].astype(np.int64)
    return data


@hydra.main(version_base=None, config_path=CONFIGS_ML_DIR, config_name="eval")
def evaluate(cfg: DictConfig) -> None:
    """GOAL evaluation entry point.

    Rebuilds the model from ``ckpt_path`` and runs ``trainer.test()`` on
    the test split.  Strategy / device are controlled via ``trainer.*``.
    """
    from goal.ml.utils.calculator import load_goal_checkpoint

    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    # Performance: TF32 for Ampere+ GPUs — falls back safely on older hardware
    has_tf32: bool = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    torch.set_float32_matmul_precision("high" if has_tf32 else "highest")

    loaded = load_goal_checkpoint(
        str(cfg.ckpt_path),
        checkpoint=str(cfg.get("checkpoint", "best")),
        device="cpu",  # the trainer moves it
        use_ema=bool(cfg.get("use_ema", True)),
    )
    print(f"[goal-eval] {cfg.ckpt_path}: {loaded.weights.upper()} weights")

    eval_cfg: DictConfig = build_eval_config(cfg, loaded.config)
    datamodule: GOALDataModule = GOALDataModule(eval_cfg)
    trainer: L.Trainer = build_eval_trainer(cfg, loaded.config)

    results: list[dict[str, float]] = trainer.test(loaded.module, datamodule=datamodule)

    summary: dict[str, typing.Any] | None = None
    predictions_file: typing.Any = cfg.get("predictions_file", None)
    if predictions_file and trainer.is_global_zero:
        data = collect_predictions(loaded.module, datamodule.test_dataloader())
        out_npz = Path(str(predictions_file))
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_npz, **data)
        summary = evaluation_metrics(data)
        print(f"[goal-eval] predictions written to {out_npz}")
        for key, value in summary.items():
            if not isinstance(value, list):
                print(f"[goal-eval]   {key}: {value:.6g}")

    output_file: typing.Any = cfg.get("output_file", None)
    if output_file and trainer.is_global_zero:
        out = Path(str(output_file))
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(cfg.ckpt_path),
            "weights": loaded.weights,
            "metrics": [{k: float(v) for k, v in r.items()} for r in results],
        }
        if summary is not None:
            payload["evaluation"] = summary
        out.write_text(json.dumps(payload, indent=2))
        print(f"[goal-eval] metrics written to {out}")


if __name__ == "__main__":
    evaluate()
