"""``goal-eval`` evaluates a checkpoint with the model the checkpoint defines.

It used to rebuild the model from the *command-line* config (no default
config existed, dataset-derived arguments were missing, EMA shadows of old
checkpoints were random) — now the model comes from the checkpoint through
the same loader as ``GOALCalculator`` and the config only picks the data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.io import write
from omegaconf import OmegaConf

from goal.ml.cli.evaluate import build_eval_config, build_eval_trainer
from goal.ml.data.datamodule import GOALDataModule
from goal.ml.utils.calculator import load_goal_checkpoint
from tests.ml.test_archive import _build_module, _tiny_config


def _dataset(path: Path, n: int = 4) -> Path:
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(n):
        atoms = molecule("CH4")
        atoms.positions += rng.normal(scale=0.03, size=atoms.positions.shape)
        atoms.calc = EMT()
        atoms.info["energy"] = atoms.get_potential_energy()
        atoms.arrays["forces"] = atoms.get_forces()
        atoms.calc = None
        frames.append(atoms)
    write(path, frames, format="extxyz")
    return path


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    data = _dataset(tmp_path / "methane.xyz")
    cfg = _tiny_config()
    cfg.data = {
        "dataset_type": "xyz",
        "cutoff": 5.0,
        "batch_size": 2,
        "train_paths": [str(data)],
        "val_paths": [str(data)],
        "test_paths": [str(data)],
    }
    cfg.training.ema = {"enabled": True, "decay": 0.99}
    module = _build_module(cfg)
    ckpt = tmp_path / "run.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": {"config": cfg},
            "goal_ema": {"shadow": [p.detach().clone() for p in module.parameters()]},
        },
        ckpt,
    )
    return ckpt


def test_eval_uses_the_checkpoints_model_and_data(checkpoint: Path) -> None:
    loaded = load_goal_checkpoint(checkpoint)
    assert loaded.weights == "ema"
    cli_cfg = OmegaConf.create({"data": {}, "trainer": {"accelerator": "cpu", "devices": 1}})
    eval_cfg = build_eval_config(cli_cfg, loaded.config)
    assert eval_cfg.data.test_paths == loaded.config.data.test_paths

    trainer = build_eval_trainer(cli_cfg, loaded.config)
    results = trainer.test(loaded.module, datamodule=GOALDataModule(eval_cfg))
    assert results and all(np.isfinite(v) for v in results[0].values())


def test_cli_data_overrides_merge_onto_the_checkpoints(checkpoint: Path, tmp_path: Path) -> None:
    other = _dataset(tmp_path / "other.xyz", n=2)
    loaded = load_goal_checkpoint(checkpoint)
    cli_cfg = OmegaConf.create({"data": {"test_paths": [str(other)], "batch_size": 1}})
    merged = build_eval_config(cli_cfg, loaded.config)
    assert list(merged.data.test_paths) == [str(other)]
    assert merged.data.batch_size == 1
    assert merged.data.dataset_type == "xyz"  # untouched keys come from the checkpoint


def test_plain_trainer_defaults(checkpoint: Path) -> None:
    loaded = load_goal_checkpoint(checkpoint)
    trainer = build_eval_trainer(
        OmegaConf.create({"trainer": {"accelerator": "cpu"}}), loaded.config
    )
    assert trainer.test_loop.inference_mode is False  # forces need autograd


@pytest.mark.slow
def test_goal_eval_command_end_to_end(checkpoint: Path, tmp_path: Path) -> None:
    out = tmp_path / "metrics.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "goal.ml.cli.evaluate",
            f"ckpt_path={checkpoint}",
            "trainer.accelerator=cpu",
            f"output_file={out}",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    payload = json.loads(out.read_text())
    assert payload["weights"] == "ema"
    assert payload["metrics"] and payload["metrics"][0]
