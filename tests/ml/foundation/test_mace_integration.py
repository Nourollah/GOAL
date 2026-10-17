"""Integration test: real MACE-MP fine-tuning through the GOAL stack.

Requires ``mace-torch`` installed (use the standalone ``.venv-mace`` — mace
pins e3nn==0.4.4, incompatible with GOAL's pixi env). Skipped otherwise.

Run with::

    .venv-mace/bin/python -m pytest -m integration tests/ml/foundation/test_mace_integration.py
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.integration

_HAS_MACE = importlib.util.find_spec("mace") is not None


def _methane_dataset(n: int) -> list:
    """``n`` slightly-rattled methane graphs with (fake) energy/force labels."""
    from ase.build import molecule

    from goal.ml.data.graph import AtomicGraph

    rng = np.random.default_rng(0)
    graphs = []
    for _ in range(n):
        atoms = molecule("CH4")
        atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
        graphs.append(
            AtomicGraph.from_ase(
                atoms,
                cutoff=6.0,
                energy=float(rng.normal(-40.0, 0.5)),
                forces=rng.normal(scale=0.1, size=(len(atoms), 3)),
            )
        )
    return graphs


@pytest.mark.skipif(not _HAS_MACE, reason="mace-torch not installed")
def test_mace_head_only_one_epoch(tmp_path) -> None:
    import lightning as L
    import torch

    # e3nn 0.4.4 (pinned by mace-torch) loads its Wigner-3j constants via a bare
    # torch.load, which torch>=2.6 blocks by default. Allow-list `slice` before
    # importing mace. Harmless on torch versions that don't need it.
    try:
        torch.serialization.add_safe_globals([slice])
    except Exception:  # noqa: BLE001
        pass

    from omegaconf import OmegaConf
    from torch_geometric.loader import DataLoader

    from goal.ml.nn.models.foundation.mace import MACEFinetune
    from goal.ml.training.callbacks.checkpoint_manager import GOALCheckpointManager
    from goal.ml.training.loss import (
        CompositeLoss,
        EnergyLoss,
        ForcesLoss,
        WeightedLoss,
    )
    from goal.ml.training.module import GOALModule

    torch.manual_seed(0)
    train = _methane_dataset(10)

    backbone = MACEFinetune(
        checkpoint="mace-mp-0-small", strategy="head_only", dtype="float64", reestimate_e0s=True
    )
    # Re-baseline E0s to this dataset (normally done by FoundationE0Callback).
    backbone.reestimate_atomic_energies(train)

    loss = CompositeLoss(
        [
            WeightedLoss(EnergyLoss("mae"), 1.0, "energy"),
            WeightedLoss(ForcesLoss("mae"), 10.0, "forces"),
        ]
    )
    cfg = OmegaConf.create(
        {
            # model/data sections are what GOALCalculator reads back from the
            # saved checkpoint to reconstruct the backbone (as real goal-train
            # configs carry them).
            "model": {
                "backbone": {
                    "name": "mace_finetune",
                    "checkpoint": "mace-mp-0-small",
                    "strategy": "head_only",
                    "dtype": "float64",
                    "reestimate_e0s": True,
                },
                "head": None,
            },
            "data": {"cutoff": 6.0},
            "training": {
                "ema": {"enabled": False},
                "optimizer": {"lr": 1e-4, "weight_decay": 0.0, "scheduler_type": "cosine"},
                "gradient_clip": 0.0,
            },
            "trainer": {"max_epochs": 1},
        }
    )
    module = GOALModule(backbone=backbone, head=None, loss=loss, config=cfg)

    ckpt_dir = tmp_path / "checkpoints"
    manager = GOALCheckpointManager(
        dirpath=str(ckpt_dir),
        top_k={"enabled": True, "k": 1, "metric": "val/forces_mae", "mode": "min"},
        interval={"enabled": False},
        last={"enabled": True},
        save_archive=False,
    )
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        precision="64",
        inference_mode=False,
        enable_progress_bar=False,
        logger=False,
        callbacks=[manager],
    )
    loader = DataLoader(train, batch_size=2)
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    # A checkpoint was saved by GOALCheckpointManager.
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    assert ckpts, "GOALCheckpointManager wrote no checkpoint"

    # GOALCalculator can load it (bare .ckpt legacy path) and compute.
    from ase.build import molecule

    from goal.ml.utils.calculator import GOALCalculator

    calc = GOALCalculator(checkpoint_path=str(ckpt_dir / "last.ckpt"))
    atoms = molecule("CH4")
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
