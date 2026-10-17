"""Tests for self-contained checkpoints (frozen source, sidecars, archives).

Covers the full pipeline:

1. Source / config / metadata frozen at ``on_train_start`` — before any
   weights exist.
2. Code-change resilience — loading executes ``frozen_source/`` even when
   the live source file has been destroyed.
3. ``GOALCalculator`` FORMAT B (checkpoint directory) with real inference.
4. ``.simurgh`` archive round trip (FORMAT A) — outputs identical to B.
5. Legacy bare ``.ckpt`` loading (FORMAT C) warns but still works.
6. Every checkpoint write produces a ``.json`` sidecar; eviction removes it.

A tiny SIMURGH backbone (float64, lmax=1, 8 channels) keeps everything
fast enough for unit testing.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from ase.build import molecule
from omegaconf import DictConfig, OmegaConf

from goal.ml.training.archive import (
    ARCHIVE_SUFFIX,
    FROZEN_SOURCE_DIRNAME,
    inspect_checkpoint,
    pack_simurgh_archive,
    resolve_checkpoint,
)
from goal.ml.training.callbacks.checkpoint_manager import GOALCheckpointManager

# ---------------------------------------------------------------------------
# Tiny SIMURGH module + config
# ---------------------------------------------------------------------------

_DRESSING = {
    "num_elements": 120,
    "embedding_dim": 8,
    "hidden_channels": 8,
    "lmax": 1,
    "num_radial_basis": 4,
    "cutoff": 5.0,
    "radial_mlp_hidden": 8,
    "num_message_passing": 1,
    "body_order": 1,
}

_ARTISAN = {
    "scalar_channels": 4,
    "hidden_dims": [8],
    "expert_type": "linear",
}


def _tiny_config() -> DictConfig:
    """Config the GOALCalculator can rebuild the model from."""
    return OmegaConf.create(
        {
            "model": {
                "backbone": {
                    "name": "simurgh",
                    "elements": [1, 6],
                    "dressing_kwargs": dict(_DRESSING),
                    "artisan_config": dict(_ARTISAN),
                    "cutoff": 5.0,
                    "atomic_energies": {"mode": "learned"},
                    "compute_pairwise_forces": True,
                },
                "head": {
                    "name": "dual_forces",
                    "irreps_in": "8x0e+8x1o",
                    "hidden_dim": 16,
                    "mode": "pairwise",
                },
            },
            "training": {
                "losses": [
                    {"name": "energy", "weight": 1.0, "fn": "mse"},
                    {"name": "forces", "weight": 1.0, "fn": "mse"},
                ],
                "ema": {"enabled": False},
            },
            "data": {"cutoff": 5.0},
        }
    )


def _build_module(cfg: DictConfig):
    """Construct a GOALModule exactly as the calculator rebuild would."""
    from goal.ml.nn.heads.dual_forces import DualForcesHead
    from goal.ml.nn.models.simurgh.backbone import SimurghBackbone
    from goal.ml.training.loss import (
        CompositeLoss,
        EnergyLoss,
        ForcesLoss,
        WeightedLoss,
    )
    from goal.ml.training.module import GOALModule

    torch.manual_seed(0)
    backbone = SimurghBackbone(
        elements=tuple(cfg.model.backbone.elements),
        dressing_kwargs=dict(_DRESSING),
        artisan_config={**_ARTISAN, "hidden_dims": tuple(_ARTISAN["hidden_dims"])},
        cutoff=5.0,
        atomic_energies={"mode": "learned"},
        compute_pairwise_forces=True,
    ).to(torch.float64)
    head = DualForcesHead(irreps_in="8x0e+8x1o", hidden_dim=16, mode="pairwise").to(torch.float64)
    loss = CompositeLoss(
        [
            WeightedLoss(EnergyLoss(loss_fn="mse"), weight=1.0, label="energy"),
            WeightedLoss(ForcesLoss(loss_fn="mse"), weight=1.0, label="forces"),
        ]
    )
    return GOALModule(backbone=backbone, head=head, loss=loss, config=cfg)


def _methane_batch():
    from torch_geometric.data import Batch

    from goal.ml.data.graph import AtomicGraph

    graph = AtomicGraph.from_ase(molecule("CH4"), cutoff=5.0, dtype=torch.float64)
    return Batch.from_data_list([graph])


def _make_trainer(module, cfg: DictConfig, metrics: dict[str, float], epoch: int) -> MagicMock:
    """Trainer mock whose save_checkpoint writes a *real* loadable ckpt."""
    trainer = MagicMock()
    trainer.is_global_zero = True
    trainer.current_epoch = epoch
    trainer.callback_metrics = dict(metrics)

    def _save(path: str) -> None:
        torch.save(
            {
                "state_dict": module.state_dict(),
                "hyper_parameters": {"config": cfg},
                "epoch": epoch,
                "global_step": 0,
            },
            path,
        )

    trainer.save_checkpoint = MagicMock(side_effect=_save)
    return trainer


def _make_manager(dirpath: Path, **overrides) -> GOALCheckpointManager:
    kwargs = {
        "dirpath": str(dirpath),
        "top_k": {"enabled": True, "k": 2, "metric": "val_forces/forces_mae", "mode": "min"},
        "interval": {"enabled": True, "every_n_epochs": 2, "keep_last_n": 2},
        "last": {"enabled": True, "filename": "last.ckpt"},
        "save_archive": False,
    }
    kwargs.update(overrides)
    return GOALCheckpointManager(**kwargs)


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory) -> Path:
    """A checkpoint directory produced by 5 real training steps + the manager.

    Module-scoped: building the model and freezing source is the slow part,
    and the directory is read-only for every test that uses it.
    """
    dirpath = tmp_path_factory.mktemp("ckpts")
    cfg = _tiny_config()
    module = _build_module(cfg)

    trainer = _make_trainer(module, cfg, {}, epoch=0)
    manager = _make_manager(dirpath)
    manager.setup(trainer, module, "fit")
    manager.on_train_start(trainer, module)

    # 5 real optimisation steps on methane so the saved weights are "trained"
    batch = _methane_batch()
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    for _ in range(5):
        optimizer.zero_grad()
        predictions = module(batch)
        loss = predictions["energy"].pow(2).sum() + predictions["forces"].pow(2).sum()
        loss.backward()
        optimizer.step()

    trainer = _make_trainer(module, cfg, {"val_forces/forces_mae": 0.5}, epoch=0)
    manager.on_validation_epoch_end(trainer, module)
    return dirpath


def _energy_and_forces(calc) -> tuple[float, np.ndarray]:
    atoms = molecule("CH4")
    atoms.calc = calc
    return atoms.get_potential_energy(), atoms.get_forces()


# ---------------------------------------------------------------------------
# Test 1 — source frozen at training start, before any weights
# ---------------------------------------------------------------------------


class TestFreezeAtTrainingStart:
    def test_frozen_layout_exists_before_any_checkpoint(self, tmp_path: Path) -> None:
        cfg = _tiny_config()
        module = _build_module(cfg)
        manager = _make_manager(tmp_path)
        trainer = _make_trainer(module, cfg, {}, epoch=0)
        manager.setup(trainer, module, "fit")
        manager.on_train_start(trainer, module)

        assert (tmp_path / FROZEN_SOURCE_DIRNAME).is_dir()
        assert (tmp_path / "config.yaml").is_file()
        assert (tmp_path / "metadata.json").is_file()
        assert not list(tmp_path.glob("*.ckpt")), "No weights must exist yet"

        # The model's own source must be in the frozen closure
        frozen = tmp_path / FROZEN_SOURCE_DIRNAME
        assert (frozen / "goal/ml/nn/models/simurgh/backbone.py").is_file()
        assert (frozen / "goal/ml/nn/blocks/env_dressing.py").is_file()
        assert (frozen / "goal/ml/training/module.py").is_file()
        assert (frozen / "goal/__init__.py").is_file()

    def test_metadata_contents(self, trained_dir: Path) -> None:
        metadata = json.loads((trained_dir / "metadata.json").read_text())
        assert metadata["source_frozen"] is True
        assert metadata["model_class"] == "SimurghBackbone"
        assert metadata["elements"] == [1, 6]
        assert "scale" in metadata
        assert "timestamp" in metadata

    def test_config_is_resolved_yaml(self, trained_dir: Path) -> None:
        cfg = OmegaConf.load(trained_dir / "config.yaml")
        assert cfg.model.backbone.name == "simurgh"
        assert float(cfg.data.cutoff) == 5.0

    def test_resume_keeps_original_snapshot(self, tmp_path: Path) -> None:
        cfg = _tiny_config()
        module = _build_module(cfg)
        manager = _make_manager(tmp_path)
        trainer = _make_trainer(module, cfg, {}, epoch=0)
        manager.setup(trainer, module, "fit")
        manager.on_train_start(trainer, module)

        sentinel = tmp_path / FROZEN_SOURCE_DIRNAME / "SENTINEL"
        sentinel.touch()
        manager.on_train_start(trainer, module)  # simulated resume
        assert sentinel.exists(), "Resume must not re-freeze (overwrite) the snapshot"


# ---------------------------------------------------------------------------
# Test 2 — code-change resilience: frozen source wins over modified live code
# ---------------------------------------------------------------------------


class TestCodeChangeResilience:
    def test_loads_from_frozen_source_after_live_code_destroyed(self, trained_dir: Path) -> None:
        from goal.ml.utils.calculator import GOALCalculator

        calc_before = GOALCalculator(checkpoint_path=str(trained_dir), checkpoint="best")
        energy_before, forces_before = _energy_and_forces(calc_before)

        import goal.ml.nn.models.simurgh.backbone as live_backbone

        live_path = Path(live_backbone.__file__)
        original = live_path.read_text()
        try:
            live_path.write_text(
                'raise ImportError("backbone source was modified after checkpointing")\n'
            )
            calc_after = GOALCalculator(checkpoint_path=str(trained_dir), checkpoint="best")
            energy_after, forces_after = _energy_and_forces(calc_after)
        finally:
            live_path.write_text(original)

        # Provenance: the loaded class was compiled from frozen_source/, not
        # src/.  (inspect.getfile would resolve through the restored live
        # sys.modules entry, so check the code object's origin instead.)
        loaded_cls = type(calc_after._module.backbone)
        compiled_from = Path(loaded_cls.__init__.__code__.co_filename)
        assert (trained_dir / FROZEN_SOURCE_DIRNAME) in compiled_from.parents
        assert loaded_cls is not live_backbone.SimurghBackbone

        assert energy_after == pytest.approx(energy_before, abs=1e-12)
        np.testing.assert_allclose(forces_after, forces_before, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 3 — FORMAT B: load from checkpoint directory, run methane inference
# ---------------------------------------------------------------------------


class TestDirectoryLoading:
    def test_methane_inference(self, trained_dir: Path) -> None:
        from goal.ml.utils.calculator import GOALCalculator

        calc = GOALCalculator(checkpoint_path=str(trained_dir), checkpoint="best")
        energy, forces = _energy_and_forces(calc)

        assert np.isfinite(energy)
        assert forces.shape == (5, 3)
        # Pairwise-mode SIMURGH forces obey Newton's third law structurally
        np.testing.assert_allclose(forces.sum(axis=0), np.zeros(3), atol=1e-8)

    def test_checkpoint_selectors(self, trained_dir: Path) -> None:
        best = resolve_checkpoint(trained_dir, "best")
        last = resolve_checkpoint(trained_dir, "last")
        epoch0 = resolve_checkpoint(trained_dir, "epoch=0")
        assert best.name.startswith("best_")
        assert last.name == "last.ckpt"
        assert epoch0.name == "interval_epoch=0000.ckpt"
        assert resolve_checkpoint(trained_dir, best.name) == best

    def test_invalid_directory_rejected(self, tmp_path: Path) -> None:
        from goal.ml.utils.calculator import GOALCalculator

        (tmp_path / "stray.ckpt").touch()
        with pytest.raises(FileNotFoundError, match="not a valid SIMURGH"):
            GOALCalculator(checkpoint_path=str(tmp_path))


# ---------------------------------------------------------------------------
# Test 4 — FORMAT A: archive round trip, outputs identical to FORMAT B
# ---------------------------------------------------------------------------


class TestArchiveRoundTrip:
    def test_pack_and_load(self, trained_dir: Path, tmp_path: Path) -> None:
        from goal.ml.utils.calculator import GOALCalculator

        ckpt = resolve_checkpoint(trained_dir, "best")
        archive = pack_simurgh_archive(ckpt, tmp_path / f"model{ARCHIVE_SUFFIX}")
        assert archive.is_file()

        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
        assert "weights.pt" in names
        assert "config.yaml" in names
        assert "metadata.json" in names
        assert any(n.startswith(f"{FROZEN_SOURCE_DIRNAME}/") for n in names)

        calc_dir = GOALCalculator(checkpoint_path=str(trained_dir), checkpoint="best")
        calc_arc = GOALCalculator(checkpoint_path=str(archive))
        energy_dir, forces_dir = _energy_and_forces(calc_dir)
        energy_arc, forces_arc = _energy_and_forces(calc_arc)

        assert energy_arc == pytest.approx(energy_dir, abs=1e-12)
        np.testing.assert_allclose(forces_arc, forces_dir, atol=1e-12)

    def test_pack_requires_managed_directory(self, tmp_path: Path) -> None:
        stray = tmp_path / "stray.ckpt"
        stray.touch()
        with pytest.raises(FileNotFoundError, match="not a self-contained"):
            pack_simurgh_archive(stray)

    def test_inspect_archive(self, trained_dir: Path, tmp_path: Path) -> None:
        ckpt = resolve_checkpoint(trained_dir, "best")
        archive = pack_simurgh_archive(ckpt, tmp_path / f"inspect{ARCHIVE_SUFFIX}")
        info = inspect_checkpoint(archive)
        assert info["format"] == "archive"
        assert info["source_frozen"] is True
        assert info["metadata"]["model_class"] == "SimurghBackbone"
        assert info["sidecar"]["pool"] == "top_k"


# ---------------------------------------------------------------------------
# Test 5 — FORMAT C: legacy bare .ckpt warns loudly but still loads
# ---------------------------------------------------------------------------


class TestLegacyCheckpoint:
    def test_warns_and_loads(self, trained_dir: Path, caplog) -> None:
        from goal.ml.utils.calculator import GOALCalculator

        ckpt = resolve_checkpoint(trained_dir, "best")
        with caplog.at_level(logging.WARNING, logger="goal.ml.utils.calculator"):
            calc = GOALCalculator(checkpoint_path=str(ckpt))

        assert "not self-contained" in caplog.text
        energy, _ = _energy_and_forces(calc)
        assert np.isfinite(energy)


# ---------------------------------------------------------------------------
# Test 6 — sidecar written alongside every checkpoint
# ---------------------------------------------------------------------------


class TestSidecars:
    def test_every_ckpt_has_sidecar_with_correct_values(self, tmp_path: Path) -> None:
        cfg = _tiny_config()
        module = _build_module(cfg)
        manager = _make_manager(tmp_path)
        trainer = _make_trainer(module, cfg, {}, epoch=0)
        manager.setup(trainer, module, "fit")
        manager.on_train_start(trainer, module)

        values = [0.5, 0.4, 0.3]
        for epoch, value in enumerate(values):
            trainer = _make_trainer(module, cfg, {"val_forces/forces_mae": value}, epoch=epoch)
            manager.on_validation_epoch_end(trainer, module)

        ckpts = sorted(tmp_path.glob("*.ckpt"))
        assert ckpts, "Expected checkpoints on disk"
        for ckpt in ckpts:
            sidecar_path = ckpt.with_suffix(".json")
            assert sidecar_path.is_file(), f"Missing sidecar for {ckpt.name}"
            sidecar = json.loads(sidecar_path.read_text())
            assert sidecar["pool"] in ("top_k", "interval", "last")
            assert sidecar["metrics"]["val_forces/forces_mae"] == values[sidecar["epoch"]]
            assert sidecar["config_path"] == "config.yaml"
            assert sidecar["frozen_source_path"] == f"{FROZEN_SOURCE_DIRNAME}/"
            assert sidecar["metadata_path"] == "metadata.json"

        # last.ckpt sidecar reflects the most recent epoch
        last_sidecar = json.loads((tmp_path / "last.json").read_text())
        assert last_sidecar["epoch"] == 2
        assert last_sidecar["pool"] == "last"

    def test_sidecar_deleted_with_evicted_checkpoint(self, tmp_path: Path) -> None:
        cfg = _tiny_config()
        module = _build_module(cfg)
        manager = _make_manager(
            tmp_path,
            top_k={"enabled": True, "k": 1, "metric": "val_forces/forces_mae", "mode": "min"},
            interval={"enabled": False},
            last={"enabled": False},
        )
        trainer = _make_trainer(module, cfg, {}, epoch=0)
        manager.setup(trainer, module, "fit")
        manager.on_train_start(trainer, module)

        for epoch, value in enumerate([0.5, 0.1]):  # 0.1 evicts the 0.5 ckpt
            trainer = _make_trainer(module, cfg, {"val_forces/forces_mae": value}, epoch=epoch)
            manager.on_validation_epoch_end(trainer, module)

        remaining = {p.name for p in tmp_path.glob("best_*")}
        assert len(remaining) == 2, f"Expected one ckpt + one sidecar, got {remaining}"
        assert not any("0.5000" in name for name in remaining), "Evicted sidecar lingers"
