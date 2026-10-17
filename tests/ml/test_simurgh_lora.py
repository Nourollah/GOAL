"""LoRA fine-tuning of SIMURGH checkpoints (goal.ml.adapters.simurgh_lora)."""

from __future__ import annotations

import builtins
import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import molecule
from omegaconf import OmegaConf

from goal.ml.adapters.simurgh_lora import (
    LORA_SUFFIX,
    SIMURGHLoRAAdapter,
    SIMURGHLoRAConfig,
)
from goal.ml.data.graph import AtomicGraph

peft = pytest.importorskip("peft")

ARTISAN = {
    "architecture": "equivariant",
    "hidden_irreps": "4x0e + 4x1o",
    "num_rbf": 4,
    "radial_hidden": 8,
    "n_scalar_out": 4,
    "final_hidden": 4,
    "n_elements": "auto",
}


def _base_config():
    return OmegaConf.create(
        {
            "model": {
                "backbone": {
                    "name": "simurgh_arace",
                    "elements": [1, 6, 8],
                    "num_rounds": 1,
                    "cutoff": 4.0,
                    "embedding_dim": 8,
                    "num_elements": 9,
                    "avg_num_neighbors": 4.0,
                    "artisan": ARTISAN,
                },
                "head": {"name": "energy_forces", "irreps_in": "4x0e+4x1o", "hidden_dim": 8},
            },
            "training": {"losses": [], "ema": {"enabled": False}},
            "data": {"cutoff": 4.0},
        }
    )


def _base_module():
    from goal.ml.nn.heads.energy_forces import EnergyForcesHead
    from goal.ml.nn.models.simurgh.backbone_arace import SimurghAraceBackbone
    from goal.ml.training.loss import CompositeLoss
    from goal.ml.training.module import GOALModule

    cfg = _base_config()
    kwargs = {k: v for k, v in OmegaConf.to_container(cfg.model.backbone).items() if k != "name"}
    torch.manual_seed(0)
    return GOALModule(
        backbone=SimurghAraceBackbone(**kwargs).double(),
        head=EnergyForcesHead(irreps_in="4x0e+4x1o", hidden_dim=8).double(),
        loss=CompositeLoss([]),
        config=cfg,
    )


@pytest.fixture()
def base_ckpt(tmp_path: Path) -> Path:
    module = _base_module()
    path = tmp_path / "base.ckpt"
    torch.save(
        {"state_dict": module.state_dict(), "hyper_parameters": {"config": module.config}}, path
    )
    return path


def _graph(atoms=None) -> AtomicGraph:
    return AtomicGraph.from_ase(atoms or molecule("CH3OH"), cutoff=4.0, dtype=torch.float64)


def _predict(model, graph):
    with torch.enable_grad():
        return model(graph)


class TestConfig:
    def test_defaults(self) -> None:
        cfg = SIMURGHLoRAConfig()
        assert (cfg.rank, cfg.alpha, cfg.dropout, cfg.bias) == (8, 16.0, 0.0, "none")
        assert cfg.target_modules == ["artisan"] and cfg.head_trainable is True

    def test_asdict_round_trip(self) -> None:
        cfg = SIMURGHLoRAConfig(
            rank=4, target_modules=["artisan", "fragment_interaction"], checkpoint_path="x"
        )
        assert SIMURGHLoRAConfig(**dataclasses.asdict(cfg)) == cfg

    def test_invalid_values(self) -> None:
        with pytest.raises(ValueError, match="bias"):
            SIMURGHLoRAConfig(bias="some")
        with pytest.raises(ValueError, match="rank"):
            SIMURGHLoRAConfig(rank=0)


def test_missing_peft_gives_install_instruction(monkeypatch) -> None:
    real_import = builtins.__import__

    def _no_peft(name, *args, **kwargs):
        if name == "peft" or name.startswith("peft."):
            raise ImportError("No module named 'peft'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_peft)
    with pytest.raises(ImportError, match=r"pip install 'peft>=0.13'"):
        SIMURGHLoRAAdapter(SIMURGHLoRAConfig(), model=_base_module())


class TestAdapter:
    def test_starts_as_an_exact_no_op(self) -> None:
        base = _base_module().eval()
        graph = _graph()
        expected = _predict(base, graph)
        adapter = SIMURGHLoRAAdapter(SIMURGHLoRAConfig(), model=_base_module()).eval()
        got = _predict(adapter, _graph())
        torch.testing.assert_close(got["energy"], expected["energy"])
        torch.testing.assert_close(got["forces"], expected["forces"])

    def test_only_adapters_and_head_are_trainable(self) -> None:
        adapter = SIMURGHLoRAAdapter(SIMURGHLoRAConfig(), model=_base_module())
        trainable = [n for n, p in adapter.named_parameters() if p.requires_grad]
        assert trainable
        assert all(
            "lora_" in n or any(h in n for h in ("head.", "energy_mlp", "atomic_energies"))
            for n in trainable
        )
        assert 0 < adapter.trainable_parameters() < adapter.total_parameters()
        assert all("artisan" in name for name in adapter.adapted_layers())
        # head layers are trained in full, not adapted
        assert not any("energy_mlp" in name for name in adapter.adapted_layers())

    def test_head_frozen_when_requested(self) -> None:
        adapter = SIMURGHLoRAAdapter(SIMURGHLoRAConfig(head_trainable=False), model=_base_module())
        trainable = [n for n, p in adapter.named_parameters() if p.requires_grad]
        assert trainable and all("lora_" in n for n in trainable)
        assert any("energy_mlp" in name for name in adapter.adapted_layers())

    def test_no_matching_layers_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="No nn.Linear matches"):
            SIMURGHLoRAAdapter(
                SIMURGHLoRAConfig(target_modules=["does_not_exist"]), model=_base_module()
            )

    def test_equivariance_survives_nonzero_adapters(self) -> None:
        """LoRA only touches invariant linears: energies stay invariant and
        forces rotate with the molecule."""
        from scipy.spatial.transform import Rotation

        adapter = SIMURGHLoRAAdapter(
            SIMURGHLoRAConfig(head_trainable=False), model=_base_module()
        ).eval()
        torch.manual_seed(1)
        with torch.no_grad():
            for name, p in adapter.named_parameters():
                if "lora_B" in name:
                    p.normal_(std=0.3)
        atoms = molecule("CH3OH")
        rot = Rotation.from_euler("xyz", [0.3, -1.1, 2.0]).as_matrix()
        rotated = atoms.copy()
        rotated.positions = atoms.positions @ rot.T
        a, b = _predict(adapter, _graph(atoms)), _predict(adapter, _graph(rotated))
        torch.testing.assert_close(a["energy"], b["energy"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            a["forces"].detach().numpy() @ rot.T, b["forces"].detach().numpy(), atol=1e-8
        )

    def test_one_training_step_moves_only_trainable_weights(self) -> None:
        adapter = SIMURGHLoRAAdapter(SIMURGHLoRAConfig(), model=_base_module()).train()
        frozen = {
            n: p.detach().clone() for n, p in adapter.named_parameters() if not p.requires_grad
        }
        opt = torch.optim.SGD([p for p in adapter.parameters() if p.requires_grad], lr=0.1)
        out = _predict(adapter, _graph())
        (out["energy"].sum() + (out["forces"] ** 2).sum()).backward()
        opt.step()
        for n, p in adapter.named_parameters():
            if n in frozen:
                assert torch.equal(p.detach(), frozen[n]), n

    def test_save_and_load_the_delta(self, tmp_path: Path) -> None:
        adapter = SIMURGHLoRAAdapter(SIMURGHLoRAConfig(), model=_base_module()).eval()
        with torch.no_grad():
            for name, p in adapter.named_parameters():
                if p.requires_grad:
                    p.add_(0.05)
        expected = _predict(adapter, _graph())["energy"]
        path = adapter.save(tmp_path / "adapter")
        assert path.suffix == LORA_SUFFIX and path.is_file()
        delta = torch.load(path, weights_only=False)["state_dict"]
        assert sum(t.numel() for t in delta.values()) == adapter.trainable_parameters()

        restored = SIMURGHLoRAAdapter.load(path, model=_base_module()).eval()
        torch.testing.assert_close(_predict(restored, _graph())["energy"], expected)

    def test_merge_keeps_predictions(self) -> None:
        adapter = SIMURGHLoRAAdapter(
            SIMURGHLoRAConfig(head_trainable=False), model=_base_module()
        ).eval()
        with torch.no_grad():
            for name, p in adapter.named_parameters():
                if "lora_B" in name:
                    p.normal_(std=0.1)
        before = _predict(adapter, _graph())["energy"]
        adapter.merge()
        torch.testing.assert_close(_predict(adapter, _graph())["energy"], before)


class TestFromCheckpoint:
    def test_loads_base_checkpoint(self, base_ckpt: Path) -> None:
        adapter = SIMURGHLoRAAdapter(SIMURGHLoRAConfig(checkpoint_path=str(base_ckpt)))
        assert adapter.cutoff == 4.0
        assert adapter.trainable_parameters() > 0

    def test_missing_base_checkpoint(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="must stay reachable"):
            SIMURGHLoRAAdapter(SIMURGHLoRAConfig(checkpoint_path=str(tmp_path / "nope.ckpt")))

    def test_registry_backbone_checks_the_cutoff(self, base_ckpt: Path) -> None:
        import goal.ml.adapters  # noqa: F401 — registers simurgh_lora
        from goal.ml.registry import BACKBONE_REGISTRY

        cls = BACKBONE_REGISTRY.get("simurgh_lora")
        model = cls(checkpoint_path=str(base_ckpt), cutoff=4.0, rank=4)
        assert model.output_keys
        with pytest.raises(ValueError, match="data.cutoff=5.0"):
            cls(checkpoint_path=str(base_ckpt), cutoff=5.0)
        with pytest.raises(TypeError, match="unknown arguments"):
            cls(checkpoint_path=str(base_ckpt), rnak=4)

    def test_finetuned_checkpoint_loads_in_goal_calculator(
        self, base_ckpt: Path, tmp_path: Path
    ) -> None:
        """A GOALModule around simurgh_lora saves and reloads like any model."""
        import goal.ml.adapters  # noqa: F401
        from goal.ml.registry import BACKBONE_REGISTRY
        from goal.ml.training.loss import CompositeLoss
        from goal.ml.training.module import GOALModule
        from goal.ml.utils.calculator import GOALCalculator

        cfg = OmegaConf.create(
            {
                "model": {
                    "backbone": {
                        "name": "simurgh_lora",
                        "checkpoint_path": str(base_ckpt),
                        "rank": 4,
                    },
                    "head": None,
                },
                "training": {"losses": [], "ema": {"enabled": False}},
                "data": {"cutoff": 4.0},
            }
        )
        backbone = BACKBONE_REGISTRY.get("simurgh_lora")(checkpoint_path=str(base_ckpt), rank=4)
        module = GOALModule(backbone=backbone, head=None, loss=CompositeLoss([]), config=cfg)
        with torch.no_grad():
            for name, p in module.named_parameters():
                if p.requires_grad:
                    p.add_(0.02)
        ckpt = tmp_path / "finetuned.ckpt"
        torch.save({"state_dict": module.state_dict(), "hyper_parameters": {"config": cfg}}, ckpt)

        atoms = molecule("CH3OH")
        atoms.calc = GOALCalculator(checkpoint_path=str(ckpt))
        expected = float(_predict(module.eval(), _graph())["energy"])
        assert atoms.get_potential_energy() == pytest.approx(expected, rel=1e-10)


@pytest.mark.slow
def test_goal_finetune_lora_end_to_end(base_ckpt: Path, tmp_path: Path) -> None:
    """``goal-finetune`` with the default ``finetune/simurgh_lora`` config,
    2 epochs on a tiny dataset, writes a loadable checkpoint."""
    from ase.calculators.emt import EMT
    from ase.io import write

    rng = np.random.default_rng(0)
    frames = []
    for _ in range(6):
        atoms = molecule("CH3OH")
        atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
        atoms.calc = EMT()
        atoms.info["energy"] = atoms.get_potential_energy()
        atoms.arrays["forces"] = atoms.get_forces()
        atoms.calc = None
        frames.append(atoms)
    data = tmp_path / "data.xyz"
    write(data, frames, format="extxyz")

    run_dir = tmp_path / "run"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "goal.ml.cli.finetune",
            f"model.backbone.checkpoint_path={base_ckpt}",
            "model.backbone.rank=4",
            f"data.train_paths=[{data}]",
            f"data.val_paths=[{data}]",
            "data.cutoff=4.0",
            "data.batch_size=3",
            "trainer.max_epochs=2",
            "trainer.accelerator=cpu",
            "trainer.precision=64",
            "trainer.num_sanity_val_steps=0",
            f"hydra.run.dir={run_dir}",
            "~callbacks",
            "~logger",
        ],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    ckpts = list(run_dir.rglob("*.ckpt"))
    assert ckpts, f"no checkpoint written under {run_dir}"
