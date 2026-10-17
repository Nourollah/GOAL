"""Batched simulation backend (goal.md.core.batched) and goal-simulate-batch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import bulk, molecule
from ase.io import read, write

from goal.md.core import batched
from goal.md.core.batched import BatchSimulationConfig, run_batched_simulation
from goal.ml.utils import hardware


@pytest.fixture()
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    from tests.ml.test_archive import _build_module, _tiny_config

    cfg = _tiny_config()
    module = _build_module(cfg)
    ckpt = tmp_path / "m.ckpt"
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": {"config": cfg}}, ckpt)
    frames = tmp_path / "frames.xyz"
    write(
        frames, [molecule("CH4"), molecule("C2H6"), bulk("C", "diamond", a=3.57)], format="extxyz"
    )
    return ckpt, frames


def _config(inputs, tmp_path: Path, **kwargs) -> BatchSimulationConfig:
    ckpt, frames = inputs
    base = {
        "checkpoint": str(ckpt),
        "structures": [str(frames)],
        "steps": 20,
        "log_every": 5,
        "timestep_fs": 0.25,
        "backend": "ase",
        "device": "cpu",
        "output_dir": str(tmp_path / "out"),
    }
    base.update(kwargs)
    return BatchSimulationConfig(**base)


class TestBackendResolution:
    @pytest.fixture(autouse=True)
    def _clear(self):
        hardware.clear_cache()
        yield
        hardware.clear_cache()

    def test_auto_falls_back_to_ase(self, monkeypatch) -> None:
        monkeypatch.setattr(hardware, "nvalchemi_available", lambda: False)
        assert batched.resolve_backend("auto") == "ase"

    def test_auto_prefers_nvalchemi(self, monkeypatch) -> None:
        monkeypatch.setattr(hardware, "nvalchemi_available", lambda: True)
        assert batched.resolve_backend("auto") == "nvalchemi"

    def test_explicit_nvalchemi_degrades(self, monkeypatch) -> None:
        monkeypatch.setattr(hardware, "nvalchemi_available", lambda: False)
        assert batched.resolve_backend("nvalchemi") == "ase"


class TestASEBackend:
    def test_md_writes_trajectories_and_final_structures(self, inputs, tmp_path) -> None:
        result = run_batched_simulation(_config(inputs, tmp_path))
        assert result.backend == "ase" and len(result.final_structures) == 3
        assert all(np.isfinite(result.final_energies))
        traj = read(tmp_path / "out" / "traj_0002.extxyz", index=":")
        assert [a.info["step"] for a in traj] == [5, 10, 15, 20]
        assert len(read(tmp_path / "out" / "final.extxyz", index=":")) == 3

    def test_timestep_is_in_femtoseconds(self, inputs, tmp_path, monkeypatch) -> None:
        """The batch backend hands ASE ``timestep_fs * units.fs`` — the old
        ``langevin_ase`` builder passed the raw number (≈ 10 fs per 1.0)."""
        from ase import units
        from ase.md import langevin

        seen = {}
        real = langevin.Langevin.__init__

        def _spy(self, atoms, timestep, *args, **kwargs):
            seen["dt"] = timestep
            real(self, atoms, timestep, *args, **kwargs)

        monkeypatch.setattr(langevin.Langevin, "__init__", _spy)
        run_batched_simulation(_config(inputs, tmp_path, steps=1, log_every=1))
        assert seen["dt"] == pytest.approx(0.25 * units.fs)

    def test_relax_lowers_the_forces(self, inputs, tmp_path) -> None:
        ckpt, frames = inputs
        start = read(frames, index=0)
        result = run_batched_simulation(
            _config(inputs, tmp_path, mode="relax", steps=200, fmax=0.01)
        )
        from goal.ml.utils.calculator import GOALCalculator

        start.calc = GOALCalculator(checkpoint_path=str(ckpt))
        f0 = np.abs(start.get_forces()).max()
        f1 = np.abs(result.final_structures[0].arrays["forces"]).max()
        assert f1 < f0

    def test_zarr_trajectory(self, inputs, tmp_path) -> None:
        pytest.importorskip("zarr", minversion="3")
        result = run_batched_simulation(_config(inputs, tmp_path, trajectory_format="zarr"))
        from goal.ml.data.zarr_dataset import ZarrDataset

        store = next(f for f in result.output_files if f.endswith(".zarr"))
        ds = ZarrDataset(root=store, cutoff=5.0, split="trajectory")
        assert len(ds) == 3 * 4
        # ordered by structure, then step, with both recorded per frame
        info = [ds.structure_info(i) for i in range(len(ds))]
        assert [(d["structure_index"], d["step"]) for d in info] == [
            (s, step) for s in range(3) for step in (5, 10, 15, 20)
        ]

    def test_steps_completed_is_the_real_count(self, inputs, tmp_path) -> None:
        result = run_batched_simulation(
            _config(inputs, tmp_path, mode="relax", steps=500, fmax=0.5)
        )
        assert result.steps_completed < 500

    def test_auto_falls_back_to_ase_when_nvalchemi_cannot_run(
        self, inputs, tmp_path, monkeypatch
    ) -> None:
        def _refuse(*args, **kwargs):
            raise NotImplementedError("fragment-aware checkpoint")

        monkeypatch.setattr(batched, "resolve_backend", lambda requested: "nvalchemi")
        monkeypatch.setattr(batched, "_run_nvalchemi", _refuse)
        result = run_batched_simulation(
            _config(inputs, tmp_path, backend="auto", steps=2, log_every=1)
        )
        assert result.backend == "ase"
        with pytest.raises(ValueError, match="cannot run this checkpoint"):
            run_batched_simulation(_config(inputs, tmp_path, backend="nvalchemi", steps=2))


class TestValidationAndStubs:
    @pytest.mark.parametrize("sampler", ["metadynamics", "remd"])
    def test_roadmap_samplers_raise(self, inputs, tmp_path, sampler) -> None:
        with pytest.raises(NotImplementedError, match="roadmap"):
            run_batched_simulation(_config(inputs, tmp_path, sampler=sampler))

    def test_bad_values(self, inputs, tmp_path) -> None:
        with pytest.raises(ValueError, match="mode"):
            run_batched_simulation(_config(inputs, tmp_path, mode="npt"))
        with pytest.raises(ValueError, match="checkpoint"):
            run_batched_simulation(_config(inputs, tmp_path, checkpoint=""))

    def test_enhanced_sampling_stubs(self) -> None:
        from goal.ml.utils.enhanced_sampling import (
            UPSTREAM_URL,
            run_metadynamics,
            run_remd,
        )

        for fn in (run_metadynamics, run_remd):
            with pytest.raises(NotImplementedError, match=UPSTREAM_URL):
                fn()


class TestCLI:
    def test_runs(self, inputs, tmp_path, capsys) -> None:
        from goal.md.cli.simulate_batch import main

        ckpt, frames = inputs
        rc = main(
            [
                "--checkpoint",
                str(ckpt),
                "--structures",
                str(frames),
                "--steps",
                "4",
                "--log-every",
                "2",
                "--backend",
                "ase",
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path / "cli"),
            ]
        )
        assert rc == 0
        assert "backend=ase" in capsys.readouterr().out

    def test_sampler_stub_is_a_clean_error(self, inputs, tmp_path, capsys) -> None:
        from goal.md.cli.simulate_batch import main

        ckpt, frames = inputs
        rc = main(
            ["--checkpoint", str(ckpt), "--structures", str(frames), "--sampler", "metadynamics"]
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "not implemented yet" in err and "Traceback" not in err

    def test_yaml_config_with_flag_override(self, inputs, tmp_path) -> None:
        from goal import CONFIGS_DIR
        from goal.md.cli.simulate_batch import build_config

        cfg = build_config(
            [
                "--config",
                f"{CONFIGS_DIR}/md/simulations/batch.yaml",
                "--steps",
                "7",
                "--temperature",
                "500",
            ]
        )
        assert (cfg.steps, cfg.temperature_K, cfg.timestep_fs) == (7, 500.0, 0.5)


def test_distill_stub(capsys) -> None:
    from goal.ml.cli.distill import main, run_distillation

    with pytest.raises(NotImplementedError):
        run_distillation()
    assert main([]) == 1
    assert "roadmap" in capsys.readouterr().err
