"""nvalchemi integration: model wrapper, collate functions, fallbacks.

The ``AtomicData`` → GOAL batch mapping is checked here on any hardware
with a stand-in data object; the tests marked ``gpu`` run the real
nvalchemi engine and are skipped unless nvalchemi-toolkit and an NVIDIA
GPU are present.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import bulk, molecule
from torch_geometric.data import Batch

from goal.ml.adapters import nvalchemi_wrapper as wrapper_mod
from goal.ml.adapters.nvalchemi_wrapper import (
    INSTALL_HINT,
    SIMURGHNvalchemiWrapper,
    atomic_data_to_goal_batch,
    goal_outputs_to_model_outputs,
)
from goal.ml.data.collate import (
    batch_from_tensors,
    goal_collate_fn,
    nvalchemi_collate_fn,
)
from goal.ml.data.graph import AtomicGraph
from goal.ml.utils import hardware
from goal.ml.utils.calculator import GOALCalculator
from tests.ml.test_archive import _build_module, _tiny_config

STRUCTURES = [molecule("CH4"), molecule("C2H6"), molecule("C2H2")]
PERIODIC = [bulk("C", "diamond", a=3.57), bulk("C", "diamond", a=3.6) * (2, 1, 1)]


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    cfg = _tiny_config()
    module = _build_module(cfg)
    path = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": {"config": cfg}}, path)
    return path


def _fake_atomic_data(structures, periodic: bool):
    """Stand-in with nvalchemi Batch's field names and shapes."""
    positions = torch.tensor(np.concatenate([a.positions for a in structures]))
    numbers = torch.tensor(np.concatenate([a.numbers for a in structures]), dtype=torch.int32)
    batch_idx = torch.cat(
        [torch.full((len(a),), i, dtype=torch.int32) for i, a in enumerate(structures)]
    )
    data = types.SimpleNamespace(positions=positions, atomic_numbers=numbers, batch_idx=batch_idx)
    if periodic:
        data.cell = torch.tensor(np.stack([np.array(a.cell) for a in structures]))
        data.pbc = torch.tensor(np.stack([a.pbc for a in structures]))
    return data


class TestWithoutNvalchemi:
    def test_module_imports_everywhere(self) -> None:
        assert SIMURGHNvalchemiWrapper.__name__ == "SIMURGHNvalchemiWrapper"

    def test_construction_raises_with_instructions(self, monkeypatch, checkpoint) -> None:
        monkeypatch.setattr(wrapper_mod, "nvalchemi_available", lambda: False)
        with pytest.raises(RuntimeError, match="nvalchemi-toolkit") as exc:
            SIMURGHNvalchemiWrapper(checkpoint)
        assert "git+https://github.com/NVIDIA/nvalchemi-toolkit" in str(exc.value)
        assert "GOALCalculator" in INSTALL_HINT

    def test_collate_falls_back_to_pyg(self, monkeypatch) -> None:
        monkeypatch.setattr(hardware, "nvalchemi_available", lambda: False)
        graphs = [AtomicGraph.from_ase(a, cutoff=4.0) for a in STRUCTURES]
        batch = nvalchemi_collate_fn(graphs)
        assert isinstance(batch, Batch) and batch.num_graphs == 3


class TestFieldMapping:
    """``AtomicData`` → GOAL batch must reproduce GOALCalculator exactly."""

    @pytest.mark.parametrize("periodic", [False, True])
    def test_energies_and_forces_match_goal_calculator(self, checkpoint, periodic: bool) -> None:
        structures = PERIODIC if periodic else STRUCTURES
        calc = GOALCalculator(checkpoint_path=str(checkpoint))
        goal_batch, positions = atomic_data_to_goal_batch(
            _fake_atomic_data(structures, periodic),
            calc.cutoff,
            calc.dtype,
            neighbor_backend="torch",
        )
        with torch.enable_grad():
            out = calc.module(goal_batch)
        model_out = goal_outputs_to_model_outputs(out, {"energy", "forces"})
        assert model_out["energy"].shape == (len(structures), 1)
        assert model_out["forces"].shape == positions.shape

        for i, atoms in enumerate(structures):
            atoms = atoms.copy()
            atoms.calc = calc
            assert float(model_out["energy"][i]) == pytest.approx(
                atoms.get_potential_energy(), abs=1e-10
            )
        ref_forces = np.concatenate([_forces(calc, a) for a in structures])
        np.testing.assert_allclose(model_out["forces"].detach().numpy(), ref_forces, atol=1e-10)


def _forces(calc, atoms):
    atoms = atoms.copy()
    atoms.calc = calc
    return atoms.get_forces()


class TestBatchFromTensors:
    def test_same_graphs_as_from_ase(self) -> None:
        data = _fake_atomic_data(PERIODIC, periodic=True)
        batch = batch_from_tensors(
            data.positions, data.atomic_numbers, 4.0, data.batch_idx, data.cell, data.pbc, "torch"
        )
        reference = goal_collate_fn([AtomicGraph.from_ase(a, cutoff=4.0) for a in PERIODIC])
        assert batch.num_graphs == reference.num_graphs

        def key(b):
            return sorted(zip(*b.edge_index.tolist(), *b.unit_shifts.T.tolist()))

        assert key(batch) == key(reference)

    def test_positions_keep_their_autograd_link(self) -> None:
        pos = torch.tensor(STRUCTURES[0].positions, requires_grad=True)
        batch = batch_from_tensors(
            pos, torch.tensor(STRUCTURES[0].numbers), 4.0, neighbor_backend="torch"
        )
        batch.pos.sum().backward()
        assert pos.grad is not None and torch.all(pos.grad == 1)


@pytest.mark.gpu
class TestRealNvalchemi:
    @pytest.fixture(autouse=True)
    def _require(self):
        hardware.clear_cache()
        if not hardware.nvalchemi_available():
            pytest.skip("needs nvalchemi-toolkit and an NVIDIA GPU")

    def _batch(self, structures):
        from nvalchemi.data import AtomicData
        from nvalchemi.data import Batch as NvBatch

        datas = []
        for a in structures:
            d = AtomicData.from_atoms(a, device="cuda", dtype=torch.float64)
            d.forces = torch.zeros(len(a), 3, device="cuda", dtype=torch.float64)
            d.energy = torch.zeros(1, 1, device="cuda", dtype=torch.float64)
            datas.append(d)
        return NvBatch.from_data_list(datas, device=torch.device("cuda"))

    def test_wrapper_matches_goal_calculator(self, checkpoint) -> None:
        from nvalchemi.models.base import BaseModelMixin

        model = SIMURGHNvalchemiWrapper(checkpoint, device="cuda")
        assert isinstance(model, BaseModelMixin) and isinstance(model, SIMURGHNvalchemiWrapper)
        out = model(self._batch(STRUCTURES))
        calc = GOALCalculator(checkpoint_path=str(checkpoint))
        energies = [_energy(calc, a) for a in STRUCTURES]
        np.testing.assert_allclose(out["energy"].cpu().numpy().ravel(), energies, atol=1e-9)

    def test_nvt_langevin_runs(self, checkpoint) -> None:
        from nvalchemi.dynamics import NVTLangevin
        from nvalchemi.dynamics.hooks import NaNDetectorHook

        model = SIMURGHNvalchemiWrapper(checkpoint, device="cuda")
        nvt = NVTLangevin(
            model=model,
            dt=0.25,
            temperature=300.0,
            friction=0.05,
            n_steps=10,
            hooks=[NaNDetectorHook()],
        )
        result = nvt.run(self._batch(STRUCTURES[:2]))
        assert torch.isfinite(result.positions).all()

    def test_collate_returns_an_nvalchemi_batch(self) -> None:
        from nvalchemi.data import Batch as NvBatch

        batch = nvalchemi_collate_fn([AtomicGraph.from_ase(a, cutoff=4.0) for a in STRUCTURES])
        assert isinstance(batch, NvBatch) and batch.num_graphs == 3


def _energy(calc, atoms) -> float:
    atoms = atoms.copy()
    atoms.calc = calc
    return atoms.get_potential_energy()
