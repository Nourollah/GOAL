"""MD stability and drift tooling: NVE builder, bond-deviation detection, summaries."""

from __future__ import annotations

import importlib.util
import json
import typing
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.md import MDLogger

from goal.md.core.batched import (
    BatchSimulationConfig,
    _StabilityTracker,
    run_batched_simulation,
)
from goal.md.core.md_factory import DynamicsFactory
from goal.md.core.observers import BondDeviationObserver
from tests.md.test_batched_simulation import inputs  # noqa: F401  (fixture)

_SCRIPT = Path(__file__).parents[2] / "scripts" / "simurgh_dev" / "md_stability.py"
_spec = importlib.util.spec_from_file_location("md_stability", _SCRIPT)
assert _spec is not None and _spec.loader is not None
md_stability = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(md_stability)


class _Spring(Calculator):
    """E = k/2 (r - r0)^2 between atoms 0 and 1."""

    implemented_properties: typing.ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, k: float = 1.0, r0: float = 1.5) -> None:
        super().__init__()
        self.k, self.r0 = k, r0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        d = self.atoms.positions[1] - self.atoms.positions[0]
        r = float(np.linalg.norm(d))
        f = self.k * (r - self.r0) * d / r
        self.results = {"energy": 0.5 * self.k * (r - self.r0) ** 2, "forces": np.array([f, -f])}


def test_velocity_verlet_conserves_energy(tmp_path: Path) -> None:
    atoms = Atoms("C2", positions=[[0, 0, 0], [1.55, 0, 0]])
    atoms.calc = _Spring()
    dyn = DynamicsFactory.create("velocity_verlet", atoms, timestep_fs=0.5, temperature_K=300.0)
    totals: list[float] = []
    dyn.attach(lambda: totals.append(atoms.get_total_energy()), interval=100)
    log = tmp_path / "nve.log"
    dyn.attach(MDLogger(dyn, atoms, str(log), header=True, peratom=True, mode="w"), interval=100)
    dyn.run(20_000)  # 10 ps
    assert np.ptp(totals) / len(atoms) < 1e-6
    assert abs(md_stability.nve_drift(log)) < 1e-3  # meV/atom/ps


def test_velocity_verlet_rejects_two_timesteps() -> None:
    with pytest.raises(ValueError, match="not both"):
        DynamicsFactory.create("velocity_verlet", molecule("H2"), timestep=0.1, timestep_fs=1.0)


class _Clock:
    nsteps = 0


def _stretched(atoms: Atoms, i: int, by: float) -> np.ndarray:
    """Positions with atom ``i`` moved ``by`` Å away from its nearest neighbour."""
    d = atoms.get_distances(i, range(len(atoms)))
    d[i] = np.inf
    j = int(np.argmin(d))
    axis = (atoms.positions[i] - atoms.positions[j]) / d[j]
    positions = atoms.positions.copy()
    positions[i] += by * axis
    return positions


def test_bond_break_is_detected_at_the_right_step() -> None:
    atoms = molecule("CH3CH2OH")
    moving, clock = atoms.copy(), _Clock()
    observer = BondDeviationObserver(moving, dynamics=clock)
    rng = np.random.default_rng(0)
    for step in range(10):
        clock.nsteps = step
        base = _stretched(atoms, len(atoms) - 1, 0.8) if step >= 6 else atoms.positions
        moving.positions = base + rng.normal(scale=0.03, size=(len(atoms), 3))
        observer()
    assert observer.onset_step == 6


def test_nonbonded_clash_is_detected() -> None:
    atoms = molecule("CH4")
    observer = BondDeviationObserver(atoms)
    squeezed = atoms.positions.copy()
    squeezed[2] = squeezed[1] + 0.6 * (squeezed[2] - squeezed[1]) / np.linalg.norm(
        squeezed[2] - squeezed[1]
    )  # two H atoms 0.6 Å apart, C-H bond lengths change by < 0.5 Å
    assert not observer.check(atoms.positions)
    assert observer.check(squeezed)


def test_batched_run_records_stability(inputs, tmp_path: Path) -> None:  # noqa: F811
    ckpt, frames = inputs
    config = BatchSimulationConfig(
        checkpoint=str(ckpt),
        structures=[str(frames)],
        steps=10,
        log_every=5,
        backend="ase",
        write_trajectory=False,
        output_dir=str(tmp_path / "out"),
    )
    result = run_batched_simulation(config)
    saved = json.loads((tmp_path / "out" / "stability.json").read_text())
    assert saved["structures"] == result.stability
    assert len(result.stability) == len(result.final_structures)
    assert all(np.isfinite(s["mean_temperature_K"]) for s in result.stability)


def test_tracker_onset_time_from_logged_frames() -> None:
    frames = [molecule("CH4"), molecule("H2O")]
    tracker = _StabilityTracker(frames, BatchSimulationConfig(timestep_fs=0.5))
    broken = frames[1].copy()
    broken.positions = _stretched(frames[1], 1, 1.0)
    tracker.add(0, frames[0], 100)
    tracker.add(1, frames[1], 100)
    tracker.add(1, broken, 200)
    first, second = tracker.summary()
    assert first["unstable_step"] is None and first["onset_ps"] is None
    assert second["unstable_step"] == 200 and second["onset_ps"] == pytest.approx(0.1)


def test_survival_summary_and_md17_temperature() -> None:
    stability = {
        "steps": 200_000,
        "timestep_fs": 0.5,
        "structures": [
            {"onset_ps": None, "mean_temperature_K": 550.0},
            {"onset_ps": 10.0, "mean_temperature_K": 560.0},
            {"onset_ps": 30.0, "mean_temperature_K": 555.0},
            {"onset_ps": None, "mean_temperature_K": 551.0},
        ],
    }
    s = md_stability.survival_summary(stability)
    assert s["survival_fraction"] == 0.5 and s["run_ps"] == 100.0
    assert s["onset_median_ps"] == 20.0 and s["onset_iqr_ps"] == 10.0
    assert md_stability.effective_md17_temperature(21) == pytest.approx(552.6, abs=0.05)
    assert md_stability.effective_md17_temperature(9) == pytest.approx(642.9, abs=0.05)
