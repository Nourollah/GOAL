"""Tests for the goal.md Simulation orchestration layer.

Covers:
- Simulation construction and attribute access
- Observer setup (trajectory, log)
- Running a simulation and receiving a RunResult
- SimulationConfig defaults
- Observers module helpers
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _water() -> Atoms:
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.calc = EMT()
    return atoms


# ---------------------------------------------------------------------------
# SimulationConfig
# ---------------------------------------------------------------------------


class TestSimulationConfig:
    def test_defaults(self) -> None:
        """Verify SimulationConfig initializes with expected defaults."""
        from goal.md.core.simulation import SimulationConfig

        cfg = SimulationConfig()
        assert cfg.steps == 1000
        assert cfg.chunk_size == 100
        assert cfg.log_interval == 10
        # temperature and timestep belong to the dynamics section now
        assert not hasattr(cfg, "temperature_K") and not hasattr(cfg, "timestep_fs")


# ---------------------------------------------------------------------------
# Simulation construction
# ---------------------------------------------------------------------------


class TestSimulationConstruction:
    def test_basic_construction(self) -> None:
        """Test Simulation initialization with atoms and calculator."""
        from goal.md.core.simulation import Simulation

        atoms = _water()
        calc = EMT()
        sim = Simulation(atoms, calc, steps=10, temperature_K=300.0)

        assert sim.atoms is atoms
        assert sim.steps == 10
        assert sim.temperature_K == 300.0
        assert sim.calculator is calc

    def test_calculator_attached_to_atoms(self) -> None:
        """Verify calculator is properly attached to atoms during Simulation init."""
        from goal.md.core.simulation import Simulation

        atoms = _water()
        calc = EMT()
        sim = Simulation(atoms, calc, steps=5)
        assert atoms.calc is calc

    def test_dynamics_property_lazy(self) -> None:
        """Test that dynamics property is lazily created on first access."""
        from goal.md.core.simulation import Simulation

        atoms = _water()
        sim = Simulation(atoms, EMT(), steps=5)
        assert sim._dynamics is None
        dyn = sim.dynamics  # triggers lazy creation
        assert dyn is not None


# ---------------------------------------------------------------------------
# Running a simulation
# ---------------------------------------------------------------------------


class TestSimulationRun:
    def test_run_returns_result(self) -> None:
        """Verify sim.run() returns a RunResult with expected properties."""
        from goal.md.core.runner import RunResult
        from goal.md.core.simulation import Simulation

        atoms = _water()
        sim = Simulation(atoms, EMT(), steps=5, show_progress=False)
        result = sim.run()

        assert isinstance(result, RunResult)
        assert result.steps_completed == 5
        assert result.final_energy is not None
        assert isinstance(result.final_energy, float)

    def test_trajectory_written(self, tmp_path: Path) -> None:
        """Test that trajectory file is written when trajectory_file is specified."""
        from goal.md.core.simulation import Simulation

        traj_path = str(tmp_path / "test.traj")
        atoms = _water()
        sim = Simulation(
            atoms,
            EMT(),
            steps=10,
            log_interval=5,
            trajectory_file=traj_path,
            output_dir=str(tmp_path),
            show_progress=False,
        )
        sim.run()
        assert Path(traj_path).exists()

    def test_log_file_written(self, tmp_path: Path) -> None:
        """Test that log file is written with simulation data."""
        from goal.md.core.simulation import Simulation

        log_path = str(tmp_path / "md.log")
        atoms = _water()
        sim = Simulation(
            atoms,
            EMT(),
            steps=10,
            log_interval=5,
            log_file=log_path,
            output_dir=str(tmp_path),
            show_progress=False,
        )
        sim.run()
        assert Path(log_path).exists()
        content = Path(log_path).read_text()
        assert len(content) > 0

    def test_energy_history_collected(self) -> None:
        """Verify energy history is collected during simulation run."""
        from goal.md.core.simulation import Simulation

        atoms = _water()
        sim = Simulation(atoms, EMT(), steps=20, log_interval=5, show_progress=False)
        result = sim.run()
        # Should have ~4 energy samples (20 steps / 5 interval)
        assert len(result.energy_history) > 0

    def test_final_temperature_reported(self) -> None:
        """Test that final temperature is properly reported in RunResult."""
        from goal.md.core.simulation import Simulation

        atoms = _water()
        sim = Simulation(atoms, EMT(), steps=5, temperature_K=500.0, show_progress=False)
        result = sim.run()
        assert result.final_temperature is not None
        assert result.final_temperature > 0


# ---------------------------------------------------------------------------
# Observers
# ---------------------------------------------------------------------------


class TestObservers:
    def test_trajectory_observer(self, tmp_path: Path) -> None:
        """Test TrajectoryObserver writes frames to trajectory file."""
        from ase import units
        from ase.md.langevin import Langevin

        from goal.md.core.observers import TrajectoryObserver

        atoms = _water()
        dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01)
        obs = TrajectoryObserver(atoms, tmp_path / "test.traj")
        dyn.attach(obs, interval=1)
        dyn.run(3)
        # Read back
        from ase.io import read

        frames = read(str(tmp_path / "test.traj"), index=":")
        assert len(frames) > 0

    def test_setup_observers_returns_dict(self, tmp_path: Path) -> None:
        """Test setup_observers returns dict with trajectory, log, and energy_collector."""
        from ase import units
        from ase.md.langevin import Langevin

        from goal.md.core.observers import setup_observers

        atoms = _water()
        dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01)
        obs = setup_observers(
            dynamics=dyn,
            atoms=atoms,
            trajectory_file=str(tmp_path / "out.traj"),
            log_file=str(tmp_path / "md.log"),
            log_interval=1,
            collect_energies=True,
        )
        assert "trajectory" in obs
        assert "log" in obs
        assert "energy_collector" in obs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestRunner:
    def test_run_dynamics_basic(self) -> None:
        """Test run_dynamics executes requested number of steps."""
        from ase import units
        from ase.md.langevin import Langevin

        from goal.md.core.runner import run_dynamics

        atoms = _water()
        dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01)
        result = run_dynamics(dyn, steps=10, chunk_size=5, progress=False)
        assert result.steps_completed == 10

    def test_run_dynamics_energy_collector(self) -> None:
        """Test run_dynamics collects energy history when collector is attached."""
        from ase import units
        from ase.md.langevin import Langevin

        from goal.md.core.observers import EnergyCollector
        from goal.md.core.runner import run_dynamics

        atoms = _water()
        dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01)
        collector = EnergyCollector(atoms)
        dyn.attach(collector, interval=1)
        result = run_dynamics(
            dyn, steps=10, chunk_size=5, progress=False, energy_collector=collector
        )
        assert len(result.energy_history) > 0

    def test_run_dynamics_thermostat_restart(self) -> None:
        """Test thermostat restart functionality during dynamics."""
        from ase import units
        from ase.md.langevin import Langevin

        from goal.md.core.runner import run_dynamics

        atoms = _water()
        dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01)
        result = run_dynamics(
            dyn,
            steps=20,
            chunk_size=5,
            restart_thermostat=True,
            temperature_K=300.0,
            progress=False,
        )
        assert result.steps_completed == 20
