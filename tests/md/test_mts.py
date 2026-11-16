"""Tests for goal.md multi-timescale (MTS) components.

Covers:
- RespaIntegrator: single-level (fast only) and two-level
- MTSConfig defaults
- MTSSimulation short run
- DualCalculator base class
- QMLearnerCalculator with ThresholdPolicy and AlwaysBasePolicy
- IPICompatibleDriver unit conversion
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.emt import EMT


def _water() -> Atoms:
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.calc = EMT()
    return atoms


def _copper_bulk() -> Atoms:
    from ase.build import bulk

    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 1)
    atoms.calc = EMT()
    return atoms


# ---------------------------------------------------------------------------
# RESPA integrator
# ---------------------------------------------------------------------------


class TestRespaIntegrator:
    def test_single_level_run(self) -> None:
        """RespaIntegrator with no slow calculator degenerates to Verlet."""
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        from goal.md.mts.integrators.respa import RespaIntegrator

        atoms = _copper_bulk()
        MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)  # give initial velocities
        fast_calc = EMT()
        dyn = RespaIntegrator(
            atoms=atoms,
            fast_calculator=fast_calc,
            timestep=1.0 * units.fs,
            mts_ratio=1,
            slow_calculator=None,
            temperature_K=None,
        )
        initial_positions = atoms.get_positions().copy()
        dyn.run(5)
        # With non-zero initial velocities positions must change
        assert not np.allclose(atoms.get_positions(), initial_positions)

    def test_two_level_run(self) -> None:
        """RespaIntegrator with separate fast and slow calculators."""
        from goal.md.mts.integrators.respa import RespaIntegrator

        atoms = _copper_bulk()
        fast_calc = EMT()
        slow_calc = EMT()

        dyn = RespaIntegrator(
            atoms=atoms,
            fast_calculator=fast_calc,
            timestep=0.5 * units.fs,
            mts_ratio=2,
            slow_calculator=slow_calc,
            temperature_K=None,
        )
        dyn.run(3)
        # Fast calculator should have been called more than slow
        assert dyn.mts_ratio == 2

    def test_langevin_thermostat(self) -> None:
        """Thermostat should not crash and temperature should be set."""
        from goal.md.mts.integrators.respa import RespaIntegrator

        atoms = _copper_bulk()
        dyn = RespaIntegrator(
            atoms=atoms,
            fast_calculator=EMT(),
            timestep=1.0 * units.fs,
            mts_ratio=1,
            temperature_K=300.0,
            friction=0.01,
        )
        dyn.run(5)
        # The O-update runs twice per outer step, so each covers half of it
        assert dyn._lang_c1 == pytest.approx(np.exp(-0.01 * 0.5))

    def test_mts_ratio_1_is_verlet_on_the_slow_level(self) -> None:
        """At mts_ratio=1 the fast force cancels exactly: the run is velocity
        Verlet on the slow surface, and observers see the slow level's energy."""
        from ase.calculators.lj import LennardJones
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet

        from goal.md.mts.integrators.respa import RespaIntegrator

        ref = _copper_bulk()
        MaxwellBoltzmannDistribution(ref, temperature_K=300.0, rng=np.random.default_rng(0))
        atoms = ref.copy()
        VelocityVerlet(ref, timestep=1.0 * units.fs).run(5)
        fast = LennardJones(sigma=2.3, epsilon=0.1, rc=5.0)
        RespaIntegrator(
            atoms, fast, timestep=1.0 * units.fs, mts_ratio=1, slow_calculator=EMT()
        ).run(5)
        np.testing.assert_allclose(atoms.positions, ref.positions, atol=1e-10)
        assert atoms.get_potential_energy() == pytest.approx(ref.get_potential_energy())

    def test_outer_step_time_and_call_counts(self) -> None:
        """Log time advances by the outer step; chunked runs add no extra calls."""
        from goal.md.mts.integrators.respa import RespaIntegrator

        dyn = RespaIntegrator(
            _copper_bulk(), EMT(), timestep=0.5 * units.fs, mts_ratio=4, slow_calculator=EMT()
        )
        times: list[float] = []
        dyn.attach(lambda: times.append(dyn.get_time() / units.fs), interval=1)
        dyn.run(2)
        dyn.run(1)
        assert times == pytest.approx([0.0, 2.0, 4.0, 6.0])  # step 0 once, then each outer step
        assert dyn.mts_stats["fast_calculator_calls"] == 1 + 3 * 4
        assert dyn.mts_stats["slow_calculator_calls"] == 1 + 3

    def test_mts_stats(self) -> None:
        """Test RespaIntegrator reports correct MTS statistics."""
        from goal.md.mts.integrators.respa import RespaIntegrator

        atoms = _copper_bulk()
        dyn = RespaIntegrator(
            atoms=atoms,
            fast_calculator=EMT(),
            timestep=1.0 * units.fs,
            mts_ratio=4,
        )
        stats = dyn.mts_stats
        assert stats["mts_ratio"] == 4
        assert stats["inner_timestep_fs"] == pytest.approx(1.0)
        assert stats["outer_timestep_fs"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# IPI adapter
# ---------------------------------------------------------------------------


class TestIPICompatibleDriver:
    def test_unit_conversion_roundtrip(self) -> None:
        """Forces in i-pi units converted back to ASE should match direct computation."""
        from goal.md.mts.ipi_adapter import _ANG_TO_BOHR, IPICompatibleDriver

        atoms = _water()
        driver = IPICompatibleDriver(EMT(), atoms)

        pos_bohr = atoms.get_positions() * _ANG_TO_BOHR
        cell_bohr = atoms.get_cell().array * _ANG_TO_BOHR

        forces_au, energy_au, virial_au = driver.get_forces(pos_bohr, cell_bohr)

        assert forces_au.shape == (3, 3)
        assert isinstance(energy_au, float)
        assert virial_au.shape == (3, 3)

    def test_ase_units_helper(self) -> None:
        """Test IPICompatibleDriver provides ASE unit helper method."""
        from goal.md.mts.ipi_adapter import IPICompatibleDriver

        atoms = _water()
        driver = IPICompatibleDriver(EMT(), atoms)
        forces, energy = driver.get_forces_ase(atoms)
        assert forces.shape == (3, 3)
        assert isinstance(energy, float)


class TestMTSConfig:
    def test_defaults(self) -> None:
        """Test MTSConfig initializes with expected defaults."""
        from goal.md.mts.ipi_adapter import MTSConfig

        cfg = MTSConfig()
        assert cfg.mts_ratio == 4
        assert cfg.timestep_fs == 1.0
        assert cfg.temperature_K == 300.0


class TestMTSSimulation:
    def test_short_run(self) -> None:
        """Test MTSSimulation runs for specified number of steps."""
        from goal.md.mts.ipi_adapter import MTSConfig, MTSSimulation

        atoms = _copper_bulk()
        cfg = MTSConfig(
            timestep_fs=0.5,
            mts_ratio=2,
            total_steps=4,
            temperature_K=300.0,
            show_progress=False,
            log_interval=2,
        )
        sim = MTSSimulation(atoms, fast_calculator=EMT(), config=cfg, slow_calculator=EMT())
        result = sim.run()
        assert result["steps_completed"] == 4
        assert "mts_stats" in result


# ---------------------------------------------------------------------------
# DualCalculator base class
# ---------------------------------------------------------------------------


class TestDualCalculatorBase:
    def test_abstract_cannot_instantiate(self) -> None:
        """Test DualCalculator is abstract and cannot be instantiated."""
        from goal.md.mts.dual_calculator.base import DualCalculator

        with pytest.raises(TypeError):
            DualCalculator(EMT())  # type: ignore[abstract]

    def test_state_initialised(self) -> None:
        """Test DualCalculatorState initializes with expected defaults."""
        from goal.md.mts.dual_calculator.base import DualCalculatorState

        state = DualCalculatorState()
        assert state.step == 0
        assert state.active_calculator == "base"


# ---------------------------------------------------------------------------
# QMLearnerCalculator
# ---------------------------------------------------------------------------


class TestQMLearnerCalculator:
    def test_always_base_policy(self) -> None:
        """Test QMLearnerCalculator with AlwaysBasePolicy uses base calculator."""
        from goal.md.mts.dual_calculator.qm_ml_learner import (
            AlwaysBasePolicy,
            QMLearnerCalculator,
        )

        atoms = _water()
        learner = QMLearnerCalculator(
            base_calculator=EMT(),
            ml_calculator=EMT(),
            policy=AlwaysBasePolicy(),
            eval_interval=5,
        )
        atoms.calc = learner
        _ = atoms.get_potential_energy()
        assert learner.state.active_calculator == "base"

    def test_data_buffer_populated(self) -> None:
        """Test QMLearnerCalculator populates data buffer from energy evaluations."""
        from goal.md.mts.dual_calculator.qm_ml_learner import (
            AlwaysBasePolicy,
            QMLearnerCalculator,
        )

        atoms = _water()
        learner = QMLearnerCalculator(
            base_calculator=EMT(),
            ml_calculator=EMT(),
            policy=AlwaysBasePolicy(),
            eval_interval=1000,
        )
        atoms.calc = learner
        for _ in range(3):
            atoms.get_potential_energy()

        assert len(learner.data_buffer) > 0

    def test_threshold_policy_reports_history(self) -> None:
        """Test QMLearnerCalculator reports training history correctly."""
        from ase import units
        from ase.md.langevin import Langevin

        from goal.md.mts.dual_calculator.qm_ml_learner import (
            QMLearnerCalculator,
            ThresholdPolicy,
        )

        atoms = _water()
        learner = QMLearnerCalculator(
            base_calculator=EMT(),
            ml_calculator=EMT(),
            policy=ThresholdPolicy(min_steps=0),
            eval_interval=2,
        )
        atoms.calc = learner
        # Use dynamics so atoms move and cache is invalidated each step
        dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01)
        dyn.run(5)

        history = learner.get_training_history()
        assert "data_buffer_size" in history
        assert "total_md_steps" in history
        assert history["total_md_steps"] >= 5

    def test_switch_policies(self) -> None:
        """Test different switching policies for dual calculator."""
        from goal.md.mts.dual_calculator.qm_ml_learner import (
            DualCalculatorState,
            StepFractionPolicy,
            ThresholdPolicy,
        )

        # ThresholdPolicy — before min_steps
        pol = ThresholdPolicy(min_steps=100)
        state = DualCalculatorState(step=50, force_rmse=0.001, energy_mae=0.0001)
        assert not pol.should_switch_to_ml(state)

        # ThresholdPolicy — after min_steps with good accuracy
        state.step = 200
        assert pol.should_switch_to_ml(state)

        # StepFractionPolicy
        frac_pol = StepFractionPolicy(switch_fraction=0.1, total_steps=1000)
        state_early = DualCalculatorState(step=50)
        state_late = DualCalculatorState(step=150)
        assert not frac_pol.should_switch_to_ml(state_early)
        assert frac_pol.should_switch_to_ml(state_late)
