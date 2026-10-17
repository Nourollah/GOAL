"""Online learning: the QM → ML switch must actually save QM calls, and the
fine-tuner must be able to reach the model it fine-tunes.

Two failures this guards against:

* ``DualCalculator.calculate`` ran the base (QM) calculator on *every*
  step and only then decided which result to use, so switching to ML
  saved nothing.
* ``GOALOnlineFinetuner`` read ``calculator.module``, which
  ``GOALCalculator`` did not have; the ``AttributeError`` was caught and
  logged, so the model was silently never fine-tuned.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.build import molecule
from ase.calculators.calculator import all_changes
from ase.calculators.emt import EMT

from goal.md.mts.dual_calculator.qm_ml_learner import (
    AlwaysBasePolicy,
    GOALOnlineFinetuner,
    QMLearnerCalculator,
    ThresholdPolicy,
)


class CountingEMT(EMT):
    """EMT that counts its (real) calculations."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        self.calls += 1
        super().calculate(atoms, properties, system_changes)


class AlwaysMLPolicy:
    """Switch to ML immediately and never come back."""

    def should_switch_to_ml(self, state) -> bool:
        return True

    def should_return_to_base(self, state) -> bool:
        return False


def _water() -> Atoms:
    return Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])


def _evaluate_n_frames(learner: QMLearnerCalculator, n: int) -> None:
    atoms = _water()
    atoms.calc = learner
    rng = np.random.default_rng(0)
    for _ in range(n):
        atoms.positions += rng.normal(scale=1e-3, size=atoms.positions.shape)
        atoms.get_potential_energy()


class TestQMSkipping:
    def test_qm_runs_only_on_recalibration_steps_after_switch(self) -> None:
        base, ml = CountingEMT(), CountingEMT()
        learner = QMLearnerCalculator(
            base_calculator=base, ml_calculator=ml, policy=AlwaysMLPolicy(), eval_interval=5
        )
        _evaluate_n_frames(learner, 20)

        assert learner.state.active_calculator == "ml"
        # One QM reference per evaluation (steps 5, 10, 15, 20), nothing else.
        assert base.calls == 4
        assert learner.get_metrics()["base_calls"] == 4
        assert learner.get_metrics()["ml_calls"] == 20

    def test_ml_results_are_returned_on_ml_steps(self) -> None:
        learner = QMLearnerCalculator(
            base_calculator=EMT(), ml_calculator=EMT(), policy=AlwaysMLPolicy(), eval_interval=1000
        )
        atoms = _water()
        atoms.calc = learner
        energy = atoms.get_potential_energy()
        reference = _water()
        reference.calc = EMT()
        assert energy == pytest.approx(reference.get_potential_energy())
        assert learner.state.base_energy is None  # no stale reference kept

    def test_base_steps_still_run_qm_every_step(self) -> None:
        base = CountingEMT()
        learner = QMLearnerCalculator(
            base_calculator=base, ml_calculator=EMT(), policy=AlwaysBasePolicy(), eval_interval=5
        )
        _evaluate_n_frames(learner, 10)
        assert base.calls == 10
        assert len(learner.data_buffer) == 10

    def test_threshold_policy_switches_and_then_saves_qm(self) -> None:
        """Identical base and ML ⇒ zero error ⇒ the threshold policy switches,
        after which QM is only called at evaluation steps."""
        base = CountingEMT()
        learner = QMLearnerCalculator(
            base_calculator=base,
            ml_calculator=EMT(),
            policy=ThresholdPolicy(min_steps=3),
            eval_interval=10,
        )
        _evaluate_n_frames(learner, 50)
        assert learner.state.active_calculator == "ml"
        assert base.calls < 50 // 2


@pytest.fixture()
def goal_calculator():
    from goal.ml.utils.calculator import GOALCalculator
    from tests.ml.test_archive import _build_module, _tiny_config

    return GOALCalculator(module=_build_module(_tiny_config()), cutoff=5.0)


class TestOnlineFinetuner:
    def _buffer(self, n: int = 6) -> deque:
        reference = EMT()
        buffer: deque = deque()
        rng = np.random.default_rng(1)
        for _ in range(n):
            atoms = molecule("CH4")
            atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
            atoms.calc = reference
            buffer.append((atoms.copy(), atoms.get_forces().copy(), atoms.get_potential_energy()))
        return buffer

    def test_finetuner_reaches_the_calculators_model(self, goal_calculator) -> None:
        tuner = GOALOnlineFinetuner(goal_calculator, min_buffer_size=2, batch_size=2)
        assert tuner.model is goal_calculator.module

    def test_update_trains_the_shared_model(self, goal_calculator) -> None:
        tuner = GOALOnlineFinetuner(
            goal_calculator,
            min_buffer_size=2,
            batch_size=3,
            epochs_per_update=1,
            learning_rate=1e-2,
        )
        before = [p.detach().clone() for p in goal_calculator.module.parameters()]

        metrics = tuner.update(self._buffer())

        assert metrics["update_count"] == 1
        assert np.isfinite(metrics["loss"])
        after = list(goal_calculator.module.parameters())
        assert any(not torch.equal(a, b.detach()) for a, b in zip(before, after))
        assert not goal_calculator.module.training  # back in inference mode

    def test_device_follows_the_calculator(self, goal_calculator) -> None:
        assert GOALOnlineFinetuner(goal_calculator).device == str(goal_calculator.device)

    def test_calculator_without_model_raises_a_clear_error(self) -> None:
        tuner = GOALOnlineFinetuner(EMT())
        with pytest.raises(AttributeError, match="GOALCalculator"):
            _ = tuner.model

    def test_learner_with_finetuner_runs_md(self, goal_calculator) -> None:
        from ase.md.verlet import VelocityVerlet

        tuner = GOALOnlineFinetuner(goal_calculator, min_buffer_size=2, batch_size=2)
        learner = QMLearnerCalculator(
            base_calculator=EMT(),
            ml_calculator=goal_calculator,
            policy=ThresholdPolicy(min_steps=1000),
            fine_tuner=tuner,
            eval_interval=3,
        )
        atoms = molecule("CH4")
        atoms.calc = learner
        VelocityVerlet(atoms, timestep=0.2 * units.fs).run(7)
        history = learner.get_training_history()
        assert history["fine_tune_steps"] >= 2
        assert all(
            np.isfinite(m["loss"]) for m in history["fine_tune_metrics"] if m["loss"] == m["loss"]
        )


class TestOnlineLearnerConfig:
    """``goal-simulate-mts --config-name md/mts/simulations/online_learner``
    used to be ignored: ``MTSSimulation.from_config`` never read the
    ``base_calculator`` / ``ml_calculator`` / ``learner`` sections."""

    @pytest.fixture(autouse=True)
    def _register_emt(self):
        from goal.md.core.calculator_factory import CalculatorBuilder, CalculatorFactory

        class _EMTBuilder(CalculatorBuilder):
            def build(self):
                return EMT()

        CalculatorFactory.register("test_emt", _EMTBuilder())
        yield
        CalculatorFactory._builders.pop("test_emt", None)

    def test_builds_a_learner_from_the_shipped_config(self, tmp_path) -> None:
        from hydra import compose, initialize_config_dir

        from goal import CONFIGS_DIR
        from goal.md.mts.ipi_adapter import MTSSimulation

        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base="1.3"):
            cfg = compose(
                config_name="md/mts/simulations/online_learner",
                overrides=[
                    # replace (not merge into) the two calculator sections
                    "~base_calculator",
                    "+base_calculator={type: test_emt}",
                    "~ml_calculator",
                    "+ml_calculator={type: test_emt}",
                    "molecule.smiles=O",
                    "mts.total_steps=4",
                    "mts.show_progress=false",
                    f"mts.output_dir={tmp_path}",
                    "learner.eval_interval=2",
                ],
            )
        sim = MTSSimulation.from_config(cfg)
        assert isinstance(sim.fast_calc, QMLearnerCalculator)
        assert sim.slow_calc is None
        assert isinstance(sim.fast_calc.policy, ThresholdPolicy)
        assert sim.fast_calc.fine_tuner is None  # EMT is not a GOAL model
        result = sim.run()
        assert result["steps_completed"] == 4
