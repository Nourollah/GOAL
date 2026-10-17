"""GOAL Molecular Dynamics (MD) Module.

Provides molecular dynamics simulation capabilities integrated with the
goal.ml machine learning framework.

Usage patterns
--------------
**Config-first (recommended)**::

    from hydra import compose, initialize_config_dir
    from goal import CONFIGS_DIR
    from goal.md.core.simulation import simulate_from_config
    with initialize_config_dir(config_dir=CONFIGS_DIR):
        cfg = compose("md/simulations/langevin_with_model")
    result = simulate_from_config(cfg)

**Direct (library)**::

    from goal.md import MoleculeFactory, CalculatorFactory, Simulation

    atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
    calc  = CalculatorFactory.create("goal_model", checkpoint="model.ckpt")
    sim   = Simulation(atoms, calc, steps=5000, temperature_K=300.0)
    result = sim.run()

**Loading goal.ml models**::

    from goal.md.adapters.model_loader import load_goal_calculator
    calc = load_goal_calculator("outputs/train/run/last.ckpt")
    atoms.calc = calc

**Multi-timescale MD (RESPA + i-pi physics)**::

    from goal.md.mts.ipi_adapter import MTSSimulation, MTSConfig

    cfg = MTSConfig(mts_ratio=4, timestep_fs=1.0, total_steps=10000)
    sim = MTSSimulation(atoms, fast_calc, config=cfg, slow_calc=qm_calc)
    result = sim.run()
"""

from goal.md.core.calculator_factory import CalculatorFactory
from goal.md.core.md_factory import DynamicsFactory
from goal.md.core.molecule_factory import MoleculeFactory
from goal.md.core.molecule_tools import (
    box_molecule,
    generate_3d_coordinates_from_smiles,
)
from goal.md.core.observers import MDLogObserver, TrajectoryObserver, setup_observers
from goal.md.core.runner import RunResult, run_dynamics
from goal.md.core.simulation import Simulation, SimulationConfig, simulate_from_config

__all__ = [
    # Factories
    "MoleculeFactory",
    "CalculatorFactory",
    "DynamicsFactory",
    # Simulation
    "Simulation",
    "SimulationConfig",
    "simulate_from_config",
    # Runner
    "RunResult",
    "run_dynamics",
    # Observers
    "setup_observers",
    "TrajectoryObserver",
    "MDLogObserver",
    # Molecule utilities
    "generate_3d_coordinates_from_smiles",
    "box_molecule",
]
