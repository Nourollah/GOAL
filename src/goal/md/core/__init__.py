"""Core molecular dynamics functionality.

Factory classes for creating molecules, calculators, and dynamics objects.
All factories support custom builder registration for extensibility.
"""

from goal.md.core.calculator_factory import CalculatorBuilder, CalculatorFactory
from goal.md.core.md_factory import DynamicsBuilder, DynamicsFactory
from goal.md.core.molecule_factory import MoleculeBuilder, MoleculeFactory
from goal.md.core.molecule_tools import (
    box_molecule,
    generate_3d_coordinates_from_smiles,
)

__all__ = [
    "CalculatorFactory",
    "CalculatorBuilder",
    "MoleculeFactory",
    "MoleculeBuilder",
    "DynamicsFactory",
    "DynamicsBuilder",
    "generate_3d_coordinates_from_smiles",
    "box_molecule",
]

# The set of paramters accepted by `ase.md.langevin` and `flashmd.ase.langevin` to check against the user input in `simulation.py`.
COMMON_DYNAMICS_PARAMS = {                                                                                                      "temperature_K",
        "timestep",
        "fixcm",
        "rng",
        "trajectory",
        "logfile",
        "loginterval",
        "append_trajectory",
        "master",
        "comm",
}

ASE_DYNAMICS_PARAMS = COMMON_DYNAMICS_PARAMS | {
        "friction",
        "temperature"
}

FLASHMD_DYNAMICS_PARAMS = COMMON_DYNAMICS_PARAMS | {
        "model_name", # FlashMD Langevin acually takes 'model'. It is internally handled by GOAL's DynamicsFactory.
        "time_constant",
        "device",
        "rescale_energy",
        "random_rotation"
}
