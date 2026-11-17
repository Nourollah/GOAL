"""The invariant utilities pass on a sound model and catch known defects.

A smooth pair potential with analytic forces is the sound model; three
corrupted copies each carry one defect the utilities must flag: an energy
step at 1.8 Å, a size-normalised (non-extensive) energy, and forces that
are not the gradient of the energy.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import molecule

from tests.ml.simurgh_dev.utils import (
    bond_scan,
    far_copy_extensivity,
    finite_difference_force_error,
    scan_jumps,
    symmetry_errors,
)

CUTOFF = 5.0


def pair_potential(
    atoms: Atoms, step: float = 0.0, per_atom: bool = False, force_gain: float = 1.0
) -> tuple[float, np.ndarray]:
    """``sum f(r_ij)`` with ``f = (r - rc)² exp(-r)`` inside the cutoff, plus optional defects."""
    pos = atoms.positions
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    i, j = np.triu_indices(len(atoms), k=1)
    rij = r[i, j]
    inside = rij < CUTOFF
    f = np.where(inside, (rij - CUTOFF) ** 2 * np.exp(-rij), 0.0)
    df = np.where(inside, (2 * (rij - CUTOFF) - (rij - CUTOFF) ** 2) * np.exp(-rij), 0.0)
    energy = f.sum() + step * (rij < 1.8).sum()
    forces = np.zeros_like(pos)
    pull = (df / np.maximum(rij, 1e-12))[:, None] * diff[i, j]
    np.add.at(forces, i, -pull)
    np.add.at(forces, j, pull)
    if per_atom:
        energy, forces = energy / len(atoms), forces / len(atoms)
    return float(energy), force_gain * forces


def test_sound_model_passes_every_check() -> None:
    ethanol, water = molecule("CH3CH2OH"), molecule("H2O")
    assert abs(far_copy_extensivity(pair_potential, ethanol)) < 1e-10
    assert abs(far_copy_extensivity(pair_potential, ethanol, water)) < 1e-10
    errors = symmetry_errors(pair_potential, ethanol)
    assert max(errors.values()) < 1e-10, errors
    assert finite_difference_force_error(pair_potential, ethanol) < 1e-6
    assert scan_jumps(*bond_scan(pair_potential, 6, 1))["energy"] < 1e-3


def test_bond_scan_flags_an_energy_step() -> None:
    r, energy, slope = bond_scan(lambda a: pair_potential(a, step=0.5), 6, 1)
    below, above = r < 1.8, r > 1.8
    assert scan_jumps(r[below], energy[below], slope[below])["energy"] < 1e-3
    assert scan_jumps(r[above], energy[above], slope[above])["energy"] < 1e-3
    assert scan_jumps(r, energy, slope)["energy"] > 0.4


def test_far_copy_flags_a_non_extensive_energy() -> None:
    error = far_copy_extensivity(lambda a: pair_potential(a, per_atom=True), molecule("CH4"))
    assert abs(error) > 1e-3, error


def test_finite_differences_flag_non_conservative_forces() -> None:
    wrong = finite_difference_force_error(
        lambda a: pair_potential(a, force_gain=1.1), molecule("CH3CH2OH")
    )
    assert wrong > 1e-3, wrong


def test_scan_jumps_flags_a_step() -> None:
    r = np.linspace(0.6, 4.0, 200)
    energy, slope = np.cos(r), -np.sin(r)
    assert scan_jumps(r, energy, slope)["energy"] < 1e-6
    energy[100:] += 0.01
    assert scan_jumps(r, energy, slope)["energy"] > 9e-3
