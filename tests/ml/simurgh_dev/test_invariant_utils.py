"""The invariant utilities pass on a sound model and catch a known defect.

* legacy SIMURGH Simplified (readouts woken up, float64) passes every check
  except bond-scan smoothness: its energy jumps where a bond crosses the
  1.8 Å covalent cutoff of fragment perception (the fragment labels flip);
* legacy NF SIMURGH, Stage 1, is not size-extensive once three or more
  fragments are present: its softmax normalises over every fragment
  partner before the envelope is applied, so a far copy dilutes the
  near-fragment weights.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from tests.ml.simurgh.test_nf_simurgh import CUTOFF as NF_CUTOFF
from tests.ml.simurgh.test_nf_simurgh import _activate, _build_model
from tests.ml.simurgh.test_simurgh_simplified import make_model, wake_corrections
from tests.ml.simurgh_dev.utils import (
    bond_scan,
    far_copy_extensivity,
    finite_difference_force_error,
    predictor,
    scan_jumps,
    symmetry_errors,
)


@pytest.fixture(scope="module")
def simplified():
    model = wake_corrections(make_model("simurgh_legacy_simplified_monolithic").double())
    return predictor(model, cutoff=5.0, compute_fragment_index=True)


def test_simplified_is_extensive(simplified) -> None:
    ethanol, water = molecule("CH3CH2OH"), molecule("H2O")
    assert abs(far_copy_extensivity(simplified, ethanol)) < 1e-5
    assert abs(far_copy_extensivity(simplified, ethanol, water)) < 1e-5


def test_simplified_symmetries_and_forces(simplified) -> None:
    ethanol = molecule("CH3CH2OH")
    errors = symmetry_errors(simplified, ethanol)
    assert max(errors.values()) < 1e-8, errors
    assert finite_difference_force_error(simplified, ethanol) < 1e-6


def test_simplified_bond_scan_jumps_only_at_fragment_perception(simplified) -> None:
    r, energy, slope = bond_scan(simplified, 6, 1)
    assert np.ptp(energy) > 0  # the scan is not trivially flat
    below, above = r < 1.8, r > 1.8
    assert scan_jumps(r[below], energy[below], slope[below])["energy"] < 1e-4
    assert scan_jumps(r[above], energy[above], slope[above])["energy"] < 1e-4
    assert scan_jumps(r, energy, slope)["energy"] > 0.1  # the legacy defect


def test_scan_jumps_flags_a_step() -> None:
    r = np.linspace(0.6, 4.0, 200)
    energy, slope = np.cos(r), -np.sin(r)
    assert scan_jumps(r, energy, slope)["energy"] < 1e-6
    energy[100:] += 0.01
    assert scan_jumps(r, energy, slope)["energy"] > 9e-3


def test_nf_stage1_far_copy_is_not_extensive() -> None:
    model = _build_model()
    _activate(model)
    nf = predictor(model, cutoff=NF_CUTOFF, compute_fragment_index=True)
    a = molecule("CH4") + Atoms("OH2", positions=[[4.0, 0, 0], [4.957, 0, 0], [3.76, 0.927, 0]])
    error = far_copy_extensivity(nf, a)  # 2 fragments -> 4 with the copy
    assert abs(error) > 1e-3, error
