"""MFC locality diagnostic: deterministic and invariant to rigid motions."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from goal.ml.data.graph import AtomicGraph
from tests.ml.simurgh_dev.test_mfc import alkane
from tests.ml.simurgh_dev.test_monolithic import awake, build

_SCRIPT = Path(__file__).parents[3] / "scripts" / "simurgh_dev" / "mfc_locality.py"
_spec = importlib.util.spec_from_file_location("mfc_locality", _SCRIPT)
assert _spec is not None and _spec.loader is not None
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


def _calc() -> SimpleNamespace:
    model = awake(build()).eval()
    return SimpleNamespace(
        module=model,
        device=torch.device("cpu"),
        build_graph=lambda atoms: AtomicGraph.from_ase(atoms, cutoff=5.0, dtype=torch.float64),
    )


def test_locality_map_is_deterministic_and_rigid_motion_invariant() -> None:
    calc = _calc()
    atoms = alkane(10)
    moved = atoms.copy()
    moved.positions = atoms.positions @ Rotation.random(random_state=7).as_matrix().T + 2.0
    a = diag.locality(calc, [atoms], buffer=5.0, random_cuts=0, seed=0)["table"]
    b = diag.locality(calc, [atoms], buffer=5.0, random_cuts=0, seed=0)["table"]
    c = diag.locality(calc, [moved], buffer=5.0, random_cuts=0, seed=0)["table"]
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, c, atol=1e-9)
    # SIMURGH returns atomic energies, so |Δe| is always there; core forces only
    # for windows whose buffered core is not empty
    assert np.isfinite(a[:, [3, 5]]).all() and np.isfinite(a[:, 4]).any()


def test_summary_and_scores() -> None:
    table = np.array(
        [
            # structure, atom, d, |de|, |dF_core|, |dF|
            [0, 0, 0.5, 0.1, 0.05, 0.2],
            [0, 1, 3.0, 0.01, 0.004, 0.02],
            [0, 2, 8.0, 1e-6, 1e-5, 1e-5],
        ]
    )
    summary = diag.summarise(table, bin_width=1.0)
    assert summary["locality_length"] == 4.0  # last bin with median >= 1 meV/Å ends at 4 Å
    scores = diag.atom_scores(table, buffer=2.0)
    assert scores == {(0, 1): 0.02, (0, 2): 1e-5}
