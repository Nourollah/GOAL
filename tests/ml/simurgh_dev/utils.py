"""Physical invariant checks shared by the SIMURGH tests.

Every check takes a *predictor*, ``atoms -> (energy, forces)``, so it
works for any GOAL model: wrap a monolithic model or a ``GOALModule``
with :func:`predictor`.  Checks return the measured error; the tests
choose the tolerance (1e-5 eV for float64 extensivity, for example).
"""

from __future__ import annotations

import typing

import numpy as np
import torch
from ase import Atoms
from scipy.spatial.transform import Rotation
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph

Predictor = typing.Callable[[Atoms], tuple[float, np.ndarray]]


def predictor(
    model: typing.Callable[[Batch], dict[str, torch.Tensor]],
    cutoff: float,
    dtype: torch.dtype = torch.float64,
    **graph_kwargs: typing.Any,
) -> Predictor:
    """Wrap ``model(batch) -> {"energy", "forces"}`` as ``atoms -> (E, F)``.

    ``graph_kwargs`` go to :meth:`AtomicGraph.from_ase` (for example
    ``compute_fragment_index=True`` for fragment-aware models).
    """

    def predict(atoms: Atoms) -> tuple[float, np.ndarray]:
        graph = AtomicGraph.from_ase(atoms, cutoff=cutoff, dtype=dtype, **graph_kwargs)
        out = model(Batch.from_data_list([graph]))
        return float(out["energy"].sum()), out["forces"].detach().cpu().double().numpy()

    return predict


def far_copy_extensivity(
    predict: Predictor, a: Atoms, b: Atoms | None = None, distance: float = 100.0
) -> float:
    """``E(A ∪ B) - E(A) - E(B)`` with ``B`` (default: a copy of ``A``) placed far away."""
    b = (a if b is None else b).copy()
    b.positions[:, 0] += a.positions[:, 0].max() + distance - b.positions[:, 0].min()
    return predict(a + b)[0] - predict(a)[0] - predict(b)[0]


def isolated_atom(predict: Predictor, z: int) -> tuple[float, np.ndarray]:
    """Energy and force of a single atom of element ``z``."""
    return predict(Atoms(numbers=[z], positions=[[0.0, 0.0, 0.0]]))


def symmetry_errors(predict: Predictor, atoms: Atoms, seed: int = 0) -> dict[str, float]:
    """Max ``abs(ΔE)`` and ``abs(ΔF)`` under a random rotation, a translation and a permutation.

    Forces are compared after mapping back (rotated by ``R``, permuted by
    the same permutation), so every entry is an error and should be ~0.
    """
    rng = np.random.default_rng(seed)
    e0, f0 = predict(atoms)
    out: dict[str, float] = {}

    rot = Rotation.random(random_state=seed).as_matrix()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rot.T
    e, f = predict(rotated)
    out["rotation_energy"], out["rotation_forces"] = abs(e - e0), np.abs(f - f0 @ rot.T).max()

    moved = atoms.copy()
    moved.positions += rng.normal(scale=5.0, size=3)
    e, f = predict(moved)
    out["translation_energy"], out["translation_forces"] = abs(e - e0), np.abs(f - f0).max()

    perm = rng.permutation(len(atoms))
    e, f = predict(atoms[perm])
    out["permutation_energy"], out["permutation_forces"] = abs(e - e0), np.abs(f - f0[perm]).max()
    return out


def finite_difference_force_error(
    predict: Predictor, atoms: Atoms, step: float = 1e-4, max_coords: int = 12, seed: int = 0
) -> float:
    """Max ``abs(F + dE/dx)`` over up to ``max_coords`` random Cartesian coordinates.

    Central differences; ``O(step²)`` error, so use float64.
    """
    _, forces = predict(atoms)
    flat = np.random.default_rng(seed).permutation(3 * len(atoms))[:max_coords]
    worst = 0.0
    for k in flat:
        i, c = divmod(int(k), 3)
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[i, c] += step
        minus.positions[i, c] -= step
        grad = (predict(plus)[0] - predict(minus)[0]) / (2 * step)
        worst = max(worst, abs(forces[i, c] + grad))
    return worst


def bond_scan(
    predict: Predictor, z1: int, z2: int, r: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diatomic scan: ``(r, E(r), dE/dr)``; defaults to 200 points over 0.6-4 Å.

    ``dE/dr`` is read from the force on atom 2 along the bond, so it is the
    model's own derivative, not a finite difference.
    """
    r = np.linspace(0.6, 4.0, 200) if r is None else np.asarray(r, dtype=float)
    energy, slope = np.empty_like(r), np.empty_like(r)
    for k, d in enumerate(r):
        e, f = predict(Atoms(numbers=[z1, z2], positions=[[0, 0, 0], [d, 0, 0]]))
        energy[k], slope[k] = e, -f[1, 0]
    return r, energy, slope


def scan_jumps(r: np.ndarray, energy: np.ndarray, slope: np.ndarray) -> dict[str, float]:
    """Discontinuity measures of a scan.

    ``energy``: max ``abs(ΔE - mean slope · Δr)`` between neighbours — a jump in E
    at continuous dE/dr shows up here and nowhere else.  ``slope``: max
    ``abs(Δ(dE/dr))`` between neighbours, for comparison with the scan's range.
    """
    dr = np.diff(r)
    trapezoid = 0.5 * (slope[1:] + slope[:-1]) * dr
    return {
        "energy": float(np.abs(np.diff(energy) - trapezoid).max()),
        "slope": float(np.abs(np.diff(slope)).max()),
    }
