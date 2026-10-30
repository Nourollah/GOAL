"""Context-sensitivity diagnostic against a model with a known Jacobian."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from ase.build import molecule

from goal.ml.data.graph import AtomicGraph

_SCRIPT = Path(__file__).parents[3] / "scripts" / "simurgh_dev" / "context_sensitivity.py"
_spec = importlib.util.spec_from_file_location("context_sensitivity", _SCRIPT)
assert _spec is not None and _spec.loader is not None
ctx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ctx)


class _PairExp(torch.nn.Module):
    """e_i = Σ_{j≠i} exp(-r_ij), so ‖∂e_i/∂r_j‖ = exp(-r_ij) exactly."""

    def forward(self, graph):
        d = torch.cdist(graph.pos, graph.pos)
        mask = ~torch.eye(d.shape[0], dtype=torch.bool)
        return {"atomic_energies": torch.where(mask, torch.exp(-d), 0.0).sum(dim=1)}


def test_binned_jacobian_matches_analytic() -> None:
    atoms = molecule("CH3CH2OH")
    graph = AtomicGraph.from_ase(atoms, cutoff=5.0, dtype=torch.float64)
    edges = np.linspace(0.0, 5.0, 11)
    sums, counts = ctx.context_sensitivity(_PairExp(), graph, edges)
    d = atoms.get_all_distances()[~np.eye(len(atoms), dtype=bool)]
    expected, _ = np.histogram(d, bins=edges, weights=np.exp(-d))
    assert counts.sum() == len(atoms) * (len(atoms) - 1)
    np.testing.assert_allclose(sums, expected, rtol=1e-12)


def test_simurgh_atomic_energies_can_be_differentiated_after_forward() -> None:
    """The model keeps its graph when the caller's positions require grad."""
    from tests.ml.simurgh_dev.test_monolithic import awake, build

    model = awake(build()).eval()
    graph = AtomicGraph.from_ase(molecule("CH3CH2OH"), cutoff=5.0, dtype=torch.float64)
    sums, counts = ctx.context_sensitivity(model, graph, np.linspace(0.0, 5.0, 11))
    assert counts.sum() == 72 and sums[counts > 0].min() > 0
