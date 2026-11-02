"""PGS edge correction (C2, arXiv:2606.15892 Eq. S9): ``w' = w + g(e) ⊙ s(r)``."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule

from goal.ml.nn.models.simurgh.dev.monolithic import _PGSEdge
from tests.ml.simurgh_dev.test_monolithic import (
    CUTOFF,
    _graph,
    awake,
    build,
    load_shared,
    predict_fn,
)

EDGE = {"readout": {"pgs_edge": True}}


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def test_fresh_correction_changes_nothing() -> None:
    ethanol = molecule("CH3CH2OH")
    on, off = build(**EDGE), build()
    load_shared(off, on)
    (e_on, f_on), (e_off, f_off) = predict_fn(on)(ethanol), predict_fn(off)(ethanol)
    assert abs(e_on - e_off) < 1e-12 and np.abs(f_on - f_off).max() < 1e-12


def test_basis_limit_and_orthonormality() -> None:
    module = _PGSEdge(4, 3, CUTOFF, 5, per_channel=False)
    m = torch.arange(1, 17, dtype=torch.float64)
    small = module.basis(torch.tensor([1e-9]))[0]
    torch.testing.assert_close(small, math.sqrt(2 / CUTOFF) * m * math.pi / CUTOFF)
    r = torch.linspace(1e-6, CUTOFF, 200001)
    b = module.basis(r)  # without the envelope: orthonormal under r^2 dr
    gram = torch.trapezoid(b[:, :, None] * b[:, None, :] * r[:, None, None] ** 2, r, dim=0)
    torch.testing.assert_close(gram, torch.eye(16), atol=1e-6, rtol=0)


def test_correction_vanishes_at_the_cutoff_with_two_derivatives() -> None:
    torch.manual_seed(0)
    module = _PGSEdge(4, 3, CUTOFF, 5, per_channel=True)
    with torch.no_grad():
        module.alpha.normal_()
        module.gate[-1].weight.normal_()
    e = torch.randn(1, 4)
    for r0 in (CUTOFF - 1e-6, CUTOFF + 1e-6):
        r = torch.tensor([r0], requires_grad=True)
        out = module(e, r).sum()
        (d1,) = torch.autograd.grad(out, r, create_graph=True)
        (d2,) = torch.autograd.grad(d1.sum(), r)
        assert abs(float(out)) < 1e-12 and abs(float(d1)) < 1e-9 and abs(float(d2)) < 1e-3


@pytest.mark.parametrize("extra", [Atoms("C", [[0.0, 1.3, 0.0]]), None], ids=["trimer", "dimer"])
def test_scan_across_the_cutoff_is_c2(extra) -> None:
    """E, dE/dr and d2E/dr2 are continuous where the far atom crosses r_c."""
    p = predict_fn(awake(build(**EDGE), gain=0.3))

    def energy(d: float) -> float:
        atoms = Atoms("O", [[0.0, 0.0, 0.0]]) + Atoms("H", [[d, 0.0, 0.0]])
        return p(atoms if extra is None else atoms + extra)[0]

    h = 1e-4
    for side in (-1, 1):
        r = CUTOFF + side * 3 * h  # a central-difference stencil entirely on one side
        e = [energy(r + k * h) for k in (-1, 0, 1)]
        if side == -1:
            below = e
        else:
            above = e
    slope = [(x[2] - x[0]) / (2 * h) for x in (below, above)]
    curv = [(x[2] - 2 * x[1] + x[0]) / h**2 for x in (below, above)]
    assert abs(slope[0] - slope[1]) < 1e-6 and abs(curv[0] - curv[1]) < 1e-2


def test_alpha_moves_after_one_step() -> None:
    model = awake(build(**EDGE)).train()
    out = model(_graph(molecule("CH3CH2OH")))
    (out["energy"].sum() + out["forces"].square().sum()).backward()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    assert all(layer.pgs_edge.alpha.abs().sum() > 0 for layer in model.atom_line.layers)


def test_ece_core_refuses_it() -> None:
    with pytest.raises(ValueError, match="core: ace"):
        build(core="ece", **EDGE)
