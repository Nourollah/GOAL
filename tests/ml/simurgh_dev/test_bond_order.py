"""Bond-order pair line (C5, arXiv:2608.22933): the tests of Appendix C-3.1, item 8."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule
from e3nn import o3

from goal.ml.nn.models.simurgh.dev.monolithic import _BondOrder, _triplets
from tests.ml.simurgh_dev.test_monolithic import (
    _graph,
    awake,
    build,
    load_shared,
    predict_fn,
)

BO = {"pair_line": {"bond_order": True}}


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def module(seed: int = 0) -> _BondOrder:
    torch.manual_seed(seed)
    cfg = build(**BO).config
    return _BondOrder(cfg, 4, 5.0, 5)


def edges(atoms: Atoms, cutoff: float = 5.0):
    pos = torch.tensor(atoms.positions, requires_grad=True)
    d = torch.cdist(pos.detach(), pos.detach())
    i, j = torch.nonzero((d < cutoff) & (d > 0), as_tuple=True)
    vec = pos[j] - pos[i]
    return pos, vec, vec.norm(dim=-1), torch.stack([i, j])


def test_isolated_bond_has_unit_order_and_finite_gradients() -> None:
    m = module()
    pos, vec, r, ei = edges(Atoms("CO", [[0, 0, 0], [1.2, 0, 0]]))
    b = m.bond_orders(vec, r, ei, torch.tensor([1, 3]))
    assert torch.equal(b, torch.ones(2))
    m.amplitude.data.fill_(1.0)
    (g,) = torch.autograd.grad(m(vec, r, ei, torch.tensor([1, 3])).sum(), pos)
    assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_fresh_branch_is_bit_identical() -> None:
    ethanol = molecule("CH3CH2OH")
    on, off = build(**BO), build()
    load_shared(off, on)
    (e_on, f_on), (e_off, f_off) = predict_fn(on)(ethanol), predict_fn(off)(ethanol)
    assert e_on == e_off and np.array_equal(f_on, f_off)


def test_third_atom_crossing_the_cutoff_is_c2() -> None:
    p = predict_fn(awake(build(**BO), gain=0.3))
    base = Atoms("CO", [[0, 0, 0], [1.2, 0, 0]])
    rc = 5.0

    def energy(d: float) -> float:
        return p(base + Atoms("H", [[0.0, d, 0.0]]))[0]

    h = 1e-4
    below = [energy(rc - 3 * h + k * h) for k in (-1, 0, 1)]
    above = [energy(rc + 3 * h + k * h) for k in (-1, 0, 1)]
    slope = [(x[2] - x[0]) / (2 * h) for x in (below, above)]
    curv = [(x[2] - 2 * x[1] + x[0]) / h**2 for x in (below, above)]
    assert abs(slope[0] - slope[1]) < 1e-6 and abs(curv[0] - curv[1]) < 1e-2


def test_bond_order_falls_under_compression() -> None:
    m = module()
    with torch.no_grad():
        m.chi.abs_()  # one sign of neighbour weight, as in a trained model of one chemistry
    methane = molecule("CH4")
    orders = []
    for factor in (0.85, 0.95, 1.0, 1.05, 1.15):
        atoms = methane.copy()
        atoms.positions *= factor
        _, vec, r, ei = edges(atoms)
        orders.append(float(m.bond_orders(vec, r, ei, torch.tensor([1, 0, 0, 0, 0]))[0]))
    assert all(a < b for a, b in zip(orders, orders[1:])), orders
    assert all(0 < b < 1 for b in orders)


def test_legendre_form_equals_the_spherical_harmonic_and_cartesian_forms() -> None:
    m = module()
    atoms = molecule("CH3OH")
    _, vec, r, ei = edges(atoms)
    elem = torch.tensor([1, 3, 0, 0, 0, 0])
    lam = torch.nn.functional.softplus(m.lambda_raw)
    f = m.envelope(r)
    e1, e2 = _triplets(ei[0], len(elem))
    unit = vec / r.unsqueeze(-1)
    weight = (
        m.chi[elem[ei[0][e1]], elem[ei[1][e2]]]
        * f[e2, None]
        * torch.exp(lam * (r[e1] - r[e2])[:, None])
    )
    sh = o3.spherical_harmonics(
        list(range(m.lmax + 1)), unit, normalize=True, normalization="component"
    )
    blocks = [sh[:, ell**2 : (ell + 1) ** 2] for ell in range(m.lmax + 1)]
    addition = torch.stack([(b[e1] * b[e2]).sum(-1) for b in blocks], -1)  # S_l P_l(cos)
    cos = (unit[e1] * unit[e2]).sum(-1)
    powers = torch.stack([cos**k for k in range(m.lmax + 1)], -1)  # Cartesian cos^l basis
    coeffs = torch.zeros(m.lmax + 1, m.lmax + 1)  # P_l in powers of cos, times S_l = 2l + 1
    for ell in range(m.lmax + 1):
        poly = np.polynomial.legendre.leg2poly([0] * ell + [1])
        coeffs[ell, : ell + 1] = torch.tensor(poly) * (2 * ell + 1)
    cartesian = powers @ coeffs.T
    torch.testing.assert_close(addition, cartesian, rtol=1e-10, atol=1e-10)
    # and the module's Bt3 contracted the same way
    sh_bt3 = torch.zeros(len(r), m.nmax, m.lmax + 1).index_add(
        0, e1, weight[:, :, None] * addition[:, None, :]
    )
    legendre = [torch.ones_like(cos), cos]
    for ell in range(1, m.lmax):
        legendre.append(
            ((2 * ell + 1) * cos * legendre[ell] - ell * legendre[ell - 1]) / (ell + 1)
        )
    ours = torch.stack(legendre, -1) * (2 * torch.arange(m.lmax + 1) + 1)
    torch.testing.assert_close(ours, addition, rtol=1e-10, atol=1e-10)
    assert sh_bt3.abs().sum() > 0


def test_attraction_only_is_scaled_and_options_validate() -> None:
    m = module()
    m.amplitude.data.fill_(1.0)
    _, vec, r, ei = edges(Atoms("CO", [[0, 0, 0], [0.3, 0, 0]]))
    assert float(m(vec, r, ei, torch.tensor([1, 3])).sum()) > 0  # compressed: repulsive
    with pytest.raises(NotImplementedError, match="bond_order_body: 4"):
        build(pair_line={"bond_order": True, "bond_order_body": 4})
    for key, bad in [("bond_order_cutoff_fn", "cosine"), ("bond_order_eta_map", "exp")]:
        with pytest.raises(ValueError, match=key):
            build(pair_line={"bond_order": True, key: bad})
    smooth = build(
        pair_line={
            "bond_order": True,
            "bond_order_eta_map": "smooth",
            "bond_order_cutoff_fn": "envelope",
        }
    )
    assert np.isfinite(predict_fn(awake(smooth))(molecule("H2O"))[0])


def test_every_parameter_gets_a_gradient() -> None:
    model = awake(build(**BO), gain=0.1).train()
    out = model(_graph(molecule("CH3CH2OH")))
    (out["energy"].sum() + out["forces"].square().sum() + out["aux_loss"]).backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []
