"""Physical long-range ablation (C4): reference vectors and the eight integration tests."""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path

import pytest
import torch
from ase import Atoms
from ase.build import molecule

import goal.ml.nn.models  # noqa: F401  (registers the ablation)
from goal.ml.nn.models.simurgh.dev.ablations import physical_long_range as lr
from goal.ml.registry import MODEL_REGISTRY
from tests.ml.simurgh_dev.test_monolithic import (
    SMALL,
    _graph,
    awake,
    build,
    energy,
    predict_fn,
)
from tests.ml.simurgh_dev.utils import (
    far_copy_extensivity,
    finite_difference_force_error,
    symmetry_errors,
)

FIXTURE = Path(__file__).parent / "fixtures" / "longrange_dispersion_reference.csv"
SYMBOL = {"H": 1, "C": 6, "N": 7, "O": 8}


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def model(**long_range):
    torch.manual_seed(0)
    cfg = {**copy.deepcopy(SMALL), "long_range": long_range}
    return MODEL_REGISTRY.get("simurgh_dev_abl_physical_long_range")(**cfg).eval()


def pair_terms(a: str, b: str, h_i: float, h_j: float, dtype=torch.float64):
    (ai, ci), (aj, cj) = lr.FREE_ATOM[SYMBOL[a]], lr.FREE_ATOM[SYMBOL[b]]
    t = lambda x: torch.tensor([x], dtype=dtype)  # noqa: E731
    return lr.dispersion_pair_terms(t(h_i * ai), t(h_j * aj), t(h_i**2 * ci), t(h_j**2 * cj))


@pytest.mark.parametrize(("dtype", "rtol"), [(torch.float64, 1e-9), (torch.float32, 1e-5)])
def test_dispersion_reproduces_the_reference_table(dtype, rtol) -> None:
    rows = list(csv.DictReader(FIXTURE.open()))
    assert len(rows) == 35
    for row in rows:
        a, b = row["pair"].split("-")
        terms = pair_terms(a, b, float(row["h_i"]), float(row["h_j"]), dtype)
        e = 1e3 * lr.dispersion_pair_energy(torch.tensor([float(row["r_A"])], dtype=dtype), terms)
        assert float(e) == pytest.approx(float(row["E_disp_meV"]), rel=rtol)
        for key, column in [
            ("alpha_ij", "alpha_ij"),
            ("C6", "C6"),
            ("osc_mw", "muw"),
            ("C8", "C8"),
            ("C10", "C10"),
            ("Rd", "Rd_bohr"),
        ]:  # 6 decimals
            assert float(terms[key]) == pytest.approx(float(row[column]), abs=2e-6, rel=1e-5)


def test_limits_and_derived_values() -> None:
    cc, hh = pair_terms("C", "C", 1, 1), pair_terms("H", "H", 1, 1)
    zero = torch.tensor([1e-12])
    assert 1e3 * float(lr.dispersion_pair_energy(zero, cc)) == pytest.approx(-3.92906, abs=1e-5)
    assert 1e3 * float(lr.dispersion_pair_energy(zero, hh)) == pytest.approx(-1.23592, abs=1e-5)
    r = torch.tensor([50.0])
    ratio = float(lr.dispersion_pair_energy(r, cc) * (r / lr.BOHR) ** 6 / (-cc["C6"] * lr.HARTREE))
    assert ratio == pytest.approx(1.00218, abs=1e-5)
    assert lr.COULOMB == pytest.approx(lr.HARTREE * lr.BOHR, rel=1e-15)
    assert lr.COULOMB == pytest.approx(14.399645351950548, rel=1e-15)
    for z, (a0, c6) in {1: (4.5, 6.5), 6: (12.0, 46.6), 7: (7.4, 24.2), 8: (5.4, 15.6)}.items():
        assert lr.FREE_ATOM[z] == (a0, c6)
    m = model()
    torch.testing.assert_close(m.alpha0_bohr3, torch.tensor([4.5, 12.0, 7.4, 5.4]))
    fluorine = {"mode": "provided", "values": {1: -1.0, 9: -2.0}}
    with pytest.raises(ValueError, match="no free-atom dispersion data"):
        MODEL_REGISTRY.get("simurgh_dev_abl_physical_long_range")(
            **{**copy.deepcopy(SMALL), "atomic_energies": fluorine}
        )


def test_combination_rule_and_closure() -> None:
    a = torch.tensor([3.1, 9.7])
    c = torch.tensor([5.2, 41.0])  # noqa: E702
    eq19 = lr.dispersion_pair_terms(a[:1], a[1:], c[:1], c[1:])["C6"]
    ts = 2 * c[0] * c[1] / (a[1] / a[0] * c[0] + a[0] / a[1] * c[1])
    assert float(eq19) == pytest.approx(float(ts), rel=1e-12)
    homo = pair_terms("O", "O", 0.7, 0.7)
    assert float(homo["C6"]) == pytest.approx(0.49 * 15.6, rel=1e-12)
    assert float(homo["alpha_ij"]) == pytest.approx(0.7 * 5.4, rel=1e-12)
    ch = pair_terms("C", "H", 1, 1)
    assert float(ch["C6"]) == pytest.approx(17.4039, abs=1e-4) and float(ch["alpha_ij"]) == 8.25
    table_s3 = {1.38: 0.58020, 2.67: 0.45423, 11.1: 0.26518, 16.8: 0.22618, 27.3: 0.18745}
    for alpha, product in table_s3.items():
        assert float(lr.oscillator_product(torch.tensor(alpha))) == pytest.approx(
            product, rel=5e-4
        )
    for z, m in {1: 0.37356, 6: 0.25739, 7: 0.30951, 8: 0.34874}.items():
        assert float(lr.oscillator_product(torch.tensor(lr.FREE_ATOM[z][0]))) == pytest.approx(
            m, abs=1e-5
        )


def test_electrostatic_pair_energies() -> None:
    def e(qi, qj, r):
        return 1e3 * float(
            lr.electrostatic_pair_energy(
                torch.tensor(qi), torch.tensor(qj), torch.tensor(r), 4.0, None
            )
        )

    assert e(0.3, -0.4, 1.0) == pytest.approx(-477.480242, abs=1e-6)
    assert e(0.3, -0.4, 4.0) == pytest.approx(-364.037777, abs=1e-6)
    assert e(0.3, -0.4, 12.0) == pytest.approx(-143.993273, abs=1e-6)
    assert e(1.0, 1.0, 0.5) == pytest.approx(4041.007173, abs=1e-6)


def test_switches_vanish_with_two_derivatives_at_the_cutoff() -> None:
    f = lr.smooth_step(torch.tensor([0.25, 0.5, 0.75]), 0.0, 1.0)
    torch.testing.assert_close(
        f, torch.tensor([0.9350308309, 0.5, 0.0649691691]), atol=1e-10, rtol=0
    )
    rc = 12.0
    terms = pair_terms("C", "O", 1, 1)
    q = torch.tensor(0.3), torch.tensor(-0.4)
    for fn in (
        lambda r: lr.dispersion_pair_energy(r, terms) * lr.smooth_step(r, rc - 2, rc),
        lambda r: lr.electrostatic_pair_energy(*q, r, 4.0, rc),
    ):
        r = torch.tensor([rc - 1e-6], requires_grad=True)
        v = fn(r).sum()
        (d1,) = torch.autograd.grad(v, r, create_graph=True)
        (d2,) = torch.autograd.grad(d1.sum(), r)
        assert abs(float(v)) < 1e-9 and abs(float(d1)) < 1e-7 and abs(float(d2)) < 1e-4
    r = torch.tensor([3.0], requires_grad=True)  # below r_on: bare forces, constant offset
    bare = lr.electrostatic_pair_energy(*q, r, 4.0, None)
    cut = lr.electrostatic_pair_energy(*q, r, 4.0, rc)
    u_c = lr.erf_coulomb(torch.tensor(rc), 4.0)[0]
    assert float(cut - bare) == pytest.approx(float(-lr.COULOMB * q[0] * q[1] * u_c), rel=1e-12)
    (g1,) = torch.autograd.grad(bare.sum(), r)
    (g2,) = torch.autograd.grad(cut.sum(), r)
    assert float(g1) == pytest.approx(float(g2), rel=1e-12)


def test_fresh_model_is_bit_identical_and_ratios_start_at_0_8() -> None:
    ethanol = molecule("CH3CH2OH")
    for constraint in ("uniform", "pair_transfer"):
        assert energy(model(charge_constraint=constraint), ethanol) == energy(build(), ethanol)
    m = model()
    ratios = m.ratios(torch.zeros(4, 8), torch.arange(4))
    torch.testing.assert_close(ratios, torch.full((4,), 0.8))


@pytest.mark.parametrize("constraint", ["uniform", "pair_transfer"])
@pytest.mark.parametrize("total", [-1.0, 0.0, 1.0])
def test_charges_conserve_the_total(constraint: str, total: float) -> None:
    m = awake(model(charge_constraint=constraint), gain=0.5)
    graph = _graph(molecule("CH3CH2OH"))
    graph.total_charge = torch.tensor([total])
    m(graph)
    pos, elem = graph.pos, m.z_to_index[graph.atomic_numbers]
    batch = torch.zeros(len(elem), dtype=torch.long)
    i, j = lr._intramolecular_pairs(batch, 1)
    h = torch.randn(len(elem), 8)
    q = m.charge_head(
        h, elem, batch, 1, torch.stack([i, j]), (pos[j] - pos[i]).norm(dim=-1), graph.total_charge
    )
    assert float(q.sum()) == pytest.approx(total, abs=1e-12)
    assert q.std() > 0


@pytest.mark.parametrize("constraint", ["uniform", "pair_transfer"])
def test_invariants_with_both_terms_on(constraint: str) -> None:
    m = awake(model(charge_constraint=constraint), gain=0.3)
    with torch.no_grad():
        m.a_disp.fill_(1.0)
        m.a_elec.fill_(1.0)
    p = predict_fn(m)
    ethanol = molecule("CH3CH2OH")
    assert max(symmetry_errors(p, ethanol).values()) < 1e-9
    assert finite_difference_force_error(p, ethanol) < 1e-6
    # neutral copies 1000 A apart (the uniform correction is the same per copy here)
    far = ethanol.copy()
    far.translate([1000.0, 0, 0])
    assert abs(p(ethanol + far)[0] - 2 * p(ethanol)[0]) < 1e-6
    assert abs(far_copy_extensivity(p, ethanol)) < 1e-5


def test_amplitudes_and_ratios_are_logged() -> None:
    m = awake(model(), gain=0.3).train()
    out = m(_graph(molecule("CH3CH2OH")))
    terms = out["aux_terms"]
    assert {"lr_a_disp", "lr_a_elec", "lr_ratio_mean_Z1", "lr_ratio_mean_Z6"} <= set(terms)
    (out["energy"].sum() + out["forces"].square().sum() + out["aux_loss"]).backward()
    assert [n for n, p in m.named_parameters() if p.grad is None] == []


def test_gaussian_pair_kernel_uses_the_wp7_module() -> None:
    m = model(electrostatics_kernel="gaussian_pair", charge_constraint="pair_transfer")
    assert m.electrostatics is not None and m.charge_head is None
    assert m.electrostatics.charge.constraint == "pair_transfer"
    with pytest.raises(ValueError, match="unknown long_range"):
        model(sigma_typo=1.0)
    assert math.isfinite(energy(m, molecule("H2O")))
