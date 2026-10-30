"""Factorised electrostatics: kernel, charge conservation, invariants."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase.build import molecule

from goal.ml.nn.models.simurgh.dev.monolithic import damped_coulomb
from tests.ml.simurgh_dev.test_mfc import alkane, batch_of
from tests.ml.simurgh_dev.test_monolithic import (
    _graph,
    awake,
    build,
    energy,
    load_shared,
    predict_fn,
)
from tests.ml.simurgh_dev.utils import (
    far_copy_extensivity,
    finite_difference_force_error,
    symmetry_errors,
)

COULOMB = 14.399645


def test_two_unit_charges_have_the_coulomb_asymptote() -> None:
    r = torch.tensor([50.0], dtype=torch.float64)
    t0, b, c = damped_coulomb(r, torch.tensor([1.0], dtype=torch.float64))
    assert float(COULOMB * t0) == pytest.approx(COULOMB / 50.0, rel=1e-12)
    assert float(b) == pytest.approx(50.0**-3, rel=1e-9)
    assert float(c) == pytest.approx(3 * 50.0**-5, rel=1e-9)


def test_kernel_factors_are_its_derivatives() -> None:
    r = torch.linspace(0.8, 6.0, 20, dtype=torch.float64, requires_grad=True)
    sigma = torch.full_like(r, 0.9)
    t0, b, c = damped_coulomb(r, sigma)
    (d1,) = torch.autograd.grad(t0.sum(), r, create_graph=True)
    (d2,) = torch.autograd.grad(d1.sum(), r)
    torch.testing.assert_close(b, -d1 / r)
    torch.testing.assert_close(c, (d2 - d1 / r) / r**2)


@pytest.mark.parametrize("dipoles", [False, True])
def test_charges_are_neutral_per_molecule(dipoles: bool) -> None:
    model = awake(build(electrostatics={"enabled": True, "dipoles": dipoles}))
    out = model(batch_of(alkane(4), molecule("CH3CH2OH"), molecule("H2O")))
    totals = torch.zeros(3, dtype=torch.float64).index_add_(
        0, torch.tensor([0] * 14 + [1] * 9 + [2] * 3), out["charges"]
    )
    assert out["charges"].abs().max() > 0
    assert totals.abs().max() < 1e-12
    assert out["energy_elec"].abs().min() > 0 and out["dipole"].shape == (3, 3)


@pytest.mark.parametrize("dipoles", [False, True])
def test_zero_init_equivalence(dipoles: bool) -> None:
    ethanol = molecule("CH3CH2OH")
    on = build(electrostatics={"enabled": True, "dipoles": dipoles})
    off = build()
    load_shared(off, on)
    assert energy(on, ethanol) == energy(off, ethanol)


@pytest.mark.parametrize("sigma", ["learned", 1.0])
def test_invariants_with_dipoles(sigma) -> None:
    model = awake(build(electrostatics={"enabled": True, "dipoles": True, "sigma": sigma}))
    p = predict_fn(model)
    ethanol = molecule("CH3CH2OH")
    assert abs(far_copy_extensivity(p, ethanol)) < 1e-5
    assert max(symmetry_errors(p, ethanol).values()) < 1e-9
    assert finite_difference_force_error(p, ethanol) < 1e-6


def test_every_parameter_gets_a_gradient() -> None:
    model = awake(build(electrostatics={"enabled": True, "dipoles": True})).train()
    out = model(_graph(molecule("CH3CH2OH")))
    (out["energy"].sum() + out["forces"].square().sum() + out["aux_loss"]).backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_charges_join_the_mfc_loss() -> None:
    model = awake(
        build(
            electrostatics={"enabled": True},
            mfc={"enabled": True, "start_epoch": 0, "warmup_epochs": 1},
        )
    ).train()
    model.mfc.principal_prob = 1.0
    model.mfc.rng = np.random.default_rng(0)
    terms = model(batch_of(alkane(16)))["aux_terms"]
    assert float(terms["mfc_charge"]) > 0


def test_bad_sigma_is_refused() -> None:
    with pytest.raises(ValueError, match="sigma"):
        build(electrostatics={"enabled": True, "sigma": "fixed"})
