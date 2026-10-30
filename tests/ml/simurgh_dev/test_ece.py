"""SIMURGH with the edge cluster expansion: core: ece and the ECE bond line."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule

from tests.ml.simurgh_dev.test_mfc import alkane, batch_of
from tests.ml.simurgh_dev.test_monolithic import (
    CUTOFF,
    E0,
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
    isolated_atom,
    symmetry_errors,
)

ECE = {
    "lmax": 2,
    "mmax": 2,
    "channels": 4,
    "bottleneck": 8,
    "message_width": 8,
    "heads": 2,
    "num_mp_layers": 2,
    "energy_mlp": [16],
    "film": {"element_dim": 3, "mlp": [8]},
}


def ece_model(activation: str = "o2_product", grid_points: int = 16, **overrides):
    ece = {**ECE, "activation": activation, "grid_points": grid_points}
    return build(core="ece", ece=ece, **overrides)


@pytest.mark.parametrize("activation", ["o2_product", "fourier"])
def test_core_ece_invariants(activation: str) -> None:
    p = predict_fn(awake(ece_model(activation), gain=0.1))
    ethanol = molecule("CH3CH2OH")
    assert abs(far_copy_extensivity(p, ethanol)) < 1e-5
    assert abs(far_copy_extensivity(p, ethanol, molecule("CH3CN"))) < 1e-5
    for z, e0 in E0.items():
        e, f = isolated_atom(p, z)
        assert e == pytest.approx(e0, abs=1e-12) and not f.any()
    pair = Atoms("CO", positions=[[0, 0, 0], [CUTOFF + 0.01, 0, 0]])
    assert p(pair)[0] == pytest.approx(E0[6] + E0[8], abs=1e-12)
    errors = symmetry_errors(p, ethanol)
    tol = 1e-9 if activation == "o2_product" else 1e-3  # fourier: aliasing under rotation
    assert errors["translation_energy"] < 1e-9 and errors["permutation_energy"] < 1e-9
    assert errors["rotation_energy"] < tol
    assert finite_difference_force_error(p, ethanol) < 1e-6


@pytest.mark.parametrize("activation", ["o2_product", "fourier"])
def test_cutoff_crossing(activation: str) -> None:
    p = predict_fn(awake(ece_model(activation), gain=0.1))
    water = molecule("H2O")

    def e(d: float) -> float:
        return p(water + Atoms("C", positions=[[d, 0.0, 0.0]]))[0]

    assert abs(e(CUTOFF - 1e-7) - e(CUTOFF + 1e-7)) < 1e-9


def _gauge_error(model, atoms: Atoms, seed: int = 0) -> float:
    frames = model.atom_line.frames
    original = frames.forward
    g = torch.Generator().manual_seed(seed)

    def turned(vectors, gauge=None):
        angles = torch.rand(vectors.shape[0], generator=g, dtype=vectors.dtype) * 2 * math.pi
        return original(vectors, angles)

    e_plain = energy(model, atoms)
    frames.forward = turned
    try:
        e_turned = energy(model, atoms)
    finally:
        frames.forward = original
    return abs(e_turned - e_plain) / len(atoms)


def test_gauge_invariance() -> None:
    """Random rotations about each bond axis: exact with o2_product, aliasing with fourier.

    Measured at weight scale 1 (ethanol, eV/atom): fourier 1.6e-6 at 10 grid
    points, 8.7e-10 at 16, below 1e-12 at 32; o2_product 0.
    """
    ethanol = molecule("CH3CH2OH")
    assert _gauge_error(awake(ece_model("o2_product"), gain=1.0), ethanol) < 1e-6
    errors = {
        points: _gauge_error(awake(ece_model("fourier", points), gain=1.0), ethanol)
        for points in (10, 16, 32)
    }
    assert errors[16] < 1e-5
    assert errors[10] > errors[16] >= errors[32]


def test_bond_along_the_singular_direction_is_finite() -> None:
    p = predict_fn(awake(ece_model(), gain=0.1))
    theta = 1e-6
    atoms = Atoms("CO", positions=[[0, 0, 0], [1.2 * math.sin(theta), -1.2 * math.cos(theta), 0]])
    e, f = p(atoms)
    assert np.isfinite(e) and np.isfinite(f).all()


@pytest.mark.parametrize("mode", ["phi+ece", "ece"])
def test_ece_bond_line_starts_at_zero(mode: str) -> None:
    ethanol = molecule("CH3CH2OH")
    line = build(pair_line={"mode": mode}, ece=ECE)
    base = build(pair_line={"enabled": mode != "ece"})
    load_shared(base, line)
    assert energy(line, ethanol) == pytest.approx(energy(base, ethanol), abs=1e-10)
    out = line(_graph(ethanol))
    assert out["edge_energies"].shape == (_graph(ethanol).edge_index.shape[1],)


@pytest.mark.parametrize(
    "overrides",
    [{}, {"electrostatics": {"enabled": True}}],
    ids=["core", "core_charges"],
)
def test_core_ece_every_parameter_gets_a_gradient(overrides: dict) -> None:
    model = awake(ece_model(**overrides), gain=0.1).train()
    out = model(_graph(molecule("CH3CH2OH")))
    (out["energy"].sum() + out["forces"].square().sum() + out["aux_loss"]).backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_ece_line_every_parameter_gets_a_gradient() -> None:
    model = awake(build(pair_line={"mode": "phi+ece"}, ece=ECE), gain=0.1).train()
    out = model(_graph(molecule("CH3CH2OH")))
    (out["energy"].sum() + out["forces"].square().sum() + out["aux_loss"]).backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_mfc_runs_on_the_ece_core() -> None:
    model = awake(
        ece_model(mfc={"enabled": True, "start_epoch": 0, "warmup_epochs": 1}), gain=0.1
    ).train()
    model.mfc.principal_prob = 1.0
    model.mfc.rng = np.random.default_rng(0)
    terms = model(batch_of(alkane(16)))["aux_terms"]
    assert float(terms["mfc_energy"]) >= 0 and torch.isfinite(terms["mfc_core_force"])


def test_core_ece_refuses_a_second_bond_line() -> None:
    with pytest.raises(ValueError, match="pair line stays phi"):
        ece_model(pair_line={"mode": "phi+ece"})
