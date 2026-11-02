"""Extrapolation grade (C6, arXiv:2609.40060): the tests of Appendix C-3.2, item 7."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule
from e3nn import o3

from goal.ml.nn.models.simurgh.dev.extrapolation import (
    GradeBuilder,
    features,
    projection,
    score,
)
from tests.ml.simurgh_dev.test_monolithic import _graph, awake, build, load_shared

CALM = {"diagnostics": {"calm": True}}


@pytest.fixture(scope="module")
def setup():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    model = awake(build(**CALM), gain=0.5)
    rng = np.random.default_rng(0)
    base = molecule("CH3CH2OH")
    frames = []
    for _ in range(300):
        a = base.copy()
        a.positions += rng.normal(scale=0.04, size=a.positions.shape)
        frames.append(a)
    zs, numbers = zip(*(z_of(model, a) for a in frames))
    z, n = torch.cat(zs), torch.cat(numbers)
    builder = GradeBuilder(z.shape[-1])
    builder.pass1(z, n)
    builder.pass2(z, n)
    builder.finish_moments()
    builder.pass3(z, n)
    grade = builder.grade(projection(1), seed=42)
    yield model, grade, frames, z, n
    torch.set_default_dtype(previous)


def z_of(model, atoms: Atoms, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(_graph(atoms))
    r = projection(out["calm_basis"].shape[-1], seed=seed)
    return features(out["calm_basis"].detach(), out["calm_layer_norms"].detach(), r), torch.tensor(
        atoms.numbers
    )


def gamma(model, grade, atoms: Atoms) -> torch.Tensor:
    return score(grade, *z_of(model, atoms))


def test_energies_and_forces_are_bit_identical_with_the_grade_on(setup) -> None:
    on, off = build(**CALM), build()
    load_shared(off, on)
    a, b = on(_graph(molecule("CH3CH2OH"))), off(_graph(molecule("CH3CH2OH")))
    assert torch.equal(a["energy"], b["energy"]) and torch.equal(a["forces"], b["forces"])
    assert "calm_basis" in a and "calm_basis" not in b


def test_gamma_is_invariant(setup) -> None:
    model, grade, frames, *_ = setup
    atoms = frames[0]
    g = gamma(model, grade, atoms)
    moved = atoms.copy()
    moved.positions = moved.positions @ o3.rand_matrix().numpy().T + 3.0
    torch.testing.assert_close(gamma(model, grade, moved), g, rtol=1e-8, atol=1e-8)
    perm = np.random.default_rng(1).permutation(len(atoms))
    torch.testing.assert_close(gamma(model, grade, atoms[perm]), g[perm], rtol=1e-8, atol=1e-8)


def test_sharded_build_equals_single_shard(setup) -> None:
    _, grade, _, z, n = setup
    builder = GradeBuilder(z.shape[-1])
    builder.pass1(z, n)
    for shard in torch.arange(len(n)).chunk(5):
        builder.pass2(z[shard], n[shard])
    builder.finish_moments()
    for shard in torch.arange(len(n)).chunk(3):
        builder.pass3(z[shard], n[shard])
    sharded = builder.grade(projection(1), seed=42)
    for zz, g in grade["elements"].items():
        for key in ("count", "mean", "covariance", "histogram"):  # the additive statistics
            torch.testing.assert_close(
                sharded["elements"][zz][key], g[key], rtol=1e-10, atol=1e-12
            )
        # pinv(Σ + 1e-6 I) of a near-singular Σ amplifies summation-order rounding
        for key in ("precision", "theta"):
            torch.testing.assert_close(sharded["elements"][zz][key], g[key], rtol=1e-5, atol=1e-6)


def test_flagged_fraction_on_the_build_set(setup) -> None:
    _, grade, _, z, n = setup
    pooled = score(grade, z, n)
    fraction = float((pooled > 1).mean(dtype=torch.float64))
    assert 0.0 < fraction < 0.1, fraction  # the source reports 2-5 %


def test_bond_scan_leaves_the_support(setup) -> None:
    model, grade, *_ = setup
    base = molecule("CH3CH2OH")
    c, o = 0, 2  # the C-O bond of ase's ethanol
    assert base.get_chemical_symbols()[o] == "O"
    d_eq = base.get_distance(c, o)

    def max_gamma(scale: float) -> float:
        a = base.copy()
        a.set_distance(c, o, scale * d_eq, fix=0)
        return float(gamma(model, grade, a).max())

    assert max_gamma(1.0) <= 1.0 + 1e-9
    assert max_gamma(0.75) > 1.0 and max_gamma(1.5) > 1.0


def test_isolated_atom_is_far_outside_without_nan(setup) -> None:
    model, grade, *_ = setup
    g = gamma(model, grade, Atoms("C", [[0, 0, 0]]))
    assert torch.isfinite(g).all() and float(g[0]) > 10.0


def test_gamma_is_unchanged_by_a_molecule_beyond_the_receptive_field(setup) -> None:
    model, grade, frames, *_ = setup
    far = molecule("H2O")
    far.translate([100.0, 0, 0])
    g = gamma(model, grade, frames[3])
    torch.testing.assert_close(gamma(model, grade, frames[3] + far)[: len(frames[3])], g)


def test_projection_seeds_agree(setup) -> None:
    model, _, frames, *_ = setup
    fractions = []
    for seed in (42, 43, 44):
        zs, ns = zip(*(z_of(model, a, seed) for a in frames))
        z, n = torch.cat(zs), torch.cat(ns)
        builder = GradeBuilder(z.shape[-1])
        builder.pass1(z, n)
        builder.pass2(z, n)
        builder.finish_moments()
        builder.pass3(z, n)
        fractions.append(
            float((score(builder.grade(projection(1), seed), z, n) > 1).double().mean())
        )
    assert max(fractions) - min(fractions) < 0.013, fractions  # 1.3 percentage points
