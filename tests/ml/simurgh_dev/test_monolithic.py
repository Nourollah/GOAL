"""SIMURGH (``simurgh_dev``): physical invariants and section contracts.

Float64 on CPU throughout.  Optional sections start at zero, so the tests
that need a non-trivial model first give every zero-initialised parameter
random values (:func:`awake`).
"""

from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.models.simurgh.dev.monolithic import (
    SimurghDev,
    _es_kernel,
    _SymmetricProductBasis,
    symmetric_coupling_basis,
)
from goal.ml.registry import MODEL_REGISTRY
from tests.ml.simurgh_dev.utils import (
    far_copy_extensivity,
    finite_difference_force_error,
    isolated_atom,
    predictor,
    symmetry_errors,
)

E0 = {1: -13.595298, 6: -1028.481145, 7: -1483.895483, 8: -2041.117735}
CUTOFF = 5.0
SMALL: dict = {
    "num_channels": 8,
    "message_irreps": "8x0e + 8x1o + 8x2e",
    "embedding_dim": 8,
    "radial_mlp": [16, 16],
    "avg_num_neighbors": 6.0,
    "scale": 0.7,
    "dtype": "float64",
    "atomic_energies": {"mode": "provided", "values": E0},
}

#: Flag combinations of the ablation programme implemented so far.
VARIANTS: dict[str, dict] = {
    "base": {},
    "softmax_null": {"attention": {"mode": "softmax_null"}},
    "no_attention": {"attention": {"mode": "none"}},
    "no_pgs": {"readout": {"pgs": False}},
    "no_pair_line": {"pair_line": {"enabled": False}},
    "lowrank_bspline_pair": {"pair_line": {"coefficients": "lowrank", "basis": "bspline"}},
    "correlation_2_embedding_weights": {"correlation": 2, "element_weights": "embedding"},
    "one_layer_nu4": {"num_interactions": 1, "correlation": 4},
    "prior": {"a_basis": {"regularity_prior_sigma": 0.5, "prior_weight_decay": 1e-3}},
    "per_layer_linear_env": {"radial": {"per_layer": ["mlp", "linear"], "env_conditioned": True}},
    "no_pair_code": {"pair_conditioned_radial": False},
}


def build(seed: int = 0, **overrides) -> SimurghDev:
    torch.manual_seed(seed)
    cfg = copy.deepcopy(SMALL)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return SimurghDev(**cfg).eval()


def awake(model: SimurghDev, seed: int = 1, gain: float = 0.2) -> SimurghDev:
    """Give every zero-initialised parameter random values."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.dtype.is_floating_point and not p.abs().sum():
                p.copy_(gain * torch.randn(p.shape, generator=generator, dtype=p.dtype))
    return model


def load_shared(dst: SimurghDev, src: SimurghDev) -> None:
    """Copy ``src`` weights into ``dst``; a wider input layer contributes its leading columns."""
    source = src.state_dict()
    target = dst.state_dict()
    for key, value in target.items():
        if key in source:
            s = source[key]
            value.copy_(s if s.shape == value.shape else s[..., : value.shape[-1]])


def predict_fn(model: SimurghDev):
    return predictor(model, cutoff=CUTOFF)


def energy(model: SimurghDev, atoms: Atoms) -> float:
    return predict_fn(model)(atoms)[0]


@pytest.fixture(scope="module")
def ethanol() -> Atoms:
    return molecule("CH3CH2OH")


# ---------------------------------------------------------------------------
# Section 5: physical invariants, for every implemented flag combination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_extensive_isolated_and_symmetric(variant: str, ethanol: Atoms) -> None:
    p = predict_fn(awake(build(**VARIANTS[variant])))
    assert abs(far_copy_extensivity(p, ethanol)) < 1e-5
    assert abs(far_copy_extensivity(p, ethanol, molecule("CH3CN"))) < 1e-5
    for z, e0 in E0.items():
        e, f = isolated_atom(p, z)
        assert e == pytest.approx(e0, abs=1e-12)
        assert not f.any()
    errors = symmetry_errors(p, ethanol)
    assert max(errors.values()) < 1e-9, errors


@pytest.mark.parametrize("variant", ["base", "softmax_null", "lowrank_bspline_pair", "prior"])
def test_forces_match_finite_differences(variant: str, ethanol: Atoms) -> None:
    p = predict_fn(awake(build(**VARIANTS[variant])))
    assert finite_difference_force_error(p, ethanol) < 1e-6


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_cutoff_crossing_is_smooth(variant: str) -> None:
    """Section 5 tests 6, 12 and 17: an atom crossing r_c changes nothing abruptly."""
    p = predict_fn(awake(build(**VARIANTS[variant])))
    water = molecule("H2O")

    def config(d: float) -> Atoms:
        return water + Atoms("C", positions=[[d, 0.0, 0.0]])

    d_in, d_out = CUTOFF - 1e-7, CUTOFF + 1e-7  # nearest atom (O at the origin)
    e_in, f_in = p(config(d_in + 0.0))
    e_out, f_out = p(config(d_out))
    assert abs(e_in - e_out) < 1e-9
    assert np.abs(f_in - f_out).max() < 1e-6
    # beyond every cutoff the C atom is exactly a separated atom
    far = config(CUTOFF + 2.0)
    assert p(far)[0] == pytest.approx(p(water)[0] + E0[6], abs=1e-9)
    # a 1000-point scan across r_c has no jump in E or F
    distances = np.linspace(CUTOFF - 0.5, CUTOFF + 0.5, 1000)
    energies = np.array([p(config(d))[0] for d in distances[::10]])
    assert np.abs(np.diff(energies)).max() < 1e-3


def test_separated_pair_is_sum_of_e0() -> None:
    p = predict_fn(awake(build()))
    pair = Atoms("CO", positions=[[0, 0, 0], [CUTOFF + 0.01, 0, 0]])
    assert p(pair)[0] == pytest.approx(E0[6] + E0[8], abs=1e-12)


# ---------------------------------------------------------------------------
# Zero-initialised sections: switching one on leaves a fresh model unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section",
    [
        {"pair_line": {"enabled": True}},
        {"attention": {"mode": "gate"}},
        {"readout": {"pgs": True}},
        {"radial": {"env_conditioned": True}},
    ],
    ids=["pair_line", "attention_gate", "pgs", "env_conditioned"],
)
def test_zero_init_equivalence(section: dict, ethanol: Atoms) -> None:
    off = {
        key: {k: (False if isinstance(v, bool) else "none") for k, v in value.items()}
        for key, value in section.items()
    }
    with_section, without = build(**section), build(**off)
    load_shared(without, with_section)
    assert energy(with_section, ethanol) == energy(without, ethanol)


def test_attention_gate_is_exactly_uniform_when_fresh(ethanol: Atoms) -> None:
    gate, uniform = build(attention={"mode": "gate"}), build(attention={"mode": "none"})
    load_shared(uniform, gate)
    out_gate, out_uniform = gate(_graph(ethanol)), uniform(_graph(ethanol))
    assert torch.equal(out_gate["energy"], out_uniform["energy"])
    assert torch.equal(out_gate["forces"], out_uniform["forces"])


def _graph(atoms: Atoms) -> Batch:
    return Batch.from_data_list([AtomicGraph.from_ase(atoms, cutoff=CUTOFF, dtype=torch.float64)])


def test_softmax_null_lone_neighbour_fades_smoothly() -> None:
    model = awake(build(attention={"mode": "softmax_null"}))
    gate = model.atom_line.layers[0].attention
    messages = torch.ones(1, gate.dim, dtype=torch.float64)
    h = torch.ones(2, model.config["num_channels"], dtype=torch.float64)
    receiver = torch.tensor([1])
    weights = []
    for d in (CUTOFF - 0.2, CUTOFF - 0.1, CUTOFF - 0.01, CUTOFF - 1e-4):
        r = torch.tensor([d], dtype=torch.float64, requires_grad=True)
        env = torch.clamp(1 - r / CUTOFF, min=0) ** 6  # any envelope -> 0 at r_c
        rbf = model.atom_line.rbf(r)
        a = gate(messages, h, receiver, rbf, env, 2, 1.0)
        weights.append(float(a[1, 0]))
        score_grad = torch.autograd.grad(a[1, 0], gate.temperature, retain_graph=True)[0]
    assert weights == sorted(weights, reverse=True) and weights[-1] < 1e-12
    assert all(np.isfinite(weights))
    assert score_grad is not None


# ---------------------------------------------------------------------------
# WP2: product basis and regularity prior
# ---------------------------------------------------------------------------


def test_product_basis_multiplicities_are_exact() -> None:
    """Sym³ of an l = 1 block holds exactly one L = 1 and one L = 3; no L = 2."""
    counts = {L: symmetric_coupling_basis([(1, -1)], L, (-1) ** L, 3).shape[-1] for L in range(4)}
    assert counts == {0: 0, 1: 1, 2: 0, 3: 1}
    # Sym² of l = 1: L = 0 and L = 2 (the antisymmetric L = 1 part is excluded)
    assert [symmetric_coupling_basis([(1, -1)], L, 1, 2).shape[-1] for L in range(3)] == [1, 0, 1]


def test_product_basis_is_homogeneous_and_bias_free() -> None:
    torch.manual_seed(0)
    from e3nn import o3

    irreps = o3.Irreps("4x0e + 4x1o + 4x2e")
    basis = _SymmetricProductBasis(irreps, irreps, 3, 2, "index", 4).double()
    a = torch.randn(5, 4, 9, dtype=torch.float64)
    elem, emb = torch.tensor([0, 1, 0, 1, 1]), torch.zeros(2, 4, dtype=torch.float64)
    ts = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0], dtype=torch.float64)
    outs = torch.stack([basis(t * a, elem, emb) for t in ts])  # (5, N, C, D)
    vander = torch.stack([ts, ts**2, ts**3], dim=1)  # degrees 1..3, no constant term
    coef = torch.linalg.lstsq(vander, outs.reshape(5, -1)).solution
    assert torch.allclose(vander @ coef, outs.reshape(5, -1), atol=1e-10)
    assert not basis(torch.zeros_like(a), elem, emb).any()


def test_hidden_features_are_equivariant(ethanol: Atoms) -> None:
    from e3nn import o3

    model = awake(build())
    line = model.atom_line
    rot = o3.rand_matrix(dtype=torch.float64)

    def features(atoms: Atoms) -> torch.Tensor:
        g = _graph(atoms)
        elem = model.z_to_index[g.atomic_numbers]
        vec = g.pos[g.edge_index[1]] - g.pos[g.edge_index[0]]
        h = line.embed_to_h(line.embedding.weight[elem])
        sh = o3.spherical_harmonics(line.irreps_sh, vec, normalize=True, normalization="component")
        r = vec.norm(dim=-1)
        from goal.ml.nn.models.simurgh.dev.monolithic import polynomial_envelope

        env = polynomial_envelope(r / CUTOFF, 5)
        e = line.embedding.weight
        ei, ej = e[elem[g.edge_index[0]]], e[elem[g.edge_index[1]]]
        code = line.pair_code(torch.cat([ei + ej, ei * ej], dim=-1))
        return line.layers[0](
            h, elem, e, g.edge_index, sh * line.sh_factor, line.rbf(r), code, env, 6.0
        )

    rotated = ethanol.copy()
    rotated.positions = ethanol.positions @ rot.numpy().T
    d = model.atom_line.irreps_a.D_from_matrix(rot)
    torch.testing.assert_close(features(rotated), features(ethanol) @ d.T, atol=1e-6, rtol=0)


def test_regularity_prior_factors_and_weight_decay_requirement() -> None:
    sigma, rc = 0.5, CUTOFF
    model = build(a_basis={"regularity_prior_sigma": sigma, "prior_weight_decay": 1e-3})
    n = torch.arange(1, 9, dtype=torch.float64)
    expected = torch.exp(-5 * sigma**2 * n**2 / (2 * rc**2))
    torch.testing.assert_close(model.atom_line.rbf.factor, expected)
    l_of_component = torch.cat([torch.full((2 * ell + 1,), ell) for ell in range(4)]).double()
    torch.testing.assert_close(
        model.atom_line.sh_factor, torch.exp(-(sigma**2) * l_of_component**2 / (2 * rc**2))
    )
    with pytest.raises(ValueError, match="prior_weight_decay"):
        build(a_basis={"regularity_prior_sigma": sigma})


# ---------------------------------------------------------------------------
# WP8 A: PGS readout
# ---------------------------------------------------------------------------


def test_es_kernel_vanishes_with_two_continuous_derivatives() -> None:
    z = torch.tensor([1.0 - 1e-4, 1.0, 1.0 + 1e-4], dtype=torch.float64, requires_grad=True)
    k = _es_kernel(z, 4.0)
    (d1,) = torch.autograd.grad(k.sum(), z, create_graph=True)
    (d2,) = torch.autograd.grad(d1.sum(), z)
    assert k.abs().max() < 1e-7 and d1.abs().max() < 1e-4 and d2.abs().max() < 1e-1
    assert torch.isfinite(d2).all()


def test_pgs_sets_its_width_on_the_first_training_batch(ethanol: Atoms) -> None:
    model = build()
    assert not bool(model.atom_line.pgs.initialised)
    model.train()(_graph(ethanol))
    assert bool(model.atom_line.pgs.initialised)
    assert (model.atom_line.pgs.width != 1).any()


# ---------------------------------------------------------------------------
# WP3: pair line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coefficients", ["table", "lowrank"])
def test_pair_potential_symmetric_and_smooth_at_cutoff(coefficients: str) -> None:
    model = awake(build(pair_line={"coefficients": coefficients}))
    pl, emb = model.pair_line, model.atom_line.embedding.weight
    r = torch.linspace(0.7, CUTOFF + 0.5, 50, dtype=torch.float64, requires_grad=True)
    a = torch.full_like(r, 1, dtype=torch.long)
    b = torch.full_like(r, 3, dtype=torch.long)
    ab, ba = pl.learned(r, a, b, emb), pl.learned(r, b, a, emb)
    torch.testing.assert_close(ab, ba, rtol=0, atol=0)
    (slope,) = torch.autograd.grad(ab.sum(), r)
    beyond = r >= CUTOFF
    assert not ab[beyond].any() and not slope[beyond].any()
    edge = torch.tensor([CUTOFF - 1e-6], dtype=torch.float64, requires_grad=True)
    value = pl.learned(edge, a[:1], b[:1], emb)
    (edge_slope,) = torch.autograd.grad(value.sum(), edge)
    assert abs(float(value)) < 1e-12 and abs(float(edge_slope)) < 1e-9


def test_pair_dispatch_does_not_scale_with_element_pairs() -> None:
    """One gather per edge: the op count is the same for 2 and 4 elements."""
    from torch.profiler import ProfilerActivity, profile

    def count_ops(values: dict) -> int:
        model = build(atomic_energies={"mode": "provided", "values": values})
        r = torch.rand(100, dtype=torch.float64) + 1.0
        idx = torch.zeros(100, dtype=torch.long)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            model.pair_line.learned(r, idx, idx, model.atom_line.embedding.weight)
        return len(prof.events())

    two = {z: E0[z] for z in (1, 6)}
    assert count_ops(two) == count_ops(E0)


def test_zbl_wall_acts_only_below_r_min() -> None:
    model = build()
    pl = model.pair_line
    r = torch.tensor([0.3, 0.5, 0.6, 0.9, 2.0], dtype=torch.float64)
    a = torch.full((5,), 1, dtype=torch.long)
    wall = pl.wall_energy(r, a, a)
    assert (wall[:2] > 0).all() and not wall[2:].any()


# ---------------------------------------------------------------------------
# Fragment decomposition (output only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_fragment_energies_sum_to_the_local_energy(dtype: str) -> None:
    model = awake(build(dtype=dtype))
    atoms = molecule("CH3CH2OH")
    graph = AtomicGraph.from_ase(atoms, cutoff=CUTOFF, dtype=torch.float64)
    graph.fragment_index = torch.tensor([0, 0, 1, 1, 0, 0, 0, 0, 0])
    out = model(Batch.from_data_list([graph]))
    k = out["fragment_energies"]
    assert k.dtype == torch.float64
    assert k.shape == (2,)
    torch.testing.assert_close(k.sum(), out["atomic_energies"].sum())
    pair = out["fragment_pair_energies"]
    torch.testing.assert_close(pair, pair.T)
    assert pair.diagonal().abs().max() == 0


# ---------------------------------------------------------------------------
# WP1: D3(BJ) dispersion term
# ---------------------------------------------------------------------------


def test_d3_matches_dftd3_and_has_no_parameters(ethanol: Atoms) -> None:
    from dftd3.interface import DispersionModel, RationalDampingParam

    with_d3 = build(dispersion={"enabled": True, "backend": "precomputed"})
    assert not list(with_d3.dispersion.parameters())
    without = build()
    load_shared(without, with_d3)
    e_with, f_with = predict_fn(with_d3)(ethanol)
    e_without, f_without = predict_fn(without)(ethanol)
    bohr, hartree = 0.52917721067, 27.211386245988
    ref = DispersionModel(ethanol.numbers, ethanol.positions / bohr).get_dispersion(
        RationalDampingParam(s6=1.0, a1=0.4289, s8=0.7875, a2=4.4407, s9=0.0), grad=True
    )
    assert e_with - e_without == pytest.approx(ref["energy"] * hartree, abs=1e-9)
    np.testing.assert_allclose(f_with - f_without, -ref["gradient"] * hartree / bohr, atol=1e-9)
    assert finite_difference_force_error(predict_fn(with_d3), ethanol) < 1e-6
    assert max(symmetry_errors(predict_fn(with_d3), ethanol).values()) < 1e-8


def test_d3_disabled_is_bit_identical(ethanol: Atoms) -> None:
    a, b = build(), build(dispersion={"enabled": False, "backend": "precomputed"})
    assert energy(a, ethanol) == energy(b, ethanol)


def test_d3_torch_backend_needs_its_package() -> None:
    try:
        import tad_dftd3  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="tad-dftd3"):
            build(dispersion={"enabled": True, "backend": "torch"})
    else:  # pragma: no cover - only where tad-dftd3 is installed
        pytest.skip("tad-dftd3 installed: parity is tested in test_d3_backends_agree")


# ---------------------------------------------------------------------------
# Configuration contract
# ---------------------------------------------------------------------------


def test_registry_and_config_validation() -> None:
    assert MODEL_REGISTRY.get("simurgh_dev") is SimurghDev
    with pytest.raises(ValueError, match="unknown SIMURGH config key"):
        build(not_a_key=1)
    with pytest.raises(ValueError, match="unknown SIMURGH config key"):
        build(pair_line={"typo": 1})
    with pytest.raises(ValueError, match="avg_num_neighbors"):
        build(avg_num_neighbors=None)
    with pytest.raises(ValueError, match="scale"):
        build(scale="force_rms")
    with pytest.raises(ValueError, match="atomic_energies"):
        build(atomic_energies={"mode": "average", "values": None})
    with pytest.raises(ValueError, match="E0"):
        build(elements=[1, 6, 9])


def test_unknown_element_raises() -> None:
    with pytest.raises(ValueError, match="built for elements"):
        energy(build(), Atoms("HF", positions=[[0, 0, 0], [0.9, 0, 0]]))


def test_float32_model_keeps_float64_totals(ethanol: Atoms) -> None:
    single, double = build(dtype="float32"), build()
    out = single(_graph(ethanol))
    assert out["energy"].dtype == torch.float64 and out["forces"].dtype == torch.float32
    assert float(out["energy"]) == pytest.approx(energy(double, ethanol), abs=1e-3)


def test_every_parameter_gets_a_gradient(ethanol: Atoms) -> None:
    """DDP without find_unused_parameters: every parameter must reach the loss."""
    model = awake(build()).train()
    out = model(_graph(ethanol))
    loss = out["energy"].sum() + out["forces"].square().sum() + out["aux_loss"]
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_pair_combinations_cover_the_lookup() -> None:
    model = build()
    n = len(model.elements)
    ids = {int(model.pair_line.pair_id[a, b]) for a, b in itertools.product(range(n), repeat=2)}
    assert ids == set(range(n * (n + 1) // 2))


def test_pair_potential_export() -> None:
    import importlib.util
    from pathlib import Path

    script = Path(__file__).parents[3] / "scripts" / "simurgh_dev" / "export_pair_potentials.py"
    spec = importlib.util.spec_from_file_location("export_pair_potentials", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.pair_potentials(awake(build()), points=50)
    assert sorted(k for k in data if k != "r") == sorted(
        ["H-H", "H-C", "H-N", "H-O", "C-C", "C-N", "C-O", "N-N", "N-O", "O-O"]
    )
    co = data["C-O"]
    np.testing.assert_allclose(co["total"], co["phi"] + co["wall"])
    assert co["total"][-1] == pytest.approx(0.0, abs=1e-12)  # at the pair cutoff
