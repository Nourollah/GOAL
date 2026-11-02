"""PAN gate (C3, arXiv:2606.15892 Eqs. S4-S8, S14-S15): descriptors, statistics and gate."""

from __future__ import annotations

import itertools

import pytest
import torch
from ase import Atoms
from ase.build import molecule
from e3nn import o3

from goal.ml.data.graph import AtomicGraph
from goal.ml.data.statistics import compute_atom_statistics
from goal.ml.nn.models.simurgh.dev import SimurghDev
from goal.ml.nn.models.simurgh.dev.monolithic import _PANGate, pan_descriptors
from tests.ml.simurgh_dev.test_monolithic import (
    CUTOFF,
    SMALL,
    _graph,
    awake,
    build,
    predict_fn,
)

PAN = {"attention": {"pan_gate": True}}


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _environment(n: int, seed: int, dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    vectors = torch.randn(n, 3, generator=g, dtype=torch.float64)
    vectors = (
        vectors / vectors.norm(dim=-1, keepdim=True) * (1 + 3.5 * torch.rand(n, 1, generator=g))
    )
    return vectors.to(dtype), vectors.norm(dim=-1).to(dtype)


def _pair_sum(vectors: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-6) -> float:
    w = (
        1
        - (lengths / CUTOFF) ** 5 * 21
        + 35 * (lengths / CUTOFF) ** 6
        - 15 * (lengths / CUTOFF) ** 7
    )
    w = torch.where(lengths < CUTOFF, w, torch.zeros_like(w)).double()
    unit = (vectors / lengths.unsqueeze(-1)).double()
    pairs = [(j, k) for j, k in itertools.permutations(range(len(w)), 2)]
    weight = sum(w[j] * w[k] for j, k in pairs)
    mean = sum(w[j] * w[k] * unit[j] @ unit[k] for j, k in pairs) / (eps + weight)
    return float(
        sum(w[j] * w[k] * (unit[j] @ unit[k] - mean) ** 2 for j, k in pairs) / (eps + weight)
    )


@pytest.mark.parametrize(("dtype", "tol"), [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_moment_form_equals_the_pair_sum(dtype, tol) -> None:
    for seed in range(20):
        vectors, lengths = _environment(7, seed, dtype)
        receiver = torch.zeros(7, dtype=torch.long)
        _, eta = pan_descriptors(vectors, lengths, receiver, 1, CUTOFF, 5)[0]
        assert abs(float(eta) - _pair_sum(vectors, lengths)) < tol


def test_descriptor_values_and_invariance() -> None:
    vectors, lengths = _environment(6, 3)
    receiver = torch.zeros(6, dtype=torch.long)
    env = (
        1
        - 21 * (lengths / CUTOFF) ** 5
        + 35 * (lengths / CUTOFF) ** 6
        - 15 * (lengths / CUTOFF) ** 7
    )
    desc = pan_descriptors(vectors, lengths, receiver, 1, CUTOFF, 5)[0]
    assert float(desc[0]) == pytest.approx(float(env.clamp_min(0).sum()), rel=1e-12)
    for few in (1, 2):  # eta vanishes with fewer than three neighbours
        assert (
            abs(
                float(
                    pan_descriptors(vectors[:few], lengths[:few], receiver[:few], 1, CUTOFF, 5)[
                        0, 1
                    ]
                )
            )
            < 1e-12
        )
    torch.manual_seed(0)
    rot = o3.rand_matrix()
    for transform in (rot, -torch.eye(3)):
        moved = pan_descriptors(vectors @ transform.T, lengths, receiver, 1, CUTOFF, 5)[0]
        torch.testing.assert_close(moved, desc, rtol=1e-12, atol=1e-12)
    perm = torch.randperm(6, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(
        pan_descriptors(vectors[perm], lengths[perm], receiver, 1, CUTOFF, 5)[0], desc
    )


def test_eta_and_energy_are_c2_when_a_third_neighbour_crosses_the_cutoff() -> None:
    p = predict_fn(awake(build(**PAN), gain=0.3))
    base = Atoms("OHH", [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])

    def at(d: float) -> tuple[float, float]:
        atoms = base + Atoms("C", [[0.0, -d, 0.0]])
        vec = torch.tensor(atoms.positions[1:] - atoms.positions[0])
        eta = pan_descriptors(
            vec, vec.norm(dim=-1), torch.zeros(3, dtype=torch.long), 1, CUTOFF, 5
        )[0, 1]
        return float(eta), p(atoms)[0]

    h = 1e-4
    for k in range(2):  # eta, then E
        below = [at(CUTOFF - 3 * h + i * h)[k] for i in (-1, 0, 1)]
        above = [at(CUTOFF + 3 * h + i * h)[k] for i in (-1, 0, 1)]
        slope = [(x[2] - x[0]) / (2 * h) for x in (below, above)]
        curv = [(x[2] - 2 * x[1] + x[0]) / h**2 for x in (below, above)]
        assert abs(slope[0] - slope[1]) < 1e-6 and abs(curv[0] - curv[1]) < 1e-2


def test_gate_scales_only_the_scalar_slots_within_0_2() -> None:
    irreps = o3.Irreps("4x0e + 4x1o + 4x0e + 2x2e")
    gate = _PANGate(irreps, 5, 3, descriptors=True)
    assert gate.scalar.numel() == irreps.dim and int(gate.scalar.sum()) == 8
    with torch.no_grad():
        gate.f[-1].weight.normal_(std=50.0)  # saturate the sigmoid
    e, elem = torch.randn(6, 5), torch.tensor([0, 1, 2, 1])
    edges = torch.tensor([[0, 1, 2, 3, 0, 2], [1, 0, 3, 2, 2, 0]])
    messages = torch.randn(6, irreps.dim)
    out = gate(messages, e, elem, edges, torch.randn(4, 2))
    assert torch.equal(out[:, ~gate.scalar], messages[:, ~gate.scalar])  # l > 0 bit-identical
    ratio = out[:, gate.scalar] / messages[:, gate.scalar]
    assert float(ratio.min()) >= 0.0 and float(ratio.max()) <= 2.0


def test_statistics_are_training_split_buffers() -> None:
    model = build(**PAN)
    names = dict(model.named_buffers())
    assert "atom_line.pan_stats" in names and "atom_line.pan_stats" in model.state_dict()
    assert not any("pan_stats" in n for n, _ in model.named_parameters())
    with pytest.raises(ValueError, match="atom_statistics"):
        SimurghDev(**{**SMALL, **PAN, "atom_statistics": None})
    SimurghDev(
        **{
            **SMALL,
            "attention": {"pan_gate": True, "pan_gate_descriptors": "generic"},
            "atom_statistics": None,
        }
    )
    with pytest.raises(ValueError, match="core: ace"):
        build(core="ece", **PAN)


def test_requested_statistics_are_means_over_every_atom() -> None:
    frames = [molecule(name) for name in ("CH3CH2OH", "H2O", "CH3CN")]
    graphs = [AtomicGraph.from_ase(a, cutoff=CUTOFF, dtype=torch.float64) for a in frames]
    descriptors = SimurghDev.atom_statistics({**SMALL, **PAN})
    assert sorted(descriptors) == ["pan_eta", "pan_rho"]
    assert SimurghDev.atom_statistics(SMALL) == {}  # nothing requested without the gate
    stats = compute_atom_statistics(graphs, descriptors)
    rho = torch.cat([descriptors["pan_rho"](g) for g in graphs])
    assert stats["pan_rho"]["mean"] == pytest.approx(float(rho.mean()), rel=1e-12)
    assert stats["pan_rho"]["std"] == pytest.approx(float(rho.std(unbiased=False)), rel=1e-12)
    out = build(**PAN)(_graph(frames[0]))
    assert torch.isfinite(out["energy"]).all()
