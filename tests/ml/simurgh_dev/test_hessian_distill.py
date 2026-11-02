"""Hessian distillation (C8, arXiv:2501.09009): label format, data layer, loss; the nine tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule
from e3nn import o3
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph
from goal.ml.data.hessian_rows import HessianRows, with_hessian_rows
from tests.ml.simurgh_dev.test_monolithic import CUTOFF, awake, build, load_shared

KD = {"aux": {"hessian_distill": True}}
SCRIPT = Path(__file__).parents[3] / "scripts" / "labels" / "hessian_labels.py"


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def tool():
    spec = importlib.util.spec_from_file_location("hessian_labels", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HarmonicTeacher(torch.nn.Module):
    """Analytic toy teacher: ``E = Σ_{i<j} k (r_ij - r0)²`` over all pairs."""

    def forward(self, graph):
        i, j = torch.triu_indices(len(graph.pos), len(graph.pos), offset=1)
        r = (graph.pos[j] - graph.pos[i]).norm(dim=-1)
        return {"energy": (0.7 * (r - 1.3) ** 2).sum().reshape(1)}


def graph_of(atoms: Atoms) -> AtomicGraph:
    return AtomicGraph.from_ase(atoms, cutoff=CUTOFF, dtype=torch.float64)


def student(model, atoms_list, rows):
    """Training-mode forward with differentiable forces, and the sampled student rows."""
    graphs = []
    for atoms, index in zip(atoms_list, rows):
        g = graph_of(atoms)
        g.hessian_row_number = torch.tensor([index])
        graphs.append(g)
    batch = Batch.from_data_list(graphs)
    batch.pos.requires_grad_(True)
    out = model.train()(batch)
    return model.hessian_rows(batch, batch.pos, out["forces"], batch.batch, len(graphs)), batch


def full_hessian(model, atoms: Atoms) -> torch.Tensor:
    batch = Batch.from_data_list([graph_of(atoms)])
    batch.pos.requires_grad_(True)
    forces = model.train()(batch)["forces"].reshape(-1)
    rows = [
        -torch.autograd.grad(forces[k], batch.pos, retain_graph=True)[0].reshape(-1)
        for k in range(forces.numel())
    ]
    return torch.stack(rows).detach()


@pytest.fixture(scope="module")
def model():
    torch.set_default_dtype(torch.float64)
    return awake(build(**KD), gain=0.3)


def test_vjp_rows_equal_the_full_hessian(model) -> None:
    ethanol = molecule("CH3CH2OH")
    h = full_hessian(model, ethanol)
    rows, _ = student(model, [ethanol], [[4, 17]])
    torch.testing.assert_close(rows[:, 0].reshape(-1).detach(), h[4], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(rows[:, 1].reshape(-1).detach(), h[17], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(h, h.T, rtol=1e-8, atol=1e-8)  # symmetric
    sums = h.reshape(9 * 3, 9, 3).sum(dim=1)  # translation sum rule
    assert float(sums.abs().max()) < 1e-8


def test_hessian_rotates_and_batched_rows_match(model) -> None:
    water = molecule("H2O")
    torch.manual_seed(0)
    rot = o3.rand_matrix()
    turned = water.copy()
    turned.positions = water.positions @ rot.numpy().T
    big = torch.block_diag(*[rot] * 3)
    torch.testing.assert_close(
        full_hessian(model, turned), big @ full_hessian(model, water) @ big.T, rtol=1e-8, atol=1e-8
    )
    ethanol = molecule("CH3CH2OH")
    together, _ = student(model, [water, ethanol], [[2], [11]])
    alone_w, _ = student(model, [water], [[2]])
    alone_e, _ = student(model, [ethanol], [[11]])
    torch.testing.assert_close(together[:3].detach(), alone_w.detach())
    torch.testing.assert_close(together[3:].detach(), alone_e.detach())


def test_finite_difference_rows_converge(model) -> None:
    ethanol = molecule("CH3CH2OH")
    exact, _ = student(model, [ethanol], [[7]])
    fd_model = awake(
        build(aux={"hessian_distill": True, "hessian_mode": "finite_difference"}), gain=0.3
    )
    fd, _ = student(fd_model, [ethanol], [[7]])
    error = float((fd - exact).abs().max())
    assert error < 0.05 * float(exact.abs().max())  # right difference, h = 1e-3 A


def test_hessian_is_continuous_across_the_cutoff(model) -> None:
    def row(d: float) -> torch.Tensor:
        return full_hessian(model, Atoms("CO", [[0, 0, 0], [d, 0, 0]]))

    assert float((row(CUTOFF - 1e-7) - row(CUTOFF + 1e-7)).abs().max()) < 1e-5


def test_label_tool_round_trip(tmp_path) -> None:
    from ase.io import read, write

    labels = tool()
    frames = [molecule("H2O"), molecule("CH4")]
    labelled = labels.label(frames, HarmonicTeacher(), CUTOFF, "harmonic-toy", "analytic")
    write(tmp_path / "h.xyz", labelled, format="extxyz")
    back = read(tmp_path / "h.xyz", index=":")
    for atoms, original in zip(back, frames):
        n = len(atoms)
        stored = atoms.arrays["hessian"].reshape(3 * n, 3 * n)
        again = labels.hessian(HarmonicTeacher(), original, CUTOFF)
        np.testing.assert_allclose(stored, again, rtol=1e-6, atol=1e-5)  # float32 storage
        assert atoms.info["hessian_units"] == "eV/A^2" and atoms.info["hessian_dtype"] == "float32"
        g = graph_of(atoms)  # the data layer reads it, and samples rows for batching
        assert g.hessian.shape == (3 * n, 3 * n)
    dataset = [graph_of(a) for a in back]
    wrapped = with_hessian_rows(dataset)
    assert (
        isinstance(wrapped, HessianRows)
        and with_hessian_rows([graph_of(molecule("H2O"))]) is not None
    )
    item = wrapped[1]
    assert item.hessian_row.shape == (5, 1, 3) and not hasattr(item, "hessian")
    j = int(item.hessian_row_number)
    np.testing.assert_allclose(item.hessian_row[:, 0].reshape(-1), dataset[1].hessian[j])
    batch = Batch.from_data_list([wrapped[0], wrapped[1]])
    assert batch.hessian_row.shape == (8, 1, 3) and batch.hessian_row_number.shape == (2, 1)
    plain = [graph_of(molecule("H2O"))]
    assert with_hessian_rows(plain) is plain


def test_zero_weight_is_base_training_and_the_loss_trains() -> None:
    from ase.io import write  # noqa: F401

    labels = tool()
    frames = labels.label(
        [molecule("H2O"), molecule("CH4")], HarmonicTeacher(), CUTOFF, "toy", "analytic"
    )
    graphs = [HessianRows([graph_of(a) for a in frames])[k] for k in range(2)]
    batch = Batch.from_data_list(graphs)
    on = awake(build(aux={"hessian_distill": True, "hessian_weight": 0.0})).train()
    off = awake(build()).train()
    load_shared(off, on)
    grads = []
    for m in (on, off):
        out = m(batch.clone())
        (out["energy"].sum() + out["forces"].square().sum() + out["aux_loss"]).backward()
        grads.append({n: p.grad.clone() for n, p in m.named_parameters()})
    assert all(torch.equal(grads[0][k], grads[1][k]) for k in grads[1])
    m = awake(build(**KD)).train()
    out = m(batch.clone())
    assert float(out["aux_terms"]["hessian"]) > 0
    (out["aux_loss"]).backward()  # third-order derivatives through every kernel
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
