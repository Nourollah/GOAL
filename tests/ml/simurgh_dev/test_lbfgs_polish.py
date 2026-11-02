"""Float64 L-BFGS stage (C9, arXiv:2311.16326 App. C): the tests of Appendix C-4.3, item 7."""

from __future__ import annotations

import copy
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import molecule
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph
from tests.ml.simurgh_dev.test_monolithic import CUTOFF, awake, build, predict_fn
from tests.ml.simurgh_dev.utils import finite_difference_force_error

SCRIPT = Path(__file__).parents[3] / "scripts" / "simurgh_dev" / "lbfgs_polish.py"


def lbfgs():
    spec = importlib.util.spec_from_file_location("lbfgs_polish", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def data(n: int = 8, seed: int = 0, chunk: int = 4) -> list[Batch]:
    """Jittered ethanol labelled by a fixed random reference model, in chunks."""
    reference = awake(build(seed=7), gain=0.3)
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(n):
        atoms = molecule("CH3CH2OH")
        atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
        g = AtomicGraph.from_ase(atoms, cutoff=CUTOFF, dtype=torch.float64)
        out = reference(Batch.from_data_list([g]))
        g.energy, g.forces = out["energy"].detach(), out["forces"].detach()
        graphs.append(g)
    return [Batch.from_data_list(graphs[k : k + chunk]) for k in range(0, n, chunk)]


def test_float64_and_chunked_gradient_equals_single_pass() -> None:
    m = lbfgs()
    model = m.deterministic(awake(build(dtype="float32"), gain=0.3))
    assert all(p.dtype == torch.float64 for p in model.parameters())
    assert all(b.dtype == torch.float64 for b in model.buffers() if b.is_floating_point())
    chunks = data()
    l2 = m.l2_parameters(model, "product_basis")
    loss_c = m.objective(model, chunks, (1.0, 500.0, 5e-5), l2)
    grad_c = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    whole = data(chunk=8)  # the same eight structures in one pass
    loss_w = m.objective(model, whole, (1.0, 500.0, 5e-5), l2)
    assert loss_c == pytest.approx(loss_w, rel=1e-10)
    for a, b in zip(grad_c, (p.grad for p in model.parameters())):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-14)


def test_l2_reaches_only_its_group() -> None:
    m = lbfgs()
    model = m.deterministic(awake(build()))
    selected = m.l2_parameters(model, "product_basis")
    assert selected and all(
        ".product." in n for n, p in model.named_parameters() if any(p is q for q in selected)
    )
    m.objective(model, data(4), (0.0, 0.0, 1.0), selected)
    for p in model.parameters():
        touched = p.grad is not None and bool(p.grad.abs().sum())
        assert touched == any(p is q for q in selected)
    assert len(m.l2_parameters(model, "all_non_embedding")) > len(selected)


def test_zero_iterations_are_bit_identical_and_runs_are_deterministic() -> None:
    m = lbfgs()
    chunks = data()
    model = awake(build(), gain=0.3)
    before = copy.deepcopy(model.state_dict())
    assert m.polish(model, chunks, chunks, max_iter=0) == []
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    runs = []
    for _ in range(2):
        twin = awake(build(), gain=0.3)
        losses = m.polish(twin, chunks, chunks[:1], max_iter=6, every=2)
        runs.append((losses, copy.deepcopy(twin.state_dict())))
    assert runs[0][0] == runs[1][0]
    assert all(torch.equal(runs[0][1][k], runs[1][1][k]) for k in runs[0][1])
    losses = runs[0][0]
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(losses))  # accepted iterations


def test_convex_quadratic_reaches_its_minimiser() -> None:
    m = lbfgs()

    class Quadratic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.x = torch.nn.Parameter(torch.zeros(3))

        def forward(self, batch):
            target = torch.tensor([1.0, -2.0, 0.5])
            residual = (self.x - target) * torch.tensor([1.0, 3.0, 10.0]).sqrt()
            return {"energy": torch.zeros(1), "forces": residual.reshape(1, 3)}

    class Chunk:
        num_graphs, num_nodes = 1, 1
        energy = torch.zeros(1)
        forces = torch.zeros(1, 3)

        def clone(self):
            return self

    q = Quadratic()
    m.polish(q, [Chunk()], [Chunk()], kappa=(0.0, 1.0, 0.0), max_iter=50, every=10**9)
    torch.testing.assert_close(q.x.detach(), torch.tensor([1.0, -2.0, 0.5]), atol=1e-6, rtol=0)


def test_polished_model_keeps_conservative_smooth_forces() -> None:
    m = lbfgs()
    chunks = data()
    model = awake(build(), gain=0.3)
    m.polish(model, chunks, chunks[:1], max_iter=5, every=5)
    p = predict_fn(model.eval())
    ethanol = molecule("CH3CH2OH")
    assert finite_difference_force_error(p, ethanol) < 1e-6
    water = molecule("H2O")
    from ase import Atoms

    def e(d: float) -> float:
        return p(water + Atoms("C", [[d, 0, 0]]))[0]

    assert abs(e(CUTOFF - 1e-7) - e(CUTOFF + 1e-7)) < 1e-9


def test_refuses_rtx_gpus_without_override(monkeypatch) -> None:
    m = lbfgs()
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA RTX 6000 Ada")
    with pytest.raises(SystemExit, match="allow-rtx"):
        m.refuse_consumer_gpu("cuda:0", allow=False)
    m.refuse_consumer_gpu("cuda:0", allow=True)
    m.refuse_consumer_gpu("cpu", allow=False)
