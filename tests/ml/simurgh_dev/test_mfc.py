"""Multi-scale fragment consistency (MFC): windows, objectives, gradients, DDP."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from ase import Atoms
from scipy.spatial.transform import Rotation
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.models.simurgh.dev.monolithic import (
    core_mask,
    halfspace_windows,
    sphere_windows,
)
from tests.ml.simurgh_dev.test_monolithic import awake, build

BUFFER = 5.0


def alkane(n: int, seed: int = 0) -> Atoms:
    """Zig-zag CnH2n+2, slightly rattled (about 1.27 Å per carbon along x)."""
    pos, sym = [], []
    for k in range(n):
        y = 0.43 if k % 2 else -0.43
        pos.append([1.27 * k, y, 0.0])
        sym.append("C")
        for z in (0.89, -0.89):
            pos.append([1.27 * k, y + (0.51 if y > 0 else -0.51), z])
            sym.append("H")
    pos.append([-1.0, -0.43, 0.0])
    pos.append([1.27 * (n - 1) + 1.0, 0.43 if (n - 1) % 2 else -0.43, 0.0])
    sym += ["H", "H"]
    atoms = Atoms("".join(sym), positions=pos)
    atoms.rattle(0.02, seed=seed)
    return atoms


def batch_of(*structures: Atoms) -> Batch:
    return Batch.from_data_list(
        [AtomicGraph.from_ase(a, cutoff=5.0, dtype=torch.float64) for a in structures]
    )


def mfc_model(local: bool = False, **mfc) -> torch.nn.Module:
    cfg = {"enabled": True, "start_epoch": 0, "warmup_epochs": 1, "buffer": BUFFER, **mfc}
    model = build(num_interactions=1 if local else 2, mfc=cfg)
    model.mfc.principal_prob = 1.0
    return awake(model).train()


def mfc_terms(model, batch: Batch) -> dict[str, torch.Tensor]:
    model.mfc.rng = np.random.default_rng(0)
    return model(batch)["aux_terms"]


# ---------------------------------------------------------------------------
# Window engine
# ---------------------------------------------------------------------------


def test_core_mask_is_the_buffered_interior() -> None:
    pos = np.array([[0.0, 0, 0], [3.0, 0, 0], [6.0, 0, 0], [12.0, 0, 0]])
    inside = np.array([True, True, True, False])
    core = core_mask(pos, inside, np.ones(4, dtype=bool), 5.0)
    assert core.tolist() == [True, True, True, False]  # 12, 9 and 6 Å from the removed atom
    assert core_mask(pos, inside, np.ones(4, dtype=bool), 6.5).tolist() == [
        True,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize("levels", [2, 3])
def test_halfspace_windows_are_nested_and_valid(levels: int) -> None:
    atoms = alkane(22)
    heavy = atoms.numbers > 1
    windows = halfspace_windows(atoms.positions, heavy, BUFFER, levels, np.random.default_rng(0))
    assert len(windows) == levels - 1
    parent = set(range(len(atoms)))
    for w in windows:  # largest first
        nodes, core = set(w.nodes.tolist()), set(w.core.tolist())
        assert core <= nodes < parent
        removed = atoms.positions[sorted(parent - nodes)]
        d = np.linalg.norm(atoms.positions[w.core][:, None] - removed[None], axis=-1)
        assert d.min() >= BUFFER and heavy[w.core].any() and heavy[sorted(parent - nodes)].any()
        parent = nodes


def test_sphere_windows_follow_the_radii() -> None:
    atoms = alkane(20)
    windows = sphere_windows(
        atoms.positions, atoms.numbers > 1, BUFFER, [8.0, 14.0], np.random.default_rng(1)
    )
    assert [w.parent for w in windows] == [-1, 0]
    assert len(windows[0].nodes) > len(windows[1].nodes)


def test_short_molecule_gets_no_window() -> None:
    atoms = alkane(3)
    assert (
        halfspace_windows(atoms.positions, atoms.numbers > 1, BUFFER, 2, np.random.default_rng(0))
        == []
    )


def test_windows_are_permutation_and_rigid_motion_equivariant() -> None:
    atoms = alkane(16)
    heavy = atoms.numbers > 1

    def nodes(positions: np.ndarray, is_heavy: np.ndarray) -> list[list[int]]:
        rng = np.random.default_rng(3)
        ws = halfspace_windows(positions, is_heavy, BUFFER, 2, rng, principal_prob=1.0)
        return [sorted(w.nodes.tolist()) for w in ws]

    rot = Rotation.random(random_state=2).as_matrix()
    moved = atoms.positions @ rot.T + np.array([3.0, -1.0, 7.0])
    assert nodes(moved, heavy) == nodes(atoms.positions, heavy)
    perm = np.random.default_rng(4).permutation(len(atoms))
    got = nodes(atoms.positions[perm], heavy[perm])
    assert [sorted(perm[np.array(n)].tolist()) for n in got] == nodes(atoms.positions, heavy)


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window_type", ["halfspace", "sphere"])
def test_strictly_local_model_has_zero_mfc(window_type: str) -> None:
    """(a): one layer, cutoff <= buffer: energy and core_force objectives vanish."""
    model = mfc_model(local=True, window_type=window_type, radii=[8.0])
    terms = mfc_terms(model, batch_of(alkane(16), alkane(18, seed=1)))
    assert float(terms["mfc_valid_fraction"]) > 0
    assert float(terms["mfc_energy"]) < 1e-20
    assert float(terms["mfc_core_force"]) < 1e-20


def test_message_passing_model_is_penalised() -> None:
    terms = mfc_terms(mfc_model(), batch_of(alkane(16)))
    assert float(terms["mfc_energy"]) > 0 and float(terms["mfc_core_force"]) > 0


def test_three_levels_and_force_objective_run() -> None:
    model = mfc_model(levels=3, objective=["energy", "core_force", "force"])
    terms = mfc_terms(model, batch_of(alkane(26)))
    assert {"mfc_energy", "mfc_core_force", "mfc_force"} <= set(terms)
    assert all(torch.isfinite(v) for v in terms.values())


def test_mfc_does_not_mutate_the_batch() -> None:
    batch = batch_of(alkane(16))
    before = {k: v.clone() for k, v in batch.to_dict().items() if torch.is_tensor(v)}
    mfc_terms(mfc_model(), batch)
    after = batch.to_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_mfc_is_invariant_to_rigid_motion() -> None:
    model = mfc_model()
    atoms = alkane(16)
    moved = atoms.copy()
    moved.positions = atoms.positions @ Rotation.random(random_state=5).as_matrix().T + 4.0
    a, b = mfc_terms(model, batch_of(atoms)), mfc_terms(model, batch_of(moved))
    for key in ("mfc_energy", "mfc_core_force"):
        assert float(a[key]) == pytest.approx(float(b[key]), rel=1e-6)


def test_stop_grad_parent_keeps_the_teacher_out_of_the_gradient() -> None:
    """(g): no gradient of L_MFC reaches the parent (full-structure) energies."""
    for stop_grad, expect_grad in (("parent", False), ("none", True)):
        model = mfc_model(stop_grad=stop_grad)
        model.mfc.rng = np.random.default_rng(0)
        batch = batch_of(alkane(16))
        pos = batch.pos.clone().requires_grad_(True)
        from goal.ml.nn.models.simurgh.geometry import differentiable_edges

        vec, r = differentiable_edges(batch, pos)
        elem = model.z_to_index[batch.atomic_numbers]
        e_full, _ = model.atom_line(elem, batch.edge_index, vec, r, model.avg_num_neighbors)
        out = model.mfc(model, batch, pos, pos, elem, batch.batch, e_full)
        (grad,) = torch.autograd.grad(out["mfc"], e_full, allow_unused=True)
        got = grad is not None and bool(grad.abs().sum() > 0)
        assert got == expect_grad


def test_core_force_matches_finite_differences() -> None:
    """(i): the core force is -d(Σ_core s e_k)/dr on the window graph."""
    model = mfc_model()
    atoms = alkane(16)
    w = halfspace_windows(
        atoms.positions, atoms.numbers > 1, BUFFER, 2, np.random.default_rng(0), 1.0
    )[0]
    sub = atoms[w.nodes]
    core = np.searchsorted(w.nodes, w.core)

    def q(positions: torch.Tensor) -> torch.Tensor:
        g = Batch.from_data_list([AtomicGraph.from_ase(sub, cutoff=5.0, dtype=torch.float64)])
        vec = positions[g.edge_index[1]] - positions[g.edge_index[0]]
        elem = model.z_to_index[g.atomic_numbers]
        e, _ = model.atom_line(elem, g.edge_index, vec, vec.norm(dim=-1), model.avg_num_neighbors)
        return (model.scale * e[torch.as_tensor(core)]).sum()

    x = torch.tensor(sub.positions, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(q(x), x)
    for i, c in ((int(core[0]), 0), (0, 1), (len(sub) - 1, 2)):
        h = 1e-5
        xp, xm = x.detach().clone(), x.detach().clone()
        xp[i, c] += h
        xm[i, c] -= h
        fd = (q(xp) - q(xm)) / (2 * h)
        assert float(grad[i, c]) == pytest.approx(float(fd), abs=1e-7)


def test_lambda_warms_up_after_start_epoch() -> None:
    model = build(mfc={"enabled": True, "start_epoch": 20, "warmup_epochs": 10, "weight": 2.0})
    assert model.mfc.weight(19) == 0.0
    assert model.mfc.weight(20) == pytest.approx(0.2)
    assert model.mfc.weight(40) == pytest.approx(2.0)


def test_mfc_terms_are_logged_and_added_to_aux_loss() -> None:
    model = mfc_model()
    batch = batch_of(alkane(16))
    model.mfc.rng = np.random.default_rng(0)
    with_mfc = model(batch)
    assert "mfc_energy" in with_mfc["aux_terms"]
    mfc, model.mfc = model.mfc, None
    without = model(batch)["aux_loss"]
    model.mfc = mfc
    assert float(with_mfc["aux_loss"]) > float(without)


def test_fraction_and_every_n_steps_reduce_the_work() -> None:
    model = mfc_model(fraction=0.5)
    graphs = model.mfc.structures(np.repeat(np.arange(6), 3))
    assert len(graphs) == 3 and len(set(graphs)) == 3
    assert model.mfc.structures(np.zeros(4, dtype=int)).tolist() == [0]  # at least one
    terms = mfc_terms(model, batch_of(*(alkane(16, seed=k) for k in range(4))))
    assert float(terms["mfc_energy"]) > 0
    skipping = mfc_model(every_n_steps=2)
    batch = batch_of(alkane(16))
    assert "mfc_energy" not in mfc_terms(skipping, batch)  # step 1 of 2
    assert "mfc_energy" in mfc_terms(skipping, batch)
    with pytest.raises(ValueError, match="fraction"):
        build(mfc={"enabled": True, "fraction": 0.0})


def test_calibrated_tolerance_names_the_file_it_needs() -> None:
    with pytest.raises(NotImplementedError, match="tolerance_file.*d_min_A, d_max_A"):
        build(mfc={"enabled": True, "tolerance": "calibrated"})


def ester(seed: int = 0) -> Atoms:
    """Pentyl pentanoate: BRICS cuts the ester into two groups of about 16 atoms."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCC(=O)OCCCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    pos = mol.GetConformer().GetPositions()
    return Atoms(numbers=[a.GetAtomicNum() for a in mol.GetAtoms()], positions=pos)


def test_chemical_windows_are_capped_groups() -> None:
    from goal.ml.nn.models.simurgh.dev.monolithic import CAP_LENGTH, chemical_windows

    atoms = ester()
    pos, numbers = atoms.positions, atoms.numbers
    windows = chemical_windows(pos, numbers, 1.0, 3, np.random.default_rng(0), "brics")
    assert len(windows) == 2 and windows[1].parent == 0
    assert set(windows[1].nodes) < set(windows[0].nodes) < set(range(len(atoms)))
    for w in windows:
        assert w.caps is not None and len(w.caps) >= 1
        kept = pos[w.nodes]
        d = np.linalg.norm(w.caps[:, None] - kept[None], axis=-1)
        x = w.nodes[d.argmin(axis=1)]  # each cap sits on the line from its kept atom X
        lengths = d.min(axis=1)
        assert np.allclose(lengths, [CAP_LENGTH[z] for z in numbers[x]])
        removed = np.setdiff1d(np.arange(len(atoms)), w.nodes)
        far = np.concatenate([pos[removed], w.caps])
        assert np.linalg.norm(pos[w.core][:, None] - far[None], axis=-1).min() >= 1.0
    with pytest.raises(ValueError, match="H must be in elements"):
        build(mfc={"enabled": True, "window_type": "chemical"}, elements=[6, 8])


def test_chemical_windows_vanish_for_a_strictly_local_model() -> None:
    """Test (f) of WP6 with caps: one layer, cutoff <= buffer."""
    cfg = {"enabled": True, "start_epoch": 0, "warmup_epochs": 1, "buffer": 2.0}
    cfg |= {"window_type": "chemical", "chemical_scheme": "brics", "levels": 2}
    model = awake(build(num_interactions=1, cutoff=2.0, mfc=cfg)).train()
    terms = mfc_terms(model, batch_of(ester(), ester(seed=1)))
    assert float(terms["mfc_valid_fraction"]) > 0
    assert float(terms["mfc_energy"]) < 1e-20 and float(terms["mfc_core_force"]) < 1e-20
    deep = awake(build(cutoff=2.0, mfc=cfg)).train()
    assert float(mfc_terms(deep, batch_of(ester()))["mfc_energy"]) > 0


@pytest.mark.parametrize("levels", [2, 3])
def test_separate_parameters(levels: int) -> None:
    from tests.ml.simurgh_dev.test_monolithic import load_shared

    cfg = {"enabled": True, "start_epoch": 0, "warmup_epochs": 1, "buffer": BUFFER}
    cfg |= {"levels": levels, "radii": [8.0], "parameters": "separate"}
    separate, shared = build(mfc=cfg), build(mfc={**cfg, "parameters": "shared"})
    load_shared(shared, separate)
    chain = batch_of(alkane(16))
    assert torch.equal(separate.eval()(chain)["energy"], shared.eval()(chain)["energy"])
    model = awake(separate).train()
    model.mfc.principal_prob = 1.0
    out = model(chain)
    assert not torch.equal(out["energy"], shared.train()(chain)["energy"])
    (out["energy"].sum() + out["forces"].square().sum() + out["aux_loss"]).backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_violation_weights() -> None:
    model = mfc_model(window_weighting="violation", violation_every=1)
    mfc = model.mfc
    assert np.array_equal(mfc.structure_weights([1, 2]), [1.0, 1.0])  # nothing scored yet
    mfc.violation = {1: 1.0, 2: 3.0, 3: 100.0, 4: 1e-3}
    w = mfc.structure_weights([1, 2, 3, 4, 5])
    assert w[4] == 1.0 and w[:4].mean() == pytest.approx(1.0)
    raw = np.clip(np.array([1.0, 3.0, 100.0, 1e-3]) / np.mean([1.0, 3.0, 100.0, 1e-3]), 0.25, 4)
    assert np.allclose(w[:4], raw / raw.mean())
    model.mfc.violation = {}
    batch = batch_of(alkane(16), alkane(18, seed=1))
    terms = mfc_terms(model, batch)  # epoch 0: a scoring epoch
    assert len(model.mfc.violation) >= 1 and "mfc_weight_max" in terms
    again = mfc_model(window_weighting="violation", violation_every=1)
    mfc_terms(again, batch)
    assert again.mfc.violation == model.mfc.violation  # deterministic for a fixed seed
    picks = mfc_model(window_weighting="violation", fraction=0.5).mfc
    chosen = [
        tuple(picks.structures(np.repeat(np.arange(4), 3), np.array([1e-9, 1e-9, 1.0, 1.0])))
    ]
    assert chosen == [(2, 3)]


# ---------------------------------------------------------------------------
# DDP: every parameter reaches the loss on every step (static graph)
# ---------------------------------------------------------------------------


def _ddp_worker(rank: int, world: int, port: int, results: dict) -> None:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = mfc_model()
    ddp = DistributedDataParallel(model, find_unused_parameters=False, static_graph=True)
    opt = torch.optim.SGD(ddp.parameters(), lr=1e-4)
    # rank 1's second batch is too short for any window: MFC is skipped there
    batches = [batch_of(alkane(16)), batch_of(alkane(16 if rank == 0 else 3))]
    for batch in batches * 2:
        out = ddp(batch)
        loss = out["forces"].square().mean() + out["aux_loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    gathered = [torch.zeros_like(flat) for _ in range(world)]
    dist.all_gather(gathered, flat)
    results[rank] = bool(torch.equal(gathered[0], gathered[1]))
    dist.destroy_process_group()


def test_ddp_static_graph_two_processes() -> None:
    import torch.multiprocessing as mp

    port = 29500 + os.getpid() % 1000
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_ddp_worker, args=(2, port, results), nprocs=2, join=True)
    assert dict(results) == {0: True, 1: True}
