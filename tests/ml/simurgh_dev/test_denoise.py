"""Auxiliary denoising (C7, rung D1): the applicable tests of Appendix C-4.1, item 9."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase.build import molecule
from e3nn import o3

from tests.ml.simurgh_dev.test_mfc import alkane, batch_of
from tests.ml.simurgh_dev.test_monolithic import (
    _graph,
    awake,
    build,
    load_shared,
    predict_fn,
)
from tests.ml.simurgh_dev.utils import finite_difference_force_error

DN = {"aux": {"denoise": True}}


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def labelled(atoms, seed: int = 0):
    g = _graph(atoms)
    g.forces = torch.randn(len(atoms), 3, generator=torch.Generator().manual_seed(seed))
    return g


def test_eval_outputs_are_bit_identical_with_the_branch_on() -> None:
    ethanol = molecule("CH3CH2OH")
    on, off = awake(build(**DN)), awake(build())
    load_shared(off, on)
    (e1, f1), (e2, f2) = predict_fn(on)(ethanol), predict_fn(off)(ethanol)
    assert e1 == e2 and np.array_equal(f1, f2)
    assert finite_difference_force_error(predict_fn(on), ethanol) < 1e-6


def test_zero_encoding_and_fresh_branch_leave_the_trunk_unchanged() -> None:
    model = build(**DN)
    line, dn = model.atom_line, model.denoise
    assert torch.equal(dn.encode(torch.zeros(3, 9)), torch.zeros(3, dn.encode.irreps_out.dim))
    atoms = molecule("CH3CH2OH")
    pos = torch.tensor(atoms.positions)
    ei = torch.nonzero((torch.cdist(pos, pos) < 5.0) & ~torch.eye(9, dtype=torch.bool)).T
    vec = pos[ei[1]] - pos[ei[0]]
    elem = model.z_to_index[torch.tensor(atoms.numbers)]
    code = dn.encode(torch.randn(9, 9))  # zero-initialised map: zero shift
    a, ha = line(elem, ei, vec, vec.norm(dim=-1), model.avg_num_neighbors, code)
    b, hb = line(elem, ei, vec, vec.norm(dim=-1), model.avg_num_neighbors)
    assert torch.equal(a, b) and torch.equal(ha, hb)


def test_head_and_encoding_are_equivariant() -> None:
    model = awake(build(aux={"denoise": True, "denoise_p": 1.0, "denoise_r": 1.0}), gain=0.3)
    dn = model.denoise
    torch.manual_seed(1)
    rot = o3.rand_matrix()
    f = torch.randn(9, 3)

    def code(v: torch.Tensor) -> torch.Tensor:
        size = v.norm(dim=-1, keepdim=True)
        return size * o3.spherical_harmonics(
            dn.irreps_code, v / size, normalize=False, normalization="component"
        )

    d = dn.irreps_code.D_from_matrix(rot)
    torch.testing.assert_close(code(f @ rot.T), code(f) @ d.T)
    enc_d = dn.encode.irreps_out.D_from_matrix(rot)
    torch.testing.assert_close(dn.encode(code(f @ rot.T)), dn.encode(code(f)) @ enc_d.T)
    # the head: o(R x~, R f~) = R o(x~, f~)
    atoms = molecule("CH3CH2OH")
    x = torch.tensor(atoms.positions)
    edges = torch.nonzero((torch.cdist(x, x) < 5.0) & ~torch.eye(9, dtype=torch.bool)).T
    elem = model.z_to_index[torch.tensor(atoms.numbers)]
    o = dn.predict(model, elem, x, edges, f)
    o_rot = dn.predict(model, elem, x @ rot.T + 1.0, edges, f @ rot.T)
    torch.testing.assert_close(o_rot, o @ o3.Irreps("1x1o").D_from_matrix(rot).T)
    assert o.abs().sum() > 0


def test_mask_rate_copies_and_rebuilt_neighbour_list() -> None:
    model = build(aux={"denoise": True, "denoise_p": 0.5, "denoise_r": 0.25})
    dn = model.denoise
    torch.manual_seed(0)
    structures = [alkane(10, seed=k) for k in range(40)]
    batch = batch_of(*structures)
    _, x_t, mask, _, owner, edges = dn.corrupt(batch.pos, batch.batch, 40)
    copies = int(owner.max()) + 1
    assert 10 <= copies <= 30  # Binomial(40, 0.5)
    assert abs(float(mask.double().mean()) - 0.25) < 0.05
    d = (x_t[edges[1]] - x_t[edges[0]]).norm(dim=-1)
    assert float(d.max()) < 5.0 and torch.equal(owner[edges[0]], owner[edges[1]])
    full = torch.cdist(x_t, x_t)
    expected = int(((full < 5.0) & (owner[:, None] == owner[None, :])).sum()) - len(x_t)
    assert edges.shape[1] == expected  # every pair within r_c of x~, and only those
    model.eval()
    model.denoise.last_copies = -1
    out = model(batch_of(*structures[:3]))  # validation and test: no copies
    assert "aux_terms" not in out and model.denoise.last_copies == -1


def test_zero_weight_gives_the_base_gradients() -> None:
    g = labelled(molecule("CH3CH2OH"))
    on = awake(build(aux={"denoise": True, "denoise_p": 1.0, "denoise_scale": 0.0})).train()
    off = awake(build()).train()
    load_shared(off, on)
    grads = []
    for model in (on, off):
        out = model(g)
        (out["energy"].sum() + out["forces"].square().sum() + out["aux_loss"]).backward()
        grads.append(
            {n: p.grad.clone() for n, p in model.named_parameters() if not n.startswith("denoise")}
        )
    assert grads[0].keys() == grads[1].keys()
    assert all(torch.equal(grads[0][k], grads[1][k]) for k in grads[1])


def test_weight_decays_and_every_parameter_learns() -> None:
    model = awake(build(aux={"denoise": True, "denoise_p": 1.0, "denoise_epochs": 10}), gain=0.3)
    dn = model.denoise
    assert dn.weight(0, 2.0) == pytest.approx(0.0625 * 100 * 4)
    assert dn.weight(5, 2.0) == pytest.approx(0.0625 * 100 * 4 / 2) and dn.weight(10, 2.0) == 0
    out = model.train()(labelled(molecule("CH3CH2OH")))
    assert float(out["aux_terms"]["denoise"]) > 0
    (out["energy"].sum() + out["forces"].square().sum() + out["aux_loss"]).backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []
    with pytest.raises(NotImplementedError, match="D2"):
        build(aux={"denoise": True, "denoise_rung": "D2"})
