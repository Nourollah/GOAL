"""ECE branches next to the ACE core: deep fusion (B) and two energy heads (C)."""

from __future__ import annotations

import copy

import numpy as np
import torch
from ase.build import molecule
from scipy.spatial.transform import Rotation

from goal.ml.nn.models.simurgh.dev.ablations.ace_ece_fusion import SimurghAceEceFusion
from goal.ml.nn.models.simurgh.dev.ablations.ace_ece_heads import SimurghAceEceHeads
from goal.ml.training.loss import EnergyLoss, ForcesLoss
from tests.ml.simurgh_dev.test_ece import ECE
from tests.ml.simurgh_dev.test_mfc import alkane, batch_of
from tests.ml.simurgh_dev.test_monolithic import SMALL, _graph, awake, build


def _cfg(**overrides) -> dict:
    cfg = copy.deepcopy(SMALL)
    for key, value in overrides.items():
        cfg[key] = {**cfg[key], **value} if isinstance(cfg.get(key), dict) else value
    return cfg


def fusion(seed: int = 0, **overrides) -> SimurghAceEceFusion:
    torch.manual_seed(seed)
    ece = {**ECE, **overrides.pop("ece", {})}
    return SimurghAceEceFusion(**_cfg(pair_line={"mode": "phi+ece"}, ece=ece, **overrides)).eval()


def heads(seed: int = 0, **overrides) -> SimurghAceEceHeads:
    torch.manual_seed(seed)
    return SimurghAceEceHeads(**_cfg(ece=ECE, **overrides)).eval()


# ---------------------------------------------------------------------------
# B: deep fusion
# ---------------------------------------------------------------------------


def test_fresh_fusion_equals_the_plain_ece_line_model() -> None:
    """Zero cross-links and a zero ECE head: the ACE-only energy, bit for bit."""
    plain = build(pair_line={"mode": "phi+ece"}, ece=ECE)
    fused = fusion()
    fused.load_state_dict(plain.state_dict(), strict=False)
    g = _graph(molecule("CH3CH2OH"))
    a, b = plain(g), fused(g)
    assert torch.allclose(a["energy"], b["energy"], atol=1e-12, rtol=0)
    assert torch.allclose(a["forces"], b["forces"], atol=1e-12, rtol=0)


def test_fusion_every_parameter_gets_a_gradient() -> None:
    model = awake(fusion(), gain=0.1).train()
    out = model(_graph(molecule("CH3CH2OH")))
    (out["energy"].sum() + out["forces"].square().sum() + out["aux_loss"]).backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_fusion_is_rotation_invariant_and_equivariant() -> None:
    model = awake(fusion(), gain=0.1)
    atoms = molecule("CH3CH2OH")
    rot = Rotation.random(random_state=3).as_matrix()
    turned = atoms.copy()
    turned.positions = atoms.positions @ rot.T
    a, b = model(_graph(atoms)), model(_graph(turned))
    r = torch.as_tensor(rot, dtype=torch.float64)
    assert torch.allclose(a["energy"], b["energy"], atol=1e-9)
    assert torch.allclose(a["forces"] @ r.T, b["forces"], atol=1e-8)


def test_fusion_refuses_mismatched_rounds() -> None:
    import pytest

    with pytest.raises(ValueError, match="num_mp_layers"):
        fusion(ece={**ECE, "num_mp_layers": 1})


def test_mfc_windows_run_the_fused_forward() -> None:
    model = fusion(mfc={"enabled": True, "start_epoch": 0, "warmup_epochs": 1, "buffer": 5.0})
    model = awake(model, gain=0.1).train()
    model.mfc.principal_prob = 1.0
    model.mfc.rng = np.random.default_rng(0)
    calls = []
    original = model.line_energies
    model.line_energies = lambda *a: calls.append(a[0].shape[0]) or original(*a)
    terms = model(batch_of(alkane(16)))["aux_terms"]
    assert len(calls) >= 2 and torch.isfinite(terms["mfc_energy"])  # full structure + windows


# ---------------------------------------------------------------------------
# C: two energy heads
# ---------------------------------------------------------------------------


def test_heads_predict_the_mean_and_stay_conservative() -> None:
    model = awake(heads(), gain=0.1)
    atoms = molecule("CH3CH2OH")
    out = model(_graph(atoms))
    assert torch.allclose(out["energy"], 0.5 * (out["energy_ace"] + out["energy_ece"]))
    assert torch.allclose(out["forces"], 0.5 * (out["forces_ace"] + out["forces_ece"]))
    assert not torch.allclose(out["energy_ace"], out["energy_ece"])
    h, k = 1e-4, (2, 1)  # finite difference of the averaged energy, one coordinate
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions[k] += h
    minus.positions[k] -= h
    fd = -(model(_graph(plus))["energy"] - model(_graph(minus))["energy"]) / (2 * h)
    assert abs(float(fd) - float(out["forces"][k])) < 1e-6


def test_heads_every_parameter_gets_a_gradient() -> None:
    model = awake(heads(), gain=0.1).train()
    out = model(_graph(molecule("CH3CH2OH")))
    (out["energy"].sum() + out["forces"].square().sum() + out["aux_loss"]).backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_heads_aux_trains_each_head_on_the_labels() -> None:
    """With GOAL's MSE losses: 1/2 [L(ACE) + L(ECE)] = L(mean) + 1/4 D (no agreement yet)."""
    model = awake(heads(mfc={"enabled": False}), gain=0.1).train()
    model.set_epoch(0)  # before mfc.start_epoch: agreement weight 0
    w = {"energy": 3.0, "forces": 100.0}
    model.set_loss_weights(w)
    g = _graph(molecule("CH3CH2OH"))
    g.energy = torch.tensor([-0.3], dtype=torch.float64)
    g.forces = torch.randn(
        g.num_nodes, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    out = model(g)
    e_loss, f_loss = EnergyLoss("mse"), ForcesLoss("mse", error="norm")

    def loss(energy, forces):
        pred = {"energy": energy, "forces": forces, "num_atoms": out["num_atoms"]}
        return w["energy"] * e_loss(pred, g) + w["forces"] * f_loss(pred, g)

    each = 0.5 * (
        loss(out["energy_ace"], out["forces_ace"]) + loss(out["energy_ece"], out["forces_ece"])
    )
    reg = model._regularisers  # the aux loss also holds the regularisers: take them out
    elem = model.z_to_index[g.atomic_numbers]
    model.head = "ace"
    extra = reg(
        elem,
        g.edge_index,
        torch.zeros(g.num_nodes, dtype=torch.float64),
        model.atom_line.embedding.weight,
    )
    mean = loss(out["energy"], out["forces"]) + (out["aux_loss"] - extra)
    assert torch.allclose(each, mean, rtol=1e-9, atol=1e-12)


def test_mfc_runs_on_each_head() -> None:
    model = heads(mfc={"enabled": True, "start_epoch": 0, "warmup_epochs": 1, "buffer": 5.0})
    model = awake(model, gain=0.1).train()
    model.mfc.principal_prob = 1.0
    model.mfc.rng = np.random.default_rng(0)
    terms = model(batch_of(alkane(16)))["aux_terms"]
    assert {"mfc_energy_ace", "mfc_energy_ece", "agreement_energy", "agreement_forces"} <= set(
        terms
    )
    assert float(terms["mfc_energy_ace"]) != float(terms["mfc_energy_ece"])
