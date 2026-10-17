"""Unit tests for the MACE fine-tune backbone (mocked — no mace-torch)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from goal.ml.nn.models.foundation.mace import MACEFinetune

# ---------------------------------------------------------------------------
# Fine-tuning strategies
# ---------------------------------------------------------------------------


def test_head_only_freezes_interactions(fake_mace) -> None:
    bb = MACEFinetune(model=fake_mace, strategy="head_only")
    grad = {n: p.requires_grad for n, p in bb._model.named_parameters()}
    assert all(not v for n, v in grad.items() if "interactions" in n)
    assert all(v for n, v in grad.items() if "readouts" in n)


def test_full_unfreezes_all(fake_mace) -> None:
    bb = MACEFinetune(model=fake_mace, strategy="full")
    assert all(p.requires_grad for p in bb._model.parameters())


def test_lora_config_created_with_rank(fake_mace, monkeypatch) -> None:
    """With peft mocked, LoRA config must be built with the requested rank."""
    recorded: dict[str, object] = {}

    def fake_lora_config(**kwargs):
        recorded.update(kwargs)
        return MagicMock(name="LoraConfig")

    def fake_get_peft_model(model, config):
        return model  # leave the model unchanged for the test

    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = fake_lora_config
    fake_peft.get_peft_model = fake_get_peft_model
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    MACEFinetune(model=fake_mace, strategy="lora", lora_rank=8, lora_alpha=32.0)
    assert recorded["r"] == 8
    assert recorded["lora_alpha"] == 32.0
    assert isinstance(recorded["target_modules"], list) and recorded["target_modules"]


# ---------------------------------------------------------------------------
# Data adapters
# ---------------------------------------------------------------------------


def test_adapt_input_methane(fake_mace, methane_graph) -> None:
    bb = MACEFinetune(model=fake_mace, strategy="head_only")
    data = bb._adapt_input(methane_graph)
    for key in (
        "positions",
        "node_attrs",
        "edge_index",
        "shifts",
        "unit_shifts",
        "cell",
        "batch",
        "ptr",
    ):
        assert key in data, f"missing MACE key {key!r}"
    n = methane_graph.pos.shape[0]
    e = methane_graph.edge_index.shape[1]
    # one-hot node_attrs (N, n_elements), each row sums to 1
    assert data["node_attrs"].shape == (n, len(fake_mace.atomic_numbers))
    assert torch.allclose(
        data["node_attrs"].sum(-1), torch.ones(n, dtype=data["node_attrs"].dtype)
    )
    # non-periodic methane → shifts (E, 3) all zeros
    assert data["shifts"].shape == (e, 3)
    assert torch.all(data["shifts"] == 0)


def test_adapt_output_shapes(fake_mace, methane_graph) -> None:
    bb = MACEFinetune(model=fake_mace, strategy="head_only")
    batch = next(iter(DataLoader([methane_graph, methane_graph], batch_size=2)))
    out = bb(batch)
    assert set(out) >= {"energy", "forces", "num_atoms"}
    assert out["energy"].shape == (2,)  # (B,)
    assert out["forces"].shape == (batch.pos.shape[0], 3)  # (N, 3)


# ---------------------------------------------------------------------------
# GOALModule compatibility (forward → loss → backward)
# ---------------------------------------------------------------------------


def test_goalmodule_training_step(fake_mace, methane_graph) -> None:
    from omegaconf import OmegaConf

    from goal.ml.training.loss import (
        CompositeLoss,
        EnergyLoss,
        ForcesLoss,
        WeightedLoss,
    )
    from goal.ml.training.module import GOALModule

    loss = CompositeLoss(
        [
            WeightedLoss(EnergyLoss("mae"), 1.0, "energy"),
            WeightedLoss(ForcesLoss("mae"), 1.0, "forces"),
        ]
    )
    cfg = OmegaConf.create({"training": {"ema": {"enabled": False}}})
    bb = MACEFinetune(model=fake_mace, strategy="head_only")
    module = GOALModule(backbone=bb, head=None, loss=loss, config=cfg)
    module.train()

    batch = next(iter(DataLoader([methane_graph, methane_graph], batch_size=2)))
    preds = module(batch)  # GOALModule.forward → monolithic backbone
    losses = module.loss(preds, batch)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    # A trainable (readout) parameter received a gradient.
    assert any(p.grad is not None for n, p in bb._model.named_parameters() if "readouts" in n)


# ---------------------------------------------------------------------------
# E0 re-estimation
# ---------------------------------------------------------------------------


def test_e0_reestimation_returns_finite(fake_mace) -> None:
    from ase.build import molecule

    from goal.ml.data.graph import AtomicGraph
    from goal.ml.nn.models.foundation.base import estimate_reference_energies

    graphs = []
    for e in (-40.0, -41.5, -42.3):
        atoms = molecule("CH4")
        graphs.append(
            AtomicGraph.from_ase(atoms, cutoff=5.0, energy=e, forces=np.zeros((len(atoms), 3)))
        )
    refs = estimate_reference_energies(graphs)
    assert set(refs) == {1, 6}  # H and C
    assert all(np.isfinite(v) for v in refs.values())


def test_set_atomic_energies_in_place(fake_mace) -> None:
    bb = MACEFinetune(model=fake_mace, strategy="head_only")
    bb.set_atomic_energies({1: -13.6, 6: -1028.5})
    buf = bb._model.atomic_energies_fn.atomic_energies
    # z-table is (1, 6, 7, 8) → columns 0 and 1 updated.
    assert float(buf[0, 0]) == pytest.approx(-13.6)
    assert float(buf[0, 1]) == pytest.approx(-1028.5)
