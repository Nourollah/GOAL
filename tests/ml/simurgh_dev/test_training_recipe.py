"""SIMURGH training recipe: optimisers, weight-decay groups, stages and hooks."""

from __future__ import annotations

import pytest
import torch
from ase.build import molecule
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.models.simurgh.dev.monolithic import _newton_schulz
from goal.ml.training.loss import CompositeLoss, EnergyLoss, ForcesLoss, WeightedLoss
from goal.ml.training.module import GOALModule
from tests.ml.simurgh_dev.test_monolithic import awake, build


def _batch() -> Batch:
    graphs = []
    for name in ("CH3CH2OH", "H2O", "CH4"):
        atoms = molecule(name)
        atoms.rattle(0.05, seed=1)
        g = AtomicGraph.from_ase(atoms, cutoff=5.0, dtype=torch.float64)
        g.energy = torch.tensor([0.0], dtype=torch.float64)
        g.forces = torch.randn(
            len(atoms), 3, dtype=torch.float64, generator=torch.Generator().manual_seed(len(atoms))
        )
        graphs.append(g)
    return Batch.from_data_list(graphs)


def _loss(model, batch) -> torch.Tensor:
    out = model(batch)
    return (out["forces"] - batch.forces).square().mean() + out["aux_loss"]


@pytest.mark.parametrize("name", ["amsgrad", "muon", "block_rms_adam"])
def test_every_optimiser_reduces_the_loss(name: str) -> None:
    model = awake(build()).train()
    batch = _batch()
    opt = model.build_optimizer({"name": name, "lr": 1e-3, "muon_lr": 1e-3, "weight_decay": 1e-4})
    before = float(_loss(model, batch))
    for _ in range(10):
        opt.zero_grad()
        _loss(model, batch).backward()
        opt.step()
    assert float(_loss(model, batch)) < 0.99 * before


def test_parameter_groups_follow_the_decay_rules() -> None:
    model = build(a_basis={"regularity_prior_sigma": 0.5, "prior_weight_decay": 0.03})
    opt = model.build_optimizer({"name": "muon", "lr": 1e-3, "weight_decay": 0.01})
    decay = {id(p): g["weight_decay"] for g in opt.param_groups for p in g["params"]}
    kind = {id(p): g["kind"] for g in opt.param_groups for p in g["params"]}
    assert decay[id(model.atom_line.embedding.weight)] == 0.0
    assert decay[id(model.pair_line.table)] == 0.0
    layer = model.atom_line.layers[0]
    assert decay[id(layer.radial[0].weight)] == 0.03  # prior group
    assert all(decay[id(p)] == 0.03 for p in layer.product.parameters())
    assert kind[id(layer.linear_up.weight)] == "muon"
    assert kind[id(layer.radial[-1].weight)] == "muon"
    assert kind[id(model.atom_line.readouts[0].weight)] == "adam"  # a vector, not a matrix
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))


def test_newton_schulz_is_nearly_orthogonal() -> None:
    g = torch.randn(32, 16, dtype=torch.float64)
    s = torch.linalg.svdvals(_newton_schulz(g, 5))
    assert 0.6 < float(s.min()) and float(s.max()) < 1.3


def test_unknown_optimiser_is_refused() -> None:
    with pytest.raises(ValueError, match="optimizer.name"):
        build().build_optimizer({"name": "sgd", "lr": 1e-3})


def test_pair_first_stage_freezes_the_atom_line() -> None:
    model = awake(build(pair_line={"pair_first_epochs": 2})).train()
    batch = _batch()
    model.set_epoch(0)
    _loss(model, batch).backward()
    atom = [p.grad for n, p in model.named_parameters() if n.startswith("atom_line.layers")]
    assert all(g is not None and not g.any() for g in atom)
    assert model.pair_line.table.grad.any()
    model.zero_grad()
    model.set_epoch(2)
    _loss(model, batch).backward()
    assert any(
        p.grad.any() for n, p in model.named_parameters() if n.startswith("atom_line.layers")
    )


def _module(stages: list) -> GOALModule:
    cfg = OmegaConf.create(
        {
            "training": {
                "ema": {"enabled": False},
                "stages": stages,
                "optimizer": {
                    "name": "amsgrad",
                    "lr": 0.01,
                    "weight_decay": 0.0,
                    "scheduler_type": "cosine",
                },
            },
            "trainer": {"max_epochs": 20},
        }
    )
    loss = CompositeLoss(
        [
            WeightedLoss(EnergyLoss("huber", delta=0.001), weight=1.0, label="energy"),
            WeightedLoss(
                ForcesLoss("huber", error="norm", delta=0.01), weight=100.0, label="forces"
            ),
        ]
    )
    return GOALModule(backbone=build(), head=None, loss=loss, config=cfg)


def test_goal_module_uses_the_model_optimiser_and_stage_lr() -> None:
    from types import SimpleNamespace

    stages = [
        {"name": "energy_only", "epochs": 5, "loss_weights": {"energy": 1.0, "forces": 0.0}},
        {"name": "main", "epochs": 10, "loss_weights": {"energy": 1.0, "forces": 100.0}},
        {"name": "switch", "epochs": None, "lr": 2.5e-4, "loss_weights": {"energy": 400.0}},
    ]
    module = _module(stages)
    optimizer = module.configure_optimizers()["optimizer"]
    assert type(optimizer).__name__ == "_SimurghOptimizer"
    module._trainer = SimpleNamespace(optimizers=[optimizer])
    module._maybe_advance_stage(0)
    assert module._weighted_loss_by_label["forces"].weight == 0.0
    assert module.backbone.epoch == 0
    module._maybe_advance_stage(16)
    assert module.backbone.epoch == 16
    assert all(g["lr"] == 2.5e-4 for g in optimizer.param_groups)
    assert module._weighted_loss_by_label["energy"].weight == 400.0
