"""``gate_reg`` must see the artisan gates of every SIMURGH family.

``GOALModule.forward`` used to look for ``backbone.artisan_bank.experts`` —
the pre-rename attribute (it is ``.artisans`` now) — and only on the
modular path, so ``gate_values`` was never injected and ``gate_reg`` was
identically zero for every model.  The gates are now found by artisan
class, on the modular and the monolithic path alike.
"""

from __future__ import annotations

import pytest
import torch
from ase.build import molecule
from omegaconf import OmegaConf
from torch import nn

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.heads.energy_forces import EnergyForcesHead
from goal.ml.nn.models.simurgh.arace import MonolithicArace
from goal.ml.nn.models.simurgh.backbone_arace import SimurghAraceBackbone
from goal.ml.training.loss import CompositeLoss, GateRegLoss, WeightedLoss
from goal.ml.training.module import GOALModule

ELEMENTS = (1, 6, 8)
N_PAIRS = len(ELEMENTS) * (len(ELEMENTS) + 1) // 2  # 6
ARTISAN = {
    "architecture": "equivariant",
    "hidden_irreps": "4x0e + 4x1o",
    "num_rbf": 4,
    "radial_hidden": 8,
    "n_scalar_out": 4,
    "final_hidden": 4,
}
COMMON = {
    "elements": ELEMENTS,
    "num_rounds": 2,
    "artisan": ARTISAN,
    "cutoff": 4.0,
    "embedding_dim": 8,
    "num_elements": 9,
    "avg_num_neighbors": 4.0,
}


def _config():
    return OmegaConf.create({"training": {"ema": {"enabled": False}}})


def _loss() -> CompositeLoss:
    return CompositeLoss([WeightedLoss(GateRegLoss(), weight=1.0, label="gate_reg")])


def _modular() -> GOALModule:
    torch.manual_seed(0)
    return GOALModule(
        backbone=SimurghAraceBackbone(**COMMON).double(),
        head=EnergyForcesHead(irreps_in="4x0e+4x1o", hidden_dim=8).double(),
        loss=_loss(),
        config=_config(),
    )


def _monolithic() -> GOALModule:
    torch.manual_seed(0)
    return GOALModule(
        backbone=MonolithicArace(**COMMON).double(), head=None, loss=_loss(), config=_config()
    )


def _legacy() -> GOALModule:
    from tests.ml.test_archive import _build_module, _tiny_config

    return _build_module(_tiny_config())


def _graph() -> AtomicGraph:
    return AtomicGraph.from_ase(molecule("CH3OH"), cutoff=4.0, dtype=torch.float64)


@pytest.mark.parametrize(
    ("factory", "n_gates"),
    [(_modular, 2 * N_PAIRS), (_monolithic, 2 * N_PAIRS), (_legacy, 3)],
    ids=["simurgh_arace", "monolithic_arace", "legacy_simurgh"],
)
class TestGateValues:
    def test_gate_values_injected(self, factory, n_gates: int) -> None:
        module = factory()
        with torch.enable_grad():
            pred = module(_graph())
        assert "gate_values" in pred
        assert pred["gate_values"].shape == (n_gates,)

    def test_gate_reg_is_nonzero_and_trains_the_gates(self, factory, n_gates: int) -> None:
        module = factory().train()
        with torch.enable_grad():
            pred = module(_graph())
        loss = GateRegLoss()(pred, {})
        # Gates initialise at 1.0, so L = sum(g^2) = number of gates.
        assert float(loss) == pytest.approx(n_gates)
        loss.backward()
        gates = [m.gate for m, attr in module._gated_artisans() if attr == "gate"]
        assert gates and all(
            g.grad is not None and float(g.grad) == pytest.approx(2.0) for g in gates
        )


def test_backbone_without_artisans_has_no_gate_values() -> None:
    from goal.ml.nn.models.invariant import InvariantGNN

    module = GOALModule(
        backbone=InvariantGNN(num_elements=10, hidden_channels=8, num_interactions=1).double(),
        head=EnergyForcesHead(irreps_in="8x0e", hidden_dim=8).double(),
        loss=_loss(),
        config=_config(),
    )
    with torch.enable_grad():
        pred = module(_graph())
    assert "gate_values" not in pred
    assert float(GateRegLoss()(pred, {})) == 0.0


def test_frozen_source_copies_are_recognised_by_class_name() -> None:
    """A model rebuilt from a checkpoint's frozen source carries *copies* of
    the artisan classes, which ``isinstance`` against the live ones rejects."""

    class _AracePairArtisan(nn.Module):  # same name, different class object
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Parameter(torch.tensor(1.0))

    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.artisans = nn.ModuleList([_AracePairArtisan(), _AracePairArtisan()])

    module = GOALModule(backbone=_Backbone(), head=None, loss=_loss(), config=_config())
    assert module._gate_values().shape == (2,)


def test_monolithic_model_keeps_its_own_aux_loss() -> None:
    """The monolithic path now gets the side channels too — but never
    overwrites values the model reported itself."""

    class _Model(nn.Module):
        last_aux_loss = torch.tensor(5.0)

        def forward(self, graph):
            return {"energy": torch.zeros(1), "aux_loss": torch.tensor(1.0)}

    module = GOALModule(backbone=_Model(), head=None, loss=_loss(), config=_config())
    assert float(module(_graph())["aux_loss"]) == 1.0
