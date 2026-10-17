"""HyperSpec / InvariantGNN must predict non-zero, conservative forces.

Both models used to read the neighbour list's precomputed ``edge_attr`` /
``edge_weight``, which carry no autograd link to ``graph.pos``.  The
energy/forces head then found ``graph.pos.requires_grad`` False and returned
``zeros_like(pos)``, so the force loss was identically zero and no force
learning happened.  They now re-derive edge geometry from positions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.heads.energy_forces import EnergyForcesHead
from goal.ml.nn.models.hyperspec import HyperSpecModel
from goal.ml.nn.models.invariant import InvariantGNN
from goal.ml.training.loss import CompositeLoss
from goal.ml.training.module import GOALModule


def _hyperspec() -> tuple[torch.nn.Module, torch.nn.Module]:
    torch.manual_seed(0)
    backbone = HyperSpecModel(
        num_elements=10,
        embedding_dim=8,
        hidden_channels=4,
        num_interactions=2,
        lmax=1,
        cutoff=4.0,
        num_radial_basis=4,
    )
    head = EnergyForcesHead(irreps_in="4x0e+4x1o", hidden_dim=8)
    return backbone.double(), head.double()


def _invariant() -> tuple[torch.nn.Module, torch.nn.Module]:
    torch.manual_seed(0)
    backbone = InvariantGNN(
        num_elements=10,
        embedding_dim=8,
        hidden_channels=8,
        num_interactions=2,
        cutoff=4.0,
        num_radial_basis=6,
    )
    head = EnergyForcesHead(irreps_in="8x0e", hidden_dim=8)
    return backbone.double(), head.double()


BUILDERS = {"hyperspec": _hyperspec, "invariant_gnn": _invariant}


def _module(name: str) -> GOALModule:
    from omegaconf import OmegaConf

    backbone, head = BUILDERS[name]()
    cfg = OmegaConf.create({"training": {"ema": {"enabled": False}}})
    return GOALModule(backbone=backbone, head=head, loss=CompositeLoss([]), config=cfg)


def _predict(module: GOALModule, atoms: Atoms) -> dict[str, torch.Tensor]:
    graph = AtomicGraph.from_ase(atoms, cutoff=4.0, dtype=torch.float64)
    with torch.enable_grad():
        return module(graph)


@pytest.mark.parametrize("name", list(BUILDERS))
class TestBaselineForces:
    def test_three_atom_forces_are_nonzero(self, name: str) -> None:
        atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        out = _predict(_module(name).eval(), atoms)
        forces = out["forces"]
        assert forces.shape == (3, 3)
        assert torch.isfinite(forces).all()
        assert forces.abs().sum() > 1e-8

    def test_forces_are_minus_energy_gradient(self, name: str) -> None:
        """Central finite differences of the energy reproduce the forces."""
        module = _module(name).eval()
        atoms = molecule("CH3OH")
        forces = _predict(module, atoms)["forces"].detach().numpy()

        h = 1e-5
        for atom, axis in [(0, 0), (1, 2), (4, 1)]:
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[atom, axis] += h
            minus.positions[atom, axis] -= h
            e_plus = float(_predict(module, plus)["energy"])
            e_minus = float(_predict(module, minus)["energy"])
            fd = -(e_plus - e_minus) / (2 * h)
            assert forces[atom, axis] == pytest.approx(fd, rel=1e-4, abs=1e-7)

    def test_net_force_vanishes(self, name: str) -> None:
        """Translation invariance of the energy ⇒ forces sum to zero."""
        forces = _predict(_module(name).eval(), molecule("CH3OH"))["forces"]
        np.testing.assert_allclose(forces.sum(0).detach().numpy(), 0.0, atol=1e-10)

    def test_force_loss_has_gradient_in_training(self, name: str) -> None:
        """create_graph=True in train mode: a force loss reaches the weights."""
        module = _module(name).train()
        out = _predict(module, molecule("CH3OH"))
        (out["forces"] ** 2).sum().backward()
        grads = [p.grad for p in module.backbone.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)

    def test_no_grad_context_does_not_crash(self, name: str) -> None:
        graph = AtomicGraph.from_ase(molecule("CH3OH"), cutoff=4.0, dtype=torch.float64)
        with torch.no_grad():
            out = _module(name).eval()(graph)
        assert out["energy"].shape == (1,)
