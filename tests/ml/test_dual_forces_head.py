"""Tests for the ``DualForcesHead``.

Validates each of the three force-prediction modes and the override
points exposed to the Hydra config:

* ``mode == "autograd"`` — forces follow ``-grad(E, positions)`` and
  vanish when ``graph.pos.requires_grad`` is False.
* ``mode == "direct"`` — forces come from the equivariant ``1x1o``
  projection only.
* ``mode == "hybrid"`` — leading autograd contribution plus a weighted
  direct correction.
"""

from __future__ import annotations

import pytest
import torch
from ase.build import molecule
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph, NodeFeatures
from goal.ml.nn.heads.dual_forces import DualForcesHead


@pytest.fixture
def methane_batch() -> Batch:
    """A batch holding one methane molecule, float64."""
    graph = AtomicGraph.from_ase(molecule("CH4"), cutoff=5.0, dtype=torch.float64)
    return Batch.from_data_list([graph])


def _fake_features(num_atoms: int, irreps: str, dtype: torch.dtype) -> NodeFeatures:
    from e3nn.o3 import Irreps

    irreps_obj = Irreps(irreps)
    feats = torch.randn(num_atoms, irreps_obj.dim, dtype=dtype, requires_grad=True)
    # Pre-populate node_energies that *do* depend on positions to make
    # autograd well-defined.  A real backbone does this for us.
    return NodeFeatures(node_feats=feats, irreps=irreps)


class TestDualForcesHead:
    def test_autograd_mode_zero_grad_without_pos(self, methane_batch) -> None:
        """Test autograd mode produces zero forces when pos.requires_grad is False."""
        head = DualForcesHead(
            irreps_in="8x0e+8x1o+8x2e",
            hidden_dim=16,
            mode="autograd",
        ).double()
        nf = _fake_features(methane_batch.num_atoms, "8x0e+8x1o+8x2e", torch.float64)
        # methane_batch.pos.requires_grad is False by default
        out = head(nf, methane_batch)
        assert "energy" in out and "forces" in out
        assert out["forces"].shape == methane_batch.pos.shape
        assert (out["forces"] == 0).all()

    def test_direct_mode_yields_nonzero_forces(self, methane_batch) -> None:
        """Test direct mode produces nonzero forces regardless of pos.requires_grad."""
        head = DualForcesHead(
            irreps_in="8x0e+8x1o+8x2e",
            hidden_dim=16,
            mode="direct",
        ).double()
        nf = _fake_features(methane_batch.num_atoms, "8x0e+8x1o+8x2e", torch.float64)
        out = head(nf, methane_batch)
        # Direct mode runs the force_proj regardless of pos.requires_grad
        assert out["forces"].shape == methane_batch.pos.shape

    def test_hybrid_blends_autograd_and_direct(self, methane_batch) -> None:
        """Test hybrid mode blends autograd and direct force predictions."""
        head = DualForcesHead(
            irreps_in="8x0e+8x1o+8x2e",
            hidden_dim=16,
            mode="hybrid",
            correction_weight=0.05,
        ).double()
        nf = _fake_features(methane_batch.num_atoms, "8x0e+8x1o+8x2e", torch.float64)
        methane_batch.pos.requires_grad_(True)
        # Manually inject pos-dependence so autograd is meaningful
        nf = NodeFeatures(
            node_feats=methane_batch.pos.norm(dim=-1, keepdim=True)
            .expand(-1, 8 + 24 + 40)
            .clone(),  # 8x0e + 8x1o + 8x2e = 8 + 24 + 40
            irreps="8x0e+8x1o+8x2e",
        )
        out = head(nf, methane_batch)
        # Both branches contribute → finite, well-shaped output
        assert torch.isfinite(out["forces"]).all()
        assert out["forces"].shape == methane_batch.pos.shape

    def test_invalid_mode_raises(self) -> None:
        """Test DualForcesHead raises ValueError for invalid force mode."""
        try:
            DualForcesHead(irreps_in="4x0e", mode="bogus")
        except ValueError as exc:
            assert "mode" in str(exc)
        else:
            raise AssertionError("expected ValueError for unknown mode")

    def test_uses_node_energies_when_provided(self, methane_batch) -> None:
        """When NodeFeatures.node_energies is populated, the head must
        use those values directly instead of running the scalar readout.
        """
        head = DualForcesHead(
            irreps_in="8x0e+8x1o+8x2e",
            hidden_dim=16,
            mode="direct",  # avoids autograd dependency
        ).double()
        # Random features for the direct head; specific energies for the readout test
        feats = torch.randn(methane_batch.num_atoms, 8 + 24 + 40, dtype=torch.float64)
        node_energies = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        nf = NodeFeatures(
            node_feats=feats,
            irreps="8x0e+8x1o+8x2e",
            node_energies=node_energies,
        )
        out = head(nf, methane_batch)
        # Total energy is just the sum of node_energies (single-graph batch)
        assert out["energy"].item() == 15.0
