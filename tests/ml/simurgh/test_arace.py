"""Tests for the ARACE SIMURGH variant (Part 3).

Coverage:

* Test 5 — forward pass on methane: finite scalar energy, non-zero
  forces, Newton's third law.
* Test 6 — rotation equivariance: energy invariant, forces rotate.
* Test 7 — layer contributions: every artisan layer contributes a
  non-zero energy (not all energy from a single layer).
* Test 8 — shared vs independent artisan weights: both run, and the
  parameter counts differ by exactly ``(num_rounds - 1) ×
  n_artisan_params``.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from goal.ml.nn.models.simurgh.arace import MonolithicArace
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY
from tests.ml.simurgh.conftest import random_so3

CUTOFF = 5.0
NUM_ROUNDS = 2

ARTISAN_SUBCONFIG: dict = {
    "architecture": "equivariant",
    "hidden_irreps": "8x0e + 8x1o + 8x2e",
    "num_layers": 1,
    "num_rbf": 4,
    "radial_hidden": 16,
    "n_scalar_out": 8,
    "final_hidden": 8,
    "element_conditioned": True,
}


def _build_model(
    num_rounds: int = NUM_ROUNDS,
    share_artisan_weights: bool = False,
) -> MonolithicArace:
    return MonolithicArace(
        elements=(1, 6, 7, 8),
        num_rounds=num_rounds,
        share_artisan_weights=share_artisan_weights,
        artisan=dict(ARTISAN_SUBCONFIG),
        cutoff=CUTOFF,
        embedding_dim=16,
        num_elements=9,
    ).double()


def _rotate_batch(batch: Batch, R: torch.Tensor) -> Batch:
    rotated = batch.clone()
    rotated.pos = batch.pos.detach() @ R.T
    if getattr(rotated, "edge_vectors", None) is not None:
        rotated.edge_vectors = batch.edge_vectors @ R.T
    return rotated


# ----------------------------------------------------------------------
# Test 5 — forward pass on methane
# ----------------------------------------------------------------------


class TestForwardPass:
    def test_methane_energy_forces_newton(self, methane_batch) -> None:  # noqa: ANN001
        model = _build_model(num_rounds=2)
        model.eval()
        out = model(methane_batch)

        assert out["energy"].shape == (1,)
        assert torch.isfinite(out["energy"]).item(), "Energy is not finite"

        assert out["forces"].shape == methane_batch.pos.shape
        assert torch.isfinite(out["forces"]).all().item()
        assert (out["forces"].abs() > 0).any().item(), "Forces are identically zero"

        net_force = out["forces"].sum(dim=0).norm().item()
        assert net_force < 1e-5, f"Newton violated: ‖Σ F‖ = {net_force:.3e}"

    def test_registered_in_registries(self) -> None:
        assert BACKBONE_REGISTRY.get("monolithic_arace") is MonolithicArace
        assert MODEL_REGISTRY.get("monolithic_arace") is MonolithicArace


# ----------------------------------------------------------------------
# Test 6 — rotation equivariance
# ----------------------------------------------------------------------


class TestRotationEquivariance:
    def test_energy_invariant_forces_equivariant(self, methylamine_batch) -> None:  # noqa: ANN001
        # Methylamine rather than methane for the same reason as the
        # equivariant-artisan test: perfectly tetrahedral methane sits
        # at a high-symmetry point that is numerically pathological for
        # norm layers acting on symmetry-suppressed l>0 blocks.
        model = _build_model(num_rounds=2)
        model.eval()

        out0 = model(methylamine_batch)
        e0, f0 = out0["energy"], out0["forces"]

        R = random_so3()
        rotated = _rotate_batch(methylamine_batch, R)
        out1 = model(rotated)
        e1, f1 = out1["energy"], out1["forces"]

        e_diff = (e0 - e1).abs().max().item()
        assert e_diff < 1e-5, f"Energy not invariant: {e_diff:.3e}"

        f_diff = (f1.detach() - f0.detach() @ R.T).abs().max().item()
        assert f_diff < 1e-5, f"Forces not equivariant: {f_diff:.3e}"


# ----------------------------------------------------------------------
# Test 7 — layer contributions
# ----------------------------------------------------------------------


class TestLayerContributions:
    def test_every_layer_contributes(self, methane_batch) -> None:  # noqa: ANN001
        model = _build_model(num_rounds=3)
        model.eval()
        out = model(methane_batch)

        layer_e = out["layer_energies"]  # (L, B)
        assert layer_e.shape[0] == 3
        per_layer = layer_e.abs().sum(dim=1)  # (L,)
        for layer_idx, contribution in enumerate(per_layer.tolist()):
            assert contribution > 1e-12, (
                f"Artisan layer {layer_idx} contributes no energy " f"(|E_L| = {contribution:.3e})"
            )
        # Not all energy from a single layer.
        total = per_layer.sum().item()
        assert per_layer.max().item() < total, "All energy comes from a single artisan layer"


# ----------------------------------------------------------------------
# Test 8 — shared vs independent artisan weights
# ----------------------------------------------------------------------


class TestWeightSharing:
    @pytest.mark.parametrize("share", [True, False])
    def test_valid_outputs(self, methane_batch, share: bool) -> None:  # noqa: ANN001
        model = _build_model(num_rounds=2, share_artisan_weights=share)
        model.eval()
        out = model(methane_batch)
        assert torch.isfinite(out["energy"]).item()
        assert torch.isfinite(out["forces"]).all().item()
        assert (out["forces"].abs() > 0).any().item()

    def test_parameter_counts(self) -> None:
        num_rounds = 3
        shared = _build_model(num_rounds=num_rounds, share_artisan_weights=True)
        independent = _build_model(num_rounds=num_rounds, share_artisan_weights=False)

        def n_params(module: torch.nn.Module) -> int:
            # named_parameters deduplicates shared tensors by identity.
            return sum(p.numel() for _, p in module.named_parameters())

        # All artisans share one config → every bank has the same size.
        first_block = independent.backbone.blocks[0]
        n_artisan_params = n_params(first_block.artisans)

        diff = n_params(independent) - n_params(shared)
        expected = (num_rounds - 1) * n_artisan_params
        assert diff == expected, (
            f"Parameter-count difference {diff} != "
            f"(num_rounds - 1) × n_artisan_params = {expected}"
        )
