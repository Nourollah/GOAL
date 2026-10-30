"""End-to-end tests for the SIMURGH backbone and monolithic variants.

Coverage:

* Smoke test on methane — forward runs, energy is a finite scalar,
  forces are non-zero and sum to ~0 (Newton's third law).
* Equivariance test — rotating positions by ``R`` rotates forces by
  ``R`` and leaves the energy invariant.
* Both modular (backbone + head) and monolithic variants are
  exercised.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.heads.dual_forces import DualForcesHead
from goal.ml.nn.models.simurgh.legacy import SimurghBackbone, SimurghMonolithic
from tests.ml.simurgh.conftest import random_so3


def _build_modular(cutoff: float = 5.0) -> tuple[SimurghBackbone, DualForcesHead]:
    backbone = SimurghBackbone(
        elements=(1, 6, 7, 8),
        dressing_kwargs={
            "num_elements": 9,
            "embedding_dim": 16,
            "hidden_channels": 16,
            "lmax": 2,
            "num_radial_basis": 4,
            "cutoff": cutoff,
            "radial_mlp_hidden": 16,
        },
        artisan_config={
            "scalar_channels": 8,
            "hidden_dims": (32, 16),
            "expert_type": "linear",
        },
        cutoff=cutoff,
    ).double()
    head = DualForcesHead(
        irreps_in=str(backbone.irreps_out),
        hidden_dim=16,
        mode="autograd",
    ).double()
    return backbone, head


def _build_monolithic(
    cutoff: float = 5.0,
    forces_mode: str = "autograd",
    correction_weight: float = 0.0,
    expert_type: str = "linear",
) -> SimurghMonolithic:
    return SimurghMonolithic(
        elements=(1, 6, 7, 8),
        dressing_kwargs={
            "num_elements": 9,
            "embedding_dim": 16,
            "hidden_channels": 16,
            "lmax": 2,
            "num_radial_basis": 4,
            "cutoff": cutoff,
            "radial_mlp_hidden": 16,
        },
        artisan_config={
            "scalar_channels": 8,
            "hidden_dims": (32, 16) if expert_type == "linear" else (16, 16),
            "expert_type": expert_type,
            "transformer_heads": 2,
            "transformer_layers": 1,
        },
        cutoff=cutoff,
        forces_mode=forces_mode,
        correction_weight=correction_weight,
    ).double()


# ----------------------------------------------------------------------
# Smoke tests on methane
# ----------------------------------------------------------------------


class TestMethaneSmoke:
    def test_modular_methane(self, methane_batch) -> None:  # noqa: ANN001
        """Test modular backbone+head on methane produces finite energy and reasonable forces."""
        backbone, head = _build_modular()
        methane_batch.pos.requires_grad_(True)
        nf = backbone(methane_batch)
        out = head(nf, methane_batch)
        assert torch.isfinite(out["energy"]).item()
        assert out["energy"].shape == (1,)
        assert out["forces"].shape == methane_batch.pos.shape
        # Forces non-zero
        assert (out["forces"].abs() > 0).any().item()
        # Newton's third law — total force on the molecule is ~0
        force_sum = out["forces"].sum(dim=0)
        assert force_sum.abs().max().item() < 1e-9

    def test_monolithic_methane(self, methane_batch) -> None:  # noqa: ANN001
        """Test monolithic model produces finite energy and valid forces."""
        model = _build_monolithic()
        out = model(methane_batch)
        assert torch.isfinite(out["energy"]).item()
        assert out["forces"].shape == methane_batch.pos.shape
        assert (out["forces"].abs() > 0).any().item()
        assert out["forces"].sum(dim=0).abs().max().item() < 1e-9

    def test_num_artisans_is_ten(self) -> None:
        """Verify SIMURGH model has 10 experts (pairs for H, C, N, O)."""
        model = _build_monolithic()
        assert model.num_artisans == 10
        gates = model.gates()
        assert set(gates.keys()) == {
            "H-H",
            "H-C",
            "H-N",
            "H-O",
            "C-C",
            "C-N",
            "C-O",
            "N-N",
            "N-O",
            "O-O",
        }

    @pytest.mark.parametrize("expert_type", ["linear", "transformer"])
    def test_expert_backends(self, methane_batch, expert_type: str) -> None:  # noqa: ANN001
        """Test SIMURGH works with both linear and transformer expert backends."""
        model = _build_monolithic(expert_type=expert_type)
        out = model(methane_batch)
        assert torch.isfinite(out["energy"]).item()
        assert torch.isfinite(out["forces"]).all().item()


# ----------------------------------------------------------------------
# Equivariance test
# ----------------------------------------------------------------------


def _rotate_batch(batch: Batch, R: torch.Tensor) -> Batch:
    """Return a fresh ``Batch`` with positions rotated by ``R``."""
    rotated_positions = batch.pos.detach() @ R.T
    graph = AtomicGraph(
        positions=rotated_positions,
        atomic_numbers=batch.atomic_numbers,
        cell=torch.zeros(3, 3, dtype=batch.pos.dtype),
        pbc=torch.zeros(3, dtype=torch.bool),
        edge_index=batch.edge_index,
        # edge_vectors / edge_lengths are recomputed inside the model so
        # we can leave the legacy fields as-is (they will be overridden).
        edge_vectors=batch.edge_attr.detach() @ R.T,
        edge_lengths=batch.edge_weight.detach(),
    )
    return Batch.from_data_list([graph])


class TestEquivariance:
    def _check_rotation_equivariance(
        self,
        model: SimurghMonolithic,
        batch: Batch,
        atol_energy: float = 1e-5,
        atol_forces: float = 1e-5,
    ) -> None:
        # Make sure the model is in eval mode so dropout etc. are off
        model.eval()
        out_orig = model(batch)
        e0, f0 = out_orig["energy"], out_orig["forces"]

        R = random_so3()
        rotated = _rotate_batch(batch, R)
        out_rot = model(rotated)
        e1, f1 = out_rot["energy"], out_rot["forces"]

        # Energy invariance
        assert (
            e0 - e1
        ).abs().max().item() < atol_energy, (
            f"Energy not invariant: orig={e0.item()} rotated={e1.item()}"
        )
        # Forces equivariance: f_rotated == R @ f_original
        rotated_f0 = f0.detach() @ R.T
        diff = (f1.detach() - rotated_f0).abs().max().item()
        assert diff < atol_forces, f"Forces not equivariant — max diff {diff:.3e}"

    def test_modular_equivariance(self, methylamine_batch) -> None:  # noqa: ANN001
        """Test rotation equivariance for modular backbone+head combination."""
        # Build a self-contained "model-like" wrapper that runs backbone
        # + head so we can reuse the rotation-equivariance helper.
        backbone, head = _build_modular()
        backbone.eval()
        head.eval()

        class _Wrap(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, graph: Batch) -> dict[str, torch.Tensor]:
                graph.pos.requires_grad_(True)
                nf = self.backbone(graph)
                return self.head(nf, graph)

        wrapper = _Wrap()
        wrapper.eval()
        self._check_rotation_equivariance(
            wrapper,  # type: ignore[arg-type]
            methylamine_batch,
            atol_energy=1e-5,
            atol_forces=1e-5,
        )

    def test_monolithic_equivariance(self, methane_batch) -> None:  # noqa: ANN001
        """Test rotation equivariance for monolithic SIMURGH model."""
        model = _build_monolithic()
        self._check_rotation_equivariance(model, methane_batch)

    def test_transformer_expert_equivariance(self, methane_batch) -> None:  # noqa: ANN001
        """Test rotation equivariance with transformer expert backend."""
        model = _build_monolithic(expert_type="transformer")
        # Transformer expert uses invariant scalars only, so the full
        # network must remain E(3)-equivariant.
        self._check_rotation_equivariance(model, methane_batch)


# ----------------------------------------------------------------------
# Pre-existing primitive sanity (regression test for the
# WeightedTensorProduct fix that this PR includes).
# ----------------------------------------------------------------------


class TestWeightedTensorProductFix:
    def test_radial_weighted_tp_accepts_per_edge_weights(self) -> None:
        """Test WeightedTensorProduct handles per-edge weights correctly."""
        from e3nn.o3 import spherical_harmonics

        from goal.ml.nn.primitives.tp import WeightedTensorProduct

        tp = WeightedTensorProduct(
            irreps_in1="8x0e+8x1o",
            irreps_in2="1x0e+1x1o",
            irreps_out="8x0e+8x1o",
        ).double()
        E = 4
        x1 = torch.randn(E, tp.irreps_in1.dim, dtype=torch.float64)
        edges = torch.randn(E, 3, dtype=torch.float64)
        x2 = spherical_harmonics(tp.irreps_in2, edges, normalize=True, normalization="component")
        weights = torch.randn(E, tp.weight_numel, dtype=torch.float64)
        out = tp(x1, x2, weights)
        assert out.shape == (E, tp.irreps_out.dim)
