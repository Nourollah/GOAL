"""Tests for the equivariant PotentialArtisan architecture (Part 2).

Coverage:

* Test 1 — pair symmetry: swapping ``features_A`` and ``features_B``
  leaves the equivariant artisan energy unchanged (guaranteed by the
  shared symmetric node embedding).
* Test 2 — rotation equivariance of the full monolithic model for both
  ``architecture="scalar"`` and ``architecture="equivariant"``: energy
  invariant, forces rotate with the molecule.
* Test 3 — zero masking: zero input tensors produce exactly zero
  output from the equivariant artisan (all maps are bias-free).
* Test 4 — Newton's third law on methane for both architectures.
"""

from __future__ import annotations

import pytest
import torch
from e3nn.o3 import Irreps, spherical_harmonics
from torch_geometric.data import Batch

from goal.ml.nn.blocks.artisans import (
    ArtisanConfig,
    PotentialArtisan,
    _EquivariantArtisanCore,
)
from goal.ml.nn.models.simurgh import SimurghMonolithic
from tests.ml.simurgh.conftest import random_so3

IRREPS_IN = Irreps("8x0e + 8x1o + 8x2e")
CUTOFF = 5.0

EQUIVARIANT_SUBCONFIG: dict = {
    "hidden_irreps": "8x0e + 8x1o + 8x2e",
    "num_layers": 2,
    "num_rbf": 4,
    "radial_hidden": 16,
    "n_scalar_out": 8,
    "final_hidden": 8,
    "element_conditioned": True,
}


def _make_equivariant_artisan() -> PotentialArtisan:
    cfg = ArtisanConfig(
        architecture="equivariant",
        equivariant=dict(EQUIVARIANT_SUBCONFIG),
    )
    return PotentialArtisan(IRREPS_IN, cfg, cutoff=CUTOFF).double()


def _random_edge_inputs(
    num_edges: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random features / geometry for a batch of edges (float64)."""
    g = torch.Generator().manual_seed(7)
    feats_a = torch.randn(num_edges, IRREPS_IN.dim, generator=g, dtype=torch.float64)
    feats_b = torch.randn(num_edges, IRREPS_IN.dim, generator=g, dtype=torch.float64)
    vecs = torch.randn(num_edges, 3, generator=g, dtype=torch.float64)
    lengths = vecs.norm(dim=-1)
    core = _EquivariantArtisanCore(IRREPS_IN, CUTOFF, **EQUIVARIANT_SUBCONFIG)
    edge_sh = spherical_harmonics(
        core.irreps_edge, vecs, normalize=True, normalization="component"
    ).double()
    z_dst = torch.tensor([1, 6, 1, 6, 1, 6][:num_edges], dtype=torch.long)
    return feats_a, feats_b, edge_sh, lengths, z_dst


def _build_monolithic(architecture: str) -> SimurghMonolithic:
    artisan_config: dict = {
        "architecture": architecture,
        "scalar_channels": 8,
        "hidden_dims": (32, 16),
        "expert_type": "linear",
        "equivariant": dict(EQUIVARIANT_SUBCONFIG),
    }
    return SimurghMonolithic(
        elements=(1, 6, 7, 8),
        dressing_kwargs={
            "num_elements": 9,
            "embedding_dim": 16,
            "hidden_channels": 8,
            "lmax": 2,
            "num_radial_basis": 4,
            "cutoff": CUTOFF,
            "radial_mlp_hidden": 16,
        },
        artisan_config=artisan_config,
        cutoff=CUTOFF,
        forces_mode="autograd",
    ).double()


def _rotate_batch(batch: Batch, R: torch.Tensor) -> Batch:
    rotated = batch.clone()
    rotated.pos = batch.pos.detach() @ R.T
    if getattr(rotated, "edge_vectors", None) is not None:
        rotated.edge_vectors = batch.edge_vectors @ R.T
    return rotated


# ----------------------------------------------------------------------
# Test 1 — pair symmetry
# ----------------------------------------------------------------------


class TestPairSymmetry:
    def test_swap_features_leaves_energy_unchanged(self) -> None:
        """E(A, B) == E(B, A) for the equivariant artisan, per edge."""
        artisan = _make_equivariant_artisan()
        artisan.eval()
        feats_a, feats_b, edge_sh, lengths, z_dst = _random_edge_inputs()

        e_ab = artisan.forward_equivariant(feats_a, feats_b, edge_sh, lengths, z_dst)
        e_ba = artisan.forward_equivariant(feats_b, feats_a, edge_sh, lengths, z_dst)

        diff = (e_ab - e_ba).abs().max().item()
        assert diff < 1e-6, f"Pair symmetry violated: max |E(A,B) - E(B,A)| = {diff:.3e}"


# ----------------------------------------------------------------------
# Test 2 — rotation equivariance (both architectures)
# ----------------------------------------------------------------------


class TestRotationEquivariance:
    @pytest.mark.parametrize("architecture", ["scalar", "equivariant"])
    def test_energy_invariant_forces_equivariant(
        self, methylamine_batch, architecture: str  # noqa: ANN001
    ) -> None:
        # Methylamine rather than methane: the perfectly tetrahedral
        # methane fixture sits at a high-symmetry point where the
        # dressing's EquivariantLayerNorm amplifies symmetry-suppressed
        # (numerically ~zero) l>0 blocks, polluting force gradients for
        # any consumer of the full equivariant features.  The existing
        # modular equivariance test uses methylamine for the same reason.
        model = _build_monolithic(architecture)
        model.eval()

        out0 = model(methylamine_batch)
        e0, f0 = out0["energy"], out0["forces"]

        R = random_so3()
        rotated = _rotate_batch(methylamine_batch, R)
        out1 = model(rotated)
        e1, f1 = out1["energy"], out1["forces"]

        e_diff = (e0 - e1).abs().max().item()
        assert e_diff < 1e-5, f"[{architecture}] energy not invariant: {e_diff:.3e}"

        f_diff = (f1.detach() - f0.detach() @ R.T).abs().max().item()
        assert f_diff < 1e-5, f"[{architecture}] forces not equivariant: {f_diff:.3e}"


# ----------------------------------------------------------------------
# Test 3 — zero masking
# ----------------------------------------------------------------------


class TestZeroMasking:
    def test_zero_inputs_give_zero_output(self) -> None:
        """All maps are bias-free → zero input must yield exactly zero."""
        artisan = _make_equivariant_artisan()
        artisan.eval()
        assert artisan.equivariant_core is not None

        num_edges = 4
        feats = torch.zeros(num_edges, IRREPS_IN.dim, dtype=torch.float64)
        edge_sh = torch.zeros(
            num_edges, artisan.equivariant_core.irreps_edge.dim, dtype=torch.float64
        )
        lengths = torch.zeros(num_edges, dtype=torch.float64)
        z_dst = torch.zeros(num_edges, dtype=torch.long)

        out = artisan.forward_equivariant(feats, feats, edge_sh, lengths, z_dst)
        assert torch.isfinite(out).all(), "Zero-masked edges produced non-finite output"
        assert out.abs().max().item() == 0.0, (
            f"Zero input did not produce zero output: max |out| = " f"{out.abs().max().item():.3e}"
        )


# ----------------------------------------------------------------------
# Test 4 — Newton's third law (both architectures)
# ----------------------------------------------------------------------


class TestNewtonsThirdLaw:
    @pytest.mark.parametrize("architecture", ["scalar", "equivariant"])
    def test_forces_sum_to_zero_on_methane(
        self, methane_batch, architecture: str  # noqa: ANN001
    ) -> None:
        model = _build_monolithic(architecture)
        model.eval()
        out = model(methane_batch)
        assert torch.isfinite(out["energy"]).item()
        assert (out["forces"].abs() > 0).any().item(), "Forces are identically zero"
        net_force = out["forces"].sum(dim=0).norm().item()
        assert net_force < 1e-5, f"[{architecture}] ‖Σ F‖ = {net_force:.3e}"
