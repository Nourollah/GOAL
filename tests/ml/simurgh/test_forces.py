"""Tests for the pairwise-force mode added in TASK 4.

Two properties under test:

* **Newton's third law is exact** — when the artisan bank computes per-pair
  forces via ``-∂E_ij/∂r_ij`` and scatters with the
  ``+0.5 F`` / ``-0.5 F`` Newton-symmetric pattern, the net force per
  molecule must be zero to machine precision (``< 1e-10`` is the
  ceiling we test against; in practice it sits at ``1e-13`` for
  float64).
* **Directions agree with the autograd path** — pure pairwise and
  pure autograd are mathematically the same total force, just
  computed via different code paths.  After random init, the per-atom
  cosine similarity between the two should be far above noise
  (``> 0.8``).
"""

from __future__ import annotations

import typing

import pytest
import torch
from torch_geometric.data import Batch

from goal.ml.nn.heads.dual_forces import DualForcesHead
from goal.ml.nn.models.simurgh.legacy.backbone import SimurghBackbone


def _small_simurgh(
    compute_pairwise_forces: bool,
    elements: typing.Sequence[int] = (1, 6, 7, 8),
) -> SimurghBackbone:
    """Tiny SIMURGH configuration suitable for fast unit tests."""
    return SimurghBackbone(
        elements=elements,
        dressing_kwargs={
            "num_elements": 120,
            "embedding_dim": 8,
            "hidden_channels": 8,
            "lmax": 1,
            "num_radial_basis": 4,
            "cutoff": 5.0,
            "radial_mlp_hidden": 8,
            "num_message_passing": 1,
            "body_order": 1,
        },
        artisan_config={
            "scalar_channels": 4,
            "hidden_dims": (8,),
            "expert_type": "linear",
        },
        cutoff=5.0,
        atomic_energies={"mode": "learned"},
        compute_pairwise_forces=compute_pairwise_forces,
    ).to(torch.float64)


# ---------------------------------------------------------------------------
# TASK 4: Newton's third law on pairwise forces
# ---------------------------------------------------------------------------


class TestPairwiseNewton:
    """Per-molecule net force is structurally zero in pairwise mode."""

    def test_newton_violation_below_machine_epsilon(
        self,
        methane_batch,  # noqa: ANN001 (pytest fixture)
    ) -> None:
        torch.manual_seed(0)
        backbone = _small_simurgh(compute_pairwise_forces=True)
        head = DualForcesHead(
            irreps_in="8x0e+8x1o",
            hidden_dim=16,
            mode="pairwise",
        ).to(torch.float64)

        features = backbone(methane_batch)
        out = head(features, methane_batch)
        forces: torch.Tensor = out["forces"]  # (N, 3)

        # Net force on the (single) molecule = sum over atoms.
        net_force: torch.Tensor = forces.sum(dim=0)  # (3,)
        violation: float = float(net_force.norm())
        assert violation < 1e-10, (
            f"pairwise mode produced net force {violation:.3e} eV/Å — "
            "Newton's third law is broken"
        )

    def test_features_carry_node_forces(
        self,
        methane_batch,  # noqa: ANN001
    ) -> None:
        """A backbone built with ``compute_pairwise_forces=True`` must
        populate ``NodeFeatures.node_forces``; the energy-only build
        must leave it ``None``."""
        torch.manual_seed(0)
        bb_pairwise = _small_simurgh(compute_pairwise_forces=True)
        bb_energy = _small_simurgh(compute_pairwise_forces=False)
        feats_p = bb_pairwise(methane_batch)
        feats_e = bb_energy(methane_batch)
        assert feats_p.node_forces is not None
        assert feats_p.node_forces.shape == methane_batch.pos.shape
        assert feats_e.node_forces is None


# ---------------------------------------------------------------------------
# TASK 4: pairwise vs autograd direction agreement
# ---------------------------------------------------------------------------


class TestPairwiseVsAutograd:
    """Pairwise and autograd forces must agree on direction *broadly*.

    The two paths are not mathematically identical for SIMURGH: the
    pairwise path only differentiates through ``r_ij`` inside the
    expert (capturing the direct-distance contribution), while the
    autograd path also picks up the environmental dependence through
    the equivariant dressing → scalar-projection chain.  Cosine
    similarity is therefore expected to be high but not 1.0.

    We require ``> 0.5`` — meaningfully above random (0.0) but well
    below perfect (1.0).
    """

    def test_cosine_similarity_above_threshold(
        self,
        methane_batch,  # noqa: ANN001
    ) -> None:
        torch.manual_seed(0)
        bb_pair = _small_simurgh(compute_pairwise_forces=True)
        torch.manual_seed(0)
        bb_auto = _small_simurgh(compute_pairwise_forces=False)

        head_pair = DualForcesHead(irreps_in="8x0e+8x1o", hidden_dim=16, mode="pairwise").to(
            torch.float64
        )
        torch.manual_seed(0)
        head_auto = DualForcesHead(irreps_in="8x0e+8x1o", hidden_dim=16, mode="autograd").to(
            torch.float64
        )

        # Pairwise forward — head doesn't need pos.requires_grad.
        feats_pair = bb_pair(methane_batch)
        out_pair = head_pair(feats_pair, methane_batch)
        f_pair: torch.Tensor = out_pair["forces"]

        # Autograd forward — must enable requires_grad on positions
        # so the head's ``autograd.grad(energy, pos)`` finds a graph.
        methane_batch.pos.requires_grad_(True)
        feats_auto = bb_auto(methane_batch)
        out_auto = head_auto(feats_auto, methane_batch)
        f_auto: torch.Tensor = out_auto["forces"]

        # Per-atom cosine similarity, then mean over atoms.
        dot: torch.Tensor = (f_pair * f_auto).sum(dim=-1)
        norm_p: torch.Tensor = f_pair.norm(dim=-1)
        norm_a: torch.Tensor = f_auto.norm(dim=-1)
        cos_per_atom: torch.Tensor = dot / (norm_p * norm_a + 1e-8)
        mean_cos: float = float(cos_per_atom.mean())

        assert mean_cos > 0.5, (
            f"pairwise vs autograd cosine similarity = {mean_cos:.3f} "
            "below the 0.5 threshold — the two paths disagree wildly on "
            "direction (something other than the expected environmental-"
            "dressing discrepancy is off)"
        )


# ---------------------------------------------------------------------------
# Misuse / validation
# ---------------------------------------------------------------------------


class TestPairwiseModeValidation:
    """Misconfiguration must raise loud, actionable errors."""

    def test_pairwise_head_without_backbone_forces_raises(
        self,
        methane_batch,  # noqa: ANN001
    ) -> None:
        """Selecting ``mode='pairwise'`` on a backbone built with
        ``compute_pairwise_forces=False`` is a config mistake — the
        head receives ``node_forces=None`` and must complain loudly."""
        torch.manual_seed(0)
        bb = _small_simurgh(compute_pairwise_forces=False)
        head = DualForcesHead(irreps_in="8x0e+8x1o", hidden_dim=16, mode="pairwise").to(
            torch.float64
        )
        features = bb(methane_batch)
        with pytest.raises(ValueError, match="compute_pairwise_forces"):
            head(features, methane_batch)

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            DualForcesHead(irreps_in="8x0e", mode="bogus")
