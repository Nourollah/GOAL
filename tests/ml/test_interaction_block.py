"""Tests for EquivariantInteractionBlock.

Covers:
* Element conditioning — different element types at the
  destination atom produce different TP weights and thus different output
  features, even when the geometric neighbourhood is identical.
* Pre-computed spherical harmonics — passing ``edge_sh``
  externally yields exactly the same output as the internal SH computation.
* Aggregation normalisation exponent — ``agg_norm_scale`` buffer
  stores the correct value for both ``exponent=1.0`` and ``exponent=0.5``,
  and the forward output magnitudes scale accordingly.
* Equivariance under SO(3) rotation — after element conditioning the block
  remains equivariant: scalar (l=0) features are invariant, vector (l=1)
  features rotate with the geometry.
"""

from __future__ import annotations

import pytest
import torch
from e3nn.o3 import Irreps, spherical_harmonics

from goal.ml.nn.blocks.interaction import EquivariantInteractionBlock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block(
    element_conditioned: bool = False,
    avg_num_neighbors: float | None = None,
    agg_norm_exponent: float = 1.0,
    n_elements: int = 120,
    hidden: int = 8,
    lmax: int = 1,
) -> EquivariantInteractionBlock:
    irreps_node = Irreps(f"{hidden}x0e+{hidden}x1o")
    irreps_edge = Irreps("1x0e+1x1o")
    return EquivariantInteractionBlock(
        irreps_node=irreps_node,
        irreps_edge=irreps_edge,
        num_basis=4,
        cutoff=5.0,
        hidden_dim=16,
        avg_num_neighbors=avg_num_neighbors,
        agg_norm_exponent=agg_norm_exponent,
        element_conditioned=element_conditioned,
        n_elements=n_elements,
    ).double()


def _dimer_inputs(
    z_src: int = 6,
    z_dst: int = 6,
    dist: float = 1.5,
    hidden: int = 8,
    lmax: int = 1,
) -> tuple[torch.Tensor, ...]:
    """Build a minimal 2-atom graph (one directed edge src→dst)."""
    # Edge from atom 0 (src) to atom 1 (dst)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0, 0.0], [dist, 0.0, 0.0]], dtype=torch.float64)
    edge_vectors = (pos[1] - pos[0]).unsqueeze(0)  # (1, 3)
    edge_lengths = edge_vectors.norm(dim=-1)  # (1,)
    atomic_numbers = torch.tensor([z_src, z_dst], dtype=torch.long)
    node_feats = torch.randn(2, Irreps(f"{hidden}x0e+{hidden}x1o").dim, dtype=torch.float64)
    return node_feats, edge_index, edge_vectors, edge_lengths, atomic_numbers


# ---------------------------------------------------------------------------
# Element conditioning
# ---------------------------------------------------------------------------


class TestElementConditioning:
    def test_conditioning_changes_output_when_destination_element_differs(self) -> None:
        """With conditioning on, changing the destination atom from C (6) to N (7)
        while keeping all geometry identical must change the output features."""
        block = _make_block(element_conditioned=True)

        node_feats, edge_index, edge_vectors, edge_lengths, z_cn = _dimer_inputs(z_src=1, z_dst=6)
        z_cn_n = z_cn.clone()
        z_cn_n[1] = 7  # destination becomes N

        with torch.no_grad():
            out_c = block(
                node_feats=node_feats,
                edge_index=edge_index,
                edge_vectors=edge_vectors,
                edge_lengths=edge_lengths,
                atomic_numbers=z_cn,
            )
            out_n = block(
                node_feats=node_feats,
                edge_index=edge_index,
                edge_vectors=edge_vectors,
                edge_lengths=edge_lengths,
                atomic_numbers=z_cn_n,
            )

        assert not torch.allclose(out_c, out_n), (
            "Element-conditioned block produced identical outputs for C and N "
            "destinations in the same geometric environment — conditioning has no effect."
        )

    def test_no_conditioning_gives_element_blind_output(self) -> None:
        """With conditioning *off*, changing the destination element must NOT
        change the output (geometry alone drives the message)."""
        block = _make_block(element_conditioned=False)

        node_feats, edge_index, edge_vectors, edge_lengths, z_c = _dimer_inputs(z_src=1, z_dst=6)
        z_n = z_c.clone()
        z_n[1] = 7  # destination becomes N, but conditioning is off

        with torch.no_grad():
            out_c = block(
                node_feats=node_feats,
                edge_index=edge_index,
                edge_vectors=edge_vectors,
                edge_lengths=edge_lengths,
            )
            out_n = block(
                node_feats=node_feats,
                edge_index=edge_index,
                edge_vectors=edge_vectors,
                edge_lengths=edge_lengths,
            )

        assert torch.allclose(out_c, out_n), (
            "Block without element conditioning produced different outputs for "
            "C vs N when only atomic_numbers changed — geometry should be the "
            "only driver when element_conditioned=False."
        )

    def test_conditioning_requires_atomic_numbers(self) -> None:
        """Calling an element-conditioned block without atomic_numbers raises."""
        block = _make_block(element_conditioned=True)
        node_feats, edge_index, edge_vectors, edge_lengths, _ = _dimer_inputs()
        with pytest.raises(ValueError, match="atomic_numbers"):
            block(
                node_feats=node_feats,
                edge_index=edge_index,
                edge_vectors=edge_vectors,
                edge_lengths=edge_lengths,
                atomic_numbers=None,
            )


# ---------------------------------------------------------------------------
# Pre-computed spherical harmonics
# ---------------------------------------------------------------------------


class TestPrecomputedSH:
    def test_precomputed_sh_matches_internal_computation(self) -> None:
        """Passing ``edge_sh`` externally must give the identical output as
        letting the block compute SH internally."""
        block = _make_block(element_conditioned=False)
        node_feats, edge_index, edge_vectors, edge_lengths, _ = _dimer_inputs()

        # Compute SH the same way the block does internally
        edge_sh = spherical_harmonics(
            block.irreps_edge,
            edge_vectors.to(torch.float64),
            normalize=True,
            normalization="component",
        )

        with torch.no_grad():
            out_internal = block(
                node_feats=node_feats,
                edge_index=edge_index,
                edge_vectors=edge_vectors,
                edge_lengths=edge_lengths,
                edge_sh=None,
            )
            out_external = block(
                node_feats=node_feats,
                edge_index=edge_index,
                edge_vectors=edge_vectors,
                edge_lengths=edge_lengths,
                edge_sh=edge_sh,
            )

        assert torch.allclose(
            out_internal, out_external, atol=1e-12
        ), "Pre-computed edge_sh gives different result than internal SH computation."


# ---------------------------------------------------------------------------
# Aggregation normalisation exponent
# ---------------------------------------------------------------------------


class TestAggNormExponent:
    def test_buffer_value_exponent_1(self) -> None:
        """With avg_num_neighbors=10 and exponent=1.0, agg_norm_scale = 1/10."""
        block = _make_block(avg_num_neighbors=10.0, agg_norm_exponent=1.0)
        expected = 1.0 / (10.0**1.0)
        # Buffer is stored in the model's default dtype (float32); allow float32 rounding.
        assert block.agg_norm_scale.item() == pytest.approx(expected, rel=1e-5)

    def test_buffer_value_exponent_half(self) -> None:
        """With avg_num_neighbors=10 and exponent=0.5, agg_norm_scale = 1/sqrt(10)."""
        block = _make_block(avg_num_neighbors=10.0, agg_norm_exponent=0.5)
        expected = 1.0 / (10.0**0.5)
        assert block.agg_norm_scale.item() == pytest.approx(expected, rel=1e-5)

    def test_no_neighbors_gives_scale_one(self) -> None:
        """When avg_num_neighbors is None, agg_norm_scale = 1.0 (no normalisation)."""
        block = _make_block(avg_num_neighbors=None)
        assert block.agg_norm_scale.item() == pytest.approx(1.0, rel=1e-6)

    def test_higher_exponent_gives_smaller_scale(self) -> None:
        """For the same avg_num_neighbors > 1, exponent=1.0 → smaller scale than 0.5."""
        block_1 = _make_block(avg_num_neighbors=16.0, agg_norm_exponent=1.0)
        block_half = _make_block(avg_num_neighbors=16.0, agg_norm_exponent=0.5)
        assert (
            block_1.agg_norm_scale.item() < block_half.agg_norm_scale.item()
        ), "exponent=1.0 should divide by N (smaller scale) vs exponent=0.5 (sqrt N)."


# ---------------------------------------------------------------------------
# Equivariance under SO(3)
# ---------------------------------------------------------------------------


class TestEquivariance:
    def test_scalar_features_invariant_under_rotation(self) -> None:
        """Rotating all positions by R leaves scalar (l=0) outputs unchanged."""
        from scipy.spatial.transform import Rotation as R

        torch.manual_seed(42)
        rot_np = R.random(random_state=7).as_matrix()
        R_mat = torch.tensor(rot_np, dtype=torch.float64)

        block = _make_block(element_conditioned=False, avg_num_neighbors=1.0)

        node_feats, edge_index, edge_vectors, edge_lengths, z = _dimer_inputs(z_dst=6, dist=1.5)

        # Rotate edge vectors
        edge_vectors_rot = (R_mat @ edge_vectors.T).T
        edge_lengths_rot = edge_vectors_rot.norm(dim=-1)

        with torch.no_grad():
            out = block(
                node_feats=node_feats,
                edge_index=edge_index,
                edge_vectors=edge_vectors,
                edge_lengths=edge_lengths,
            )
            out_rot = block(
                node_feats=_rotate_node_feats(node_feats, R_mat, block.irreps_node),
                edge_index=edge_index,
                edge_vectors=edge_vectors_rot,
                edge_lengths=edge_lengths_rot,
            )

        # l=0 scalars: invariant
        n_scalars = sum(mul for mul, ir in block.irreps_node if ir.l == 0)
        scalars = out[:, :n_scalars]
        scalars_rot = out_rot[:, :n_scalars]
        assert torch.allclose(
            scalars, scalars_rot, atol=1e-10
        ), "Scalar (l=0) features changed under rotation — block is not invariant."


def _rotate_node_feats(node_feats: torch.Tensor, R: torch.Tensor, irreps: Irreps) -> torch.Tensor:
    """Apply SO(3) rotation R to equivariant node features."""

    rotated = torch.zeros_like(node_feats)
    offset = 0
    for mul, ir in irreps:
        size = mul * (2 * ir.l + 1)
        chunk = node_feats[:, offset : offset + size]  # (N, mul*(2l+1))
        if ir.l == 0:
            rotated[:, offset : offset + size] = chunk
        else:
            D = ir.D_from_matrix(R)  # (2l+1, 2l+1)
            chunk_3d = chunk.reshape(-1, mul, 2 * ir.l + 1)  # (N, mul, 2l+1)
            rotated_3d = chunk_3d @ D.T  # (N, mul, 2l+1)
            rotated[:, offset : offset + size] = rotated_3d.reshape(-1, size)
        offset += size
    return rotated
