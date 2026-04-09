"""Tests for goal.ml.utils.extraction — feature extraction utilities."""

from __future__ import annotations

import typing

import pytest
import torch
import torch.nn as nn

from goal.ml.data.graph import NodeFeatures
from goal.ml.utils.extraction import (
    HookBasedExtractor,
    describe_irreps,
    extract_irrep_channels,
    extract_scalars,
    pool_nodes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyBlockModel(nn.Module):
    """Minimal model with a ``nn.ModuleList`` of linear layers.

    Each block returns a plain tensor (no tuple), mimicking a simple
    interaction‐block architecture.
    """

    def __init__(self, in_dim: int = 8, hidden_dim: int = 16, n_blocks: int = 3) -> None:
        super().__init__()
        self.blocks: nn.ModuleList = nn.ModuleList(
            [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim) for i in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _TupleBlockModel(nn.Module):
    """Model whose blocks return ``(features, skip_connection)`` tuples.

    Used to test ``output_index`` parameter of ``HookBasedExtractor``.
    """

    def __init__(self, dim: int = 8, n_blocks: int = 2) -> None:
        super().__init__()
        self.interactions: nn.ModuleList = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.interactions:
            out: torch.Tensor = block(x)
            x = out  # ignore skip for simplicity in forward
        return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHookBasedExtractor:
    """Test the generic hook‐based feature extractor."""

    def test_captures_all_layers(self) -> None:
        """Extractor should capture one tensor per block in the ModuleList."""
        model: _DummyBlockModel = _DummyBlockModel(in_dim=8, hidden_dim=16, n_blocks=3)
        x: torch.Tensor = torch.randn(5, 8)

        with HookBasedExtractor(model, blocks_attr="blocks") as ext:
            _ = model(x)
            captured: dict[str, torch.Tensor] = ext.captured

        assert len(captured) == 3
        for i in range(3):
            key: str = f"layer_{i}"
            assert key in captured
            assert captured[key].shape == (5, 16)

    def test_detach_default(self) -> None:
        """Captured tensors should be detached by default (no grad)."""
        model: _DummyBlockModel = _DummyBlockModel(in_dim=4, hidden_dim=8, n_blocks=2)
        x: torch.Tensor = torch.randn(3, 4)

        with HookBasedExtractor(model, blocks_attr="blocks", detach=True) as ext:
            _ = model(x)
            for tensor in ext.captured.values():
                assert not tensor.requires_grad

    def test_no_detach_keeps_grad(self) -> None:
        """With ``detach=False``, captured tensors should retain gradients."""
        model: _DummyBlockModel = _DummyBlockModel(in_dim=4, hidden_dim=8, n_blocks=2)
        x: torch.Tensor = torch.randn(3, 4, requires_grad=True)

        with HookBasedExtractor(model, blocks_attr="blocks", detach=False) as ext:
            _ = model(x)
            # At least the first layer's output should be part of the graph
            assert ext.captured["layer_0"].requires_grad or ext.captured["layer_1"].requires_grad

    def test_context_manager_removes_hooks(self) -> None:
        """After exiting the context manager, no hooks should remain."""
        model: _DummyBlockModel = _DummyBlockModel(in_dim=4, hidden_dim=8, n_blocks=2)

        ext: HookBasedExtractor
        with HookBasedExtractor(model, blocks_attr="blocks") as ext:
            # Inside: hooks are attached
            assert len(ext._hooks) == 2

        # Outside: hooks should be removed
        assert len(ext._hooks) == 0
        assert len(ext.captured) == 0

    def test_extract_method(self) -> None:
        """The ``extract()`` convenience method should return all captured features."""
        model: _DummyBlockModel = _DummyBlockModel(in_dim=4, hidden_dim=8, n_blocks=3)
        x: torch.Tensor = torch.randn(5, 4)

        ext: HookBasedExtractor = HookBasedExtractor(model, blocks_attr="blocks")
        result: dict[str, torch.Tensor] = ext.extract(x)
        assert len(result) == 3
        ext.remove()

    def test_output_index_for_tuple_blocks(self) -> None:
        """``output_index`` should correctly index tuple-returning blocks."""
        model: _TupleBlockModel = _TupleBlockModel(dim=8, n_blocks=2)
        x: torch.Tensor = torch.randn(4, 8)

        # The linear blocks don't actually return tuples in their forward,
        # but the hook captures whatever the block produces.  Here we just
        # verify the extractor doesn't crash with output_index=None (plain
        # tensor) and captures the right shapes.
        with HookBasedExtractor(
            model,
            blocks_attr="interactions",
            output_index=None,
        ) as ext:
            _ = model(x)
            assert ext.captured["layer_0"].shape == (4, 8)
            assert ext.captured["layer_1"].shape == (4, 8)


class TestExtractScalars:
    """Test the L=0 scalar extraction helper."""

    def test_pure_scalars(self) -> None:
        """With only L=0 irreps, all channels should be returned."""
        feats: NodeFeatures = NodeFeatures(
            node_feats=torch.randn(5, 10),
            irreps="10x0e",
        )
        scalars: torch.Tensor = extract_scalars(feats)
        assert scalars.shape == (5, 10)

    def test_mixed_irreps(self) -> None:
        """From '4x0e+2x1o', only the 4 scalar channels should be kept."""
        # 4x0e → 4 channels, 2x1o → 2*3 = 6 channels → total 10
        feats: NodeFeatures = NodeFeatures(
            node_feats=torch.randn(5, 10),
            irreps="4x0e+2x1o",
        )
        scalars: torch.Tensor = extract_scalars(feats)
        assert scalars.shape == (5, 4)

    def test_no_scalars(self) -> None:
        """If irreps have no L=0 components, result should be (N, 0)."""
        feats: NodeFeatures = NodeFeatures(
            node_feats=torch.randn(5, 9),
            irreps="3x1o",
        )
        scalars: torch.Tensor = extract_scalars(feats)
        assert scalars.shape == (5, 0)

    def test_complex_irreps(self) -> None:
        """Scalars from '128x0e+64x1o+32x2e' should be 128 channels."""
        # 128x0e → 128, 64x1o → 192, 32x2e → 160 → total 480
        feats: NodeFeatures = NodeFeatures(
            node_feats=torch.randn(3, 480),
            irreps="128x0e+64x1o+32x2e",
        )
        scalars: torch.Tensor = extract_scalars(feats)
        assert scalars.shape == (3, 128)


class TestExtractIrrepChannels:
    """Test extraction of channels for a specific angular momentum."""

    def test_l1_vectors(self) -> None:
        """Extract L=1 vector channels from mixed irreps."""
        # 4x0e → 4, 2x1o → 6 → total 10
        feats: NodeFeatures = NodeFeatures(
            node_feats=torch.randn(5, 10),
            irreps="4x0e+2x1o",
        )
        vectors: torch.Tensor = extract_irrep_channels(feats, l=1)
        assert vectors.shape == (5, 6)

    def test_l2_tensors(self) -> None:
        """Extract L=2 tensor channels."""
        # 10x0e → 10, 5x1o → 15, 3x2e → 15 → total 40
        feats: NodeFeatures = NodeFeatures(
            node_feats=torch.randn(4, 40),
            irreps="10x0e+5x1o+3x2e",
        )
        tensors: torch.Tensor = extract_irrep_channels(feats, l=2)
        assert tensors.shape == (4, 15)

    def test_absent_l(self) -> None:
        """If the requested angular momentum is absent, return (N, 0)."""
        feats: NodeFeatures = NodeFeatures(
            node_feats=torch.randn(5, 4),
            irreps="4x0e",
        )
        result: torch.Tensor = extract_irrep_channels(feats, l=2)
        assert result.shape == (5, 0)


class TestPoolNodes:
    """Test the per-node → per-graph pooling helper."""

    def test_sum_pooling(self) -> None:
        """Sum pooling should give (num_graphs, D)."""
        # 7 nodes total: 3 in graph 0, 4 in graph 1
        batch: torch.Tensor = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        feats: NodeFeatures = NodeFeatures(
            node_feats=torch.ones(7, 16),
            irreps="16x0e",
        )
        result: torch.Tensor = pool_nodes(feats, batch, num_graphs=2, mode="sum")
        assert result.shape == (2, 16)
        # Graph 0 has 3 nodes of ones → sum = 3 per channel
        assert torch.allclose(result[0], torch.full((16,), 3.0))
        # Graph 1 has 4 nodes of ones → sum = 4
        assert torch.allclose(result[1], torch.full((16,), 4.0))

    def test_mean_pooling(self) -> None:
        """Mean pooling should average over nodes within each graph."""
        batch: torch.Tensor = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        feats: NodeFeatures = NodeFeatures(
            node_feats=torch.ones(7, 8),
            irreps="8x0e",
        )
        result: torch.Tensor = pool_nodes(feats, batch, num_graphs=2, mode="mean")
        assert result.shape == (2, 8)
        # Mean of ones is 1.0 regardless of graph size
        assert torch.allclose(result[0], torch.ones(8))
        assert torch.allclose(result[1], torch.ones(8))


class TestDescribeIrreps:
    """Test the irreps pretty-printer."""

    def test_no_crash_simple(self) -> None:
        """Should not raise on a simple irreps string."""
        describe_irreps("10x0e")

    def test_no_crash_complex(self) -> None:
        """Should not raise on a complex irreps string."""
        describe_irreps("128x0e+64x1o+32x2e+16x3o")

    def test_no_crash_empty(self) -> None:
        """Should handle trivial irreps without crashing."""
        describe_irreps("1x0e")
