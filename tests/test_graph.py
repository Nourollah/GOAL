"""Tests for AtomicGraph — the core data contract."""

from __future__ import annotations

import torch
import pytest


class TestAtomicGraph:
    """Verify AtomicGraph construction, properties, and batching."""

    def _make_graph(self, num_atoms: int = 5) -> "AtomicGraph":
        """Create a minimal valid AtomicGraph for testing."""
        from gmd.data.graph import AtomicGraph

        positions = torch.randn(num_atoms, 3, dtype=torch.float64)
        atomic_numbers = torch.randint(1, 30, (num_atoms,))
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        pbc = torch.tensor([True, True, True])

        # Simple linear chain edges
        row = torch.arange(num_atoms - 1)
        col = torch.arange(1, num_atoms)
        edge_index = torch.stack([
            torch.cat([row, col]),
            torch.cat([col, row]),
        ])
        num_edges = edge_index.shape[1]
        edge_vectors = torch.randn(num_edges, 3, dtype=torch.float64)
        edge_lengths = edge_vectors.norm(dim=-1)

        return AtomicGraph(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            edge_index=edge_index,
            edge_vectors=edge_vectors,
            edge_lengths=edge_lengths,
            energy=torch.tensor([-10.0], dtype=torch.float64),
            forces=torch.randn(num_atoms, 3, dtype=torch.float64),
        )

    def test_construction(self):
        """AtomicGraph should be constructible with required fields."""
        graph = self._make_graph()
        assert graph.num_atoms == 5
        assert graph.pos.shape == (5, 3)
        assert graph.z.shape == (5,)

    def test_properties(self):
        """Property accessors should return the correct underlying tensors."""
        graph = self._make_graph()
        assert torch.equal(graph.positions, graph.pos)
        assert torch.equal(graph.atomic_numbers, graph.z)
        assert torch.equal(graph.edge_vectors, graph.edge_attr)
        assert torch.equal(graph.edge_lengths, graph.edge_weight)

    def test_batching(self):
        """PyG batching should concatenate multiple graphs correctly."""
        from torch_geometric.data import Batch

        g1 = self._make_graph(num_atoms=3)
        g2 = self._make_graph(num_atoms=4)
        batch = Batch.from_data_list([g1, g2])

        assert batch.num_nodes == 7  # 3 + 4
        assert batch.num_graphs == 2
        assert batch.batch.shape == (7,)
        assert (batch.batch[:3] == 0).all()
        assert (batch.batch[3:] == 1).all()

    def test_energy_optional(self):
        """Energy should be optional (None for inference)."""
        from gmd.data.graph import AtomicGraph

        positions = torch.randn(3, 3, dtype=torch.float64)
        atomic_numbers = torch.tensor([1, 6, 8])
        cell = torch.zeros(3, 3, dtype=torch.float64)
        pbc = torch.tensor([False, False, False])
        edge_index = torch.tensor([[0, 1], [1, 0]])
        edge_vectors = torch.randn(2, 3, dtype=torch.float64)
        edge_lengths = edge_vectors.norm(dim=-1)

        graph = AtomicGraph(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            edge_index=edge_index,
            edge_vectors=edge_vectors,
            edge_lengths=edge_lengths,
        )
        assert graph.energy is None
        assert graph.forces is None


class TestNodeFeatures:
    """Verify the NodeFeatures container."""

    def test_construction(self):
        """NodeFeatures should hold typed fields."""
        from gmd.data.graph import NodeFeatures

        nf = NodeFeatures(
            node_feats=torch.randn(10, 64),
            irreps="64x0e",
        )
        assert nf.node_feats.shape == (10, 64)
        assert nf.irreps == "64x0e"
        assert nf.node_energies is None


class TestMinimumImageConvention:
    """Verify _apply_mic wraps edge vectors correctly."""

    def test_mic_wraps_vectors(self):
        """Vectors exceeding half the cell should be wrapped."""
        from gmd.data.graph import _apply_mic

        cell = torch.eye(3, dtype=torch.float64) * 10.0
        pbc = torch.tensor([True, True, True])

        # Vector of (6, 0, 0) should wrap to (-4, 0, 0) in a 10 Å cell
        edge_vectors = torch.tensor([[6.0, 0.0, 0.0]], dtype=torch.float64)
        wrapped = _apply_mic(edge_vectors, cell, pbc)
        assert torch.allclose(wrapped, torch.tensor([[-4.0, 0.0, 0.0]], dtype=torch.float64))

    def test_mic_no_pbc(self):
        """With no PBC, vectors should not be wrapped."""
        from gmd.data.graph import _apply_mic

        cell = torch.eye(3, dtype=torch.float64) * 10.0
        pbc = torch.tensor([False, False, False])

        edge_vectors = torch.tensor([[6.0, 0.0, 0.0]], dtype=torch.float64)
        result = _apply_mic(edge_vectors, cell, pbc)
        assert torch.allclose(result, edge_vectors)
