"""Tests for AtomicGraph — the core data contract."""

from __future__ import annotations

import pytest
import torch


class TestAtomicGraph:
    """Verify AtomicGraph construction, properties, and batching."""

    def _make_graph(self, num_atoms: int = 5) -> AtomicGraph:  # noqa: F821
        """Create a minimal valid AtomicGraph for testing."""
        from goal.ml.data.graph import AtomicGraph

        positions = torch.randn(num_atoms, 3, dtype=torch.float64)
        atomic_numbers = torch.randint(1, 30, (num_atoms,))
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        pbc = torch.tensor([True, True, True])

        # Simple linear chain edges
        row = torch.arange(num_atoms - 1)
        col = torch.arange(1, num_atoms)
        edge_index = torch.stack(
            [
                torch.cat([row, col]),
                torch.cat([col, row]),
            ]
        )
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
        from goal.ml.data.graph import AtomicGraph

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
        from goal.ml.data.graph import NodeFeatures

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
        from goal.ml.data.graph import _apply_mic

        cell = torch.eye(3, dtype=torch.float64) * 10.0
        pbc = torch.tensor([True, True, True])

        # Vector of (6, 0, 0) should wrap to (-4, 0, 0) in a 10 Å cell
        edge_vectors = torch.tensor([[6.0, 0.0, 0.0]], dtype=torch.float64)
        wrapped = _apply_mic(edge_vectors, cell, pbc)
        assert torch.allclose(wrapped, torch.tensor([[-4.0, 0.0, 0.0]], dtype=torch.float64))

    def test_mic_no_pbc(self):
        """With no PBC, vectors should not be wrapped."""
        from goal.ml.data.graph import _apply_mic

        cell = torch.eye(3, dtype=torch.float64) * 10.0
        pbc = torch.tensor([False, False, False])

        edge_vectors = torch.tensor([[6.0, 0.0, 0.0]], dtype=torch.float64)
        result = _apply_mic(edge_vectors, cell, pbc)
        assert torch.allclose(result, edge_vectors)


class TestPeriodicNeighborList:
    """Verify that the ASE-backed neighbor list finds cross-boundary pairs."""

    def test_cross_boundary_pair_found(self):
        """Two atoms near opposite cell boundaries must be neighbors under PBC.

        Setup: 5 Å cubic cell, atoms at x=0.2 and x=4.9.
        - Cartesian distance: 4.7 Å  (> cutoff 4.0 Å → missed by radius_graph)
        - Minimum-image distance: 0.3 Å (< cutoff 4.0 Å → correct)
        """
        pytest.importorskip("ase", reason="ase required for this test")
        from ase import Atoms

        from goal.ml.data.graph import AtomicGraph

        cell_size = 5.0
        atoms = Atoms(
            numbers=[11, 17],  # Na, Cl
            positions=[[0.2, 0.0, 0.0], [4.9, 0.0, 0.0]],
            cell=[[cell_size, 0, 0], [0, cell_size, 0], [0, 0, cell_size]],
            pbc=True,
        )
        cutoff = 4.0
        graph = AtomicGraph.from_ase(atoms, cutoff=cutoff, neighbor_list_backend="ase")

        assert graph.edge_index.shape[1] > 0, "No edges found in periodic structure"

        # All reported edge lengths must respect the cutoff
        assert (graph.edge_lengths <= cutoff + 1e-6).all(), (
            "Some edge lengths exceed the cutoff"
        )

        # The cross-boundary pair must be present in both directions
        src, dst = graph.edge_index
        pairs = set(zip(src.tolist(), dst.tolist()))
        assert (0, 1) in pairs or (1, 0) in pairs, (
            "Cross-boundary neighbor pair (0.3 Å MIC distance) not found. "
            "This indicates the PBC fix is not working."
        )

    def test_unit_shifts_stored(self):
        """unit_shifts must be stored on the graph and have the right shape."""
        pytest.importorskip("ase", reason="ase required for this test")
        from ase import Atoms

        from goal.ml.data.graph import AtomicGraph

        atoms = Atoms(
            numbers=[6, 6],
            positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            cell=[[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]],
            pbc=True,
        )
        graph = AtomicGraph.from_ase(atoms, cutoff=2.5, neighbor_list_backend="ase")

        assert graph.unit_shifts is not None, "unit_shifts should not be None"
        assert graph.unit_shifts.shape == (graph.edge_index.shape[1], 3)
        assert graph.unit_shifts.dtype == torch.long

    def test_edge_lengths_within_cutoff(self):
        """All neighbor distances must be strictly within the cutoff."""
        pytest.importorskip("ase", reason="ase required for this test")
        from ase import Atoms
        from ase.build import bulk

        from goal.ml.data.graph import AtomicGraph

        # FCC aluminium unit cell
        atoms = bulk("Al", "fcc", a=4.05)
        cutoff = 3.5
        graph = AtomicGraph.from_ase(atoms, cutoff=cutoff, neighbor_list_backend="ase")

        assert graph.edge_index.shape[1] > 0
        assert (graph.edge_lengths <= cutoff + 1e-6).all()


class TestNvalchemiopsBackend:
    """Tests for the nvalchemi-toolkit-ops GPU neighbor-list backend.

    All tests are automatically skipped when the package is not installed
    or no CUDA GPU is available.
    """

    @pytest.fixture(autouse=True)
    def _require_nvalchemiops_and_cuda(self):
        pytest.importorskip(
            "nvalchemiops",
            reason="nvalchemi-toolkit-ops not installed — skipping GPU backend tests",
        )
        if not pytest.importorskip("torch").cuda.is_available():
            pytest.skip("No CUDA GPU available — nvalchemiops backend is GPU-only")

    def test_cross_boundary_pair_found_gpu(self):
        """nvalchemiops must find the same cross-boundary pair as ASE.

        Setup: 5 Å cubic cell, atoms at x=0.2 and x=4.9.
        MIC distance = 0.3 Å  < cutoff = 4.0 Å  → pair must be found.
        Cartesian distance = 4.7 Å > cutoff → would be missed without PBC.
        """
        from ase import Atoms

        from goal.ml.data.graph import AtomicGraph

        atoms = Atoms(
            numbers=[11, 17],
            positions=[[0.2, 0.0, 0.0], [4.9, 0.0, 0.0]],
            cell=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
            pbc=True,
        )
        cutoff = 4.0
        graph = AtomicGraph.from_ase(
            atoms, cutoff=cutoff, neighbor_list_backend="nvalchemiops"
        )

        assert graph.edge_index.shape[1] > 0, "No edges found"
        assert (graph.edge_lengths <= cutoff + 1e-6).all(), "Edge exceeds cutoff"

        src, dst = graph.edge_index
        pairs = set(zip(src.tolist(), dst.tolist()))
        assert (0, 1) in pairs or (1, 0) in pairs, (
            "Cross-boundary pair (MIC distance 0.3 Å) not found by nvalchemiops backend"
        )

    def test_unit_shifts_stored_gpu(self):
        """unit_shifts must have shape (E, 3) and dtype torch.long."""
        from ase import Atoms

        from goal.ml.data.graph import AtomicGraph

        atoms = Atoms(
            numbers=[6, 6],
            positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            cell=[[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]],
            pbc=True,
        )
        graph = AtomicGraph.from_ase(
            atoms, cutoff=2.5, neighbor_list_backend="nvalchemiops"
        )

        assert graph.unit_shifts is not None
        assert graph.unit_shifts.shape == (graph.edge_index.shape[1], 3)
        assert graph.unit_shifts.dtype == torch.long

    def test_matches_ase_backend(self):
        """nvalchemiops and ASE backends must agree on neighbor counts and lengths.

        Uses FCC Al — a well-known periodic structure with an exact neighbour shell.
        Both backends should return the same number of edges and the same
        sorted edge-length distribution (within float32 tolerance).
        """
        from ase.build import bulk

        from goal.ml.data.neighbor_list import build_neighbor_list

        atoms = bulk("Al", "fcc", a=4.05)
        cutoff = 3.5

        nl_ase = build_neighbor_list(atoms, cutoff, backend="ase")
        nl_gpu = build_neighbor_list(atoms, cutoff, backend="nvalchemiops")

        assert nl_ase.edge_index.shape[1] == nl_gpu.edge_index.shape[1], (
            f"Edge count mismatch: ASE={nl_ase.edge_index.shape[1]}, "
            f"nvalchemiops={nl_gpu.edge_index.shape[1]}"
        )
        # Sorted lengths must agree within float32 precision
        ase_lengths = nl_ase.edge_lengths.sort().values
        gpu_lengths = nl_gpu.edge_lengths.sort().values
        assert torch.allclose(ase_lengths.float(), gpu_lengths.float(), atol=1e-4), (
            "Edge lengths differ between ASE and nvalchemiops backends"
        )

    def test_no_pbc_non_periodic(self):
        """nvalchemiops backend must work for non-periodic (molecule) systems."""
        from ase import Atoms

        from goal.ml.data.neighbor_list import build_neighbor_list

        # Simple water-like triangle: 3 atoms, no PBC
        atoms = Atoms(
            numbers=[8, 1, 1],
            positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
            pbc=False,
        )
        cutoff = 1.5
        nl = build_neighbor_list(atoms, cutoff, backend="nvalchemiops")

        assert nl.edge_index.shape[1] > 0
        assert (nl.edge_lengths <= cutoff + 1e-6).all()
        # unit_shifts must be all-zero for non-periodic systems
        assert (nl.unit_shifts == 0).all(), "unit_shifts should be zero for non-PBC systems"
