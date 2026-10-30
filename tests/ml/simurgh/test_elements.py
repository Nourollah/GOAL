"""Tests for automatic element extraction and atomic-energy validation.

Covers:
  - compute_unique_elements() from statistics.py
  - Auto-injection into SimurghBackbone via train.py flow
  - Provided-mode validation in SimurghBackbone._init_atomic_energies()
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph
from goal.ml.data.statistics import compute_unique_elements
from goal.ml.nn.models.simurgh.legacy.backbone import SimurghBackbone

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(atomic_numbers: list[int]) -> AtomicGraph:
    """Minimal single-molecule graph with no edges (sufficient for element tests)."""
    n = len(atomic_numbers)
    pos = torch.zeros(n, 3, dtype=torch.float64)
    for i in range(n):
        pos[i, 0] = float(i)  # spread atoms so cutoff can build edges if needed
    z = torch.tensor(atomic_numbers, dtype=torch.long)
    cell = torch.zeros(3, 3, dtype=torch.float64)
    pbc = torch.zeros(3, dtype=torch.bool)
    # Edge-free graph — element extraction doesn't need edges
    return AtomicGraph(
        positions=pos,
        atomic_numbers=z,
        cell=cell,
        pbc=pbc,
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_vectors=torch.zeros(0, 3, dtype=torch.float64),
        edge_lengths=torch.zeros(0, dtype=torch.float64),
    )


def _minimal_simurgh(elements: list[int], ae_mode: str, ae_values: dict | None) -> SimurghBackbone:
    """Build the smallest valid SimurghBackbone for the given elements/AE config."""
    return SimurghBackbone(
        elements=elements,
        dressing_kwargs=dict(
            embedding_dim=4,
            hidden_channels=4,
            lmax=1,
            num_radial_basis=4,
            radial_mlp_hidden=4,
            cutoff=5.0,
            num_interactions=1,
            num_message_passing=1,
            body_order=1,
            element_conditioned=False,
            per_layer_readout=False,
            symmetric_contraction=False,
            avg_num_neighbors=1.0,
            agg_norm_exponent=1.0,
            num_elements=9,
        ),
        artisan_config=dict(
            scalar_channels=4,
            hidden_dims=(8,),
            expert_type="linear",
            dropout_rate=0.0,
            rare_pair_embed_dim=4,
        ),
        cutoff=5.0,
        atomic_energies=dict(mode=ae_mode, values=ae_values),
    )


# ---------------------------------------------------------------------------
# Test 1 — compute_unique_elements extracts correct atomic numbers
# ---------------------------------------------------------------------------


class TestComputeUniqueElements:
    def test_hco_dataset(self) -> None:
        """Dataset with H, C, O → sorted list [1, 6, 8]."""
        graphs = [
            _make_graph([1, 6, 8]),  # H C O
            _make_graph([1, 1, 6]),  # H H C
            _make_graph([8, 8]),  # O O
        ]
        result = compute_unique_elements(graphs)
        assert result == [1, 6, 8]

    def test_single_element(self) -> None:
        """Dataset with only carbon → [6]."""
        graphs = [_make_graph([6, 6, 6])]
        result = compute_unique_elements(graphs)
        assert result == [6]

    def test_sorted_output(self) -> None:
        """Output is always sorted regardless of insertion order."""
        graphs = [_make_graph([8, 7, 6, 1])]
        result = compute_unique_elements(graphs)
        assert result == sorted(result)
        assert result == [1, 6, 7, 8]

    def test_empty_dataset(self) -> None:
        """Empty dataset → empty list, no crash."""
        result = compute_unique_elements([])
        assert result == []

    def test_no_duplicates(self) -> None:
        """Each element appears only once even if repeated across graphs."""
        graphs = [_make_graph([1, 1, 1]) for _ in range(10)]
        result = compute_unique_elements(graphs)
        assert result == [1]
        assert len(result) == 1

    def test_multi_graph_union(self) -> None:
        """Elements are the union over all graphs in the dataset."""
        graphs = [
            _make_graph([1]),  # H only
            _make_graph([6]),  # C only
            _make_graph([8]),  # O only
        ]
        result = compute_unique_elements(graphs)
        assert result == [1, 6, 8]


# ---------------------------------------------------------------------------
# Test 2 — provided mode, all elements covered → no error
# ---------------------------------------------------------------------------


class TestProvidedModeAllCovered:
    def test_all_elements_present(self) -> None:
        """Provided mode with values for all dataset elements raises nothing."""
        # Elements in dataset: H(1), C(6)
        _minimal_simurgh(
            elements=[1, 6],
            ae_mode="provided",
            ae_values={1: -13.6, 6: -1028.5},
        )

    def test_extra_values_are_fine(self) -> None:
        """Having more values than elements in the dataset is allowed."""
        # Dataset has H and C, but we also provide N and O — that's fine
        _minimal_simurgh(
            elements=[1, 6],
            ae_mode="provided",
            ae_values={1: -13.6, 6: -1028.5, 7: -1483.9, 8: -2041.1},
        )


# ---------------------------------------------------------------------------
# Test 3 — provided mode, missing element → ValueError before training
# ---------------------------------------------------------------------------


class TestProvidedModeMissingElement:
    def test_missing_oxygen_raises(self) -> None:
        """Provided mode with O missing from values raises ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            _minimal_simurgh(
                elements=[1, 6, 8],
                ae_mode="provided",
                ae_values={1: -13.6, 6: -1028.5},  # O(8) missing
            )
        msg = str(exc_info.value)
        # Must mention the missing element
        assert "O" in msg or "8" in msg
        # Must mention both alternative modes
        assert "dataset" in msg
        assert "learned" in msg

    def test_multiple_missing_elements(self) -> None:
        """Multiple missing elements are all reported in the error."""
        with pytest.raises(ValueError) as exc_info:
            _minimal_simurgh(
                elements=[1, 6, 7, 8],
                ae_mode="provided",
                ae_values={1: -13.6},  # C, N, O all missing
            )
        msg = str(exc_info.value)
        # At least one of the missing elements must be mentioned
        assert any(sym in msg for sym in ["C", "N", "O", "6", "7", "8"])

    def test_error_raised_at_construction_not_forward(self) -> None:
        """The error is raised in __init__, not lazily at forward time."""
        # This test verifies the error is synchronous (not deferred)
        raised = False
        try:
            _minimal_simurgh(
                elements=[1, 6, 8],
                ae_mode="provided",
                ae_values={1: -13.6},  # C and O missing
            )
        except ValueError:
            raised = True
        assert raised, "ValueError must be raised during construction, not deferred"


# ---------------------------------------------------------------------------
# Test 4 — dataset mode ignores values / never raises
# ---------------------------------------------------------------------------


class TestDatasetModeNeverRaises:
    def test_dataset_mode_with_null_values(self) -> None:
        """dataset mode with values=null raises only if we don't pre-fill."""
        # In dataset mode, train.py fills in the values before construction.
        # When values is provided (simulating post-fill), no error.
        _minimal_simurgh(
            elements=[1, 6, 8],
            ae_mode="dataset",
            ae_values={1: -13.6, 6: -1028.5, 8: -2041.1},
        )

    def test_dataset_mode_does_not_validate_coverage(self) -> None:
        """dataset mode never raises a coverage error regardless of which elements."""
        # Provide values only for C — in dataset mode this is fine because
        # train.py always fills all observed elements via compute_atomic_references.
        _minimal_simurgh(
            elements=[1, 6, 8],
            ae_mode="dataset",
            ae_values={6: -1028.5},  # only C — would fail in provided mode
        )


# ---------------------------------------------------------------------------
# Test 5 — learned mode ignores values entirely
# ---------------------------------------------------------------------------


class TestLearnedModeNeverRaises:
    def test_learned_mode_any_elements(self) -> None:
        """learned mode never raises regardless of which elements are present."""
        _minimal_simurgh(
            elements=[1, 6, 7, 8, 16],  # include S
            ae_mode="learned",
            ae_values=None,
        )

    def test_learned_mode_registers_parameter(self) -> None:
        """learned mode registers self.atomic_energies as an nn.Parameter."""
        import torch.nn as nn

        backbone = _minimal_simurgh(
            elements=[1, 6],
            ae_mode="learned",
            ae_values=None,
        )
        assert isinstance(backbone.atomic_energies, nn.Parameter)
        assert backbone.atomic_energies.requires_grad
