"""Many-body symmetric contraction inside the ARACE rounds.

An ARACE round aggregates artisan edge features onto nodes to form the
one-particle basis ``A_i`` and then applies a linear and a residual.
That makes a round a two-body message and a stack of rounds a plain
equivariant MPNN.  ``correlation`` inserts MACE's ``A → B`` step —
``B_i = Σ_n W_n[Z_i] · CG_n(A_i)`` with per-element couplings — which
raises the body order *within* a round.

What these tests pin down:

* ``correlation: null`` builds nothing, so the model is exactly what it
  was before this option existed;
* equivariance survives the contraction — energy invariant, forces
  rotating with the frame, to float64 precision;
* every order is reachable and grows the model;
* the DDP contract holds: the contraction's parameters get gradients.
"""

from __future__ import annotations

import typing

import pytest
import torch
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.models.simurgh.nf_simurgh import MonolithicNFSimurgh
from tests.ml.simurgh.conftest import random_so3

CUTOFF: float = 5.0

ARTISAN: dict[str, typing.Any] = {
    "architecture": "equivariant",
    "hidden_irreps": "8x0e + 8x1o + 8x2e",
    "num_layers": 1,
    "num_rbf": 4,
    "radial_hidden": 8,
    "n_scalar_out": 8,
    "final_hidden": 8,
    "element_conditioned": True,
}


def _model(correlation: int | None, seed: int = 0) -> typing.Any:
    torch.manual_seed(seed)
    return MonolithicNFSimurgh(
        elements=[1, 6, 8],
        cutoff=CUTOFF,
        artisan=dict(ARTISAN),
        num_elements=9,
        virtual_node={"vnode_cutoff": 10.0},
        correlation=correlation,
    ).double()


def _batch(positions: torch.Tensor) -> Batch:
    """Five atoms in two fragments, fully connected."""
    count: int = positions.shape[0]
    rows, cols = zip(*[(i, j) for i in range(count) for j in range(count) if i != j])
    graph = AtomicGraph(
        positions=positions,
        atomic_numbers=torch.tensor([6, 1, 1, 8, 6]),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.zeros(3, dtype=torch.bool),
        edge_index=torch.tensor([rows, cols], dtype=torch.long),
        fragment_index=torch.tensor([0, 0, 0, 1, 1]),
    )
    return Batch.from_data_list([graph])


POSITIONS: torch.Tensor = torch.tensor(
    [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [0.6, 0.6, 1.0], [2.0, 1.0, 0.3]],
    dtype=torch.float64,
)


class TestDefaultIsUnchanged:
    """``correlation: null`` must leave the model exactly as it was."""

    def test_no_module_is_built(self):
        for block in _model(None).backbone.arace.blocks:
            assert block.symmetric_contraction is None

    def test_correlation_property_reports_none(self):
        assert _model(None).backbone.arace.correlation is None

    def test_parameter_count_is_unchanged_by_the_option(self):
        """Adding the *option* must not add parameters when it is off."""
        keys_off = set(_model(None).state_dict())
        assert not any("symmetric_contraction" in k for k in keys_off)


class TestOrders:
    """Each supported order builds and grows the model."""

    @pytest.mark.parametrize("correlation", [1, 2, 3])
    def test_builds(self, correlation: int):
        model = _model(correlation)
        for block in model.backbone.arace.blocks:
            assert block.symmetric_contraction is not None
            assert block.symmetric_contraction.correlation == correlation
        assert model.backbone.arace.correlation == correlation

    def test_higher_order_means_more_parameters(self):
        counts = [sum(p.numel() for p in _model(c).parameters()) for c in (None, 1, 2, 3)]
        assert counts == sorted(counts)
        assert counts[0] < counts[-1]

    def test_unsupported_order_is_refused(self):
        with pytest.raises(ValueError, match="correlation must be one of"):
            _model(4)


class TestEquivariance:
    """The contraction must not break the symmetry the model is built on."""

    @pytest.mark.parametrize("correlation", [2, 3])
    def test_energy_is_rotation_invariant(self, correlation: int):
        model = _model(correlation)
        rotation = random_so3().double()

        plain = _batch(POSITIONS.clone())
        plain.pos.requires_grad_(True)
        rotated = _batch(POSITIONS @ rotation.T)
        rotated.pos.requires_grad_(True)

        assert float(model(plain)["energy"].sum()) == pytest.approx(
            float(model(rotated)["energy"].sum()), abs=1e-12
        )

    @pytest.mark.parametrize("correlation", [2, 3])
    def test_forces_rotate_with_the_frame(self, correlation: int):
        model = _model(correlation)
        rotation = random_so3().double()

        plain = _batch(POSITIONS.clone())
        plain.pos.requires_grad_(True)
        rotated = _batch(POSITIONS @ rotation.T)
        rotated.pos.requires_grad_(True)

        forces_plain: torch.Tensor = model(plain)["forces"].detach()
        forces_rotated: torch.Tensor = model(rotated)["forces"].detach()
        assert torch.allclose(forces_plain @ rotation.T, forces_rotated, atol=1e-11)

    def test_newtons_third_law_survives(self):
        model = _model(3)
        batch = _batch(POSITIONS.clone())
        batch.pos.requires_grad_(True)
        forces: torch.Tensor = model(batch)["forces"]
        assert torch.allclose(forces.sum(dim=0), torch.zeros(3, dtype=forces.dtype), atol=1e-9)


class TestGradients:
    """Every contraction parameter must receive a gradient each step."""

    def test_parameters_get_gradients(self):
        model = _model(3)
        batch = _batch(POSITIONS.clone())
        batch.pos.requires_grad_(True)
        model(batch)["energy"].sum().backward()

        seen: int = 0
        for block in model.backbone.arace.blocks:
            for name, parameter in block.symmetric_contraction.named_parameters():
                assert parameter.grad is not None, f"no grad for {name}"
                seen += 1
        assert seen > 0

    def test_it_actually_changes_the_output(self):
        """Not a no-op dressed as a feature."""
        batch = _batch(POSITIONS.clone())
        batch.pos.requires_grad_(True)
        plain = float(_model(None)(batch)["energy"].sum())

        batch = _batch(POSITIONS.clone())
        batch.pos.requires_grad_(True)
        contracted = float(_model(3)(batch)["energy"].sum())
        assert plain != pytest.approx(contracted)
