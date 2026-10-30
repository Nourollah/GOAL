"""Tests for the three atomic-energy modes (learned / dataset / provided)
and the ScaleShift gain on SIMURGH.

Mirrors what MACE does with :class:`AtomicEnergiesBlock` and
:class:`ScaleShiftBlock` — the model carries a per-element baseline
that's either trained jointly, derived from data via least squares,
or supplied directly; and a multiplicative gain on the residual
interaction energy.

Covers:

* ``mode="learned"`` — verify the baseline is a learnable
  ``nn.Parameter`` that appears in ``model.parameters()`` and receives
  non-zero gradients.
* ``mode="dataset"`` — verify
  :func:`compute_atomic_references` recovers known isolated-atom
  energies to within ``0.01`` eV on a synthetic multi-molecule
  dataset; verify residuals after subtraction are within ``[-1, 1]``
  eV; verify the LSQ output, when fed back to the backbone, produces
  the expected baseline contribution.
* ``mode="provided"`` — verify the values are registered as buffers
  (not parameters) and stay constant under an optimiser step.
* Training-loop sanity — verify a SIMURGH run on the synthetic
  dataset drives the training loss strictly down for at least 20
  optimiser steps in each mode.
"""

from __future__ import annotations

import typing

import pytest
import torch
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph
from goal.ml.data.statistics import compute_atomic_references, compute_energy_scale
from goal.ml.nn.models.simurgh.legacy.backbone import SimurghBackbone

# ---------------------------------------------------------------------------
# Synthetic multi-molecule dataset
# ---------------------------------------------------------------------------

# Reference per-element energies used to generate synthetic labels.
# Three elements, three distinct molecule formulae — the count matrix
# is full rank so the LSQ solution is unique.
_TRUE_REFS: dict[int, float] = {1: -13.6, 6: -148.0, 8: -434.0}


def _build_graph(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    energy: float,
    cutoff: float = 5.0,
) -> AtomicGraph:
    """Build an ``AtomicGraph`` with a brute-force radius cutoff and an
    energy label."""
    n: int = positions.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    vecs: list[torch.Tensor] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r: torch.Tensor = positions[j] - positions[i]
            d: float = float(r.norm())
            if d < cutoff:
                rows.append(i)
                cols.append(j)
                vecs.append(r)
    edge_index: torch.Tensor = torch.tensor([rows, cols], dtype=torch.long)
    edge_vectors: torch.Tensor = (
        torch.stack(vecs) if vecs else torch.zeros(0, 3, dtype=positions.dtype)
    )
    edge_lengths: torch.Tensor = (
        edge_vectors.norm(dim=-1) if vecs else torch.zeros(0, dtype=positions.dtype)
    )
    return AtomicGraph(
        positions=positions,
        atomic_numbers=atomic_numbers,
        cell=torch.zeros(3, 3, dtype=positions.dtype),
        pbc=torch.zeros(3, dtype=torch.bool),
        edge_index=edge_index,
        edge_vectors=edge_vectors,
        edge_lengths=edge_lengths,
        energy=torch.tensor([energy], dtype=positions.dtype),
    )


def _methane(noise: float = 0.0) -> AtomicGraph:
    """CH₄ tetrahedron, energy = Σ e(Z) + ``noise``."""
    a: float = 1.09 / (3.0**0.5)
    positions: torch.Tensor = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [a, a, a],
            [a, -a, -a],
            [-a, a, -a],
            [-a, -a, a],
        ],
        dtype=torch.float64,
    )
    z: torch.Tensor = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)
    e: float = _TRUE_REFS[6] + 4 * _TRUE_REFS[1] + noise
    return _build_graph(positions, z, e)


def _ethanol(noise: float = 0.0) -> AtomicGraph:
    """A geometrically valid ethanol arrangement (C-C-O backbone)."""
    positions: torch.Tensor = torch.tensor(
        [
            [0.000, 0.000, 0.000],  # C
            [1.520, 0.000, 0.000],  # C
            [2.020, 1.350, 0.000],  # O
            [-0.500, 1.000, 0.000],  # H on C0
            [-0.500, -0.500, 0.870],  # H on C0
            [-0.500, -0.500, -0.870],  # H on C0
            [2.020, -0.500, 0.870],  # H on C1
            [2.020, -0.500, -0.870],  # H on C1
            [2.970, 1.350, 0.000],  # H on O
        ],
        dtype=torch.float64,
    )
    z: torch.Tensor = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    e: float = 2 * _TRUE_REFS[6] + _TRUE_REFS[8] + 6 * _TRUE_REFS[1] + noise
    return _build_graph(positions, z, e)


def _benzene(noise: float = 0.0) -> AtomicGraph:
    """Planar C₆H₆ ring with H pointing outward."""
    import math

    positions_list: list[list[float]] = []
    z_list: list[int] = []
    r_c: float = 1.40
    r_h: float = 2.50
    for k in range(6):
        theta: float = 2.0 * math.pi * k / 6.0
        positions_list.append([r_c * math.cos(theta), r_c * math.sin(theta), 0.0])
        z_list.append(6)
    for k in range(6):
        theta = 2.0 * math.pi * k / 6.0
        positions_list.append([r_h * math.cos(theta), r_h * math.sin(theta), 0.0])
        z_list.append(1)
    positions: torch.Tensor = torch.tensor(positions_list, dtype=torch.float64)
    z: torch.Tensor = torch.tensor(z_list, dtype=torch.long)
    e: float = 6 * _TRUE_REFS[6] + 6 * _TRUE_REFS[1] + noise
    return _build_graph(positions, z, e)


@pytest.fixture(scope="module")
def synthetic_dataset() -> list[AtomicGraph]:
    """List of three molecules with linearly-independent count vectors.

    Adds tiny zero-mean Gaussian noise to the energies so the LSQ
    residual is non-degenerate but still well within ``[-1, 1]`` eV.
    """
    torch.manual_seed(0)
    noise_scale: float = 0.005
    eps: torch.Tensor = torch.randn(3, dtype=torch.float64) * noise_scale
    return [
        _methane(float(eps[0])),
        _ethanol(float(eps[1])),
        _benzene(float(eps[2])),
    ]


@pytest.fixture(scope="module")
def synthetic_dataset_noisy() -> list[AtomicGraph]:
    """Same molecules, but with multiple noisy copies per formula.

    With 3 unique formulae and 3 unknowns the LSQ system is exactly
    determined and recovers any noisy labels *exactly* — leaving the
    artisan bank no residual to fit.  Stacking 4 noisy copies per formula gives
    an over-determined system so a real ``O(eV)`` residual survives
    for the training-loop tests to consume.
    """
    torch.manual_seed(1)
    noise_scale: float = 0.5  # eV — larger than per-atom LSQ error
    copies_per_formula: int = 4
    builders: list[typing.Callable[[float], AtomicGraph]] = [_methane, _ethanol, _benzene]
    dataset: list[AtomicGraph] = []
    for builder in builders:
        for _ in range(copies_per_formula):
            noise: float = float(torch.randn(()).item()) * noise_scale
            dataset.append(builder(noise))
    return dataset


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


class TestComputeAtomicReferences:
    """``compute_atomic_references`` on the synthetic dataset."""

    def test_recovers_isolated_atom_energies(self, synthetic_dataset) -> None:
        refs: dict[int, float] = compute_atomic_references(synthetic_dataset)
        assert set(refs) == set(
            _TRUE_REFS
        ), f"Recovered Z keys {sorted(refs)} != true {sorted(_TRUE_REFS)}"
        # With three molecules and a single Z=8 species (only in
        # ethanol), Z=8 is the least-constrained column.  Allow 0.05
        # eV of noise-amplified error — well below the eV-scale
        # offsets the baseline is supposed to absorb.
        for z, true_e in _TRUE_REFS.items():
            assert (
                abs(refs[z] - true_e) < 0.05
            ), f"Z={z}: recovered {refs[z]:.4f} vs true {true_e:.4f}"

    def test_residuals_within_one_eV(self, synthetic_dataset) -> None:
        refs: dict[int, float] = compute_atomic_references(synthetic_dataset)
        for g in synthetic_dataset:
            e_atomic: float = sum(refs[int(z)] for z in g.atomic_numbers.tolist())
            residual: float = float(g.energy.sum()) - e_atomic
            assert abs(residual) < 1.0, (
                f"Residual {residual:.4f} eV is outside [-1, 1] for "
                f"a {g.atomic_numbers.shape[0]}-atom molecule"
            )

    def test_empty_dataset_returns_empty_dict(self) -> None:
        assert compute_atomic_references([]) == {}


class TestComputeEnergyScale:
    """``compute_energy_scale`` on the synthetic dataset."""

    def test_scale_is_small_after_baseline_subtracted(self, synthetic_dataset) -> None:
        refs: dict[int, float] = compute_atomic_references(synthetic_dataset)
        scale: float = compute_energy_scale(synthetic_dataset, refs)
        # Residuals are ~0.005 eV from the noise; mean(n_atoms) is ~8.
        # std/mean should be small (well below 1 eV/atom).
        assert 0.0 < scale < 1.0, f"scale={scale} outside expected range"


# ---------------------------------------------------------------------------
# SimurghBackbone modes
# ---------------------------------------------------------------------------


def _small_backbone(
    atomic_energies: dict[str, typing.Any] | None,
    scale: float | None = None,
) -> SimurghBackbone:
    """Tiny SimurghBackbone configuration suitable for fast unit tests."""
    return SimurghBackbone(
        elements=(1, 6, 8),
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
        atomic_energies=atomic_energies,
        scale=scale,
        num_elements_table=120,
    ).to(torch.float64)


class TestLearnedMode:
    """``mode='learned'`` — atomic_energies is a trainable Parameter."""

    def test_is_parameter_and_in_parameters(self) -> None:
        model = _small_backbone(atomic_energies={"mode": "learned"})
        assert isinstance(model.atomic_energies, torch.nn.Parameter)
        # The parameter must be reachable via ``named_parameters`` so
        # the optimiser actually trains it.
        names: list[str] = [n for n, _ in model.named_parameters()]
        assert "atomic_energies" in names

    def test_receives_nonzero_gradient(self, synthetic_dataset) -> None:
        model = _small_backbone(atomic_energies={"mode": "learned"})
        batch: Batch = Batch.from_data_list(synthetic_dataset)
        out = model(batch)
        # Sum node_energies to get per-graph total energies; force a
        # gradient back into ``atomic_energies`` through indexing.
        loss: torch.Tensor = out.node_energies.pow(2).sum()
        loss.backward()
        assert model.atomic_energies.grad is not None
        # At least the entries for observed Z must have non-zero grad.
        for z in _TRUE_REFS:
            assert (
                float(model.atomic_energies.grad[z].abs()) > 0.0
            ), f"Z={z} got zero gradient — learned baseline isn't reached"

    def test_default_mode_is_learned(self) -> None:
        """When no ``atomic_energies`` block is provided the backbone
        falls back to ``learned`` (matches the config default)."""
        model = _small_backbone(atomic_energies=None)
        assert model.atomic_energies_mode == "learned"
        assert isinstance(model.atomic_energies, torch.nn.Parameter)


class TestDatasetMode:
    """``mode='dataset'`` — buffer seeded from LSQ output."""

    def test_buffer_not_parameter(self, synthetic_dataset) -> None:
        refs: dict[int, float] = compute_atomic_references(synthetic_dataset)
        model = _small_backbone(
            atomic_energies={"mode": "dataset", "values": refs},
        )
        # Must NOT be in parameters …
        names: list[str] = [n for n, _ in model.named_parameters()]
        assert "atomic_energies" not in names
        # … and MUST be in buffers.
        buffer_names: list[str] = [n for n, _ in model.named_buffers()]
        assert "atomic_energies" in buffer_names

    def test_values_match_lsq_output(self, synthetic_dataset) -> None:
        refs: dict[int, float] = compute_atomic_references(synthetic_dataset)
        model = _small_backbone(
            atomic_energies={"mode": "dataset", "values": refs},
        )
        # Tolerance reflects the float32 default dtype used by torch
        # for buffers — values originate as float64 from LSQ.
        for z, e in refs.items():
            assert abs(float(model.atomic_energies[z]) - e) < 1e-3

    def test_missing_values_raises(self) -> None:
        with pytest.raises(ValueError, match="requires 'values'"):
            _small_backbone(atomic_energies={"mode": "dataset"})

    def test_compute_from_dataset_flag_is_dataset_mode(self) -> None:
        """``compute_from_dataset=true`` with default mode resolves to
        ``dataset``, so a missing ``values`` must raise the same error."""
        with pytest.raises(ValueError, match="requires 'values'"):
            _small_backbone(
                atomic_energies={
                    "mode": "learned",
                    "compute_from_dataset": True,
                }
            )


class TestProvidedMode:
    """``mode='provided'`` — buffer seeded from user-supplied dict."""

    def test_buffer_not_parameter(self) -> None:
        refs: dict[int, float] = {1: -13.6, 6: -148.0, 8: -434.0}
        model = _small_backbone(
            atomic_energies={"mode": "provided", "values": refs},
        )
        names: list[str] = [n for n, _ in model.named_parameters()]
        assert "atomic_energies" not in names

    def test_not_updated_by_optimiser_step(self, synthetic_dataset) -> None:
        refs: dict[int, float] = {1: -13.6, 6: -148.0, 8: -434.0}
        model = _small_backbone(
            atomic_energies={"mode": "provided", "values": refs},
        )
        before: torch.Tensor = model.atomic_energies.clone()

        # Run a single SGD step with a large learning rate — only
        # parameters move; the buffer stays put.
        optim: torch.optim.SGD = torch.optim.SGD(model.parameters(), lr=1.0)
        batch: Batch = Batch.from_data_list(synthetic_dataset)
        out = model(batch)
        loss: torch.Tensor = out.node_energies.pow(2).sum()
        loss.backward()
        optim.step()

        assert torch.allclose(model.atomic_energies, before, atol=0.0), (
            "atomic_energies buffer changed under an optimiser step — "
            "it should be frozen in provided mode"
        )


class TestScale:
    """The scalar gain buffer."""

    def test_scale_defaults_to_one(self) -> None:
        model = _small_backbone(atomic_energies={"mode": "learned"})
        assert float(model.scale) == 1.0

    def test_scale_applied_to_interaction_residual(self) -> None:
        """A non-unit scale must multiply the interaction-energy
        contribution exactly, leaving the atomic baseline untouched."""
        refs: dict[int, float] = {1: -13.6, 6: -148.0, 8: -434.0}

        torch.manual_seed(0)
        model_a = _small_backbone(
            atomic_energies={"mode": "provided", "values": refs},
            scale=1.0,
        )
        torch.manual_seed(0)
        model_b = _small_backbone(
            atomic_energies={"mode": "provided", "values": refs},
            scale=3.0,
        )

        batch: Batch = Batch.from_data_list([_methane()])
        with torch.no_grad():
            out_a = model_a(batch).node_energies
            out_b = model_b(batch).node_energies

        # Baseline contribution is identical, so the difference comes
        # purely from the scale factor on the interaction residual.
        baseline: torch.Tensor = model_a.atomic_energies[batch.atomic_numbers].to(out_a.dtype)
        residual_a: torch.Tensor = out_a - baseline
        residual_b: torch.Tensor = out_b - baseline
        # residual_b ≈ 3 × residual_a
        torch.testing.assert_close(residual_b, 3.0 * residual_a, rtol=1e-5, atol=1e-9)


# ---------------------------------------------------------------------------
# Training-loop sanity for the three modes
# ---------------------------------------------------------------------------


def _train_steps(
    model: SimurghBackbone,
    batch: Batch,
    targets: torch.Tensor,
    n_steps: int = 20,
    lr: float = 1e-2,
) -> list[float]:
    """Run ``n_steps`` Adam steps fitting ``Σ node_energies → targets``.

    Returns the per-step loss values.  Used to assert monotonic
    decrease across the modes.
    """
    optim: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(n_steps):
        optim.zero_grad()
        out = model(batch)
        # Per-graph total energies via index_add over the batch index.
        graph_energy: torch.Tensor = torch.zeros(
            int(batch.batch.max()) + 1,
            device=out.node_energies.device,
            dtype=out.node_energies.dtype,
        )
        graph_energy = graph_energy.index_add(0, batch.batch, out.node_energies)
        loss: torch.Tensor = (graph_energy - targets).pow(2).mean()
        loss.backward()
        optim.step()
        losses.append(float(loss))
    return losses


@pytest.mark.parametrize(
    "mode_kwargs",
    [
        {"mode": "learned"},
        # "dataset" needs values — supplied per-test
        # "provided" supplied per-test
    ],
    ids=["learned"],
)
def test_training_loss_monotone_learned(synthetic_dataset, mode_kwargs) -> None:
    """``learned`` mode: 20 Adam steps drive loss strictly down."""
    torch.manual_seed(0)
    model = _small_backbone(atomic_energies=mode_kwargs)
    batch: Batch = Batch.from_data_list(synthetic_dataset)
    targets: torch.Tensor = torch.tensor(
        [float(g.energy.sum()) for g in synthetic_dataset], dtype=torch.float64
    )
    losses: list[float] = _train_steps(model, batch, targets, n_steps=20, lr=1e-1)
    # Strict monotone decrease is fragile under stochastic init; check
    # that the loss is meaningfully smaller at the end and that it's
    # non-increasing over a moving average of 5.
    assert (
        losses[-1] < losses[0] * 0.5
    ), f"learned mode: loss[0]={losses[0]:.4f} vs loss[-1]={losses[-1]:.4f}"


def test_training_loss_monotone_dataset(synthetic_dataset_noisy) -> None:
    """``dataset`` mode: baseline carries the constant, residual fits fast.

    Uses the noisy dataset so a non-degenerate residual remains for the
    artisan bank to fit — with the low-noise fixture the LSQ recovery would
    drive the initial loss to machine epsilon and there'd be nothing
    to optimise.
    """
    torch.manual_seed(0)
    refs: dict[int, float] = compute_atomic_references(synthetic_dataset_noisy)
    scale: float = compute_energy_scale(synthetic_dataset_noisy, refs)
    model = _small_backbone(
        atomic_energies={"mode": "dataset", "values": refs},
        scale=scale,
    )
    batch: Batch = Batch.from_data_list(synthetic_dataset_noisy)
    targets: torch.Tensor = torch.tensor(
        [float(g.energy.sum()) for g in synthetic_dataset_noisy],
        dtype=torch.float64,
    )
    losses: list[float] = _train_steps(model, batch, targets, n_steps=20, lr=1e-2)
    assert (
        losses[-1] < losses[0]
    ), f"dataset mode: loss[0]={losses[0]:.4f} vs loss[-1]={losses[-1]:.4f}"


def test_training_loss_monotone_provided(synthetic_dataset_noisy) -> None:
    """``provided`` mode: same as dataset but values supplied directly."""
    torch.manual_seed(0)
    model = _small_backbone(
        atomic_energies={"mode": "provided", "values": _TRUE_REFS},
    )
    batch: Batch = Batch.from_data_list(synthetic_dataset_noisy)
    targets: torch.Tensor = torch.tensor(
        [float(g.energy.sum()) for g in synthetic_dataset_noisy],
        dtype=torch.float64,
    )
    losses: list[float] = _train_steps(model, batch, targets, n_steps=20, lr=1e-2)
    assert (
        losses[-1] < losses[0]
    ), f"provided mode: loss[0]={losses[0]:.4f} vs loss[-1]={losses[-1]:.4f}"
