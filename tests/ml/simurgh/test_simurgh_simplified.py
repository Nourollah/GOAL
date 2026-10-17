"""Tests for SIMURGH Simplified — ``simurgh_simplified_monolithic`` and
``simurgh_simplified_routed``.

Everything is parametrised over both versions unless the class name says
``Routed``.  Float64 throughout: e3nn's SH / Wigner-D floor is ~1e-6
relative, so every equivariance tolerance below sits just above that
floor rather than at a hand-picked constant.
"""

from __future__ import annotations

import copy
import io
import math
import typing
from pathlib import Path

import pytest
import torch
from e3nn import o3
from omegaconf import OmegaConf
from torch_geometric.data import Batch

import goal.ml.nn.models  # noqa: F401  (auto-discovery registers the models)
from goal.ml.data.graph import AtomicGraph
from goal.ml.data.statistics import compute_pair_counts
from goal.ml.nn.models.simurgh.simurgh_simplified_layers import (
    LearnedCombination,
    LineReadout,
    SimurghSimplifiedModel,
)
from goal.ml.nn.models.simurgh.simurgh_simplified_routed import (
    RoutedSimurghSimplified,
    element_receiver_counts,
)
from goal.ml.registry import MODEL_REGISTRY

VERSIONS: tuple[str, str] = ("simurgh_simplified_monolithic", "simurgh_simplified_routed")
CUTOFF: float = 5.0
CONFIG_DIR: Path = Path(__file__).parents[3] / "src" / "goal" / "configs" / "ml"

#: Training-split pair counts for the routed model.  Chosen so that the
#: rare paths are exercised on ordinary molecules: receiver counts are
#: H 625 / C 270 / O 105 (O = 10.5% < 15% → rare atom artisan) and C-O,
#: O-O are below 5% of edges (→ rare pair artisan).
PAIR_COUNTS: dict[str, int] = {
    "1_1": 400,
    "1_6": 300,
    "6_6": 100,
    "1_8": 150,
    "6_8": 40,
    "8_8": 10,
}

SMALL: dict[str, typing.Any] = {
    "cutoff": CUTOFF,
    "embedding_dim": 8,
    "num_rbf": 6,
    "radial_hidden": 16,
    "atom_line": {"hidden_irreps": "8x0e + 8x1o + 8x2e", "num_layers": 3},
    "pair_line": {"hidden_irreps": "8x0e + 8x1e + 8x2e", "num_layers": 2},
    "fragment_line": {
        "hidden_irreps": "8x0e + 8x1o + 8x2e",
        "num_layers": 2,
        "vnode_cutoff": 15.0,
        "fragment_radius": 3.0,
    },
    "readout": {"n_scalar": 8, "hidden": 8},
}

ROUTED_EXTRA: dict[str, typing.Any] = {
    "artisan": {
        "rare_artisan": {
            "enabled": True,
            "min_element_frequency": 0.15,
            "min_pair_frequency": 0.05,
            "type_embed_dim": 4,
        },
        "pair_counts": PAIR_COUNTS,
    },
    "fragment_experts": {"num_experts": 3, "top_k": 2, "balance_loss_weight": 0.01},
}


@pytest.fixture(autouse=True)
def _float64() -> typing.Iterator[None]:
    previous: torch.dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_model(version: str, seed: int = 0, **overrides: typing.Any) -> SimurghSimplifiedModel:
    torch.manual_seed(seed)
    kwargs: dict[str, typing.Any] = copy.deepcopy(SMALL)
    if version == "simurgh_simplified_routed":
        kwargs.update(copy.deepcopy(ROUTED_EXTRA))
    kwargs.update(overrides)
    return typing.cast(SimurghSimplifiedModel, MODEL_REGISTRY.get(version)(**kwargs))


def wake_corrections(model: SimurghSimplifiedModel, seed: int = 1) -> SimurghSimplifiedModel:
    """Give the zero-initialised readouts random weights so their lines contribute."""
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (LineReadout, LearnedCombination)):
                last = typing.cast(torch.nn.Linear, module.mlp[-1])
                last.weight.copy_(0.3 * torch.randn(last.weight.shape, generator=generator))
    return model


def build_graph(
    positions: torch.Tensor,
    atomic_numbers: typing.Sequence[int],
    fragments: typing.Sequence[int],
    cutoff: float = CUTOFF,
) -> AtomicGraph:
    n: int = positions.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n):
        for j in range(n):
            if i != j and float((positions[j] - positions[i]).norm()) < cutoff:
                rows.append(i)
                cols.append(j)
    edge_index: torch.Tensor = torch.tensor([rows, cols], dtype=torch.long).reshape(2, -1)
    vectors: torch.Tensor = positions[edge_index[1]] - positions[edge_index[0]]
    return AtomicGraph(
        positions=positions.clone(),
        atomic_numbers=torch.tensor(list(atomic_numbers), dtype=torch.long),
        cell=torch.zeros(3, 3),
        pbc=torch.zeros(3, dtype=torch.bool),
        edge_index=edge_index,
        edge_vectors=vectors,
        edge_lengths=vectors.norm(dim=-1),
        fragment_index=torch.tensor(list(fragments), dtype=torch.long),
    )


def batch_of(*graphs: AtomicGraph) -> Batch:
    return Batch.from_data_list(list(graphs))


# Methanol (fragment 0) next to a water molecule (fragment 1) — no
# symmetry, three elements, and a C-O pair (rare in the routed model).
METHANOL_WATER_POS: torch.Tensor = torch.tensor(
    [
        [0.000, 0.000, 0.000],  # C
        [1.410, 0.050, -0.020],  # O
        [-0.360, 1.020, 0.110],  # H
        [-0.390, -0.520, 0.880],  # H
        [-0.330, -0.480, -0.930],  # H
        [1.720, -0.840, 0.150],  # H (hydroxyl)
        [3.900, 0.400, 0.300],  # O (water)
        [4.300, 1.250, 0.500],  # H
        [4.450, -0.200, 0.850],  # H
    ],
    dtype=torch.float64,  # built at import time, before the float64 fixture runs
)
METHANOL_WATER_Z: list[int] = [6, 8, 1, 1, 1, 1, 8, 1, 1]
METHANOL_WATER_FRAG: list[int] = [0, 0, 0, 0, 0, 0, 1, 1, 1]


def methanol_water(positions: torch.Tensor | None = None) -> Batch:
    pos: torch.Tensor = METHANOL_WATER_POS if positions is None else positions
    return batch_of(build_graph(pos, METHANOL_WATER_Z, METHANOL_WATER_FRAG))


def water_pair(separation: float) -> Batch:
    """Two water fragments ``separation`` Å apart along x."""
    water: torch.Tensor = torch.tensor([[0.0, 0.0, 0.0], [0.957, 0.0, 0.0], [-0.240, 0.927, 0.0]])
    shifted: torch.Tensor = water + torch.tensor([separation, 0.3, -0.2])
    return batch_of(
        build_graph(torch.cat([water, shifted]), [8, 1, 1, 8, 1, 1], [0, 0, 0, 1, 1, 1])
    )


def energy_and_forces(
    model: SimurghSimplifiedModel, batch: Batch
) -> tuple[torch.Tensor, torch.Tensor]:
    out: dict[str, torch.Tensor] = model(batch)
    return out["energy"].detach(), out["forces"].detach()


def equivariance_floor(reference: torch.Tensor) -> float:
    """e3nn 0.6.0 float64 floor (~1e-6 relative), padded ×10."""
    return 1e-5 * (1.0 + float(reference.abs().max()))


# ---------------------------------------------------------------------------
# Equivariance — intermediate features and outputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", VERSIONS)
class TestIntermediateEquivariance:
    def test_every_line_feature_rotates_with_its_irreps(self, version: str) -> None:
        model = wake_corrections(make_model(version))
        torch.manual_seed(7)
        rotation: torch.Tensor = o3.rand_matrix()
        reference = model.line_features(methanol_water())
        rotated = model.line_features(methanol_water(METHANOL_WATER_POS @ rotation.T))
        for line, records in reference.items():
            assert records, f"{line} line recorded no features"
            for (name, irreps, x), (name_r, _, x_rot) in zip(records, rotated[line]):
                assert name == name_r
                expected: torch.Tensor = x.detach() @ irreps.D_from_matrix(rotation).to(x.dtype).T
                error: float = float((x_rot.detach() - expected).abs().max())
                assert error < equivariance_floor(x), f"{name}: error {error:.2e}"

    def test_every_intermediate_irreps_carries_l_gt_0(self, version: str) -> None:
        records = make_model(version).line_features(methanol_water())
        for line, items in records.items():
            for name, irreps, _ in items:
                assert any(ir.l > 0 for _, ir in irreps), f"{name} has no l > 0 channels"

    def test_l_gt_0_channels_are_populated_in_every_line(self, version: str) -> None:
        """Declared is not enough: the final features of each line hold real l > 0 content."""
        records = make_model(version).line_features(methanol_water())
        for line, items in records.items():
            name, irreps, x = items[-1]
            vector_part: torch.Tensor = torch.cat(
                [x[:, s] for (_, ir), s in zip(irreps, irreps.slices()) if ir.l > 0], dim=-1
            )
            assert float(vector_part.abs().max()) > 1e-6, f"{name}: l > 0 channels are empty"


@pytest.mark.parametrize("version", VERSIONS)
class TestSymmetry:
    def test_energy_invariant_forces_equivariant_under_rotation(self, version: str) -> None:
        model = wake_corrections(make_model(version))
        torch.manual_seed(11)
        rotation: torch.Tensor = o3.rand_matrix()
        energy, forces = energy_and_forces(model, methanol_water())
        energy_r, forces_r = energy_and_forces(
            model, methanol_water(METHANOL_WATER_POS @ rotation.T)
        )
        assert torch.allclose(energy, energy_r, atol=equivariance_floor(energy), rtol=0.0)
        assert torch.allclose(forces_r, forces @ rotation.T, atol=equivariance_floor(forces))

    def test_translation_invariance(self, version: str) -> None:
        model = wake_corrections(make_model(version))
        energy, forces = energy_and_forces(model, methanol_water())
        shift: torch.Tensor = torch.tensor([37.0, -12.5, 101.0])
        energy_t, forces_t = energy_and_forces(model, methanol_water(METHANOL_WATER_POS + shift))
        assert torch.allclose(energy, energy_t, atol=1e-9, rtol=0.0)
        assert torch.allclose(forces, forces_t, atol=1e-8)

    def test_permutation_invariance(self, version: str) -> None:
        model = wake_corrections(make_model(version))
        energy, forces = energy_and_forces(model, methanol_water())
        perm: torch.Tensor = torch.tensor([6, 2, 0, 8, 5, 1, 3, 7, 4])
        permuted: Batch = batch_of(
            build_graph(
                METHANOL_WATER_POS[perm],
                [METHANOL_WATER_Z[i] for i in perm],
                [METHANOL_WATER_FRAG[i] for i in perm],
            )
        )
        energy_p, forces_p = energy_and_forces(model, permuted)
        assert torch.allclose(energy, energy_p, atol=1e-9, rtol=0.0)
        assert torch.allclose(forces[perm], forces_p, atol=1e-8)

    def test_forces_match_finite_differences(self, version: str) -> None:
        model = wake_corrections(make_model(version))
        _, forces = energy_and_forces(model, methanol_water())
        step: float = 1e-4
        for atom, axis in ((0, 0), (1, 2), (6, 1), (8, 0)):
            plus: torch.Tensor = METHANOL_WATER_POS.clone()
            minus: torch.Tensor = METHANOL_WATER_POS.clone()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            e_plus, _ = energy_and_forces(model, methanol_water(plus))
            e_minus, _ = energy_and_forces(model, methanol_water(minus))
            numeric: float = -float(e_plus - e_minus) / (2.0 * step)
            assert math.isclose(float(forces[atom, axis]), numeric, abs_tol=1e-6), (atom, axis)

    def test_pair_energy_is_symmetric(self, version: str) -> None:
        model = make_model(version)
        batch: Batch = methanol_water()
        epsilon, _ = model.pair_line(model.shared_inputs(batch))
        edges: list[tuple[int, int]] = [tuple(e) for e in batch.edge_index.T.tolist()]
        position: dict[tuple[int, int], int] = {e: k for k, e in enumerate(edges)}
        reverse: torch.Tensor = torch.tensor([position[(j, i)] for i, j in edges])
        assert float((epsilon - epsilon[reverse]).abs().max()) <= 1e-12
        assert float(epsilon.abs().max()) > 0.0


# ---------------------------------------------------------------------------
# Readout and combination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", VERSIONS)
class TestCombination:
    @pytest.mark.parametrize("combination", ["additive", "learned"])
    def test_energy_is_the_sum_of_the_lines_and_e0(self, version: str, combination: str) -> None:
        model = wake_corrections(
            make_model(
                version,
                combination=combination,
                atomic_energies={"mode": "provided", "values": {1: -13.6, 6: -1028.5, 8: -2041.1}},
                scale=0.37,
            )
        )
        out: dict[str, torch.Tensor] = model(methanol_water())
        lines: torch.Tensor = (
            out["energy_atom_line"] + out["energy_pair_line"] + out["energy_fragment_line"]
        )
        assert torch.allclose(out["energy"].detach(), lines + out["energy_baseline"], atol=1e-9)
        assert all(
            float(out[f"energy_{n}_line"].abs().max()) > 0 for n in ("atom", "pair", "fragment")
        )

    def test_learned_equals_additive_at_initialisation(self, version: str) -> None:
        batch: Batch = methanol_water()
        additive, _ = energy_and_forces(make_model(version, combination="additive"), batch)
        learned, _ = energy_and_forces(
            make_model(version, combination="learned"), methanol_water()
        )
        assert torch.allclose(additive, learned, atol=1e-12, rtol=0.0)

    @pytest.mark.parametrize("combination", ["additive", "learned"])
    def test_both_modes_train_end_to_end(self, version: str, combination: str) -> None:
        model = make_model(version, combination=combination)
        model.train()
        out: dict[str, torch.Tensor] = model(methanol_water())
        loss: torch.Tensor = out["energy"].pow(2).sum() + out["forces"].pow(2).mean()
        if "aux_loss" in out:
            loss = loss + out["aux_loss"]
        loss.backward()
        missing: list[str] = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing, f"parameters without gradient: {missing}"

    def test_fragment_line_starts_at_zero(self, version: str) -> None:
        out: dict[str, torch.Tensor] = make_model(version)(methanol_water())
        assert float(out["energy_fragment_line"].abs().max()) == 0.0


# ---------------------------------------------------------------------------
# Locality: separability and fragment range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", VERSIONS)
class TestLocality:
    def test_far_separated_copies_are_additive(self, version: str) -> None:
        """E(A + A far away) == 2 E(A) for the atom and pair lines.

        The fragment line is asserted too: its aggregation is a plain sum
        and the envelope is exactly zero beyond ``vnode_cutoff`` (15 Å),
        so at 100 Å it is separable as well.
        """
        model = wake_corrections(make_model(version))
        single: dict[str, torch.Tensor] = model(methanol_water())
        far: torch.Tensor = METHANOL_WATER_POS + torch.tensor([100.0, 0.0, 0.0])
        doubled: dict[str, torch.Tensor] = model(
            batch_of(
                build_graph(
                    torch.cat([METHANOL_WATER_POS, far]),
                    METHANOL_WATER_Z * 2,
                    METHANOL_WATER_FRAG + [f + 2 for f in METHANOL_WATER_FRAG],
                )
            )
        )
        for line in ("atom", "pair", "fragment"):
            key: str = f"energy_{line}_line"
            assert torch.allclose(doubled[key], 2.0 * single[key], atol=1e-9), line

    def test_fragment_line_sees_beyond_the_atom_cutoff(self, version: str) -> None:
        model = wake_corrections(make_model(version))
        near: dict[str, torch.Tensor] = model(water_pair(8.0))
        far: dict[str, torch.Tensor] = model(water_pair(10.0))
        # Both separations exceed the 5 Å atom cutoff: the local lines are blind ...
        for line in ("atom", "pair"):
            key: str = f"energy_{line}_line"
            assert torch.allclose(near[key], far[key], atol=1e-10), line
        # ... the fragment line is not (both are inside vnode_cutoff = 15 Å).
        gap: float = float((near["energy_fragment_line"] - far["energy_fragment_line"]).abs())
        assert gap > 1e-6
        # Beyond vnode_cutoff the fragment interaction vanishes.
        assert torch.allclose(
            model(water_pair(20.0))["energy_fragment_line"],
            model(water_pair(25.0))["energy_fragment_line"],
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# Training sanity and robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", VERSIONS)
class TestTraining:
    def test_tiny_batch_overfits(self, version: str) -> None:
        model = make_model(version)
        model.train()
        torch.manual_seed(3)
        graphs: list[AtomicGraph] = []
        for k in range(3):
            pos: torch.Tensor = METHANOL_WATER_POS + 0.05 * torch.randn(9, 3)
            graphs.append(build_graph(pos, METHANOL_WATER_Z, METHANOL_WATER_FRAG))
        # Targets from a differently seeded teacher of the same version: a
        # conservative (E, F) pair the student can actually represent —
        # independent random forces are not the gradient of any energy.
        teacher = wake_corrections(make_model(version, seed=99), seed=5)
        target_e, target_f = energy_and_forces(teacher, batch_of(*copy.deepcopy(graphs)))
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)

        def loss_of() -> torch.Tensor:
            out: dict[str, torch.Tensor] = model(batch_of(*copy.deepcopy(graphs)))
            return (out["energy"] - target_e).pow(2).mean() + (out["forces"] - target_f).pow(
                2
            ).mean()

        initial: float = float(loss_of())
        for _ in range(150):
            optimiser.zero_grad()
            loss: torch.Tensor = loss_of()
            loss.backward()
            optimiser.step()
        assert float(loss_of()) < 0.1 * initial

    def test_single_fragment_and_lone_atom_batch(self, version: str) -> None:
        """K = 1 (no fragment pairs) and an isolated atom (no edges) still run,
        and every parameter still reaches the loss (DDP static graph)."""
        model = make_model(version)
        model.train()
        lone: AtomicGraph = build_graph(torch.zeros(1, 3), [8], [0])
        one_fragment: AtomicGraph = build_graph(
            METHANOL_WATER_POS[:6], METHANOL_WATER_Z[:6], [0] * 6
        )
        out: dict[str, torch.Tensor] = model(batch_of(lone, one_fragment))
        assert torch.isfinite(out["energy"]).all() and torch.isfinite(out["forces"]).all()
        loss: torch.Tensor = out["energy"].sum() + out["forces"].pow(2).sum()
        if "aux_loss" in out:
            loss = loss + out["aux_loss"]
        loss.backward()
        assert not [n for n, p in model.named_parameters() if p.grad is None]

    def test_float32_model_on_float64_data(self, version: str) -> None:
        """The training setup: datasets ship float64 positions (the
        ``BaseDataset`` default) while nothing sets torch's default dtype,
        so the model's parameters are float32.  The model must compute in
        its own dtype rather than the positions' (regression: e3nn's
        Linear raised 'both inputs should have same dtype')."""
        torch.set_default_dtype(torch.float32)
        model = make_model(
            version,
            atomic_energies={"mode": "provided", "values": {1: -13.6, 6: -1028.5, 8: -2041.1}},
        )
        torch.set_default_dtype(torch.float64)
        model.train()
        batch: Batch = methanol_water()
        assert batch.pos.dtype == torch.float64
        out: dict[str, torch.Tensor] = model(batch)
        assert out["energy"].dtype == torch.float32
        assert torch.isfinite(out["energy"]).all() and torch.isfinite(out["forces"]).all()
        assert out["forces"].shape == (9, 3)
        loss: torch.Tensor = out["energy"].sum() + out["forces"].pow(2).sum()
        if "aux_loss" in out:
            loss = loss + out["aux_loss"]
        loss.backward()
        assert not [n for n, p in model.named_parameters() if p.grad is None]

    def test_missing_fragment_index_raises(self, version: str) -> None:
        graph: AtomicGraph = build_graph(METHANOL_WATER_POS, METHANOL_WATER_Z, METHANOL_WATER_FRAG)
        graph.fragment_index = None
        with pytest.raises(ValueError, match="fragment_index"):
            make_model(version)(batch_of(graph))

    def test_scalar_only_line_is_refused(self, version: str) -> None:
        with pytest.raises(ValueError, match="l > 0"):
            make_model(version, atom_line={"hidden_irreps": "16x0e", "num_layers": 3})


def test_monolithic_rejects_routing_sections() -> None:
    with pytest.raises(TypeError):
        make_model("simurgh_simplified_monolithic", artisan=ROUTED_EXTRA["artisan"])


def test_pair_line_refuses_odd_parity() -> None:
    with pytest.raises(ValueError, match="odd-parity"):
        make_model(
            "simurgh_simplified_monolithic",
            pair_line={"hidden_irreps": "8x0e + 8x1o + 8x2e", "num_layers": 2},
        )


# ---------------------------------------------------------------------------
# Routed only — routing, continuity, utilisation
# ---------------------------------------------------------------------------


def routed(**overrides: typing.Any) -> RoutedSimurghSimplified:
    return typing.cast(
        RoutedSimurghSimplified, make_model("simurgh_simplified_routed", **overrides)
    )


class TestRoutedRouting:
    def test_routing_table_matches_the_thresholds(self) -> None:
        model = routed()
        assert model.atom_processor.dedicated_elements == (1, 6)
        assert model.pair_processor.dedicated_pairs == ((1, 1), (1, 6), (1, 8), (6, 6))

    def test_every_atom_and_edge_has_exactly_one_artisan(self) -> None:
        model = routed()
        batch: Batch = methanol_water()
        report: dict[str, dict[str, int]] = model.utilisation(batch)
        assert sum(report["atom"].values()) == batch.num_nodes
        assert sum(report["pair"].values()) == batch.edge_index.shape[1]
        atoms: torch.Tensor = model.atom_processor.route(batch.atomic_numbers)
        assert bool(((atoms >= 0) & (atoms <= model.atom_processor.rare_index)).all())
        # The rare paths are genuinely exercised by this molecule.
        assert report["atom"]["rare"] == 2  # both oxygens
        assert report["pair"]["rare"] > 0  # C-O and O-O edges

    def test_unseen_types_route_to_rare_without_error(self) -> None:
        model = routed()
        # F (Z=9) never occurs in the counts; Z=200 is beyond the table.
        z: torch.Tensor = torch.tensor([9, 1, 200, 6])
        atom_rare: int = model.atom_processor.rare_index
        assert model.atom_processor.route(z).tolist() == [atom_rare, 0, atom_rare, 1]
        pairs: torch.Tensor = torch.tensor([[1, 9], [9, 9], [1, 200], [1, 6]])
        pair_rare: int = model.pair_processor.rare_index
        assert model.pair_processor.route(pairs).tolist() == [pair_rare, pair_rare, pair_rare, 1]
        fluorinated: Batch = batch_of(
            build_graph(
                torch.tensor([[0.0, 0.0, 0.0], [1.35, 0.0, 0.0], [-0.4, 1.0, 0.1]]),
                [6, 9, 1],
                [0, 0, 0],
            )
        )
        out: dict[str, torch.Tensor] = model(fluorinated)
        assert torch.isfinite(out["energy"]).all()

    def test_element_frequencies_count_received_messages_exactly(self) -> None:
        graphs: list[AtomicGraph] = [
            build_graph(METHANOL_WATER_POS, METHANOL_WATER_Z, METHANOL_WATER_FRAG),
            build_graph(METHANOL_WATER_POS[:6] * 1.1, METHANOL_WATER_Z[:6], [0] * 6),
        ]
        derived: dict[int, float] = element_receiver_counts(compute_pair_counts(graphs))
        brute: dict[int, float] = {}
        for g in graphs:
            for receiver in g.edge_index[1].tolist():
                z: int = int(g.atomic_numbers[receiver])
                brute[z] = brute.get(z, 0.0) + 1.0
        assert derived == brute

    def test_routing_uses_the_training_split_only(self) -> None:
        """The map is a function of the counts it is given; val/test data never enter.

        Counts from a training split without oxygen leave O routed to the
        rare artisan even though a validation structure is full of it.
        """
        train: list[AtomicGraph] = [
            build_graph(METHANOL_WATER_POS[[0, 2, 3, 4]], [6, 1, 1, 1], [0, 0, 0, 0])
        ]
        counts: dict[tuple[int, int], int] = compute_pair_counts(train)
        extra: dict[str, typing.Any] = copy.deepcopy(ROUTED_EXTRA)
        extra["artisan"]["pair_counts"] = {f"{a}_{b}": c for (a, b), c in counts.items()}
        model = routed(**extra)
        assert 8 not in model.atom_processor.dedicated_elements
        validation: Batch = water_pair(3.0)  # oxygen-rich, never counted
        report: dict[str, dict[str, int]] = model.utilisation(validation)
        assert report["atom"]["rare"] == 2

    def test_routing_survives_a_checkpoint_round_trip(self) -> None:
        model = wake_corrections(routed())
        buffer: io.BytesIO = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        state: dict[str, torch.Tensor] = torch.load(buffer, weights_only=True)
        assert "atom_line.processor.element_route" in state
        assert "pair_line.processor.pair_route" in state

        restored = routed(seed=123)  # different initial weights, same config
        restored.load_state_dict(state)
        assert torch.equal(
            restored.atom_processor.element_route, model.atom_processor.element_route
        )
        assert torch.equal(restored.pair_processor.pair_route, model.pair_processor.pair_route)
        energy, forces = energy_and_forces(model, methanol_water())
        energy_r, forces_r = energy_and_forces(restored, methanol_water())
        assert torch.allclose(energy, energy_r, atol=1e-12) and torch.allclose(
            forces, forces_r, atol=1e-12
        )

    def test_loaded_buffer_is_authoritative(self) -> None:
        """Same module layout, different assignment: loading restores the trained map."""
        model = routed()
        table: torch.Tensor = model.atom_processor.element_route.clone()
        table[1], table[6] = table[6].clone(), table[1].clone()
        model.atom_processor.element_route.copy_(table)
        fresh = routed()
        fresh.load_state_dict(model.state_dict())
        assert torch.equal(fresh.atom_processor.element_route, table)

    def test_routing_needs_counts_and_the_enable_flag(self) -> None:
        extra: dict[str, typing.Any] = copy.deepcopy(ROUTED_EXTRA)
        extra["artisan"]["pair_counts"] = None
        with pytest.raises(ValueError, match="pair_counts"):
            routed(**extra)
        extra = copy.deepcopy(ROUTED_EXTRA)
        extra["artisan"]["rare_artisan"]["enabled"] = False
        with pytest.raises(ValueError, match="enabled"):
            routed(**extra)


class TestRoutedContinuity:
    """Gate for open question 4: no energy jumps along a fine coordinate scan."""

    @pytest.mark.parametrize("top_k", [1, 2])
    def test_pes_scan_has_no_jumps(self, top_k: int) -> None:
        extra: dict[str, typing.Any] = copy.deepcopy(ROUTED_EXTRA)
        extra["fragment_experts"]["top_k"] = top_k
        model = wake_corrections(routed(**extra))
        # Pull the water away from the methanol through the atom cutoff
        # region (edges leave the 5 Å sphere) in 1e-3 Å steps.
        step: float = 1e-3
        n_steps: int = 1500
        chunk: int = 250
        energies: list[torch.Tensor] = []
        slopes: list[torch.Tensor] = []  # dE/ds along the scan = -Σ F_x of the moved atoms
        # Scan geometries are independent structures, so they are batched
        # (a structure's energy cannot depend on its batch neighbours).
        for start in range(0, n_steps, chunk):
            graphs: list[AtomicGraph] = []
            for k in range(start, min(start + chunk, n_steps)):
                pos: torch.Tensor = METHANOL_WATER_POS.clone()
                pos[6:] += torch.tensor([k * step, 0.0, 0.0])
                graphs.append(build_graph(pos, METHANOL_WATER_Z, METHANOL_WATER_FRAG))
            energy, forces = energy_and_forces(model, batch_of(*graphs))
            energies.append(energy)
            slopes.append(-forces.reshape(len(graphs), 9, 3)[:, 6:, 0].sum(dim=-1))
        deltas: torch.Tensor = torch.cat(energies).diff()
        slope: torch.Tensor = torch.cat(slopes)
        # Trapezoid prediction of each step from the forces at both ends.
        predicted: torch.Tensor = 0.5 * (slope[:-1] + slope[1:]) * step
        residual: torch.Tensor = (deltas - predicted).abs()
        # A smooth E leaves a third-order residual (~1e-9 here); a jump in
        # E, or in F, shows up as a spike of order the jump itself.
        assert float(residual.max()) < 1e-6, float(residual.max())

    def test_gate_does_not_depend_on_positions(self) -> None:
        model = routed()
        base: dict[str, int] = model.utilisation(methanol_water())["fragment"]
        torch.manual_seed(5)
        jittered: torch.Tensor = METHANOL_WATER_POS + 0.3 * torch.randn(9, 3)
        assert model.utilisation(methanol_water(jittered))["fragment"] == base


class TestRoutedUtilisation:
    def test_utilisation_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        model = routed()
        batch: Batch = batch_of(
            build_graph(METHANOL_WATER_POS, METHANOL_WATER_Z, METHANOL_WATER_FRAG),
            build_graph(METHANOL_WATER_POS * 1.05, METHANOL_WATER_Z, METHANOL_WATER_FRAG),
        )
        report: dict[str, dict[str, int]] = model.utilisation(batch)
        print("\n[utilisation]", report)
        moe = model.fragment_processor
        assert sum(report["fragment"].values()) == moe.top_k * 4  # 2 graphs × 2 fragments
        assert set(report["atom"]) == {"Z1", "Z6", "rare"}
        assert "[utilisation]" in capsys.readouterr().out

    def test_balance_loss_is_returned_only_when_weighted(self) -> None:
        assert "aux_loss" in routed()(methanol_water())
        extra: dict[str, typing.Any] = copy.deepcopy(ROUTED_EXTRA)
        extra["fragment_experts"]["balance_loss_weight"] = 0.0
        assert "aux_loss" not in routed(**extra)(methanol_water())


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

#: Every SIMURGH Simplified config pair, by dataset.
DATASETS: tuple[str, ...] = ("gmd26", "md17")
PROCESSOR_KEYS: frozenset[str] = frozenset({"name", "artisan", "fragment_experts"})


def _backbone(version: str, dataset: str) -> dict[str, typing.Any]:
    config = OmegaConf.load(CONFIG_DIR / f"{version}_{dataset}.yaml")
    return typing.cast(
        dict[str, typing.Any], OmegaConf.to_container(config.model.backbone, resolve=True)
    )


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("version", VERSIONS)
def test_config_builds_its_model(version: str, dataset: str) -> None:
    kwargs: dict[str, typing.Any] = _backbone(version, dataset)
    assert kwargs.pop("name") == version
    # What train.py injects from the training split:
    if version == "simurgh_simplified_routed":
        kwargs["artisan"]["pair_counts"] = PAIR_COUNTS
    if kwargs["atomic_energies"].get("mode") == "dataset":
        kwargs["atomic_energies"]["values"] = {1: -13.6, 6: -1028.5, 8: -2041.1}
    model: SimurghSimplifiedModel = MODEL_REGISTRY.get(version)(**kwargs)
    assert torch.isfinite(model(methanol_water())["energy"]).all()


@pytest.mark.parametrize("dataset", DATASETS)
def test_configs_differ_only_in_the_processors(dataset: str) -> None:
    """The ablation contract: every shared backbone key is identical."""
    mono: dict[str, typing.Any] = _backbone("simurgh_simplified_monolithic", dataset)
    routed_cfg: dict[str, typing.Any] = _backbone("simurgh_simplified_routed", dataset)
    shared: set[str] = (set(mono) | set(routed_cfg)) - PROCESSOR_KEYS
    for key in shared:
        assert mono.get(key) == routed_cfg.get(key), key
