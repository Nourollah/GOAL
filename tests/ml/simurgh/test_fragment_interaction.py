"""Fragment decomposition and the equivariant fragment-interaction add-on.

Coverage:

* Test 1 — ``compute_fragment_index``: one connected molecule → a single
  label; two separated molecules → two labels.
* Test 2 — batching: fragment labels are offset per graph (no collisions)
  and stay contiguous.
* Test 3 — ``fragment_interaction_config=None`` (default) is a strict
  no-op: the round's output is bit-identical to plain ARACE.
* Test 4 — a single fragment (K=1) has nothing to interact with, so the
  correction is exactly zero — while every parameter still receives a
  gradient (DDP static-graph contract).
* Test 5 — with two fragments the channel carries a real signal, on
  **all** angular orders, and it depends on the inter-fragment geometry.
* Test 6 — rotation equivariance with the fragment channel enabled.
* Plus: locality (a fragment beyond the cutoff contributes nothing),
  force flow through the centroids, and batch isolation.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.data import Batch

from goal.ml.data.fragments import compute_fragment_index
from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.blocks.ace_block import AraceRound
from goal.ml.nn.blocks.fragment_interaction import (
    EquivariantFragmentInteraction,
    build_fragment_geometry,
)
from goal.ml.nn.heads.energy_forces import EnergyForcesHead
from goal.ml.nn.models.simurgh.backbone_arace import SimurghAraceBackbone
from tests.ml.simurgh.conftest import _build_graph, random_so3

CUTOFF = 5.0
HIDDEN_IRREPS = "8x0e + 8x1o + 8x2e"
# Flat column spans of HIDDEN_IRREPS: 8 scalars, 8x3 vectors, 8x5 l=2.
L0_SLICE = slice(0, 8)
L1_SLICE = slice(8, 32)
L2_SLICE = slice(32, 72)

ARTISAN_SUBCONFIG: dict = {
    "architecture": "equivariant",
    "hidden_irreps": HIDDEN_IRREPS,
    "num_layers": 1,
    "num_rbf": 4,
    "radial_hidden": 16,
    "n_scalar_out": 8,
    "final_hidden": 8,
    "element_conditioned": True,
}

FRAGMENT_CONFIG: dict = {
    "num_rbf": 4,
    "radial_hidden": 16,
    "num_layers": 1,
    "init_zero": True,
}
#: Same module with a live (non-zero) output projection, for tests that
#: need the channel to actually do something.
ACTIVE_FRAGMENT_CONFIG: dict = {**FRAGMENT_CONFIG, "init_zero": False}


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------


def _methane_positions() -> tuple[torch.Tensor, torch.Tensor]:
    a: float = 1.09 / math.sqrt(3.0)
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
    numbers: torch.Tensor = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)
    return positions, numbers


def _water_positions(offset: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    positions: torch.Tensor = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.957, 0.0, 0.0],
            [-0.240, 0.927, 0.0],
        ],
        dtype=torch.float64,
    )
    positions = positions + torch.tensor([offset, 0.0, 0.0], dtype=torch.float64)
    numbers: torch.Tensor = torch.tensor([8, 1, 1], dtype=torch.long)
    return positions, numbers


def _water_dimer(separation: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Two water molecules *separation* Å apart — not bonded, inside the cutoff."""
    pos_a, z_a = _water_positions(offset=0.0)
    pos_b, z_b = _water_positions(offset=separation)
    return torch.cat([pos_a, pos_b]), torch.cat([z_a, z_b])


def _labelled_graph(positions: torch.Tensor, numbers: torch.Tensor) -> AtomicGraph:
    graph: AtomicGraph = _build_graph(positions, numbers, cutoff=CUTOFF)
    graph.fragment_index = compute_fragment_index(positions, numbers, covalent_cutoff=1.8)
    return graph


# ----------------------------------------------------------------------
# Test 1 — fragment index computation
# ----------------------------------------------------------------------


class TestFragmentIndexComputation:
    def test_single_connected_methane_is_one_fragment(self) -> None:
        positions, numbers = _methane_positions()
        labels = compute_fragment_index(positions, numbers, covalent_cutoff=1.8)

        assert labels.dtype == torch.long
        assert labels.shape == (5,)
        assert torch.equal(labels, torch.zeros(5, dtype=torch.long))
        assert int(labels.max()) + 1 == 1

    def test_two_separate_waters_are_two_fragments(self) -> None:
        positions, numbers = _water_dimer()
        labels = compute_fragment_index(positions, numbers, covalent_cutoff=1.8)

        assert int(labels.max()) + 1 == 2
        assert torch.equal(labels[:3], torch.zeros(3, dtype=torch.long))
        assert torch.equal(labels[3:], torch.ones(3, dtype=torch.long))

    def test_isolated_atoms_are_separate_fragments(self) -> None:
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        numbers = torch.tensor([1, 1, 1], dtype=torch.long)
        labels = compute_fragment_index(positions, numbers, covalent_cutoff=1.8)

        assert int(labels.max()) + 1 == 3
        assert torch.equal(labels, torch.tensor([0, 1, 2]))

    def test_labels_are_contiguous_and_deterministic(self) -> None:
        positions, numbers = _water_dimer()
        first = compute_fragment_index(positions, numbers)
        second = compute_fragment_index(positions, numbers)
        assert torch.equal(first, second)
        present = {int(v) for v in first.unique()}
        assert present == set(range(int(first.max()) + 1))

    def test_periodic_molecule_across_boundary_stays_one_fragment(self) -> None:
        positions = torch.tensor([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]], dtype=torch.float64)
        numbers = torch.tensor([8, 1], dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        pbc = torch.ones(3, dtype=torch.bool)

        without_pbc = compute_fragment_index(positions, numbers)
        with_pbc = compute_fragment_index(positions, numbers, cell=cell, pbc=pbc)

        assert int(without_pbc.max()) + 1 == 2, "no-PBC run should see two fragments"
        assert int(with_pbc.max()) + 1 == 1, "MIC should join the wrapped pair"

    def test_rejects_mismatched_inputs(self) -> None:
        positions, _ = _methane_positions()
        with pytest.raises(ValueError, match="different structures"):
            compute_fragment_index(positions, torch.tensor([6, 1], dtype=torch.long))

    def test_empty_and_single_atom(self) -> None:
        empty = compute_fragment_index(
            torch.zeros(0, 3, dtype=torch.float64), torch.zeros(0, dtype=torch.long)
        )
        assert empty.shape == (0,)
        single = compute_fragment_index(
            torch.zeros(1, 3, dtype=torch.float64), torch.tensor([1], dtype=torch.long)
        )
        assert torch.equal(single, torch.zeros(1, dtype=torch.long))


# ----------------------------------------------------------------------
# Test 2 — fragment index batching
# ----------------------------------------------------------------------


class TestFragmentIndexBatching:
    def test_two_methanes_do_not_collide(self) -> None:
        positions, numbers = _methane_positions()
        graphs = [_labelled_graph(positions, numbers) for _ in range(2)]
        batch = Batch.from_data_list(graphs)

        labels = batch.fragment_index
        assert labels.shape == (10,)
        assert torch.equal(labels[:5], torch.zeros(5, dtype=torch.long))
        assert torch.equal(labels[5:], torch.ones(5, dtype=torch.long))
        assert int(labels.max()) + 1 == 2

    def test_mixed_fragment_counts_stay_contiguous(self) -> None:
        methane = _labelled_graph(*_methane_positions())
        dimer = _labelled_graph(*_water_dimer())
        batch = Batch.from_data_list([methane, dimer, methane.clone()])

        labels = batch.fragment_index
        assert int(labels.max()) + 1 == 4
        assert {int(v) for v in labels.unique()} == {0, 1, 2, 3}
        assert torch.equal(labels[:5], torch.zeros(5, dtype=torch.long))
        assert torch.equal(labels[5:8], torch.ones(3, dtype=torch.long))
        assert torch.equal(labels[8:11], torch.full((3,), 2, dtype=torch.long))
        assert torch.equal(labels[11:], torch.full((5,), 3, dtype=torch.long))

    def test_labels_never_cross_graph_boundaries(self) -> None:
        methane = _labelled_graph(*_methane_positions())
        dimer = _labelled_graph(*_water_dimer())
        batch = Batch.from_data_list([methane, dimer])

        for frag in batch.fragment_index.unique():
            graphs_touched = batch.batch[batch.fragment_index == frag].unique()
            assert (
                graphs_touched.numel() == 1
            ), f"fragment {int(frag)} spans graphs {graphs_touched.tolist()}"

    def test_unlabelled_graphs_carry_no_field(self) -> None:
        graph = _build_graph(*_methane_positions(), cutoff=CUTOFF)
        assert graph.fragment_index is None
        assert graph.num_fragments == 0
        batch = Batch.from_data_list([graph, graph.clone()])
        assert batch.get("fragment_index", None) is None


# ----------------------------------------------------------------------
# Round-level helpers
# ----------------------------------------------------------------------


def _build_round(
    fragment_interaction_config: dict | None = None,
    adaptive_gate_config: dict | None = None,
    seed: int = 0,
) -> AraceRound:
    torch.manual_seed(seed)
    return AraceRound(
        elements=(1, 6, 7, 8),
        irreps_node=HIDDEN_IRREPS,
        cutoff=CUTOFF,
        artisan_kwargs={k: v for k, v in ARTISAN_SUBCONFIG.items() if k != "architecture"},
        avg_num_neighbors=4.0,
        fragment_interaction_config=fragment_interaction_config,
        adaptive_gate_config=adaptive_gate_config,
    ).double()


def _round_inputs(
    graph: AtomicGraph,
    rnd: AraceRound,
    seed: int = 1234,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(h, edge_index, edge_sh, edge_lengths)`` for a single round call."""
    from e3nn.o3 import spherical_harmonics

    torch.manual_seed(seed)
    h: torch.Tensor = torch.randn(graph.pos.shape[0], rnd.irreps_node.dim, dtype=torch.float64)
    edge_sh: torch.Tensor = spherical_harmonics(
        rnd.irreps_edge, graph.edge_vectors, normalize=True, normalization="component"
    )
    return h, graph.edge_index, edge_sh, graph.edge_lengths


def _geometry(module: EquivariantFragmentInteraction, graph: AtomicGraph):
    return module.build_geometry(
        positions=graph.pos,
        fragment_index=graph.fragment_index,
        batch=graph.get("batch", None),
    )


def _standalone_module(**overrides) -> EquivariantFragmentInteraction:
    torch.manual_seed(11)
    cfg = {**ACTIVE_FRAGMENT_CONFIG, **overrides}
    return EquivariantFragmentInteraction(irreps_node=HIDDEN_IRREPS, cutoff=CUTOFF, **cfg).double()


# ----------------------------------------------------------------------
# Test 3 — disabled by default is a strict no-op
# ----------------------------------------------------------------------


class TestFragmentInteractionDisabled:
    def test_output_bit_identical_to_plain_arace(self) -> None:
        graph = _labelled_graph(*_methane_positions())
        rnd = _build_round(fragment_interaction_config=None)
        rnd.eval()
        h, edge_index, edge_sh, edge_lengths = _round_inputs(graph, rnd)

        h_pair_ref, e_ref = rnd.artisan_layer(
            h, graph.atomic_numbers, edge_index, edge_sh, edge_lengths
        )
        h_ref = rnd.ace_block(h, h_pair_ref, edge_index)

        h_new, e_atom, gate_scores, aux_loss = rnd(
            h, graph.atomic_numbers, edge_index, edge_sh, edge_lengths
        )

        assert rnd.fragment_interaction is None
        assert torch.equal(h_new, h_ref), "disabled fragment channel changed the features"
        assert torch.equal(e_atom, e_ref), "disabled fragment channel changed the energy"
        assert gate_scores is None
        assert float(aux_loss) == 0.0

    def test_adds_no_parameters(self) -> None:
        plain = _build_round(fragment_interaction_config=None)
        assert not any("fragment_interaction" in name for name in plain.state_dict())

        enabled = _build_round(fragment_interaction_config=FRAGMENT_CONFIG)
        assert sum(p.numel() for p in enabled.parameters()) > sum(
            p.numel() for p in plain.parameters()
        )

    def test_missing_geometry_raises_a_pointed_error(self) -> None:
        graph = _labelled_graph(*_methane_positions())
        rnd = _build_round(fragment_interaction_config=FRAGMENT_CONFIG)
        h, edge_index, edge_sh, edge_lengths = _round_inputs(graph, rnd)
        with pytest.raises(ValueError, match="compute_fragment_index"):
            rnd(h, graph.atomic_numbers, edge_index, edge_sh, edge_lengths)


# ----------------------------------------------------------------------
# Test 4 — a lone fragment has nothing to interact with
# ----------------------------------------------------------------------


class TestSingleFragment:
    def test_correction_is_exactly_zero(self) -> None:
        graph = _labelled_graph(*_methane_positions())
        module = _standalone_module()
        module.eval()
        torch.manual_seed(3)
        h = torch.randn(5, module.irreps_node.dim, dtype=torch.float64)

        delta = module(h, _geometry(module, graph))

        assert delta.shape == h.shape
        # K = 1 → no fragment pairs → no message.  Not "small": zero.
        assert torch.equal(delta, torch.zeros_like(delta))

    def test_every_parameter_still_receives_a_gradient(self) -> None:
        """DDP's static graph breaks if a module contributes no gradient on
        some batches, which is exactly what a single-fragment structure
        would do without the dummy-pair path."""
        graph = _labelled_graph(*_methane_positions())
        module = _standalone_module()
        torch.manual_seed(3)
        h = torch.randn(5, module.irreps_node.dim, dtype=torch.float64)

        module(h, _geometry(module, graph)).sum().backward()

        missing = [n for n, p in module.named_parameters() if p.grad is None]
        assert not missing, f"no gradient reached: {missing}"

    def test_round_is_a_near_no_op_at_zero_init(self) -> None:
        graph = _labelled_graph(*_methane_positions())
        plain = _build_round(fragment_interaction_config=None, seed=7)
        enabled = _build_round(fragment_interaction_config=FRAGMENT_CONFIG, seed=7)
        plain.eval()
        enabled.eval()
        h, edge_index, edge_sh, edge_lengths = _round_inputs(graph, enabled)

        h_plain = plain(h, graph.atomic_numbers, edge_index, edge_sh, edge_lengths)[0]
        h_frag = enabled(
            h,
            graph.atomic_numbers,
            edge_index,
            edge_sh,
            edge_lengths,
            fragment_geometry=_geometry(enabled.fragment_interaction, graph),
        )[0]

        assert torch.allclose(
            h_frag, h_plain, atol=1e-10
        ), f"max deviation {float((h_frag - h_plain).abs().max()):.3e}"


# ----------------------------------------------------------------------
# Test 5 — two fragments: a real, geometry-aware, all-orders signal
# ----------------------------------------------------------------------


class TestTwoFragments:
    def test_correction_is_non_zero(self) -> None:
        graph = _labelled_graph(*_water_dimer())
        module = _standalone_module()
        module.eval()
        torch.manual_seed(5)
        h = torch.randn(6, module.irreps_node.dim, dtype=torch.float64)

        delta = module(h, _geometry(module, graph))
        assert delta.abs().max().item() > 1e-8

    def test_correction_reaches_every_angular_order(self) -> None:
        """The whole point of going equivariant: l=1 and l=2 get corrected
        too, not just the scalars."""
        graph = _labelled_graph(*_water_dimer())
        module = _standalone_module()
        module.eval()
        torch.manual_seed(5)
        h = torch.randn(6, module.irreps_node.dim, dtype=torch.float64)

        delta = module(h, _geometry(module, graph))

        assert delta[:, L0_SLICE].abs().max().item() > 1e-8, "no l=0 correction"
        assert delta[:, L1_SLICE].abs().max().item() > 1e-8, "no l=1 correction"
        assert delta[:, L2_SLICE].abs().max().item() > 1e-8, "no l=2 correction"

    def test_correction_depends_on_inter_fragment_geometry(self) -> None:
        """Moving the second fragment must change the first one's correction
        — this is the information the invariant version could not carry."""
        module = _standalone_module()
        module.eval()
        torch.manual_seed(5)
        h = torch.randn(6, module.irreps_node.dim, dtype=torch.float64)

        near = module(h, _geometry(module, _labelled_graph(*_water_dimer(3.0))))
        far = module(h, _geometry(module, _labelled_graph(*_water_dimer(4.5))))

        assert (near[:3] - far[:3]).abs().max().item() > 1e-8

    def test_beyond_cutoff_contributes_nothing(self) -> None:
        """Locality: a fragment past the pair cutoff is not seen at all, so
        the energy stays reproducible under a neighbour list."""
        module = _standalone_module()
        module.eval()
        torch.manual_seed(5)
        h = torch.randn(6, module.irreps_node.dim, dtype=torch.float64)

        graph = _labelled_graph(*_water_dimer(separation=20.0))
        geometry = _geometry(module, graph)

        assert geometry.n_fragments == 2
        assert geometry.pair_index.shape[1] == 0, "pairs beyond the cutoff survived"
        assert torch.equal(module(h, geometry), torch.zeros_like(h))

    def test_forces_flow_through_the_centroids(self) -> None:
        """The centroids must sit inside the autograd graph, or the fragment
        channel silently contributes nothing to the forces."""
        positions, numbers = _water_dimer()
        positions = positions.clone().requires_grad_(True)
        fragment_index = compute_fragment_index(positions.detach(), numbers)

        module = _standalone_module()
        torch.manual_seed(5)
        h = torch.randn(6, module.irreps_node.dim, dtype=torch.float64)

        geometry = build_fragment_geometry(
            positions=positions,
            fragment_index=fragment_index,
            irreps_sh=module.irreps_sh,
            cutoff=module.cutoff,
        )
        module(h, geometry).sum().backward()

        assert positions.grad is not None
        assert positions.grad.abs().max().item() > 1e-10, (
            "no gradient reached the positions — the fragment channel would "
            "contribute nothing to the forces"
        )

    def test_attention_free_module_is_confined_to_one_structure(self) -> None:
        module = _standalone_module()
        module.eval()
        torch.manual_seed(9)
        h_a = torch.randn(3, module.irreps_node.dim, dtype=torch.float64)
        h_b = torch.randn(3, module.irreps_node.dim, dtype=torch.float64)

        graph_a = _labelled_graph(*_water_positions())
        alone = module(h_a, _geometry(module, graph_a))

        # Graph A batched with an unrelated graph B sitting 4 Å away: without
        # the per-structure mask, B's fragment would be inside the cutoff and
        # would change A's correction.
        batched_graph = Batch.from_data_list(
            [graph_a, _labelled_graph(*_water_positions(offset=4.0))]
        )
        batched = module(torch.cat([h_a, h_b]), _geometry(module, batched_graph))

        assert torch.allclose(
            alone, batched[:3], atol=1e-12
        ), "graph A's correction changed when graph B joined the batch"


# ----------------------------------------------------------------------
# Test 6 — equivariance with the fragment channel enabled
# ----------------------------------------------------------------------


def _build_backbone(**kwargs) -> SimurghAraceBackbone:
    torch.manual_seed(21)
    return SimurghAraceBackbone(
        elements=(1, 6, 7, 8),
        num_rounds=2,
        share_artisan_weights=False,
        artisan=dict(ARTISAN_SUBCONFIG),
        cutoff=CUTOFF,
        embedding_dim=16,
        num_elements=9,
        **kwargs,
    ).double()


def _run(backbone: SimurghAraceBackbone, batch: Batch) -> dict[str, torch.Tensor]:
    head = EnergyForcesHead(irreps_in=HIDDEN_IRREPS, hidden_dim=16).double()
    head.eval()
    return head(backbone(batch), batch)


class TestFragmentInteractionEquivariance:
    def test_module_output_rotates_with_the_input(self) -> None:
        """Direct check on the module: rotating positions *and* features must
        rotate the correction, block by block.

        Tolerance note — the reference itself is not exact.  e3nn 0.6.0's
        Wigner-D matrices are orthogonal to ~5e-15 but compose to only
        ~7e-7 (``D(R1 R2) != D(R1) D(R2)``), and its spherical harmonics
        satisfy ``Y(Rr) = D(R) Y(r)`` only to ~6e-7 in float64, because the
        ``Jd`` rotation constants are stored at reduced precision.  That
        floor is measured in the test below and used as the tolerance, so
        this asserts "as equivariant as e3nn can express" rather than a
        number plucked from the air.
        """
        from e3nn.o3 import Irreps, spherical_harmonics

        module = _standalone_module()
        module.eval()
        irreps = Irreps(HIDDEN_IRREPS)
        positions, numbers = _water_dimer()
        torch.manual_seed(5)
        h = torch.randn(6, irreps.dim, dtype=torch.float64)

        rotation = random_so3()
        wigner = irreps.D_from_matrix(rotation).to(torch.float64)

        # e3nn's own SH equivariance error, on this very rotation.
        probe = torch.randn(8, 3, dtype=torch.float64)
        sh_irreps = Irreps("1x0e + 1x1o + 1x2e")
        sh_floor = (
            (
                spherical_harmonics(
                    sh_irreps, probe @ rotation.T, normalize=True, normalization="component"
                )
                - spherical_harmonics(sh_irreps, probe, normalize=True, normalization="component")
                @ sh_irreps.D_from_matrix(rotation).double().T
            )
            .abs()
            .max()
            .item()
        )

        delta = module(h, _geometry(module, _labelled_graph(positions, numbers)))
        delta_rot = module(
            h @ wigner.T,
            _geometry(module, _labelled_graph(positions @ rotation.T, numbers)),
        )

        diff = (delta_rot - delta @ wigner.T).abs().max().item()
        assert diff <= max(sh_floor, 1e-9), (
            f"fragment correction deviates by {diff:.3e}, worse than e3nn's own "
            f"SH/Wigner-D floor of {sh_floor:.3e} — that is a real symmetry break"
        )

    def test_energy_invariant_forces_equivariant(self) -> None:
        positions, numbers = _water_dimer()
        backbone = _build_backbone(fragment_interaction=ACTIVE_FRAGMENT_CONFIG)
        backbone.eval()
        assert backbone.fragment_interaction_enabled

        out0 = _run(backbone, Batch.from_data_list([_labelled_graph(positions, numbers)]))
        rotation = random_so3()
        out1 = _run(
            backbone,
            Batch.from_data_list([_labelled_graph(positions @ rotation.T, numbers)]),
        )

        e_diff = (out0["energy"] - out1["energy"]).abs().max().item()
        assert e_diff < 1e-5, f"Energy not invariant: {e_diff:.3e}"

        f_diff = (
            (out1["forces"].detach() - out0["forces"].detach() @ rotation.T).abs().max().item()
        )
        assert f_diff < 1e-4, f"Forces not equivariant: {f_diff:.3e}"

    def test_translation_invariance(self) -> None:
        """Centroids are absolute positions, so a shifted copy must give the
        same energy — the module may only ever use *differences*."""
        positions, numbers = _water_dimer()
        backbone = _build_backbone(fragment_interaction=ACTIVE_FRAGMENT_CONFIG)
        backbone.eval()

        shift = torch.tensor([3.7, -1.2, 0.8], dtype=torch.float64)
        out0 = _run(backbone, Batch.from_data_list([_labelled_graph(positions, numbers)]))
        out1 = _run(backbone, Batch.from_data_list([_labelled_graph(positions + shift, numbers)]))

        e_diff = (out0["energy"] - out1["energy"]).abs().max().item()
        assert e_diff < 1e-8, f"Energy not translation invariant: {e_diff:.3e}"

    def test_fragment_labels_are_rotation_invariant(self) -> None:
        positions, numbers = _water_dimer()
        rotation = random_so3()
        assert torch.equal(
            compute_fragment_index(positions, numbers),
            compute_fragment_index(positions @ rotation.T, numbers),
        )

    def test_forces_are_finite_and_newton_holds(self) -> None:
        graph = _labelled_graph(*_water_dimer())
        backbone = _build_backbone(fragment_interaction=ACTIVE_FRAGMENT_CONFIG)
        backbone.eval()

        out = _run(backbone, Batch.from_data_list([graph]))
        assert torch.isfinite(out["energy"]).all()
        assert torch.isfinite(out["forces"]).all()
        net = out["forces"].sum(dim=0).norm().item()
        assert net < 1e-5, f"Newton violated: ‖Σ F‖ = {net:.3e}"

    def test_backbone_raises_without_fragment_labels(self) -> None:
        graph = _build_graph(*_water_dimer(), cutoff=CUTOFF)  # no labels attached
        backbone = _build_backbone(fragment_interaction=FRAGMENT_CONFIG)
        backbone.eval()
        with pytest.raises(ValueError, match="compute_fragment_index"):
            _run(backbone, Batch.from_data_list([graph]))
