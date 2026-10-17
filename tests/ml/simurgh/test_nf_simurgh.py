"""NF_SIMURGH — fragment virtual nodes on top of the ARACE rounds.

Coverage:

* Test 1 — exact conservation: ``Σ_k E_base[k] == Σ_i e_atom[i]`` to the
  last bit, under every flag combination, before the correction is added.
* Test 2 — zero-init reduces to a **separately instantiated**
  ``MonolithicArace``, with the virtual-node path fully active throughout.
* Test 3 — a single fragment (K=1) is well defined forwards *and*
  backwards, and every virtual-node parameter still receives a gradient.
* Test 4 — the gate: the interaction is non-trivial when fragments are
  close and decays to nothing past ``vnode_cutoff``.
* Test 5 — cross-graph isolation: a batch neighbour cannot change a
  fragment's correction.
* Test 6 — symmetry at large magnitudes, per invariant, with Newton.
* Test 7 — every pooling mode and staging flag is actually wired.
* Test 8 — smoothness and agreement with finite-difference forces.
"""

from __future__ import annotations

import math
import typing

import pytest
import torch
from e3nn.o3 import Irreps
from torch_geometric.data import Batch

from goal.ml.data.fragments import compute_fragment_index
from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.models.simurgh.arace import MonolithicArace
from goal.ml.nn.models.simurgh.nf_simurgh import (
    MonolithicNFSimurgh,
    NFSimurghBackbone,
    NFSimurghOutputs,
    NFSimurghVirtualNodeInteraction,
    _vnode_pair_geometry,
)
from tests.ml.simurgh.conftest import _build_graph, random_so3

CUTOFF = 8.0
VNODE_CUTOFF = 15.0
HIDDEN_IRREPS = "8x0e + 8x1o + 8x2e"

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

VIRTUAL_NODE: dict = {
    "pooling": "mean",
    "use_size_scalar": True,
    "n_scalar": 8,
    "vnode_num_rbf": 4,
    "vnode_cutoff": VNODE_CUTOFF,
    "vnode_radial_hidden": 16,
    "n_heads": 2,
    "final_hidden": 8,
    "dropout": 0.0,
    "pool_l1l2": False,
    "l1l2_invariants": ["norm", "axial", "mutual", "quad"],
    "vnode_equivariant": False,
    "vnode_irreps": HIDDEN_IRREPS,
    "vnode_sh_lmax": 2,
    "vnode_type_conditioned": False,
}

BACKBONE_SHARED: dict = {
    "elements": (1, 6, 7, 8),
    "num_rounds": 2,
    "share_artisan_weights": False,
    "cutoff": CUTOFF,
    "embedding_dim": 16,
    "num_elements": 9,
    "avg_num_neighbors": 4.0,
}

#: Both stages are exercised everywhere the flag could plausibly matter.
STAGES: list[bool] = [False, True]


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
    return positions, torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)


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
    return positions, torch.tensor([8, 1, 1], dtype=torch.long)


def _water_dimer(separation: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Two waters *separation* Å apart — one graph, two fragments."""
    pos_a, z_a = _water_positions(0.0)
    pos_b, z_b = _water_positions(separation)
    return torch.cat([pos_a, pos_b]), torch.cat([z_a, z_b])


def _methane_and_water(separation: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
    pos_a, z_a = _methane_positions()
    pos_b, z_b = _water_positions(separation)
    return torch.cat([pos_a, pos_b]), torch.cat([z_a, z_b])


def _labelled_graph(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    cutoff: float = CUTOFF,
) -> AtomicGraph:
    graph: AtomicGraph = _build_graph(positions, numbers, cutoff=cutoff)
    graph.fragment_index = compute_fragment_index(positions, numbers, covalent_cutoff=1.8)
    return graph


def _batch(*graphs: AtomicGraph) -> Batch:
    return Batch.from_data_list(list(graphs))


# ----------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------


def _virtual_node(**overrides: typing.Any) -> dict:
    return {**VIRTUAL_NODE, **overrides}


def _build_model(seed: int = 0, **vnode_overrides: typing.Any) -> MonolithicNFSimurgh:
    torch.manual_seed(seed)
    return MonolithicNFSimurgh(
        artisan=dict(ARTISAN_SUBCONFIG),
        virtual_node=_virtual_node(**vnode_overrides),
        **BACKBONE_SHARED,
    ).double()


def _build_arace(seed: int = 0) -> MonolithicArace:
    """A ``MonolithicArace`` built from the same seed and the same config.

    NF_SIMURGH constructs its ARACE stack first, before any virtual-node
    module, so under the same seed both models draw identical random
    numbers for every shared parameter.
    """
    torch.manual_seed(seed)
    return MonolithicArace(artisan=dict(ARTISAN_SUBCONFIG), **BACKBONE_SHARED).double()


def _build_backbone(seed: int = 0, **vnode_overrides: typing.Any) -> NFSimurghBackbone:
    torch.manual_seed(seed)
    return NFSimurghBackbone(
        artisan=dict(ARTISAN_SUBCONFIG),
        virtual_node=_virtual_node(**vnode_overrides),
        **BACKBONE_SHARED,
    ).double()


def _run_backbone(backbone: NFSimurghBackbone, batch: Batch) -> NFSimurghOutputs:
    from goal.ml.nn.models.simurgh.geometry import differentiable_edges

    positions: torch.Tensor = batch.pos.clone().requires_grad_(True)
    edge_vectors, edge_lengths = differentiable_edges(batch, positions)
    return backbone(
        atomic_numbers=batch.atomic_numbers,
        edge_index=batch.edge_index,
        edge_vectors=edge_vectors,
        edge_lengths=edge_lengths,
        positions=positions,
        fragment_index=batch.fragment_index,
        batch=batch.get("batch", None),
        cell=batch.get("cell", None),
        pbc=batch.get("pbc", None),
    )


def _activate(model: MonolithicNFSimurgh, seed: int = 99, gain: float = 0.5) -> None:
    """Un-zero the interaction readouts.

    At initialisation the correction is exactly zero by design, which is
    the right default but gives no signal to measure.  Tests that need the
    fragment channel to actually do something perturb the readout weights
    — never a flag, because there is no flag.
    """
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        modules = [model.backbone.vnode_interaction]
        if model.backbone.vnode_equivariant_layer is not None:
            modules.append(model.backbone.vnode_equivariant_layer)
        for module in modules:
            final = typing.cast(torch.nn.Linear, module.energy_mlp[-1])
            final.weight.copy_(
                torch.randn(final.weight.shape, generator=generator, dtype=torch.float64) * gain
            )


def _sh_irreps(backbone: NFSimurghBackbone) -> Irreps:
    return backbone.irreps_sh


# ======================================================================
# Test 1 — exact conservation identity
# ======================================================================


class TestConservationIdentity:
    """``Σ_k E_base[k] == Σ_i e_atom[i]``, exactly, before the correction.

    This is the property that makes the fragment pooling safe: the
    atom-level energy is preserved *by construction* (summation is
    associative), not because a residual happens to stay small.
    """

    @pytest.mark.parametrize("equivariant", STAGES)
    @pytest.mark.parametrize("pool_l1l2", [False, True])
    def test_multi_fragment(self, equivariant: bool, pool_l1l2: bool) -> None:
        backbone = _build_backbone(vnode_equivariant=equivariant, pool_l1l2=pool_l1l2)
        backbone.eval()
        outputs = _run_backbone(backbone, _batch(_labelled_graph(*_water_dimer())))

        assert outputs.n_fragments == 2
        total_atoms = outputs.node_energies.sum()
        total_fragments = outputs.vnode_base.sum()
        assert torch.allclose(
            total_atoms, total_fragments, rtol=0.0, atol=1e-14
        ), f"conservation broken by {float((total_atoms - total_fragments).abs()):.3e}"

    @pytest.mark.parametrize("equivariant", STAGES)
    def test_single_fragment(self, equivariant: bool) -> None:
        backbone = _build_backbone(vnode_equivariant=equivariant)
        backbone.eval()
        outputs = _run_backbone(backbone, _batch(_labelled_graph(*_methane_positions())))

        assert outputs.n_fragments == 1
        assert torch.allclose(
            outputs.node_energies.sum(), outputs.vnode_base.sum(), rtol=0.0, atol=1e-14
        )

    def test_holds_per_structure_across_a_batch(self) -> None:
        backbone = _build_backbone()
        backbone.eval()
        batch = _batch(
            _labelled_graph(*_water_dimer()),
            _labelled_graph(*_methane_positions()),
            _labelled_graph(*_methane_and_water()),
        )
        outputs = _run_backbone(backbone, batch)

        for graph_id in range(3):
            atoms = outputs.node_energies[batch.batch == graph_id].sum()
            fragments = outputs.vnode_base[outputs.fragment_graph == graph_id].sum()
            assert torch.allclose(
                atoms, fragments, rtol=0.0, atol=1e-14
            ), f"structure {graph_id} does not conserve"


# ======================================================================
# Test 2 — zero-init reduces to plain ARACE
# ======================================================================


class TestReducesToAraceAtInit:
    """A fresh NF_SIMURGH *is* ``MonolithicArace``, numerically.

    The virtual-node path stays fully active throughout: this passes
    because the zero-initialised readout makes the correction exactly
    zero, not because anything was switched off.  There is no switch.
    """

    @pytest.mark.parametrize("equivariant", STAGES)
    def test_energy_and_forces_match(self, equivariant: bool) -> None:
        nf = _build_model(seed=5, vnode_equivariant=equivariant)
        arace = _build_arace(seed=5)
        nf.eval()
        arace.eval()

        positions, numbers = _water_dimer()
        nf_out = nf(_batch(_labelled_graph(positions, numbers)))
        arace_out = arace(_batch(_labelled_graph(positions, numbers)))

        energy_gap = (nf_out["energy"] - arace_out["energy"]).abs().max().item()
        assert energy_gap < 1e-12, f"energy differs by {energy_gap:.3e}"

        force_gap = (nf_out["forces"] - arace_out["forces"]).abs().max().item()
        assert force_gap < 1e-12, f"forces differ by {force_gap:.3e}"

    @pytest.mark.parametrize("equivariant", STAGES)
    def test_correction_is_exactly_zero(self, equivariant: bool) -> None:
        nf = _build_model(seed=5, vnode_equivariant=equivariant)
        nf.eval()
        out = nf(_batch(_labelled_graph(*_water_dimer())))

        assert torch.equal(
            out["vnode_interaction_energies"],
            torch.zeros_like(out["vnode_interaction_energies"]),
        )
        assert float(out["vnode_interaction_abs_mean"]) == 0.0

    def test_the_virtual_node_path_really_ran(self) -> None:
        """Guard against passing for the wrong reason.

        If some future change made the fragment machinery a no-op, the
        test above would still pass.  Perturbing the readout must break
        the agreement — that is what proves the path was live.
        """
        nf = _build_model(seed=5)
        arace = _build_arace(seed=5)
        nf.eval()
        arace.eval()
        _activate(nf)

        positions, numbers = _water_dimer()
        nf_out = nf(_batch(_labelled_graph(positions, numbers)))
        arace_out = arace(_batch(_labelled_graph(positions, numbers)))

        assert (nf_out["energy"] - arace_out["energy"]).abs().max().item() > 1e-8

    def test_missing_fragment_index_raises(self) -> None:
        """No fallback path: silent degradation to atom-level-only would
        produce plausible numbers from the wrong model."""
        nf = _build_model()
        nf.eval()
        graph = _build_graph(*_water_dimer(), cutoff=CUTOFF)  # unlabelled
        with pytest.raises(ValueError, match="compute_fragment_index"):
            nf(_batch(graph))

    @pytest.mark.parametrize(
        "key", ["use_fragments", "enable_virtual_nodes", "use_fragment_ca", "init_zero"]
    )
    def test_disabling_keys_are_rejected(self, key: str) -> None:
        """The mechanism is mandatory and the zero-init is an invariant."""
        with pytest.raises(ValueError, match=key):
            MonolithicNFSimurgh(
                artisan=dict(ARTISAN_SUBCONFIG),
                virtual_node=_virtual_node(**{key: True}),
                **BACKBONE_SHARED,
            )


# ======================================================================
# Test 3 — single fragment (K = 1)
# ======================================================================


class TestSingleFragment:
    """K = 1 is well defined in both directions.

    Attention over one item attends only to itself and the self-distance
    is zero, so this is where an unguarded ``1/d`` shows up — and it shows
    up in the *backward* pass, not the forward one.
    """

    @pytest.mark.parametrize("equivariant", STAGES)
    def test_forward_is_finite(self, equivariant: bool) -> None:
        nf = _build_model(vnode_equivariant=equivariant)
        nf.eval()
        _activate(nf)
        out = nf(_batch(_labelled_graph(*_methane_positions())))

        assert torch.isfinite(out["energy"]).all()
        assert torch.isfinite(out["forces"]).all()
        assert torch.isfinite(out["vnode_interaction_energies"]).all()
        # Nothing to interact with, so the correction is zero — not "small".
        assert torch.equal(
            out["vnode_interaction_energies"],
            torch.zeros_like(out["vnode_interaction_energies"]),
        )

    @pytest.mark.parametrize("equivariant", STAGES)
    def test_backward_is_finite(self, equivariant: bool) -> None:
        nf = _build_model(vnode_equivariant=equivariant)
        _activate(nf)
        out = nf(_batch(_labelled_graph(*_methane_positions())))
        (out["energy"].sum() + out["forces"].sum()).backward()

        bad = [
            name
            for name, param in nf.named_parameters()
            if param.grad is not None and not torch.isfinite(param.grad).all()
        ]
        assert not bad, f"non-finite gradients: {bad}"

    @pytest.mark.parametrize("equivariant", STAGES)
    def test_every_virtual_node_parameter_gets_a_gradient(self, equivariant: bool) -> None:
        """DDP requirement, not an optimisation.

        GOAL runs DDP with ``find_unused_parameters: false``, which needs
        every parameter to participate in every step.  A K=1 batch
        produces no fragment pairs, so without the zero-masked path the
        virtual-node parameters would go untouched and the static graph
        would break.
        """
        nf = _build_model(vnode_equivariant=equivariant)
        _activate(nf)
        out = nf(_batch(_labelled_graph(*_methane_positions())))
        out["energy"].sum().backward()

        missing = [
            name
            for name, param in nf.backbone.named_parameters()
            if name.startswith("vnode_") and param.grad is None
        ]
        assert not missing, f"no gradient reached: {missing}"

    def test_mixed_batch_with_a_lone_fragment(self) -> None:
        """A K=1 structure batched with a K=2 one must not disturb it."""
        nf = _build_model()
        nf.eval()
        _activate(nf)
        out = nf(
            _batch(
                _labelled_graph(*_methane_positions()),
                _labelled_graph(*_water_dimer()),
            )
        )
        corrections = out["vnode_interaction_energies"]

        assert corrections.shape == (3,)
        assert float(corrections[0]) == 0.0, "the lone fragment gained a correction"
        assert corrections[1:].abs().max().item() > 1e-10


# ======================================================================
# Test 4 — two fragments, distance sensitivity  (THE GATE)
# ======================================================================


class TestDistanceSensitivity:
    """The reason the model exists.

    Fragment-fragment interaction must be modulated by the actual
    separation.  Attention over chemical summaries alone would give a
    fragment the same pull at 4 Å and at 40 Å, which is the exact gap
    this model was built to close.
    """

    def _magnitudes(self, nf: MonolithicNFSimurgh, separations: list[float]) -> list[float]:
        values: list[float] = []
        for separation in separations:
            out = nf(_batch(_labelled_graph(*_water_dimer(separation))))
            values.append(out["vnode_interaction_energies"].abs().max().item())
        return values

    def test_interaction_decays_with_separation(self) -> None:
        nf = _build_model(seed=3)
        nf.eval()
        _activate(nf)

        separations = [4.0, 7.0, 10.0, 13.0, 20.0]
        magnitudes = self._magnitudes(nf, separations)

        assert magnitudes[0] > 1e-8, (
            "no interaction between fragments 4 Å apart — the correction is "
            "not seeing the other fragment at all"
        )
        assert magnitudes[-1] == 0.0, (
            f"fragments {separations[-1]} Å apart still interact "
            f"({magnitudes[-1]:.3e}) although vnode_cutoff={VNODE_CUTOFF}"
        )
        assert magnitudes[-2] < magnitudes[0], f"interaction did not decay: {magnitudes}"

    def test_interaction_changes_when_the_partner_moves(self) -> None:
        """Two separations well inside the cutoff must give different
        corrections — the distance has to reach the energy, not just the
        on/off decision at the boundary."""
        nf = _build_model(seed=3)
        nf.eval()
        _activate(nf)

        near = nf(_batch(_labelled_graph(*_water_dimer(4.0))))
        mid = nf(_batch(_labelled_graph(*_water_dimer(6.0))))

        gap = (
            (near["vnode_interaction_energies"] - mid["vnode_interaction_energies"])
            .abs()
            .max()
            .item()
        )
        assert gap > 1e-9, f"correction is distance-blind inside the cutoff ({gap:.3e})"

    def test_forces_flow_through_the_centroids(self) -> None:
        """The centroids must sit inside the autograd graph, or the
        fragment channel silently contributes nothing to the forces."""
        nf_zero = _build_model(seed=3)
        nf_live = _build_model(seed=3)
        nf_zero.eval()
        nf_live.eval()
        _activate(nf_live)

        positions, numbers = _water_dimer(4.0)
        forces_zero = nf_zero(_batch(_labelled_graph(positions, numbers)))["forces"]
        forces_live = nf_live(_batch(_labelled_graph(positions, numbers)))["forces"]

        assert (
            forces_live - forces_zero
        ).abs().max().item() > 1e-10, "the correction changed the energy but not the forces"


# ======================================================================
# Test 5 — cross-graph isolation
# ======================================================================


class TestCrossGraphIsolation:
    """A batch neighbour must be invisible.

    The masking comes from v1's ``build_fragment_geometry``; this checks
    it is actually reaching NF_SIMURGH's pair enumeration.  Without it, a
    molecule 4 Å away in an unrelated structure would sit well inside
    ``vnode_cutoff`` and would change this structure's energy — making
    the prediction depend on how the dataloader happened to batch.
    """

    @pytest.mark.parametrize("equivariant", STAGES)
    def test_correction_unchanged_by_a_batch_neighbour(self, equivariant: bool) -> None:
        nf = _build_model(seed=7, vnode_equivariant=equivariant)
        nf.eval()
        _activate(nf)

        graph_a = _labelled_graph(*_water_dimer(4.0))
        alone = nf(_batch(graph_a))["vnode_interaction_energies"]

        # Graph B sits 4 Å from graph A in raw coordinates: without the
        # per-structure mask its fragments would be inside vnode_cutoff.
        graph_b = _labelled_graph(*_water_dimer(4.0))
        batched = nf(_batch(graph_a, graph_b))["vnode_interaction_energies"]

        assert alone.shape == (2,)
        assert batched.shape == (4,)
        assert torch.allclose(alone, batched[:2], rtol=0.0, atol=1e-12), (
            f"graph A's correction moved by "
            f"{float((alone - batched[:2]).abs().max()):.3e} when B joined the batch"
        )

    def test_energy_unchanged_by_a_batch_neighbour(self) -> None:
        nf = _build_model(seed=7)
        nf.eval()
        _activate(nf)

        graph_a = _labelled_graph(*_water_dimer(4.0))
        alone = nf(_batch(graph_a))["energy"]
        batched = nf(_batch(graph_a, _labelled_graph(*_methane_positions())))["energy"]

        assert torch.allclose(alone, batched[:1], rtol=0.0, atol=1e-12)


# ======================================================================
# Test 6 — symmetry, at large magnitudes
# ======================================================================


def _rotate_about(
    positions: torch.Tensor,
    rotation: torch.Tensor,
    pivot: torch.Tensor,
) -> torch.Tensor:
    """Rotate about *pivot* rather than the origin.

    Rotating about the centroid of everything hides absolute-position
    bugs — the centroid maps to itself, so a module that (wrongly) used
    absolute centroids would still look invariant.  A distant pivot does
    not let that slide.
    """
    return (positions - pivot) @ rotation.T + pivot


def _sh_equivariance_floor(rotation: torch.Tensor) -> float:
    """e3nn's own SH/Wigner-D error on this rotation, in float64.

    e3nn 0.6.0 stores its ``Jd`` rotation constants at reduced precision,
    so ``Y(Rr) = D(R) Y(r)`` holds only to ~1e-6 even in float64.  Tests
    below assert "as equivariant as e3nn can express" rather than a
    tolerance plucked from the air.
    """
    from e3nn.o3 import spherical_harmonics

    probe = torch.randn(16, 3, dtype=torch.float64)
    irreps = Irreps("1x0e + 1x1o + 1x2e")
    left = spherical_harmonics(
        irreps, probe @ rotation.T, normalize=True, normalization="component"
    )
    right = (
        spherical_harmonics(irreps, probe, normalize=True, normalization="component")
        @ irreps.D_from_matrix(rotation).double().T
    )
    return (left - right).abs().max().item()


class TestSymmetry:
    @pytest.mark.parametrize("equivariant", STAGES)
    def test_rotation_about_a_distant_point(self, equivariant: bool) -> None:
        nf = _build_model(seed=11, vnode_equivariant=equivariant)
        nf.eval()
        _activate(nf)

        positions, numbers = _water_dimer(4.0)
        rotation = random_so3()
        pivot = torch.tensor([37.0, -21.0, 58.0], dtype=torch.float64)

        base = nf(_batch(_labelled_graph(positions, numbers)))
        rotated = nf(_batch(_labelled_graph(_rotate_about(positions, rotation, pivot), numbers)))

        energy_gap = (base["energy"] - rotated["energy"]).abs().max().item()
        assert energy_gap < 1e-6, f"energy not invariant: {energy_gap:.3e}"

        force_gap = (
            (rotated["forces"].detach() - base["forces"].detach() @ rotation.T).abs().max().item()
        )
        assert force_gap < 1e-4, f"forces not equivariant: {force_gap:.3e}"

    @pytest.mark.parametrize("equivariant", STAGES)
    def test_translation_by_a_hundred_angstrom(self, equivariant: bool) -> None:
        """The likelier bug in practice: an absolute centroid leaking in.

        A small shift would be swamped by float error; 100 Å is not.
        """
        nf = _build_model(seed=11, vnode_equivariant=equivariant)
        nf.eval()
        _activate(nf)

        positions, numbers = _water_dimer(4.0)
        shift = torch.tensor([-63.0, 88.0, 41.0], dtype=torch.float64)

        base = nf(_batch(_labelled_graph(positions, numbers)))
        shifted = nf(_batch(_labelled_graph(positions + shift, numbers)))

        energy_gap = (base["energy"] - shifted["energy"]).abs().max().item()
        assert energy_gap < 1e-6, f"energy not translation invariant: {energy_gap:.3e}"

        correction_gap = (
            (base["vnode_interaction_energies"] - shifted["vnode_interaction_energies"])
            .abs()
            .max()
            .item()
        )
        assert correction_gap < 1e-6, (
            f"the interaction term itself moved by {correction_gap:.3e} under a "
            f"pure translation — it is using an absolute position"
        )

    @pytest.mark.parametrize("invariant", ["norm", "axial", "mutual", "quad"])
    def test_each_pooled_invariant_is_rotation_invariant(self, invariant: str) -> None:
        """Per-quantity, so a violation is localised rather than merely detected."""
        torch.manual_seed(4)
        module = NFSimurghVirtualNodeInteraction(
            irreps_node=HIDDEN_IRREPS,
            vnode_cutoff=VNODE_CUTOFF,
            n_scalar=8,
            vnode_num_rbf=4,
            vnode_radial_hidden=16,
            n_heads=2,
            final_hidden=8,
            pool_l1l2=True,
            l1l2_invariants=[invariant],
        ).double()
        module.eval()

        irreps = Irreps(HIDDEN_IRREPS)
        positions, numbers = _water_dimer(4.0)
        fragment_index = compute_fragment_index(positions, numbers)
        rotation = random_so3()
        pivot = torch.tensor([44.0, 17.0, -62.0], dtype=torch.float64)
        wigner = irreps.D_from_matrix(rotation).double()

        torch.manual_seed(6)
        h = torch.randn(positions.shape[0], irreps.dim, dtype=torch.float64)

        def geometry(pos: torch.Tensor):
            return _vnode_pair_geometry(
                positions=pos,
                fragment_index=fragment_index,
                irreps_sh=Irreps("1x0e + 1x1o + 1x2e"),
                batch=None,
                cell=None,
                pbc=None,
                max_fragments=512,
            )

        base = module._pair_invariants(h, geometry(positions))
        rotated = module._pair_invariants(
            h @ wigner.T, geometry(_rotate_about(positions, rotation, pivot))
        )

        assert base.numel() > 0, f"'{invariant}' produced no columns"
        assert base.abs().max().item() > 1e-8, f"'{invariant}' is identically zero"
        deviation = (base - rotated).abs().max().item()
        floor = max(_sh_equivariance_floor(rotation), 1e-9)
        assert deviation <= max(floor, 1e-6), (
            f"'{invariant}' moved by {deviation:.3e} under rotation "
            f"(e3nn's own floor is {floor:.3e}) — that is a real symmetry break"
        )

    @pytest.mark.parametrize("equivariant", STAGES)
    def test_newtons_third_law(self, equivariant: bool) -> None:
        """``‖Σ F‖ = 0`` on a multi-fragment structure.

        Methane plus water: two fragments, so this exercises translation
        invariance of the *interaction* term, not only of the pre-existing
        ARACE terms.
        """
        nf = _build_model(seed=13, vnode_equivariant=equivariant)
        nf.eval()
        _activate(nf)

        out = nf(_batch(_labelled_graph(*_methane_and_water(4.0))))
        net = out["forces"].detach().sum(dim=0).norm().item()

        assert torch.isfinite(out["forces"]).all()
        assert net < 1e-5, f"Newton violated: ‖Σ F‖ = {net:.3e}"


# ======================================================================
# Test 7 — pooling and staging flags are wired
# ======================================================================


class TestFlagsAreWired:
    """Every switch must change the arithmetic.

    A config key that quietly does nothing is worse than no key at all —
    an ablation would report "no effect" for a variant that never ran.
    """

    @pytest.mark.parametrize("pooling", ["mean", "sum", "attention"])
    def test_every_pooling_mode_runs(self, pooling: str) -> None:
        nf = _build_model(seed=17, pooling=pooling)
        nf.eval()
        _activate(nf)
        out = nf(_batch(_labelled_graph(*_water_dimer())))

        assert torch.isfinite(out["energy"]).all()
        assert torch.isfinite(out["forces"]).all()
        assert out["vnode_interaction_energies"].abs().max().item() > 1e-10

    def test_pooling_modes_give_distinct_energies(self) -> None:
        energies: dict[str, float] = {}
        for pooling in ("mean", "sum", "attention"):
            nf = _build_model(seed=17, pooling=pooling)
            nf.eval()
            _activate(nf)
            batch = _batch(_labelled_graph(*_water_dimer()))
            energies[pooling] = float(nf(batch)["energy"][0])

        assert (
            len(set(energies.values())) == 3
        ), f"pooling mode did not change the result: {energies}"

    @pytest.mark.parametrize("flag", ["pool_l1l2", "vnode_equivariant"])
    def test_staging_flag_changes_the_energy(self, flag: str) -> None:
        off = _build_model(seed=19, **{flag: False})
        on = _build_model(seed=19, **{flag: True})
        off.eval()
        on.eval()
        _activate(off)
        _activate(on)

        batch_off = _batch(_labelled_graph(*_water_dimer()))
        batch_on = _batch(_labelled_graph(*_water_dimer()))
        gap = (off(batch_off)["energy"] - on(batch_on)["energy"]).abs().max().item()

        assert gap > 1e-10, f"'{flag}' is not wired — energies identical ({gap:.3e})"

    def test_size_scalar_changes_the_energy(self) -> None:
        with_size = _build_model(seed=23, use_size_scalar=True)
        without = _build_model(seed=23, use_size_scalar=False)
        with_size.eval()
        without.eval()
        _activate(with_size)
        _activate(without)

        gap = (
            (
                with_size(_batch(_labelled_graph(*_water_dimer())))["energy"]
                - without(_batch(_labelled_graph(*_water_dimer())))["energy"]
            )
            .abs()
            .max()
            .item()
        )
        assert gap > 1e-10

    def test_type_conditioning_runs(self) -> None:
        nf = _build_model(seed=29, vnode_type_conditioned=True)
        nf.eval()
        _activate(nf)
        out = nf(_batch(_labelled_graph(*_methane_and_water())))
        assert torch.isfinite(out["energy"]).all()
        assert out["vnode_interaction_energies"].abs().max().item() > 1e-10

    def test_declared_output_keys_are_present(self) -> None:
        nf = _build_model()
        nf.eval()
        out = nf(_batch(_labelled_graph(*_water_dimer())))
        for key in nf.output_keys:
            assert key in out, f"declared key '{key}' missing from forward()"
        # The collapsed-round diagnostic must keep working.
        assert out["layer_energies"].shape == (BACKBONE_SHARED["num_rounds"], 1)


# ======================================================================
# Test 8 — smoothness and finite-difference forces
# ======================================================================


class TestSmoothness:
    """Continuity of ``E`` and agreement of ``F`` with central differences.

    Cheap insurance, and the test that would catch a discontinuous
    routing scheme: a top-k router over fragments flips which expert is
    active on a small displacement, making ``E`` jump and therefore
    ``F = -∂E/∂x`` wrong near the switching boundary.  Rotation tests
    would not notice; this one would.
    """

    def _energy(self, nf: MonolithicNFSimurgh, positions: torch.Tensor, numbers) -> float:
        return float(nf(_batch(_labelled_graph(positions, numbers)))["energy"][0])

    def test_energy_is_continuous_along_a_path(self) -> None:
        nf = _build_model(seed=31)
        nf.eval()
        _activate(nf)

        positions, numbers = _water_dimer(4.0)
        direction = torch.zeros_like(positions)
        direction[3:] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        steps = torch.linspace(0.0, 1.0, 41, dtype=torch.float64)
        energies = [self._energy(nf, positions + float(t) * direction, numbers) for t in steps]
        jumps = [abs(b - a) for a, b in zip(energies, energies[1:])]

        assert all(math.isfinite(e) for e in energies)
        # Neighbouring steps are 0.025 Å apart; a genuine discontinuity
        # shows up as one jump far larger than its neighbours.
        assert max(jumps) < 20.0 * (sum(jumps) / len(jumps) + 1e-12), (
            f"energy jumped along the path: max step {max(jumps):.3e} vs mean "
            f"{sum(jumps) / len(jumps):.3e}"
        )

    def test_autograd_forces_match_central_differences(self) -> None:
        nf = _build_model(seed=31)
        nf.eval()
        _activate(nf)

        # Methane + water rather than the water dimer: the dimer is planar,
        # so every z-force vanishes by mirror symmetry and a probe along z
        # would compare zero against zero.  Methane's hydrogens are
        # genuinely three-dimensional.
        positions, numbers = _methane_and_water(4.0)
        analytic = nf(_batch(_labelled_graph(positions, numbers)))["forces"].detach()

        delta = 1e-5
        probes = [(0, 0), (1, 1), (2, 2), (3, 0), (5, 0), (6, 1), (7, 0)]
        for atom, axis in probes:
            reference = float(analytic[atom, axis])
            assert abs(reference) > 1e-5, (
                f"probe force[{atom}, {axis}] = {reference:.3e} is too small to "
                f"test anything — pick a different degree of freedom"
            )

            forward = positions.clone()
            forward[atom, axis] += delta
            backward = positions.clone()
            backward[atom, axis] -= delta
            numeric = -(
                self._energy(nf, forward, numbers) - self._energy(nf, backward, numbers)
            ) / (2.0 * delta)

            assert numeric == pytest.approx(reference, rel=1e-4, abs=1e-9), (
                f"force[{atom}, {axis}]: autograd {reference:.10f} vs finite "
                f"difference {numeric:.10f}"
            )
