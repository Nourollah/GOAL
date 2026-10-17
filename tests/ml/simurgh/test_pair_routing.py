"""Frequency routing of element pairs in the ARACE artisan bank.

The artisan bank is quadratic in the element count — 83 elements is
3,486 pairs — and because of the zero-masking schedule every artisan
costs both parameters *and* a full-edge forward pass on every step.
Routing keeps a dedicated artisan for the pairs that carry the training
set and folds the rest into one shared
:class:`~goal.ml.nn.blocks.artisans._AraceRarePairArtisan`.

What these tests pin down:

* the split itself (:func:`route_element_pairs`) and its edge cases;
* routing **off** leaves the model bit-identical to before;
* a rare pair still reaches the energy, through the shared artisan;
* the bank is exactly the pre-routing bank when nothing falls below the
  threshold — no idle shared artisan;
* while a rare artisan exists it also absorbs pairs outside ``elements``;
* the DDP contract: every parameter gets a gradient every step, even
  with no rare edges in the batch;
* permutation symmetry survives the shared artisan.
"""

from __future__ import annotations

import typing

import pytest
import torch
from torch_geometric.data import Batch

from goal.ml.data.graph import AtomicGraph
from goal.ml.nn.blocks.artisans import (
    encode_pair_counts,
    parse_pair_counts,
    route_element_pairs,
)
from goal.ml.nn.models.simurgh.arace import MonolithicArace
from tests.ml.simurgh.conftest import random_so3

CUTOFF: float = 5.0

ARTISAN_SUBCONFIG: dict[str, typing.Any] = {
    "architecture": "equivariant",
    "hidden_irreps": "8x0e + 8x1o",
    "num_layers": 1,
    "num_rbf": 4,
    "radial_hidden": 8,
    "n_scalar_out": 8,
    "final_hidden": 8,
    "element_conditioned": True,
}

#: H-H, H-C and C-C carry the corpus; everything touching N or O is rare.
PAIR_COUNTS: dict[tuple[int, int], int] = {
    (1, 1): 1000,
    (1, 6): 1000,
    (6, 6): 1000,
    (1, 7): 1,
    (6, 7): 1,
    (7, 7): 1,
    (1, 8): 1,
    (6, 8): 1,
    (7, 8): 1,
    (8, 8): 1,
}


def _build(
    elements: typing.Sequence[int] = (1, 6, 7, 8),
    routing: bool = True,
    min_pair_frequency: float = 0.01,
    seed: int = 0,
    **kwargs: typing.Any,
) -> MonolithicArace:
    """A small ARACE model, with or without pair routing."""
    artisan: dict[str, typing.Any] = dict(ARTISAN_SUBCONFIG)
    if routing:
        artisan["rare_artisan"] = {
            "enabled": True,
            "min_pair_frequency": min_pair_frequency,
            "pair_embed_dim": 4,
        }
        artisan["pair_counts"] = PAIR_COUNTS
    torch.manual_seed(seed)
    return MonolithicArace(
        elements=tuple(elements),
        num_rounds=2,
        artisan=artisan,
        cutoff=CUTOFF,
        embedding_dim=8,
        num_elements=9,
        **kwargs,
    ).double()


def _molecule(numbers: typing.Sequence[int]) -> Batch:
    """A fully connected chain of atoms 1.2 Angstrom apart."""
    count: int = len(numbers)
    positions: torch.Tensor = torch.zeros(count, 3, dtype=torch.float64)
    positions[:, 0] = torch.arange(count, dtype=torch.float64) * 1.2
    rows: list[int] = []
    cols: list[int] = []
    for i in range(count):
        for j in range(count):
            if i != j:
                rows.append(i)
                cols.append(j)
    edge_index: torch.Tensor = torch.tensor([rows, cols], dtype=torch.long)
    graph: AtomicGraph = AtomicGraph(
        positions=positions,
        atomic_numbers=torch.tensor(list(numbers), dtype=torch.long),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.zeros(3, dtype=torch.bool),
        edge_index=edge_index,
    )
    return Batch.from_data_list([graph])


def _energy(model: MonolithicArace, batch: Batch) -> float:
    batch = batch.clone()
    batch.pos.requires_grad_(True)
    return float(model(batch)["energy"].sum())


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


class TestRouteElementPairs:
    """``route_element_pairs`` — frequency in, partition out."""

    ALL: typing.ClassVar[list[tuple[int, int]]] = [(1, 1), (1, 6), (6, 6), (1, 7)]

    def test_no_counts_dedicates_everything(self):
        dedicated, rare = route_element_pairs(self.ALL, None, 0.5)
        assert dedicated == tuple(self.ALL)
        assert rare == ()

    def test_threshold_splits_by_share_of_edges(self):
        counts: dict[tuple[int, int], int] = {(1, 1): 90, (1, 6): 9, (6, 6): 1}
        dedicated, rare = route_element_pairs(self.ALL, counts, 0.05)
        assert dedicated == ((1, 1), (1, 6))
        assert rare == ((6, 6), (1, 7))

    def test_never_observed_pairs_are_rare(self):
        """Frequency zero is below any positive threshold, by construction."""
        counts: dict[tuple[int, int], int] = {(1, 1): 100}
        dedicated, rare = route_element_pairs(self.ALL, counts, 1e-9)
        assert dedicated == ((1, 1),)
        assert set(rare) == {(1, 6), (6, 6), (1, 7)}

    def test_counts_are_order_insensitive(self):
        """``(6, 1)`` in the counts must match the canonical ``(1, 6)``."""
        dedicated, _ = route_element_pairs([(1, 6)], {(6, 1): 10}, 0.5)
        assert dedicated == ((1, 6),)

    def test_partition_is_exact(self):
        dedicated, rare = route_element_pairs(self.ALL, {(1, 1): 10, (1, 6): 1}, 0.5)
        assert sorted(dedicated + rare) == sorted(self.ALL)
        assert not set(dedicated) & set(rare)

    def test_at_most_one_over_threshold(self):
        """Frequencies sum to 1, so ``n_dedicated <= 1 / threshold``."""
        counts: dict[tuple[int, int], int] = {p: 1 for p in self.ALL}
        dedicated, _ = route_element_pairs(self.ALL, counts, 0.3)
        assert len(dedicated) <= int(1 / 0.3) + 1


class TestPairCountsEncoding:
    """Element pairs are tuples; YAML keys cannot be.

    Routing decides *which artisan modules exist*, so the counts have to
    survive a round trip through the run's saved ``config.yaml`` — a
    checkpoint trained with routing cannot load into a model rebuilt
    without it.
    """

    def test_tuple_keys(self):
        assert parse_pair_counts({(1, 6): 12}) == {(1, 6): 12}

    def test_string_keys(self):
        assert parse_pair_counts({"1_6": 12, "16_16": 3}) == {(1, 6): 12, (16, 16): 3}

    def test_list_keys(self):
        assert parse_pair_counts({(1, 6): 12}) == parse_pair_counts({"1_6": 12})

    def test_triples(self):
        assert parse_pair_counts([[1, 6, 12], [6, 6, 3]]) == {(1, 6): 12, (6, 6): 3}

    def test_none_passes_through(self):
        assert parse_pair_counts(None) is None

    def test_round_trip(self):
        counts: dict[tuple[int, int], int] = {(1, 6): 12, (6, 16): 3, (16, 53): 1}
        assert parse_pair_counts(encode_pair_counts(counts)) == counts

    def test_two_digit_elements_survive(self):
        """``"16_53"`` must not be read as ``(1, 6)`` by slicing characters."""
        assert parse_pair_counts(encode_pair_counts({(16, 53): 7})) == {(16, 53): 7}

    def test_malformed_key_is_refused(self):
        with pytest.raises(ValueError, match="Zlo_Zhi"):
            parse_pair_counts({"16": 1})

    def test_malformed_triple_is_refused(self):
        with pytest.raises(ValueError, match=r"\[Z_lo, Z_hi, count\]"):
            parse_pair_counts([[1, 6]])

    def test_encoded_counts_route_identically(self):
        """The reload path must produce the same bank as the training path."""
        artisan: dict[str, typing.Any] = dict(ARTISAN_SUBCONFIG)
        artisan["rare_artisan"] = {"enabled": True, "min_pair_frequency": 0.01}

        from_tuples = MonolithicArace(
            elements=(1, 6, 7, 8),
            artisan={**artisan, "pair_counts": PAIR_COUNTS},
            cutoff=CUTOFF,
            embedding_dim=8,
            num_elements=9,
        )
        from_yaml = MonolithicArace(
            elements=(1, 6, 7, 8),
            artisan={**artisan, "pair_counts": encode_pair_counts(PAIR_COUNTS)},
            cutoff=CUTOFF,
            embedding_dim=8,
            num_elements=9,
        )
        assert from_tuples.backbone.dedicated_pairs == from_yaml.backbone.dedicated_pairs
        assert from_tuples.backbone.rare_pairs == from_yaml.backbone.rare_pairs
        assert from_tuples.state_dict().keys() == from_yaml.state_dict().keys()


# ---------------------------------------------------------------------------
# What gets built
# ---------------------------------------------------------------------------


class TestBankComposition:
    def test_routing_off_is_unchanged(self):
        """No rare artisan, every pair dedicated — the pre-routing model."""
        model = _build(routing=False)
        assert len(model.backbone.dedicated_pairs) == 10
        assert model.backbone.rare_pairs == ()
        for block in model.backbone.blocks:
            assert block.rare_artisan is None

    def test_routing_on_shrinks_the_bank(self):
        model = _build(routing=True)
        assert model.backbone.dedicated_pairs == ((1, 1), (1, 6), (6, 6))
        assert len(model.backbone.rare_pairs) == 7
        for block in model.backbone.blocks:
            assert block.rare_artisan is not None
            assert len(block.artisans) == 3

    def test_routing_saves_parameters(self):
        plain: int = sum(p.numel() for p in _build(routing=False).parameters())
        routed: int = sum(p.numel() for p in _build(routing=True).parameters())
        assert routed < plain

    def test_rare_artisan_is_shared_across_rounds_when_asked(self):
        model = _build(routing=True, share_artisan_weights=True)
        first, second = model.backbone.blocks[0], model.backbone.blocks[1]
        assert first.rare_artisan is second.rare_artisan
        assert first.artisans is second.artisans

    def test_rare_artisan_is_independent_per_round_by_default(self):
        model = _build(routing=True, share_artisan_weights=False)
        first, second = model.backbone.blocks[0], model.backbone.blocks[1]
        assert first.rare_artisan is not second.rare_artisan

    def test_gates_report_the_rare_artisan(self):
        gates: dict[str, torch.Tensor] = _build(routing=True).backbone.gates()
        assert "layer0/rare" in gates
        assert "layer1/rare" in gates
        assert "layer0/H-H" in gates

    def test_enabled_without_counts_is_refused(self):
        """Routing with no counts would dedicate everything and do nothing."""
        artisan: dict[str, typing.Any] = dict(ARTISAN_SUBCONFIG)
        artisan["rare_artisan"] = {"enabled": True}
        with pytest.raises(ValueError, match="pair_counts"):
            MonolithicArace(elements=(1, 6), artisan=artisan, cutoff=CUTOFF)


# ---------------------------------------------------------------------------
# What it computes
# ---------------------------------------------------------------------------


class TestRoutedForward:
    def test_rare_pairs_still_reach_the_energy(self):
        """A molecule made only of rare pairs must not be energetically silent."""
        model = _build(routing=True)
        batch: Batch = _molecule([7, 8])  # N-O: rare
        assert _energy(model, batch) != 0.0

    def test_rare_artisan_is_what_carries_them(self):
        """Zero its gate and the rare-only molecule's energy must change."""
        model = _build(routing=True)
        batch: Batch = _molecule([7, 8])
        before: float = _energy(model, batch)
        with torch.no_grad():
            for block in model.backbone.blocks:
                block.rare_artisan.gate.zero_()
        assert _energy(model, batch) != pytest.approx(before)

    def test_unseen_pair_is_absorbed_while_a_rare_artisan_exists(self):
        """An element outside ``elements`` reaches the rare artisan — if there is one.

        ``elements`` is the model's declared chemistry, and an edge
        outside it matches no dedicated artisan. When the threshold has
        produced a shared rare artisan, that edge lands there rather than
        contributing nothing; when it has not (see
        :class:`TestNoRarePairs`), the edge contributes zero, which is
        the declared boundary rather than a surprise.
        """
        batch: Batch = _molecule([1, 8])  # oxygen is not in `elements`

        plain = _build(elements=(1, 6), routing=False)
        routed = _build(elements=(1, 6), routing=True, min_pair_frequency=0.5)
        assert routed.backbone.rare_pairs, "fixture must actually route something"

        assert _energy(plain, batch) == pytest.approx(0.0, abs=1e-12)
        assert _energy(routed, batch) != pytest.approx(0.0, abs=1e-12)

    def test_energy_is_permutation_symmetric(self):
        """E(A, B) = E(B, A) must survive the shared artisan."""
        model = _build(routing=True)
        forward: Batch = _molecule([7, 8])
        backward: Batch = _molecule([8, 7])
        assert _energy(model, forward) == pytest.approx(_energy(model, backward), rel=1e-10)

    def test_energy_is_rotation_invariant(self):
        model = _build(routing=True)
        batch: Batch = _molecule([1, 7, 8])
        rotated: Batch = batch.clone()
        torch.manual_seed(3)
        rotated.pos = batch.pos @ random_so3().T.double()
        assert _energy(model, rotated) == pytest.approx(_energy(model, batch), rel=1e-9)

    def test_forces_are_finite_through_the_rare_path(self):
        model = _build(routing=True)
        batch: Batch = _molecule([7, 8])
        batch.pos.requires_grad_(True)
        forces: torch.Tensor = model(batch)["forces"]
        assert torch.isfinite(forces).all()
        # Newton's third law — the rare path must not break it.
        assert torch.allclose(forces.sum(dim=0), torch.zeros(3, dtype=forces.dtype), atol=1e-8)


class TestNoRarePairs:
    """Nothing below the threshold ⇒ no rare artisan at all.

    The declared ``elements`` list is the model's contract, so a bank
    whose every pair clears ``min_pair_frequency`` is exactly the bank it
    was before routing existed — a shared artisan with no pairs assigned
    would be parameters that never see an edge.
    """

    #: Every pair equally common, so none can fall below a small threshold.
    EVEN_COUNTS: typing.ClassVar[dict[tuple[int, int], int]] = {
        (1, 1): 100,
        (1, 6): 100,
        (1, 8): 100,
        (6, 6): 100,
        (6, 8): 100,
        (8, 8): 100,
    }

    def _even(self, **kwargs: typing.Any) -> MonolithicArace:
        artisan: dict[str, typing.Any] = dict(ARTISAN_SUBCONFIG)
        artisan["rare_artisan"] = {"enabled": True, "min_pair_frequency": 5.0e-5}
        artisan["pair_counts"] = self.EVEN_COUNTS
        torch.manual_seed(0)
        return MonolithicArace(
            elements=(1, 6, 8),
            num_rounds=2,
            artisan=artisan,
            cutoff=CUTOFF,
            embedding_dim=8,
            num_elements=9,
            **kwargs,
        ).double()

    def test_no_rare_artisan_is_built(self):
        model = self._even()
        assert model.backbone.rare_pairs == ()
        assert len(model.backbone.dedicated_pairs) == 6
        for block in model.backbone.blocks:
            assert block.rare_artisan is None

    def test_parameter_count_matches_routing_off(self):
        """Not merely similar — identical. No idle parameters."""
        torch.manual_seed(0)
        plain = MonolithicArace(
            elements=(1, 6, 8),
            num_rounds=2,
            artisan=dict(ARTISAN_SUBCONFIG),
            cutoff=CUTOFF,
            embedding_dim=8,
            num_elements=9,
        ).double()
        assert sum(p.numel() for p in self._even().parameters()) == sum(
            p.numel() for p in plain.parameters()
        )

    def test_state_dict_keys_match_routing_off(self):
        """So a checkpoint is interchangeable between the two."""
        torch.manual_seed(0)
        plain = MonolithicArace(
            elements=(1, 6, 8),
            num_rounds=2,
            artisan=dict(ARTISAN_SUBCONFIG),
            cutoff=CUTOFF,
            embedding_dim=8,
            num_elements=9,
        ).double()
        assert self._even().state_dict().keys() == plain.state_dict().keys()

    def test_shared_weights_build_no_rare_artisan_either(self):
        model = self._even(share_artisan_weights=True)
        for block in model.backbone.blocks:
            assert block.rare_artisan is None

    def test_gates_report_no_rare_entry(self):
        assert not [k for k in self._even().backbone.gates() if k.endswith("/rare")]

    def test_forward_still_runs(self):
        model = self._even()
        batch: Batch = _molecule([1, 6, 8])
        assert _energy(model, batch) != 0.0


class TestDDPContract:
    """Every artisan runs every step, so every parameter gets a gradient.

    The zero-masking schedule exists for this: a parameter that receives
    no gradient on some steps breaks DDP's static graph. Adding the rare
    artisan must not create such a parameter.
    """

    def test_rare_artisan_gets_gradients_without_rare_edges(self):
        model = _build(routing=True)
        batch: Batch = _molecule([1, 1, 6])  # only dedicated pairs present
        batch.pos.requires_grad_(True)
        model(batch)["energy"].sum().backward()

        for block in model.backbone.blocks:
            for name, parameter in block.rare_artisan.named_parameters():
                assert parameter.grad is not None, f"no grad for rare.{name}"

    def test_dedicated_artisans_get_gradients_without_their_edges(self):
        model = _build(routing=True)
        batch: Batch = _molecule([7, 8])  # only rare pairs present
        batch.pos.requires_grad_(True)
        model(batch)["energy"].sum().backward()

        for block in model.backbone.blocks:
            for key in block._pair_keys:
                for name, parameter in block.artisans[key].named_parameters():
                    assert parameter.grad is not None, f"no grad for {key}.{name}"

    def test_no_edges_still_touches_every_artisan(self):
        """The degenerate empty-graph path must reach the rare artisan too.

        Exercised on the backbone rather than the model: with no edges the
        energy does not depend on positions at all, so the model's autograd
        force pass has nothing to differentiate — which is pre-existing
        behaviour and not what this test is about.
        """
        backbone = _build(routing=True).backbone
        block = backbone.blocks[0]
        h: torch.Tensor = torch.zeros(1, backbone.irreps_out.dim, dtype=torch.float64)
        _, e_atom, _, _ = block(
            h=h,
            atomic_numbers=torch.tensor([6]),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_sh=torch.zeros(0, block.irreps_edge.dim, dtype=torch.float64),
            edge_lengths=torch.zeros(0, dtype=torch.float64),
        )
        e_atom.sum().backward()
        assert block.rare_artisan.pair_linear.weight.grad is not None
