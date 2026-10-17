"""Per-block tests for SIMURGH potential artisans.

Validates:

* ``enumerate_element_pairs`` enumerates unordered pairs in a
  deterministic, sorted order.
* ``cosine_cutoff`` is exactly zero at and beyond the cutoff.
* ``PotentialArtisan`` produces a single scalar per pair for both the
  Linear and Transformer backbones.
* ``SimurghArtisanBank`` runs every expert exactly once per forward pass, even
  when some pair types are absent (zero-mask rule).
* Per-pair gates are registered learnable parameters named
  ``artisans.z{a}_z{b}.gate``.
* **Static zero-masking schedule**: on a batch that only contains one
  pair type (C-C), every one of the nine absent experts still executes
  and contributes exactly ``0.0`` to the total energy.
* **Data-driven routing (CHANGE 4)**: pairs below ``min_pair_frequency``
  are routed to the shared ``RarePotentialArtisan``; common pairs keep their
  dedicated ``PotentialArtisan``.
"""

from __future__ import annotations

import typing

import pytest
import torch
from e3nn.o3 import Irreps

from goal.ml.nn.blocks.artisans import (
    ArtisanConfig,
    PotentialArtisan,
    RarePotentialArtisan,
    SimurghArtisanBank,
    cosine_cutoff,
    enumerate_element_pairs,
    pair_label,
)
from tests.ml.simurgh.conftest import requires_gmd


class TestEnumerateElementPairs:
    def test_hcno_yields_ten_pairs(self) -> None:
        pairs: list[tuple[int, int]] = enumerate_element_pairs([1, 6, 7, 8])
        assert len(pairs) == 10
        assert pairs == [
            (1, 1),
            (1, 6),
            (1, 7),
            (1, 8),
            (6, 6),
            (6, 7),
            (6, 8),
            (7, 7),
            (7, 8),
            (8, 8),
        ]

    def test_unsorted_input_is_sorted(self) -> None:
        pairs = enumerate_element_pairs([8, 1, 6, 7])
        assert pairs == enumerate_element_pairs([1, 6, 7, 8])

    def test_duplicates_collapsed(self) -> None:
        # SimurghArtisanBank collapses duplicates; helper itself does not, but check
        # K * (K + 1) / 2 formula
        pairs = enumerate_element_pairs([1, 6])
        assert len(pairs) == 3
        assert pairs == [(1, 1), (1, 6), (6, 6)]

    def test_pair_label(self) -> None:
        assert pair_label(1, 6) == "H-C"
        assert pair_label(6, 1) == "H-C"  # order-insensitive
        assert pair_label(8, 8) == "O-O"


class TestCosineCutoff:
    def test_zero_at_and_beyond_cutoff(self) -> None:
        d = torch.tensor([0.0, 2.5, 4.9999, 5.0, 5.01, 100.0], dtype=torch.float64)
        out = cosine_cutoff(d, cutoff=5.0)
        # At cutoff, the cosine envelope is 0.5 * (cos(pi) + 1) = 0 anyway,
        # but the mask should still kill anything >= cutoff.
        assert out[3].item() == 0.0
        assert out[4].item() == 0.0
        assert out[5].item() == 0.0
        # And strictly positive interior values:
        assert out[0].item() == pytest.approx(1.0, rel=1e-12)
        assert 0.0 < out[1].item() < 1.0

    def test_monotone_decreasing_inside_cutoff(self) -> None:
        d = torch.linspace(0.0, 4.99, 50, dtype=torch.float64)
        out = cosine_cutoff(d, cutoff=5.0)
        diffs = out[1:] - out[:-1]
        assert (diffs <= 1e-12).all()


class TestPotentialArtisan:
    @pytest.mark.parametrize("expert_type", ["linear", "transformer"])
    def test_outputs_scalar_per_pair(self, expert_type: str) -> None:
        irreps_in = Irreps("8x0e+8x1o+8x2e")
        cfg = ArtisanConfig(
            scalar_channels=8,
            hidden_dims=(32, 16),
            expert_type=expert_type,
            transformer_heads=2,
            transformer_layers=1,
        )
        expert = PotentialArtisan(irreps_in, cfg).double()
        # Random scalars for 5 pairs
        scalars_a = torch.randn(5, 8, dtype=torch.float64)
        scalars_b = torch.randn(5, 8, dtype=torch.float64)
        dist = torch.linspace(0.5, 4.5, 5, dtype=torch.float64)
        out = expert(scalars_a, scalars_b, dist)
        assert out.shape == (5,)

    def test_gate_is_a_parameter(self) -> None:
        irreps_in = Irreps("4x0e")
        cfg = ArtisanConfig(scalar_channels=4, hidden_dims=(8,), expert_type="linear")
        expert = PotentialArtisan(irreps_in, cfg)
        assert isinstance(expert.gate, torch.nn.Parameter)
        assert expert.gate.requires_grad is True
        # Initial value should be 1.0
        assert expert.gate.item() == pytest.approx(1.0, abs=1e-12)


class TestSimurghArtisanBank:
    def _make_bank(
        self,
        elements: typing.Sequence[int] = (1, 6, 7, 8),
        irreps_in: str = "16x0e+16x1o+16x2e",
        cutoff: float = 5.0,
    ) -> SimurghArtisanBank:
        cfg = ArtisanConfig(
            scalar_channels=8,
            hidden_dims=(32, 16),
            expert_type="linear",
        )
        return SimurghArtisanBank(
            elements=elements,
            irreps_in=irreps_in,
            artisan_config=cfg,
            cutoff=cutoff,
        ).double()

    def test_num_artisans_matches_pair_count(self) -> None:
        artisan_bank = self._make_bank(elements=(1, 6, 7, 8))
        # K = 4 → 4 * 5 / 2 = 10
        assert artisan_bank.num_artisans == 10
        # Three-element set
        bank3 = self._make_bank(elements=(1, 6, 8))
        assert bank3.num_artisans == 6

    def test_every_pair_has_a_gate(self) -> None:
        artisan_bank = self._make_bank()
        gate_names: list[str] = [
            n for n, _ in artisan_bank.named_parameters() if n.endswith(".gate")
        ]
        # 10 unordered pairs in HCNO
        assert len(gate_names) == 10
        # Pair keys use ``z{lo}_z{hi}``
        assert "artisans.z1_z6.gate" in gate_names
        assert "artisans.z8_z8.gate" in gate_names

    def test_forward_runs_with_all_pair_types(self, methylamine_batch) -> None:  # noqa: ANN001
        artisan_bank = self._make_bank()
        n_atoms = methylamine_batch.num_atoms
        feats = torch.randn(n_atoms, artisan_bank._irreps_in.dim, dtype=torch.float64)
        energies = artisan_bank(
            atom_features=feats,
            atomic_numbers=methylamine_batch.atomic_numbers,
            edge_index=methylamine_batch.edge_index,
            edge_lengths=methylamine_batch.edge_weight,
        )
        assert energies.shape == (n_atoms,)
        assert torch.isfinite(energies).all()

    def test_zero_mask_runs_absent_pairs(self, water_batch) -> None:  # noqa: ANN001
        """Water has only H and O — N-containing experts must still execute
        (their gate parameters must be in the autograd graph)."""
        artisan_bank = self._make_bank()
        n_atoms = water_batch.num_atoms
        feats = torch.randn(n_atoms, artisan_bank._irreps_in.dim, dtype=torch.float64)
        energies = artisan_bank(
            atom_features=feats,
            atomic_numbers=water_batch.atomic_numbers,
            edge_index=water_batch.edge_index,
            edge_lengths=water_batch.edge_weight,
        )
        loss = energies.sum()
        loss.backward()
        for name, p in artisan_bank.named_parameters():
            if name.endswith(".gate"):
                assert p.grad is not None, f"gate {name} got no gradient"


class TestZeroMaskCCOnly:
    """CHANGE 3 acceptance test.

    On a C-C-only batch with SIMURGH configured for H/C/N/O:

    1. All 10 experts execute (their parameters are reachable from the
       loss / appear in the autograd graph).
    2. The total contribution of each of the 9 *absent* experts is
       exactly ``0.0`` — verified both via the ``return_per_pair``
       diagnostic and by checking that tampering with the gates of
       absent experts leaves the energy unchanged.
    """

    def _make_bank(self) -> SimurghArtisanBank:
        cfg = ArtisanConfig(
            scalar_channels=8,
            hidden_dims=(32, 16),
            expert_type="linear",
        )
        return SimurghArtisanBank(
            elements=(1, 6, 7, 8),
            irreps_in="16x0e+16x1o+16x2e",
            artisan_config=cfg,
            cutoff=5.0,
        ).double()

    def test_absent_experts_contribute_exactly_zero(
        self,
        carbon_dimer_batch,  # noqa: ANN001
    ) -> None:
        artisan_bank = self._make_bank()
        n_atoms = carbon_dimer_batch.num_atoms
        feats = torch.randn(n_atoms, artisan_bank._irreps_in.dim, dtype=torch.float64)

        # Use the diagnostic return-per-pair path
        atom_energies, per_pair = artisan_bank(
            atom_features=feats,
            atomic_numbers=carbon_dimer_batch.atomic_numbers,
            edge_index=carbon_dimer_batch.edge_index,
            edge_lengths=carbon_dimer_batch.edge_weight,
            return_per_pair=True,
        )

        # Sanity: 9 absent experts contribute exactly 0.0, C-C contributes
        # whatever the model says.
        assert set(per_pair.keys()) == {
            "H-H",
            "H-C",
            "H-N",
            "H-O",
            "C-C",
            "C-N",
            "C-O",
            "N-N",
            "N-O",
            "O-O",
        }
        for label, value in per_pair.items():
            if label == "C-C":
                # C-C is the only non-trivially-zero contribution.  Its
                # value depends on the random features; just check it is
                # finite and *can* be non-zero.
                assert torch.isfinite(value).item()
            else:
                assert value.item() == 0.0, (
                    f"Absent expert '{label}' contributed {value.item()!r} "
                    f"to the energy — zero-masking is broken"
                )

    def test_absent_expert_gates_have_no_effect_on_energy(
        self,
        carbon_dimer_batch,  # noqa: ANN001
    ) -> None:
        """Tamper with every non-CC gate (set it to 1e10) and verify the
        total energy is unchanged.  If the mask leaked, the energy would
        explode."""
        artisan_bank = self._make_bank()
        n_atoms = carbon_dimer_batch.num_atoms
        feats = torch.randn(n_atoms, artisan_bank._irreps_in.dim, dtype=torch.float64)

        e_before: torch.Tensor = artisan_bank(
            atom_features=feats,
            atomic_numbers=carbon_dimer_batch.atomic_numbers,
            edge_index=carbon_dimer_batch.edge_index,
            edge_lengths=carbon_dimer_batch.edge_weight,
        )

        cc_key = SimurghArtisanBank._key(6, 6)
        with torch.no_grad():
            for key, expert in artisan_bank.artisans.items():
                if key != cc_key:
                    expert.gate.fill_(1.0e10)

        e_after: torch.Tensor = artisan_bank(
            atom_features=feats,
            atomic_numbers=carbon_dimer_batch.atomic_numbers,
            edge_index=carbon_dimer_batch.edge_index,
            edge_lengths=carbon_dimer_batch.edge_weight,
        )

        assert torch.allclose(e_before, e_after), (
            "Non-CC expert gates affected the energy on a C-C-only batch — "
            "zero-masking is broken"
        )

    def test_all_experts_are_in_autograd_graph(
        self,
        carbon_dimer_batch,  # noqa: ANN001
    ) -> None:
        """On a C-C-only batch every one of the 10 experts must have a
        registered (possibly zero-valued) gradient after backward — the
        static-shape schedule keeps every expert in the autograd graph."""
        artisan_bank = self._make_bank()
        n_atoms = carbon_dimer_batch.num_atoms
        feats = torch.randn(
            n_atoms, artisan_bank._irreps_in.dim, dtype=torch.float64, requires_grad=True
        )

        energies: torch.Tensor = artisan_bank(
            atom_features=feats,
            atomic_numbers=carbon_dimer_batch.atomic_numbers,
            edge_index=carbon_dimer_batch.edge_index,
            edge_lengths=carbon_dimer_batch.edge_weight,
        )
        energies.sum().backward()

        for key in artisan_bank.artisans.keys():
            expert: PotentialArtisan = typing.cast(PotentialArtisan, artisan_bank.artisans[key])
            assert expert.gate.grad is not None, (
                f"Gate of expert {key} did not appear in the autograd graph — "
                f"the zero-mask schedule must keep every expert reachable"
            )
            # For absent experts the gradient should be exactly zero
            if key != SimurghArtisanBank._key(6, 6):
                assert expert.gate.grad.item() == 0.0, (
                    f"Absent expert {key} received a non-zero gradient "
                    f"({expert.gate.grad.item()!r}) — zero-masking is broken"
                )


# ---------------------------------------------------------------------------
# CHANGE 4: Data-driven routing — RarePotentialArtisan
# ---------------------------------------------------------------------------


class TestRarePairRouting:
    """Tests for ``rare_pair_enabled=True`` routing in ``SimurghArtisanBank``."""

    # H/C/O elements: pairs are (H-H, H-C, H-O, C-C, C-O, O-O) = 6 total.
    # We make H-H and O-O "rare" by giving them tiny counts.
    _ELEMENTS = (1, 6, 8)
    _IRREPS_IN = "8x0e+8x1o"

    def _pair_counts_hco(self) -> dict[tuple[int, int], int]:
        """95% of edges are H-C, leaving H-H and O-O below 1% each."""
        return {
            (1, 1): 1,  # rare  (~0.09 %)
            (1, 6): 550,  # common (~50 %)
            (1, 8): 400,  # common (~36 %)
            (6, 6): 150,  # common (~13.6 %)
            (6, 8): 1,  # rare  (~0.09 %)
            (8, 8): 1,  # rare  (~0.09 %)
        }  # total ≈ 1103 directed edges

    def _make_bank_with_routing(self) -> SimurghArtisanBank:
        cfg = ArtisanConfig(
            scalar_channels=8,
            hidden_dims=(16,),
            expert_type="linear",
            rare_pair_embed_dim=4,
        )
        return SimurghArtisanBank(
            elements=self._ELEMENTS,
            irreps_in=self._IRREPS_IN,
            artisan_config=cfg,
            cutoff=5.0,
            pair_counts=self._pair_counts_hco(),
            rare_pair_enabled=True,
            min_pair_frequency=0.01,
        ).double()

    def test_common_pairs_have_dedicated_experts(self) -> None:
        artisan_bank = self._make_bank_with_routing()
        # H-C, H-O, C-C are common → dedicated PotentialArtisans
        for a, b in [(1, 6), (1, 8), (6, 6)]:
            key = SimurghArtisanBank._key(a, b)
            assert (
                key in artisan_bank.artisans
            ), f"Common pair {(a, b)} should have a dedicated PotentialArtisan."

    def test_rare_pairs_go_to_rare_artisan(self) -> None:
        artisan_bank = self._make_bank_with_routing()
        rare_pairs = set(artisan_bank._rare_pairs)
        # H-H, C-O, O-O are below 1% → handled by RarePotentialArtisan
        for p in [(1, 1), (6, 8), (8, 8)]:
            assert p in rare_pairs, f"Pair {p} should be classified as rare (freq < 1%)."

    def test_rare_artisan_is_not_none(self) -> None:
        artisan_bank = self._make_bank_with_routing()
        assert artisan_bank.rare_artisan_module is not None

    def test_num_artisans_dedicated_plus_one_shared(self) -> None:
        artisan_bank = self._make_bank_with_routing()
        n_dedicated = len(artisan_bank._dedicated_pairs)
        # num_artisans = dedicated + 1 (the shared RarePotentialArtisan)
        assert artisan_bank.num_artisans == n_dedicated + 1

    def test_forward_runs_on_hco_batch(self, methane_batch) -> None:  # noqa: ANN001
        """Smoke test: routed artisan bank runs on a batch with H and C only."""
        artisan_bank = self._make_bank_with_routing()
        n = methane_batch.num_atoms
        feats = torch.randn(n, Irreps(self._IRREPS_IN).dim, dtype=torch.float64)
        energies = artisan_bank(
            atom_features=feats,
            atomic_numbers=methane_batch.atomic_numbers,
            edge_index=methane_batch.edge_index,
            edge_lengths=methane_batch.edge_weight,
        )
        assert energies.shape == (n,)
        assert torch.isfinite(energies).all()

    def test_rare_artisan_parameters_receive_gradient(self, water_batch) -> None:
        """Backward through a batch containing rare O-O pairs (water = O + 2H).

        Water contains only H-H, H-O, and O-O pairs.  H-H and O-O are rare in
        our pair_counts, so the rare_artisan_module must fire — its parameters must
        receive gradients.
        """
        artisan_bank = self._make_bank_with_routing()
        n = water_batch.num_atoms
        feats = torch.randn(n, Irreps(self._IRREPS_IN).dim, dtype=torch.float64)
        energies = artisan_bank(
            atom_features=feats,
            atomic_numbers=water_batch.atomic_numbers,
            edge_index=water_batch.edge_index,
            edge_lengths=water_batch.edge_weight,
        )
        energies.sum().backward()

        assert artisan_bank.rare_artisan_module is not None
        # The shared backbone parameters must have gotten gradients
        backbone_grads = [
            p.grad
            for name, p in artisan_bank.rare_artisan_module.named_parameters()
            if p.grad is not None
        ]
        assert backbone_grads, (
            "RarePotentialArtisan parameters received no gradient — "
            "the rare-pair path is disconnected from the autograd graph."
        )

    def test_no_routing_when_disabled(self) -> None:
        """With rare_pair_enabled=False, all pairs get dedicated artisans."""
        cfg = ArtisanConfig(
            scalar_channels=8,
            hidden_dims=(16,),
            expert_type="linear",
        )
        artisan_bank = SimurghArtisanBank(
            elements=self._ELEMENTS,
            irreps_in=self._IRREPS_IN,
            artisan_config=cfg,
            cutoff=5.0,
            pair_counts=self._pair_counts_hco(),
            rare_pair_enabled=False,
        ).double()
        assert len(artisan_bank._rare_pairs) == 0
        assert artisan_bank.rare_artisan_module is None
        # K=3 → 6 pairs, all dedicated
        assert artisan_bank.num_artisans == 6

    @requires_gmd
    def test_routing_on_gmd_hco_batch(self, gmd_batch) -> None:  # noqa: ANN001
        """Smoke test: routed artisan bank runs on real GMD batch (H, C, O atoms)."""
        cfg = ArtisanConfig(
            scalar_channels=8,
            hidden_dims=(16,),
            expert_type="linear",
            rare_pair_embed_dim=4,
        )
        # GMD O2C4H8: mostly H-C and H-H edges
        pair_counts = {
            (1, 1): 10,
            (1, 6): 400,
            (1, 8): 50,
            (6, 6): 30,
            (6, 8): 2,
            (8, 8): 1,
        }
        artisan_bank = SimurghArtisanBank(
            elements=(1, 6, 8),
            irreps_in="8x0e+8x1o",
            artisan_config=cfg,
            cutoff=5.0,
            pair_counts=pair_counts,
            rare_pair_enabled=True,
            min_pair_frequency=0.05,
        ).double()

        feats = torch.randn(
            gmd_batch.num_nodes,
            Irreps("8x0e+8x1o").dim,
            dtype=torch.float64,
        )
        energies = artisan_bank(
            atom_features=feats,
            atomic_numbers=gmd_batch.atomic_numbers,
            edge_index=gmd_batch.edge_index,
            edge_lengths=gmd_batch.edge_weight,
        )
        assert energies.shape == (gmd_batch.num_nodes,)
        assert torch.isfinite(energies).all()


class TestRarePotentialArtisanModule:
    """Direct unit tests for the :class:`RarePotentialArtisan` module."""

    def test_output_shape(self) -> None:
        irreps_in = Irreps("8x0e+8x1o")
        cfg = ArtisanConfig(scalar_channels=8, hidden_dims=(16,), rare_pair_embed_dim=4)
        expert = RarePotentialArtisan(irreps_in, [(1, 6), (6, 8)], cfg).double()

        n = 5
        scalars_a = torch.randn(n, 8, dtype=torch.float64)
        scalars_b = torch.randn(n, 8, dtype=torch.float64)
        distances = torch.linspace(0.5, 4.0, n, dtype=torch.float64)
        pair_idx = torch.zeros(n, dtype=torch.long)  # all edges are pair 0 (H-C)

        out = expert(scalars_a, scalars_b, distances, pair_idx)
        assert out.shape == (n,)
        assert torch.isfinite(out).all()

    def test_gates_are_parameters(self) -> None:
        irreps_in = Irreps("4x0e")
        cfg = ArtisanConfig(scalar_channels=4, hidden_dims=(8,), rare_pair_embed_dim=4)
        expert = RarePotentialArtisan(irreps_in, [(1, 1), (8, 8)], cfg)
        assert isinstance(expert.gates, torch.nn.Parameter)
        assert expert.gates.shape == (2,)
        assert expert.gates.requires_grad

    def test_requires_at_least_one_pair(self) -> None:
        irreps_in = Irreps("4x0e")
        cfg = ArtisanConfig(scalar_channels=4, hidden_dims=(8,))
        with pytest.raises(ValueError, match="at least one pair"):
            RarePotentialArtisan(irreps_in, [], cfg)

    def test_different_pair_idx_gives_different_output(self) -> None:
        """Two edges with the same scalars/distance but different pair_local_idx
        should produce different outputs (the embedding distinguishes pairs)."""
        irreps_in = Irreps("8x0e")
        cfg = ArtisanConfig(scalar_channels=8, hidden_dims=(16,), rare_pair_embed_dim=8)
        expert = RarePotentialArtisan(irreps_in, [(1, 1), (8, 8)], cfg).double()

        s = torch.randn(1, 8, dtype=torch.float64)
        d = torch.tensor([1.5], dtype=torch.float64)
        idx_0 = torch.zeros(1, dtype=torch.long)
        idx_1 = torch.ones(1, dtype=torch.long)

        with torch.no_grad():
            out_0 = expert(s, s, d, idx_0)
            out_1 = expert(s, s, d, idx_1)

        assert not torch.allclose(out_0, out_1), (
            "RarePotentialArtisan returned identical outputs for different pair_local_idx — "
            "the per-pair embedding has no effect."
        )
