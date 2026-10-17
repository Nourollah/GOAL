"""Tests for the AdaptiveDepthGate add-on.

Coverage:

* Test 7 — ``adaptive_gate_config=None`` (default) is a strict no-op.
* Test 8 — soft mode (``training=True``): gates in (0,1), positive scalar
  aux loss, shape preserved, gradients flow.
* Test 9 — hard mode (``training=False``): zero aux loss, atoms above the
  threshold take ``h_new`` and the rest keep ``h_prev`` exactly.
* Test 10 — the gate starts open, and ``init_bias`` controls how open.
* Test 11 — rotation equivariance holds with the gate enabled.
* Test 12 — ``aux_loss`` reaches ``GOALModule``'s loss dict and total, on
  both the modular and the monolithic ARACE path, and ``backward()`` runs.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from goal.ml.nn.blocks.adaptive_gate import AdaptiveDepthGate
from goal.ml.nn.models.simurgh.arace import MonolithicArace
from tests.ml.simurgh.conftest import random_so3
from tests.ml.simurgh.test_fragment_interaction import (
    ACTIVE_FRAGMENT_CONFIG,
    ARTISAN_SUBCONFIG,
    CUTOFF,
    HIDDEN_IRREPS,
    _build_backbone,
    _build_round,
    _labelled_graph,
    _methane_positions,
    _round_inputs,
    _run,
    _water_dimer,
)

ADAPTIVE_GATE_CONFIG: dict = {
    "n_scalar": 8,
    "hard_threshold": 0.5,
    "aux_loss_weight": 0.01,
    "init_bias": 1.0,
}


def _build_gate(**overrides) -> AdaptiveDepthGate:
    torch.manual_seed(17)
    cfg: dict = {**ADAPTIVE_GATE_CONFIG, **overrides}
    return AdaptiveDepthGate(irreps_node=HIDDEN_IRREPS, **cfg).double()


def _features(n_atoms: int = 12, seed: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    from e3nn.o3 import Irreps

    torch.manual_seed(seed)
    dim: int = Irreps(HIDDEN_IRREPS).dim
    h_prev: torch.Tensor = torch.randn(n_atoms, dim, dtype=torch.float64)
    h_new: torch.Tensor = h_prev + 0.3 * torch.randn(n_atoms, dim, dtype=torch.float64)
    return h_new, h_prev


# ----------------------------------------------------------------------
# Test 7 — disabled by default is a strict no-op
# ----------------------------------------------------------------------


class TestAdaptiveGateDisabled:
    def test_output_bit_identical_to_plain_arace(self) -> None:
        graph = _labelled_graph(*_methane_positions())
        rnd = _build_round(adaptive_gate_config=None)
        rnd.eval()
        h, edge_index, edge_sh, edge_lengths = _round_inputs(graph, rnd)

        h_pair_ref, e_ref = rnd.artisan_layer(
            h, graph.atomic_numbers, edge_index, edge_sh, edge_lengths
        )
        h_ref = rnd.ace_block(h, h_pair_ref, edge_index)

        h_new, e_atom, gate_scores, aux_loss = rnd(
            h, graph.atomic_numbers, edge_index, edge_sh, edge_lengths
        )

        assert rnd.adaptive_gate is None
        assert torch.equal(h_new, h_ref)
        assert torch.equal(e_atom, e_ref)
        assert gate_scores is None
        assert aux_loss.shape == ()
        assert float(aux_loss) == 0.0

    def test_adds_no_parameters(self) -> None:
        plain = _build_round(adaptive_gate_config=None)
        assert not any("adaptive_gate" in name for name in plain.state_dict())

        enabled = _build_round(adaptive_gate_config=ADAPTIVE_GATE_CONFIG)
        assert sum(p.numel() for p in enabled.parameters()) > sum(
            p.numel() for p in plain.parameters()
        )

    def test_backbone_side_channels_stay_none(self) -> None:
        graph = _labelled_graph(*_methane_positions())
        backbone = _build_backbone()
        backbone.eval()
        _run(backbone, Batch.from_data_list([graph]))

        assert not backbone.adaptive_gate_enabled
        assert backbone.last_aux_loss is None
        assert all(g is None for g in backbone.last_gate_scores)


# ----------------------------------------------------------------------
# Test 8 — soft mode (training)
# ----------------------------------------------------------------------


class TestAdaptiveGateSoftMode:
    def test_gate_scores_in_open_interval(self) -> None:
        gate = _build_gate()
        h_new, h_prev = _features()
        _, _, gate_scores = gate(h_new=h_new, h_prev=h_prev, training=True)

        assert gate_scores.shape == (h_new.shape[0],)
        assert (gate_scores > 0.0).all() and (gate_scores < 1.0).all()

    def test_aux_loss_is_positive_scalar(self) -> None:
        gate = _build_gate()
        h_new, h_prev = _features()
        _, aux_loss, gate_scores = gate(h_new=h_new, h_prev=h_prev, training=True)

        assert aux_loss.shape == ()
        assert float(aux_loss) > 0.0
        expected = gate.aux_loss_weight * float(gate_scores.mean())
        assert float(aux_loss) == pytest.approx(expected)

    def test_aux_loss_weight_zero_disables_the_penalty(self) -> None:
        gate = _build_gate(aux_loss_weight=0.0)
        h_new, h_prev = _features()
        _, aux_loss, _ = gate(h_new=h_new, h_prev=h_prev, training=True)
        assert float(aux_loss) == 0.0

    def test_single_gate_interpolation(self) -> None:
        """``per_irrep=False``: one gate scales the whole feature vector."""
        gate = _build_gate(per_irrep=False)
        h_new, h_prev = _features()
        h_next, _, gate_scores = gate(h_new=h_new, h_prev=h_prev, training=True)

        assert h_next.shape == h_new.shape
        expected = gate_scores.unsqueeze(-1) * h_new + (1 - gate_scores.unsqueeze(-1)) * h_prev
        assert torch.allclose(h_next, expected)

    def test_per_irrep_gate_is_constant_within_each_block(self) -> None:
        """``per_irrep=True``: one gate per irrep block, so the interpolation
        ratio is constant *within* a block and free to differ *between*
        blocks.  A ratio varying inside a block would mean the gate is
        mixing components of the same irrep — which would break
        equivariance."""
        gate = _build_gate(per_irrep=True)
        h_new, h_prev = _features()
        h_next, _, _ = gate(h_new=h_new, h_prev=h_prev, training=True)

        assert h_next.shape == h_new.shape
        ratio = (h_next - h_prev) / (h_new - h_prev)  # per column

        start, block_means = 0, []
        for dim in gate._block_dims:
            block = ratio[:, start : start + dim]
            spread = (block - block[:, :1]).abs().max().item()
            assert spread < 1e-12, f"gate varies inside an irrep block: {spread:.3e}"
            block_means.append(block[:, 0])
            start += dim

        # 3 blocks (0e, 1o, 2e) → 3 independent gates that genuinely differ.
        assert len(block_means) == 3
        assert (
            block_means[0] - block_means[1]
        ).abs().max().item() > 1e-6, (
            "per-irrep gates are identical — no more expressive than one gate"
        )

    def test_gate_scores_is_the_mean_over_blocks(self) -> None:
        gate = _build_gate(per_irrep=True)
        h_new, h_prev = _features()
        h_next, _, gate_scores = gate(h_new=h_new, h_prev=h_prev, training=True)

        ratio = (h_next - h_prev) / (h_new - h_prev)
        start, per_block = 0, []
        for dim in gate._block_dims:
            per_block.append(ratio[:, start])
            start += dim
        assert torch.allclose(gate_scores, torch.stack(per_block, dim=-1).mean(dim=-1))

    def test_gradients_reach_the_gate_parameters(self) -> None:
        gate = _build_gate()
        h_new, h_prev = _features()
        h_next, aux_loss, _ = gate(h_new=h_new, h_prev=h_prev, training=True)
        (h_next.sum() + aux_loss).backward()

        assert all(p.grad is not None for p in gate.parameters())

    def test_identical_features_give_zero_delta_input(self) -> None:
        """h_new == h_prev → the convergence delta is 0 and output is h_prev."""
        gate = _build_gate()
        _, h_prev = _features()
        h_next, _, _ = gate(h_new=h_prev, h_prev=h_prev, training=True)
        assert torch.allclose(h_next, h_prev)

    def test_rejects_mismatched_shapes(self) -> None:
        gate = _build_gate()
        h_new, h_prev = _features()
        with pytest.raises(ValueError, match="same shape"):
            gate(h_new=h_new, h_prev=h_prev[:-1], training=True)

    def test_rejects_non_leading_scalars(self) -> None:
        with pytest.raises(ValueError, match="leading block"):
            AdaptiveDepthGate(irreps_node="8x1o + 8x0e", n_scalar=4)


# ----------------------------------------------------------------------
# Test 9 — hard mode (inference)
# ----------------------------------------------------------------------


class TestAdaptiveGateHardMode:
    def test_aux_loss_is_zero(self) -> None:
        gate = _build_gate()
        h_new, h_prev = _features()
        _, aux_loss, _ = gate(h_new=h_new, h_prev=h_prev, training=False)
        assert float(aux_loss) == 0.0

    def test_selects_h_new_above_threshold_and_h_prev_below(self) -> None:
        # A wide spread of gate logits so both branches are exercised.
        # ``per_irrep=False`` so a single gate decides the whole vector and
        # the reported score is exactly the gate that was applied.
        gate = _build_gate(init_bias=0.0, per_irrep=False)
        with torch.no_grad():
            gate.gate_mlp[-1].weight.mul_(25.0)
        h_new, h_prev = _features(n_atoms=32, seed=8)

        h_next, _, gate_scores = gate(h_new=h_new, h_prev=h_prev, training=False)
        open_mask = gate_scores > gate.hard_threshold
        assert bool(open_mask.any()) and bool(
            (~open_mask).any()
        ), "test needs both open and closed atoms to be meaningful"

        assert torch.equal(h_next[open_mask], h_new[open_mask])
        assert torch.equal(h_next[~open_mask], h_prev[~open_mask])

    def test_per_irrep_hard_gating_is_all_or_nothing_per_block(self) -> None:
        """With per-block gates, each block independently takes either
        ``h_new`` or ``h_prev`` — never a blend."""
        gate = _build_gate(init_bias=0.0, per_irrep=True)
        with torch.no_grad():
            gate.gate_mlp[-1].weight.mul_(25.0)
        h_new, h_prev = _features(n_atoms=32, seed=8)

        h_next, _, _ = gate(h_new=h_new, h_prev=h_prev, training=False)

        start = 0
        for dim in gate._block_dims:
            block_next = h_next[:, start : start + dim]
            took_new = torch.isclose(block_next, h_new[:, start : start + dim]).all(dim=-1)
            took_prev = torch.isclose(block_next, h_prev[:, start : start + dim]).all(dim=-1)
            assert bool((took_new | took_prev).all()), "hard gate produced a blend"
            start += dim

    def test_round_follows_train_eval_mode(self) -> None:
        graph = _labelled_graph(*_methane_positions())
        rnd = _build_round(adaptive_gate_config=ADAPTIVE_GATE_CONFIG, seed=19)
        h, edge_index, edge_sh, edge_lengths = _round_inputs(graph, rnd)

        rnd.train()
        _, _, _, aux_train = rnd(h, graph.atomic_numbers, edge_index, edge_sh, edge_lengths)
        rnd.eval()
        _, _, _, aux_eval = rnd(h, graph.atomic_numbers, edge_index, edge_sh, edge_lengths)

        assert float(aux_train) > 0.0, "soft mode should produce a sparsity penalty"
        assert float(aux_eval) == 0.0, "hard mode must not produce a penalty"


# ----------------------------------------------------------------------
# Test 10 — the gate starts open
# ----------------------------------------------------------------------


class TestAdaptiveGateInitialisation:
    def test_default_init_bias_starts_open(self) -> None:
        # NOTE: the mean cannot exceed sigmoid(init_bias) by much — with the
        # documented default init_bias=1.0 the gate opens to ~sigmoid(1) =
        # 0.73, not >0.8.  What matters is that it starts clearly *open*
        # (well above the 0.5 hard threshold) so the model begins as plain
        # ARACE and has to learn to close atoms.
        gate = _build_gate(init_bias=1.0)
        h_new, h_prev = _features(n_atoms=64, seed=2)
        _, _, gate_scores = gate(h_new=h_new, h_prev=h_prev, training=True)

        mean = float(gate_scores.mean())
        assert mean > 0.6, f"gate did not start open: mean={mean:.3f}"
        assert mean == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item(), abs=0.15)

    def test_larger_init_bias_opens_further(self) -> None:
        h_new, h_prev = _features(n_atoms=64, seed=2)
        means: list[float] = []
        for bias in (0.0, 1.0, 2.5):
            gate = _build_gate(init_bias=bias)
            _, _, scores = gate(h_new=h_new, h_prev=h_prev, training=True)
            means.append(float(scores.mean()))

        assert means[0] < means[1] < means[2]
        assert means[2] > 0.8, f"init_bias=2.5 should open the gate wide: {means[2]:.3f}"
        assert means[0] == pytest.approx(0.5, abs=0.15)

    def test_gate_starts_open_means_near_identity_round(self) -> None:
        """At init the gate mostly passes h_new through, so the round is close
        to plain ARACE (but not identical — that is the point of a gate)."""
        graph = _labelled_graph(*_methane_positions())
        plain = _build_round(adaptive_gate_config=None, seed=23)
        gated = _build_round(adaptive_gate_config=ADAPTIVE_GATE_CONFIG, seed=23)
        plain.eval()
        gated.eval()
        h, edge_index, edge_sh, edge_lengths = _round_inputs(graph, gated)

        h_plain = plain(h, graph.atomic_numbers, edge_index, edge_sh, edge_lengths)[0]
        h_gated, _, gate_scores, _ = gated(
            h, graph.atomic_numbers, edge_index, edge_sh, edge_lengths
        )
        # Hard mode at eval: open atoms are bit-identical to plain ARACE.
        open_mask = gate_scores > gated.adaptive_gate.hard_threshold
        assert torch.equal(h_gated[open_mask], h_plain[open_mask])


# ----------------------------------------------------------------------
# Test 11 — equivariance with the gate enabled
# ----------------------------------------------------------------------


class TestAdaptiveGateEquivariance:
    def test_energy_invariant_forces_equivariant(self) -> None:
        positions, numbers = _water_dimer()
        graph = _labelled_graph(positions, numbers)

        backbone = _build_backbone(adaptive_gate=ADAPTIVE_GATE_CONFIG)
        backbone.eval()
        assert backbone.adaptive_gate_enabled

        out0 = _run(backbone, Batch.from_data_list([graph]))
        rotation = random_so3()
        rotated = _labelled_graph(positions @ rotation.T, numbers)
        out1 = _run(backbone, Batch.from_data_list([rotated]))

        e_diff = (out0["energy"] - out1["energy"]).abs().max().item()
        assert e_diff < 1e-5, f"Energy not invariant with the gate: {e_diff:.3e}"

        f_diff = (
            (out1["forces"].detach() - out0["forces"].detach() @ rotation.T).abs().max().item()
        )
        assert f_diff < 1e-4, f"Forces not equivariant with the gate: {f_diff:.3e}"

    def test_equivariance_with_both_modules(self) -> None:
        positions, numbers = _water_dimer()
        graph = _labelled_graph(positions, numbers)

        backbone = _build_backbone(
            fragment_interaction=ACTIVE_FRAGMENT_CONFIG,
            adaptive_gate=ADAPTIVE_GATE_CONFIG,
        )
        backbone.eval()

        out0 = _run(backbone, Batch.from_data_list([graph]))
        rotation = random_so3()
        rotated = _labelled_graph(positions @ rotation.T, numbers)
        out1 = _run(backbone, Batch.from_data_list([rotated]))

        e_diff = (out0["energy"] - out1["energy"]).abs().max().item()
        assert e_diff < 1e-5, f"Energy not invariant with both add-ons: {e_diff:.3e}"

        f_diff = (
            (out1["forces"].detach() - out0["forces"].detach() @ rotation.T).abs().max().item()
        )
        assert f_diff < 1e-4, f"Forces not equivariant with both add-ons: {f_diff:.3e}"

    def test_gate_scores_are_rotation_invariant(self) -> None:
        positions, numbers = _water_dimer()
        backbone = _build_backbone(adaptive_gate=ADAPTIVE_GATE_CONFIG)
        backbone.eval()

        _run(backbone, Batch.from_data_list([_labelled_graph(positions, numbers)]))
        scores0 = [g.clone() for g in backbone.last_gate_scores]

        rotation = random_so3()
        _run(
            backbone,
            Batch.from_data_list([_labelled_graph(positions @ rotation.T, numbers)]),
        )
        scores1 = backbone.last_gate_scores

        for round_idx, (a, b) in enumerate(zip(scores0, scores1)):
            assert a is not None and b is not None
            diff = (a - b).abs().max().item()
            assert diff < 1e-8, f"round {round_idx} gate scores rotated: {diff:.3e}"


# ----------------------------------------------------------------------
# Test 12 — aux_loss reaches GOALModule
# ----------------------------------------------------------------------


def _module_config() -> OmegaConf:
    return OmegaConf.create(
        {
            "training": {
                "ema": {"enabled": False},
                "gradient_clip": 0.0,
                "optimizer": {"lr": 1.0e-3},
            }
        }
    )


def _labelled_training_batch() -> Batch:
    graph = _labelled_graph(*_water_dimer())
    batch = Batch.from_data_list([graph])
    batch.energy = torch.tensor([-152.0], dtype=torch.float64)
    batch.forces = torch.zeros_like(batch.pos)
    return batch


class TestAuxLossFlowsToGOALModule:
    def _loss(self):
        from goal.ml.training.loss import (
            CompositeLoss,
            EnergyLoss,
            ForcesLoss,
            WeightedLoss,
        )

        return CompositeLoss(
            [
                WeightedLoss(EnergyLoss(), weight=1.0, label="energy"),
                WeightedLoss(ForcesLoss(), weight=1.0, label="forces"),
            ]
        )

    def test_modular_path_adds_aux_loss_to_total(self) -> None:
        from goal.ml.nn.heads.energy_forces import EnergyForcesHead
        from goal.ml.training.module import GOALModule

        backbone = _build_backbone(
            fragment_interaction=ACTIVE_FRAGMENT_CONFIG,
            adaptive_gate=ADAPTIVE_GATE_CONFIG,
        )
        head = EnergyForcesHead(irreps_in=HIDDEN_IRREPS, hidden_dim=16).double()
        module = GOALModule(
            backbone=backbone, head=head, loss=self._loss(), config=_module_config()
        )
        module.train()
        batch = _labelled_training_batch()

        predictions = module(batch)
        assert "aux_loss" in predictions
        assert "gate_scores" in predictions
        assert predictions["gate_scores"].shape == (2, batch.pos.shape[0])

        losses = module._with_aux_loss(module.loss(predictions, batch), predictions)
        assert "aux_loss" in losses
        assert float(losses["aux_loss"]) > 0.0

        bare_total = module.loss(predictions, batch)["total"]
        assert float(losses["total"]) == pytest.approx(
            float(bare_total) + float(predictions["aux_loss"])
        )

    def test_training_step_runs_and_backward_completes(self) -> None:
        from goal.ml.nn.heads.energy_forces import EnergyForcesHead
        from goal.ml.training.module import GOALModule

        backbone = _build_backbone(
            fragment_interaction=ACTIVE_FRAGMENT_CONFIG,
            adaptive_gate=ADAPTIVE_GATE_CONFIG,
        )
        head = EnergyForcesHead(irreps_in=HIDDEN_IRREPS, hidden_dim=16).double()
        module = GOALModule(
            backbone=backbone, head=head, loss=self._loss(), config=_module_config()
        )
        module.train()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # self.log() outside a Trainer
            total = module.training_step(_labelled_training_batch(), 0)

        assert torch.isfinite(total).item()
        total.backward()

        gate_grads = [
            p.grad
            for name, p in module.named_parameters()
            if "adaptive_gate" in name and p.grad is not None
        ]
        ca_grads = [
            p.grad
            for name, p in module.named_parameters()
            if "fragment_interaction" in name and p.grad is not None
        ]
        assert gate_grads, "no gradient reached the adaptive gate"
        assert ca_grads, "no gradient reached the fragment channel"

    def test_monolithic_path_returns_aux_loss(self) -> None:
        torch.manual_seed(29)
        model = MonolithicArace(
            elements=(1, 6, 7, 8),
            num_rounds=2,
            artisan=dict(ARTISAN_SUBCONFIG),
            cutoff=CUTOFF,
            embedding_dim=16,
            num_elements=9,
            fragment_interaction=ACTIVE_FRAGMENT_CONFIG,
            adaptive_gate=ADAPTIVE_GATE_CONFIG,
        ).double()
        model.train()

        out = model(_labelled_training_batch())
        assert "aux_loss" in out and out["aux_loss"].shape == ()
        assert float(out["aux_loss"]) > 0.0
        assert out["gate_scores"].shape[0] == 2
        assert torch.isfinite(out["energy"]).all()
        assert torch.isfinite(out["forces"]).all()

    def test_no_aux_loss_key_when_modules_disabled(self) -> None:
        from goal.ml.nn.heads.energy_forces import EnergyForcesHead
        from goal.ml.training.module import GOALModule

        backbone = _build_backbone()
        head = EnergyForcesHead(irreps_in=HIDDEN_IRREPS, hidden_dim=16).double()
        module = GOALModule(
            backbone=backbone, head=head, loss=self._loss(), config=_module_config()
        )
        module.train()
        batch = _labelled_training_batch()

        predictions = module(batch)
        assert "aux_loss" not in predictions
        assert "gate_scores" not in predictions

        losses = module.loss(predictions, batch)
        assert module._with_aux_loss(losses, predictions) is losses
