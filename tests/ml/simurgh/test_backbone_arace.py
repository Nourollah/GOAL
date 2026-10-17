"""Tests for the modular ARACE backbone (``simurgh_arace``).

Coverage (Part 7 of the ARACE-modularisation task):

* Test 1 — modular backbone forward pass on methane: finite scalar
  energy, forces shape, Newton's third law, non-zero per-round energies.
* Test 2 — GOALModule compatibility: one training step, finite loss,
  ``backward()`` runs.
* Test 3 — rotation equivariance: energy invariant, forces rotate.
* Test 4 — monolithic vs modular consistency: identical config +
  copied weights → identical energy and forces.
* Test 5 — weight sharing: parameter counts differ by exactly
  ``(num_rounds - 1) × params-per-artisan-bank``.
* Test 6 — the HPO config parses and every backbone search-space key
  exists in ``SimurghAraceBackbone.__init__`` and in the composed config.
"""

from __future__ import annotations

import inspect
import os
import warnings
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from goal.ml.nn.heads.energy_forces import EnergyForcesHead
from goal.ml.nn.models.simurgh.arace import MonolithicArace
from goal.ml.nn.models.simurgh.backbone_arace import SimurghAraceBackbone
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY
from tests.ml.simurgh.conftest import random_so3

CUTOFF = 5.0
NUM_ROUNDS = 2
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

_CONFIGS_ML_DIR = Path(__file__).parents[3] / "src" / "goal" / "configs" / "ml"


def _build_backbone(
    num_rounds: int = NUM_ROUNDS,
    share_artisan_weights: bool = False,
    **kwargs,
) -> SimurghAraceBackbone:
    return SimurghAraceBackbone(
        elements=(1, 6, 7, 8),
        num_rounds=num_rounds,
        share_artisan_weights=share_artisan_weights,
        artisan=dict(ARTISAN_SUBCONFIG),
        cutoff=CUTOFF,
        embedding_dim=16,
        num_elements=9,
        **kwargs,
    ).double()


def _build_head() -> EnergyForcesHead:
    return EnergyForcesHead(irreps_in=HIDDEN_IRREPS, hidden_dim=16).double()


def _run(backbone: SimurghAraceBackbone, batch: Batch) -> dict[str, torch.Tensor]:
    head = _build_head()
    head.eval()
    features = backbone(batch)
    return head(features, batch)


def _rotate_batch(batch: Batch, R: torch.Tensor) -> Batch:
    rotated = batch.clone()
    rotated.pos = batch.pos.detach() @ R.T
    if getattr(rotated, "edge_vectors", None) is not None:
        rotated.edge_vectors = batch.edge_vectors @ R.T
    return rotated


# ----------------------------------------------------------------------
# Test 1 — modular backbone forward pass
# ----------------------------------------------------------------------


class TestModularForwardPass:
    def test_methane_energy_forces_newton(self, methane_batch) -> None:  # noqa: ANN001
        backbone = _build_backbone(num_rounds=2)
        backbone.eval()
        out = _run(backbone, methane_batch)

        assert out["energy"].shape == (1,)
        assert torch.isfinite(out["energy"]).item(), "Energy is not finite"

        assert out["forces"].shape == methane_batch.pos.shape
        assert torch.isfinite(out["forces"]).all().item()
        assert (out["forces"].abs() > 0).any().item(), "Forces are identically zero"

        net_force = out["forces"].sum(dim=0).norm().item()
        assert net_force < 1e-5, f"Newton violated: ‖Σ F‖ = {net_force:.3e}"

    def test_per_round_energies_nonzero(self, methane_batch) -> None:  # noqa: ANN001
        backbone = _build_backbone(num_rounds=2)
        backbone.eval()
        _run(backbone, methane_batch)

        round_e = backbone.last_round_energies  # (L, B), detached
        assert round_e is not None and round_e.shape == (2, 1)
        for round_idx in range(round_e.shape[0]):
            contribution = round_e[round_idx].abs().sum().item()
            assert contribution > 1e-12, (
                f"Round {round_idx} contributes no energy " f"(|E_L| = {contribution:.3e})"
            )

    def test_registered_in_registries(self) -> None:
        assert BACKBONE_REGISTRY.get("simurgh_arace") is SimurghAraceBackbone
        assert MODEL_REGISTRY.get("simurgh_arace") is SimurghAraceBackbone
        # Legacy ACE-first backbone stays reachable under both names.
        assert BACKBONE_REGISTRY.get("simurgh") is BACKBONE_REGISTRY.get("simurgh_ace_first")

    def test_gates_keyed_by_round_and_pair(self) -> None:
        backbone = _build_backbone(num_rounds=2)
        gates = backbone.gates()
        assert "round0/H-C" in gates
        assert "round1/O-O" in gates
        assert len(gates) == 2 * 10  # 2 rounds × 10 pairs of {H,C,N,O}

    def test_avg_num_neighbors_settable(self) -> None:
        backbone = _build_backbone(num_rounds=2)
        assert backbone.avg_num_neighbors is None
        backbone.avg_num_neighbors = 4.0
        assert backbone.avg_num_neighbors == 4.0
        for rnd in backbone.rounds:
            assert rnd.ace_block.agg_norm_scale.item() == pytest.approx(0.25)


# ----------------------------------------------------------------------
# Test 2 — GOALModule compatibility
# ----------------------------------------------------------------------


class TestGOALModuleCompatibility:
    def test_training_step_and_backward(self, methane_batch) -> None:  # noqa: ANN001
        from goal.ml.training.loss import (
            CompositeLoss,
            EnergyLoss,
            ForcesLoss,
            WeightedLoss,
        )
        from goal.ml.training.module import GOALModule

        backbone = _build_backbone(num_rounds=2)
        head = _build_head()
        loss = CompositeLoss(
            [
                WeightedLoss(EnergyLoss(), weight=1.0, label="energy"),
                WeightedLoss(ForcesLoss(), weight=1.0, label="forces"),
            ]
        )
        cfg = OmegaConf.create(
            {
                "training": {
                    "ema": {"enabled": False},
                    "gradient_clip": 0.0,
                    "optimizer": {"lr": 1.0e-3},
                }
            }
        )
        module = GOALModule(backbone=backbone, head=head, loss=loss, config=cfg)

        # Attach energy/forces labels to the batch.
        batch = methane_batch
        batch.energy = torch.tensor([-17.0], dtype=torch.float64)
        batch.forces = torch.zeros_like(batch.pos)

        with warnings.catch_warnings():
            # self.log() outside a Trainer emits a benign warning.
            warnings.simplefilter("ignore")
            total = module.training_step(batch, 0)

        assert torch.isfinite(total).item(), "Training loss is not finite"
        total.backward()  # must not raise
        n_with_grad = sum(1 for p in module.parameters() if p.requires_grad and p.grad is not None)
        assert n_with_grad > 0


# ----------------------------------------------------------------------
# Test 3 — rotation equivariance
# ----------------------------------------------------------------------


class TestRotationEquivariance:
    def test_energy_invariant_forces_equivariant(self, methylamine_batch) -> None:  # noqa: ANN001
        # Methylamine rather than methane: perfectly tetrahedral methane
        # sits at a high-symmetry point that is numerically pathological
        # for norm layers acting on symmetry-suppressed l>0 blocks.
        backbone = _build_backbone(num_rounds=2)
        backbone.eval()

        out0 = _run(backbone, methylamine_batch)
        e0, f0 = out0["energy"], out0["forces"]

        R = random_so3()
        rotated = _rotate_batch(methylamine_batch, R)
        out1 = _run(backbone, rotated)
        e1, f1 = out1["energy"], out1["forces"]

        e_diff = (e0 - e1).abs().max().item()
        assert e_diff < 1e-5, f"Energy not invariant: {e_diff:.3e}"

        f_diff = (f1.detach() - f0.detach() @ R.T).abs().max().item()
        assert f_diff < 1e-4, f"Forces not equivariant: {f_diff:.3e}"


# ----------------------------------------------------------------------
# Test 4 — monolithic vs modular consistency
# ----------------------------------------------------------------------


def _monolithic_to_modular_state_dict(
    mono_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rename MonolithicArace parameter keys to SimurghAraceBackbone keys.

    The two models are architecturally identical; only the module tree
    differs (``backbone.blocks.{i}.artisans`` vs
    ``rounds.{i}.artisan_layer.artisans`` etc.).
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in mono_sd.items():
        new = key
        if new.startswith("backbone."):
            new = new[len("backbone.") :]
        new = new.replace("blocks.", "rounds.")
        if ".artisans." in new:
            new = new.replace(".artisans.", ".artisan_layer.artisans.")
        new = new.replace(".ace_linear.", ".ace_block.linear.")
        new = new.replace(".ace_norm.", ".ace_block.norm.")
        new = new.replace(".agg_norm_scale", ".ace_block.agg_norm_scale")
        out[new] = value
    return out


class TestMonolithicModularConsistency:
    def test_energy_and_forces_match(self, methane_batch) -> None:  # noqa: ANN001
        mono = MonolithicArace(
            elements=(1, 6, 7, 8),
            num_rounds=NUM_ROUNDS,
            share_artisan_weights=False,
            artisan=dict(ARTISAN_SUBCONFIG),
            cutoff=CUTOFF,
            embedding_dim=16,
            num_elements=9,
        ).double()
        modular = _build_backbone(num_rounds=NUM_ROUNDS)

        mapped = _monolithic_to_modular_state_dict(mono.state_dict())
        result = modular.load_state_dict(mapped, strict=True)
        assert not result.missing_keys and not result.unexpected_keys

        mono.eval()
        modular.eval()

        out_mono = mono(methane_batch.clone())
        out_mod = _run(modular, methane_batch.clone())

        e_diff = (out_mono["energy"] - out_mod["energy"]).abs().max().item()
        assert e_diff < 1e-5, f"Energies differ: {e_diff:.3e}"

        f_diff = (out_mono["forces"].detach() - out_mod["forces"].detach()).abs().max().item()
        assert f_diff < 1e-5, f"Forces differ: {f_diff:.3e}"


# ----------------------------------------------------------------------
# Test 5 — weight sharing
# ----------------------------------------------------------------------


class TestWeightSharing:
    def test_parameter_counts(self) -> None:
        num_rounds = 3
        shared = _build_backbone(num_rounds=num_rounds, share_artisan_weights=True)
        independent = _build_backbone(num_rounds=num_rounds, share_artisan_weights=False)

        def n_params(module: torch.nn.Module) -> int:
            # named_parameters deduplicates shared tensors by identity.
            return sum(p.numel() for _, p in module.named_parameters())

        # All artisans share one config → every bank has the same size.
        first_bank = independent.rounds[0].artisan_layer.artisans
        n_bank_params = n_params(first_bank)

        diff = n_params(independent) - n_params(shared)
        expected = (num_rounds - 1) * n_bank_params
        assert diff == expected, (
            f"Parameter-count difference {diff} != "
            f"(num_rounds - 1) × n_bank_params = {expected}"
        )

    def test_shared_run_works(self, methane_batch) -> None:  # noqa: ANN001
        backbone = _build_backbone(num_rounds=2, share_artisan_weights=True)
        backbone.eval()
        out = _run(backbone, methane_batch)
        assert torch.isfinite(out["energy"]).item()
        assert torch.isfinite(out["forces"]).all().item()


# ----------------------------------------------------------------------
# Test 6 — HPO config validity
# ----------------------------------------------------------------------


class TestHpoConfig:
    def test_hpo_config_parses_and_keys_exist(self) -> None:
        from hydra import compose, initialize_config_dir

        os.environ.setdefault("PROJECT_ROOT", str(_CONFIGS_ML_DIR.parents[1]))
        with initialize_config_dir(config_dir=str(_CONFIGS_ML_DIR), version_base=None):
            cfg = compose(config_name="simurgh_gmd26_hpo")

        assert cfg.model.backbone.name == "simurgh_arace"
        assert cfg.hparams_search.method == "ray"

        sig = inspect.signature(SimurghAraceBackbone.__init__)
        for key in cfg.hparams_search.search_space:
            spec = cfg.hparams_search.search_space[key]
            assert "type" in spec, f"{key}: missing search-space 'type'"
            if not key.startswith("model.backbone."):
                # training.* keys — just check they exist in the config.
                assert OmegaConf.select(cfg, key) is not None, f"{key} not in config"
                continue
            sub = key[len("model.backbone.") :]
            top_level = sub.split(".")[0]
            assert top_level in sig.parameters, (
                f"Search-space key '{key}' has no matching "
                f"SimurghAraceBackbone.__init__ parameter '{top_level}'"
            )
            assert (
                OmegaConf.select(cfg, f"model.backbone.{sub}") is not None
            ), f"'{sub}' missing from the composed model.backbone config"

    def test_default_config_uses_arace(self) -> None:
        from hydra import compose, initialize_config_dir

        os.environ.setdefault("PROJECT_ROOT", str(_CONFIGS_ML_DIR.parents[1]))
        with initialize_config_dir(config_dir=str(_CONFIGS_ML_DIR), version_base=None):
            cfg = compose(config_name="simurgh_gmd26")
            legacy = compose(config_name="simurgh_ace_first_gmd26")
        assert cfg.model.backbone.name == "simurgh_arace"
        assert legacy.model.backbone.name in ("simurgh", "simurgh_ace_first")
