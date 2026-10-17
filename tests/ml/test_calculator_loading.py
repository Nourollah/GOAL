"""GOALCalculator checkpoint loading: EMA weights, dtype, fragment labels.

* A checkpoint written with ``training.ema.enabled`` carries the EMA
  shadow under ``goal_ema``.  Validation and best-checkpoint selection
  ran on those weights, so inference must load them too.
* The rebuilt module takes the dtype the weights were saved in (a float64
  run used to be evaluated in float32).
* A model trained with ``data.compute_fragment_index`` needs every
  inference graph to carry ``fragment_index``; the options come from the
  saved config.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from ase.build import molecule
from omegaconf import DictConfig, OmegaConf

from goal.ml.utils.calculator import (
    GOALCalculator,
    apply_ema_weights,
    fragment_options_from_config,
    load_goal_checkpoint,
)
from tests.ml.test_archive import _build_module, _tiny_config


def _ema_config() -> DictConfig:
    cfg = _tiny_config()
    cfg.training.ema = {"enabled": True, "decay": 0.99}
    return cfg


def _save_ckpt(path: Path, module, cfg: DictConfig, ema_shadow=None) -> Path:
    payload = {
        "state_dict": module.state_dict(),
        "hyper_parameters": {"config": cfg},
    }
    if ema_shadow is not None:
        payload["goal_ema"] = {"shadow": ema_shadow}
    torch.save(payload, path)
    return path


def _shifted_shadow(module, delta: float = 0.25) -> list[torch.Tensor]:
    return [p.detach().clone() + delta for p in module.parameters()]


class TestEMAWeights:
    def test_ema_shadow_is_loaded_when_present(self, tmp_path: Path) -> None:
        cfg = _ema_config()
        module = _build_module(cfg)
        shadow = _shifted_shadow(module)
        ckpt = _save_ckpt(tmp_path / "ema.ckpt", module, cfg, ema_shadow=shadow)

        calc = GOALCalculator(checkpoint_path=str(ckpt))

        assert calc.weights == "ema"
        for loaded, expected in zip(calc.module.parameters(), shadow):
            torch.testing.assert_close(loaded.detach(), expected.to(loaded.dtype))

    def test_use_ema_false_keeps_raw_weights(self, tmp_path: Path) -> None:
        cfg = _ema_config()
        module = _build_module(cfg)
        raw = [p.detach().clone() for p in module.parameters()]
        ckpt = _save_ckpt(tmp_path / "ema.ckpt", module, cfg, ema_shadow=_shifted_shadow(module))

        calc = GOALCalculator(checkpoint_path=str(ckpt), use_ema=False)

        assert calc.weights == "raw"
        for loaded, expected in zip(calc.module.parameters(), raw):
            torch.testing.assert_close(loaded.detach(), expected.to(loaded.dtype))

    def test_ema_changes_the_prediction(self, tmp_path: Path) -> None:
        """Not just bookkeeping: EMA and raw weights give different energies."""
        cfg = _ema_config()
        module = _build_module(cfg)
        ckpt = _save_ckpt(tmp_path / "ema.ckpt", module, cfg, ema_shadow=_shifted_shadow(module))

        atoms = molecule("CH4")
        atoms.calc = GOALCalculator(checkpoint_path=str(ckpt))
        e_ema = atoms.get_potential_energy()
        atoms.calc = GOALCalculator(checkpoint_path=str(ckpt), use_ema=False)
        e_raw = atoms.get_potential_energy()
        assert e_ema != pytest.approx(e_raw)

    def test_checkpoint_without_ema_loads_raw(self, tmp_path: Path) -> None:
        cfg = _tiny_config()
        module = _build_module(cfg)
        ckpt = _save_ckpt(tmp_path / "plain.ckpt", module, cfg)
        assert GOALCalculator(checkpoint_path=str(ckpt)).weights == "raw"

    def test_mismatched_shadow_falls_back_to_raw(self, tmp_path: Path, caplog) -> None:
        cfg = _ema_config()
        module = _build_module(cfg)
        shadow = _shifted_shadow(module)[:-1]  # one tensor short
        ckpt = _save_ckpt(tmp_path / "bad.ckpt", module, cfg, ema_shadow=shadow)

        with caplog.at_level(logging.WARNING, logger="goal.ml.utils.calculator"):
            calc = GOALCalculator(checkpoint_path=str(ckpt))
        assert calc.weights == "raw"
        assert "does not match" in caplog.text

    def test_module_ema_is_reseeded_from_loaded_weights(self, tmp_path: Path) -> None:
        """``ema.average_parameters()`` (validation / test) must not swap in the
        random initial weights the shadow was cloned from at construction."""
        cfg = _ema_config()
        module = _build_module(cfg)
        shadow = _shifted_shadow(module)
        ckpt = _save_ckpt(tmp_path / "ema.ckpt", module, cfg, ema_shadow=shadow)

        loaded = load_goal_checkpoint(ckpt).module
        before = [p.detach().clone() for p in loaded.parameters()]
        with loaded.ema.average_parameters():
            inside = [p.detach().clone() for p in loaded.parameters()]
        for a, b in zip(before, inside):
            torch.testing.assert_close(a, b)

    def test_apply_ema_weights_without_entry_is_a_noop(self) -> None:
        module = _build_module(_ema_config())
        before = [p.detach().clone() for p in module.parameters()]
        assert apply_ema_weights(module, {}) is False
        for a, b in zip(before, module.parameters()):
            torch.testing.assert_close(a, b.detach())


class TestDtype:
    def test_float64_checkpoint_loads_as_float64(self, tmp_path: Path) -> None:
        cfg = _tiny_config()
        module = _build_module(cfg)  # built in float64
        ckpt = _save_ckpt(tmp_path / "f64.ckpt", module, cfg)

        calc = GOALCalculator(checkpoint_path=str(ckpt))
        assert next(calc.module.parameters()).dtype == torch.float64
        assert calc.dtype == torch.float64


class TestFragmentOptions:
    def test_no_fragment_labels_by_default(self) -> None:
        assert fragment_options_from_config(_tiny_config()) == {}

    def test_options_read_from_data_section(self) -> None:
        cfg = OmegaConf.create(
            {
                "data": {
                    "compute_fragment_index": True,
                    "fragment_scheme": "custom",
                    "fragment_smarts": ["[C:1](=O)-[O:2]"],
                    "fragment_covalent_cutoff": 1.7,
                    "fragment_on_failure": "raise",
                }
            }
        )
        options = fragment_options_from_config(cfg)
        assert options == {
            "compute_fragment_index": True,
            "fragment_scheme": "custom",
            "fragment_smarts": ["[C:1](=O)-[O:2]"],
            "fragment_covalent_cutoff": 1.7,
            "fragment_on_failure": "raise",
        }
        assert isinstance(options["fragment_smarts"], list)

    def _fragment_calc(self, tmp_path: Path, **kwargs) -> GOALCalculator:
        cfg = _tiny_config()
        cfg.data.compute_fragment_index = True
        cfg.data.fragment_scheme = "connected"
        cfg.data.fragment_on_failure = "raise"
        module = _build_module(cfg)
        ckpt = _save_ckpt(tmp_path / "frag.ckpt", module, cfg)
        return GOALCalculator(checkpoint_path=str(ckpt), **kwargs)

    def test_calculator_builds_graphs_with_fragment_index(self, tmp_path: Path) -> None:
        calc = self._fragment_calc(tmp_path)
        assert calc.fragment_options["compute_fragment_index"] is True
        graph = calc.build_graph(molecule("CH4"))
        assert graph.get("fragment_index") is not None
        assert graph.fragment_index.shape == (5,)

    def test_inference_defaults_to_fallback_on_perception_failure(self, tmp_path: Path) -> None:
        assert self._fragment_calc(tmp_path).fragment_options["fragment_on_failure"] == "fallback"
        kept = self._fragment_calc(tmp_path, fragment_on_failure=None)
        assert kept.fragment_options["fragment_on_failure"] == "raise"

    def test_explicit_empty_options_disable_labels(self, tmp_path: Path) -> None:
        calc = self._fragment_calc(tmp_path, fragment_options={})
        assert calc.build_graph(molecule("CH4")).get("fragment_index") is None

    def test_module_path_reads_options_from_module_config(self) -> None:
        cfg = _tiny_config()
        cfg.data.compute_fragment_index = True
        calc = GOALCalculator(module=_build_module(cfg), cutoff=5.0)
        assert calc.fragment_options["compute_fragment_index"] is True


class TestFragmentAwareModelEndToEnd:
    """An ARACE checkpoint with ``fragment_interaction`` used to die in
    ``calculate()`` with "graphs carry no 'fragment_index'"."""

    def test_fragment_interaction_checkpoint_computes(self, tmp_path: Path) -> None:
        from goal.ml.nn.heads.energy_forces import EnergyForcesHead
        from goal.ml.nn.models.simurgh.backbone_arace import SimurghAraceBackbone
        from goal.ml.training.loss import CompositeLoss
        from goal.ml.training.module import GOALModule

        backbone_cfg = {
            "name": "simurgh_arace",
            "elements": [1, 6, 8],
            "num_rounds": 1,
            "cutoff": 5.0,
            "embedding_dim": 8,
            "num_elements": 9,
            "avg_num_neighbors": 4.0,
            "artisan": {
                "architecture": "equivariant",
                "hidden_irreps": "4x0e + 4x1o",
                "num_rbf": 4,
                "radial_hidden": 8,
                "n_scalar_out": 4,
                "final_hidden": 4,
                "n_elements": "auto",
            },
            "fragment_interaction": {"num_rbf": 4, "radial_hidden": 8, "init_zero": False},
        }
        cfg = OmegaConf.create(
            {
                "model": {
                    "backbone": backbone_cfg,
                    "head": {"name": "energy_forces", "irreps_in": "4x0e+4x1o", "hidden_dim": 8},
                },
                "training": {"losses": [], "ema": {"enabled": False}},
                "data": {
                    "cutoff": 5.0,
                    "compute_fragment_index": True,
                    "fragment_scheme": "connected",
                },
            }
        )
        kwargs = {
            k: v for k, v in OmegaConf.to_container(cfg.model.backbone).items() if k != "name"
        }
        torch.manual_seed(0)
        module = GOALModule(
            backbone=SimurghAraceBackbone(**kwargs).double(),
            head=EnergyForcesHead(irreps_in="4x0e+4x1o", hidden_dim=8).double(),
            loss=CompositeLoss([]),
            config=cfg,
        )
        ckpt = _save_ckpt(tmp_path / "frag_arace.ckpt", module, cfg)

        atoms = molecule("CH3COOH")
        atoms.calc = GOALCalculator(checkpoint_path=str(ckpt))
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        assert torch.isfinite(torch.tensor(energy))
        assert forces.shape == (len(atoms), 3)
        assert abs(forces).sum() > 0
