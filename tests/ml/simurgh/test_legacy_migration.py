"""Legacy SIMURGH namespace: deprecated registry aliases and checkpoint migration.

The older SIMURGH models moved to ``goal.ml.nn.models.simurgh.legacy`` under
``simurgh_legacy*`` keys.  Their previous keys must keep resolving (with a
``DeprecationWarning``), and a checkpoint saved under an old key must give
bit-identical predictions through the alias and after migration.
"""

from __future__ import annotations

import copy
import importlib.util
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import molecule
from omegaconf import OmegaConf

import goal.ml.nn.models  # noqa: F401  (populates the registries)
from goal.ml.nn.models.simurgh.legacy import DEPRECATED_KEYS
from goal.ml.registry import BACKBONE_REGISTRY, MODEL_REGISTRY
from goal.ml.utils.calculator import GOALCalculator
from tests.ml.simurgh.test_simurgh_simplified import SMALL, wake_corrections

_SCRIPT = Path(__file__).parents[3] / "scripts" / "simurgh_dev" / "migrate_legacy_checkpoints.py"
_spec = importlib.util.spec_from_file_location("migrate_legacy_checkpoints", _SCRIPT)
assert _spec is not None and _spec.loader is not None
migration = importlib.util.module_from_spec(_spec)
_ALIAS_WARNING = ".*deprecated .* registry name"
_spec.loader.exec_module(migration)


@pytest.mark.parametrize("old", sorted(DEPRECATED_KEYS))
def test_deprecated_key_resolves_to_legacy_class(old: str) -> None:
    new = DEPRECATED_KEYS[old]
    for registry in (MODEL_REGISTRY, BACKBONE_REGISTRY):
        assert old in registry
        assert registry.canonical(old) == new
        with pytest.warns(DeprecationWarning, match=new):
            cls = registry.get(old)
        assert cls is registry.get(new)


def test_new_keys_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("error", _ALIAS_WARNING, DeprecationWarning)
        for new in DEPRECATED_KEYS.values():
            MODEL_REGISTRY.get(new)


# ---------------------------------------------------------------------------
# A legacy SIMURGH Simplified checkpoint saved under its old key
# ---------------------------------------------------------------------------


def _simplified_checkpoint(path: Path) -> Path:
    from goal.ml.training.loss import (
        CompositeLoss,
        EnergyLoss,
        ForcesLoss,
        WeightedLoss,
    )
    from goal.ml.training.module import GOALModule

    backbone_kwargs = copy.deepcopy(SMALL)
    cfg = OmegaConf.create(
        {
            "model": {
                "backbone": {"name": "simurgh_simplified_monolithic", **backbone_kwargs},
                "head": None,
            },
            "training": {
                "losses": [
                    {"name": "energy", "weight": 1.0, "fn": "mse"},
                    {"name": "forces", "weight": 1.0, "fn": "mse"},
                ],
                "ema": {"enabled": False},
            },
            "data": {"cutoff": SMALL["cutoff"], "compute_fragment_index": True},
        }
    )
    torch.manual_seed(0)
    backbone = MODEL_REGISTRY.get("simurgh_legacy_simplified_monolithic")(**backbone_kwargs)
    wake_corrections(backbone.to(torch.float64))
    loss = CompositeLoss(
        [
            WeightedLoss(EnergyLoss(loss_fn="mse"), weight=1.0, label="energy"),
            WeightedLoss(ForcesLoss(loss_fn="mse"), weight=1.0, label="forces"),
        ]
    )
    module = GOALModule(backbone=backbone, head=None, loss=loss, config=cfg)
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": {"config": cfg}}, path)
    return path


def _predict(calc) -> tuple[float, np.ndarray]:
    atoms = molecule("CH3CH2OH")
    atoms.calc = calc
    return atoms.get_potential_energy(), atoms.get_forces()


def test_alias_and_migrated_checkpoint_are_bit_identical(tmp_path: Path) -> None:
    from goal.md.core.calculator_factory import CalculatorFactory

    old_ckpt = _simplified_checkpoint(tmp_path / "simplified.ckpt")
    with pytest.warns(DeprecationWarning):
        e_old, f_old = _predict(GOALCalculator(checkpoint_path=str(old_ckpt)))
    assert f_old.std() > 0  # the readouts are awake: a non-trivial model

    new_ckpt, n = migration.migrate(old_ckpt, apply=True)
    assert n == 1 and new_ckpt == tmp_path / "simplified.migrated.ckpt"
    raw = torch.load(new_ckpt, weights_only=False)
    assert raw["hyper_parameters"]["config"].model.backbone.name == (
        "simurgh_legacy_simplified_monolithic"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", _ALIAS_WARNING, DeprecationWarning)
        e_new, f_new = _predict(GOALCalculator(checkpoint_path=str(new_ckpt)))
    assert e_new == e_old
    np.testing.assert_array_equal(f_new, f_old)

    # GOAL's MD calculator factory loads the legacy checkpoint too.
    e_md, f_md = _predict(CalculatorFactory.create("goal_model", checkpoint=str(old_ckpt)))
    assert e_md == e_old
    np.testing.assert_array_equal(f_md, f_old)


def _managed_dir(root: Path) -> Path:
    """A checkpoint directory whose frozen registry only knows an old key."""
    root.mkdir()
    cfg = OmegaConf.create({"model": {"backbone": {"name": "nf_simurgh"}}})
    OmegaConf.save(cfg, root / "config.yaml")
    torch.save({"hyper_parameters": {"config": cfg}}, root / "last.ckpt")
    frozen = root / "frozen_source" / "goal" / "model.py"
    frozen.parent.mkdir(parents=True)
    frozen.write_text(
        '@MODEL_REGISTRY.register("nf_simurgh")\n'
        '@BACKBONE_REGISTRY.register("nf_simurgh")\n'
        "class A: ...\n"
        '@MODEL_REGISTRY.register("hyperspec")\n'
        "class B: ...\n"
    )
    return root


def test_directory_dry_run_changes_nothing(tmp_path: Path) -> None:
    d = _managed_dir(tmp_path / "ckpts")
    before = {p: p.read_bytes() for p in d.rglob("*") if p.is_file()}
    target, n = migration.migrate(d)
    assert target == d and n == 3  # last.ckpt, config.yaml, one frozen file
    assert before == {p: p.read_bytes() for p in d.rglob("*") if p.is_file()}
    assert not (tmp_path / "ckpts.migrated").exists()


def test_directory_in_place_rewrites_config_ckpt_and_frozen_registry(tmp_path: Path) -> None:
    d = _managed_dir(tmp_path / "ckpts")
    migration.migrate(d, apply=True, in_place=True)
    assert OmegaConf.load(d / "config.yaml").model.backbone.name == "simurgh_legacy_nf"
    raw = torch.load(d / "last.ckpt", weights_only=False)
    assert raw["hyper_parameters"]["config"].model.backbone.name == "simurgh_legacy_nf"
    frozen = (d / "frozen_source" / "goal" / "model.py").read_text()
    assert frozen.count('register("simurgh_legacy_nf")') == 2
    assert 'register("hyperspec")' in frozen  # current keys are untouched


def test_archive_is_migrated_to_a_new_file(tmp_path: Path) -> None:
    d = _managed_dir(tmp_path / "ckpts")
    archive = tmp_path / "model.simurgh"
    with zipfile.ZipFile(archive, "w") as zf:
        for f in d.rglob("*"):
            if f.is_file():
                arcname = "weights.pt" if f.name == "last.ckpt" else str(f.relative_to(d))
                zf.write(f, arcname)
    original = archive.read_bytes()
    target, n = migration.migrate(archive, apply=True)
    assert n == 3 and target == tmp_path / "model.migrated.simurgh"
    assert archive.read_bytes() == original
    with zipfile.ZipFile(target) as zf:
        assert b"simurgh_legacy_nf" in zf.read("config.yaml")
