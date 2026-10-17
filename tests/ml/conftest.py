"""ML-module test fixtures.

Inherits the global fixtures from ``tests/conftest.py`` and adds
ML-specific helpers.  The Hydra config path is adjusted for the new
``tests/ml/`` location (two levels up from the project root's ``configs/``
directory).
"""

import sys
from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

_root = rootutils.find_root(indicator=".project-root")
sys.path.insert(0, str(_root / "src"))


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """Default training config for ML tests."""
    with initialize(version_base="1.3", config_path="../../src/goal/configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])
        with open_dict(cfg):
            cfg.paths.root_dir = str(_root)
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None
    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """Default eval config for ML tests."""
    with initialize(version_base="1.3", config_path="../../src/goal/configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=["ckpt_path=."],
        )
        with open_dict(cfg):
            cfg.paths.root_dir = str(_root)
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None
    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """Per-test training config with temporary output paths."""
    cfg = cfg_train_global.copy()
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
    yield cfg
    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
    """Per-test eval config with temporary output paths."""
    cfg = cfg_eval_global.copy()
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
    yield cfg
    GlobalHydra.instance().clear()
