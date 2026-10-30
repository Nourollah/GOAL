"""Every SIMURGH task config composes and builds its model."""

from __future__ import annotations

import glob
import os

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from goal.ml.cli import CONFIGS_ML_DIR
from goal.ml.nn.models.simurgh.dev import SimurghDev

TASKS = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{CONFIGS_ML_DIR}/simurgh_dev_*.yaml"))


def _compose(name: str):
    with initialize_config_dir(config_dir=CONFIGS_ML_DIR, version_base=None):
        return compose(config_name=name, overrides=["paths.root_dir=/root"])


def test_every_benchmark_has_a_config() -> None:
    gmd = {f"simurgh_dev_gmd26_{t}" for t in ("FD", "FF", "FCE", "FCB", "FD_plus_alkane")}
    assert gmd <= set(TASKS)
    assert sum(t.startswith("simurgh_dev_rmd17_") for t in TASKS) == 10
    assert sum(t.startswith("simurgh_dev_md22_") for t in TASKS) == 7
    assert "simurgh_dev_3bpa" in TASKS


@pytest.mark.parametrize("name", TASKS)
def test_config_builds_the_model(name: str) -> None:
    cfg = _compose(name)
    assert cfg.task_name == name and cfg.model.backbone.name == "simurgh_dev"
    assert cfg.model.head is None
    assert cfg.model.backbone.cutoff == cfg.data.cutoff
    backbone = OmegaConf.to_container(cfg.model.backbone, resolve=True)
    backbone.pop("name")
    # what goal-train injects from the training set, at a small width
    backbone.update(avg_num_neighbors=10.0, scale=1.0, num_channels=8)
    backbone["message_irreps"] = "8x0e + 8x1o + 8x2e"
    SimurghDev(**backbone)
    stages = [s.name for s in cfg.training.stages]
    assert stages == ["energy_only", "main"] and cfg.training.stages[0].epochs == 5
    losses = {loss.name: loss for loss in cfg.training.losses}
    assert losses["forces"].error == "norm" and losses["forces"].fn == "huber"
    assert cfg.checkpoint_manager.top_k.metric.startswith("val_")  # never select on OOD
    # MFC: on in the GMD configs, off on the single-system benchmarks
    assert cfg.model.backbone.mfc.enabled == ("gmd26" in name)


@pytest.mark.parametrize("name", [t for t in TASKS if "gmd26" in t])
def test_gmd_configs_use_directory_or_split_paths(name: str) -> None:
    data = _compose(name).data
    if name.endswith("plus_alkane"):
        assert data.train_dir is None and len(data.train_paths) == 8
    else:
        assert data.val_dir.endswith("/TestID") and data.test_dir.endswith("/TestOOD")


ARMS = sorted(
    "ablations/simurgh_dev/" + os.path.basename(p)[:-5]
    for p in glob.glob(f"{CONFIGS_ML_DIR}/ablations/simurgh_dev/s*.yaml")
)


@pytest.mark.parametrize("name", ARMS)
def test_ablation_arm_builds_its_model(name: str) -> None:
    import goal.ml.nn.models  # noqa: F401
    from goal.ml.registry import MODEL_REGISTRY

    cfg = _compose(name)
    assert cfg.task_name.startswith("simurgh_dev_abl_s") and cfg.data.val_dir.endswith("TestID")
    backbone = OmegaConf.to_container(cfg.model.backbone, resolve=True)
    name_key = backbone.pop("name")
    channels = 8
    backbone.update(avg_num_neighbors=10.0, scale=1.0, num_channels=channels)
    backbone["message_irreps"] = backbone["message_irreps"].replace("128x", f"{channels}x")
    MODEL_REGISTRY.get(name_key)(**backbone)
