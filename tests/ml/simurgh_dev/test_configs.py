"""Every SIMURGH task config composes and builds its model."""

from __future__ import annotations

import glob
import os

import hydra.utils
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from goal.ml.cli import CONFIGS_ML_DIR
from goal.ml.nn.models.simurgh.dev import SimurghDev

TASKS = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{CONFIGS_ML_DIR}/simurgh_dev_*.yaml"))


#: What goal-train injects for models that request per-atom statistics.
STATS = {"pan_rho": {"mean": 3.0, "std": 1.5}, "pan_eta": {"mean": 0.2, "std": 0.1}}


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
    backbone.update(avg_num_neighbors=10.0, scale=1.0, num_channels=8, atom_statistics=STATS)
    if backbone["atomic_energies"]["mode"] == "dataset":
        backbone["atomic_energies"]["values"] = {1: -1.0, 6: -2.0, 7: -3.0, 8: -4.0}
    backbone["message_irreps"] = "8x0e + 8x1o + 8x2e"
    SimurghDev(**backbone)
    stages = [s.name for s in cfg.training.stages]
    assert stages == ["warmup", "main"] and cfg.training.stages[0].epochs == 5
    # forces on in the warm-up (at weight 0 the energy surface runs away to NaN), and
    # the same force weight in every stage (a jump of it diverges under AMSGrad)
    force_weights = {s.loss_weights.forces for s in cfg.training.stages}
    assert len(force_weights) == 1 and force_weights.pop() > 0
    assert dict(cfg.training.safeguards) == {
        "lr_warmup_steps": 1000,
        "weight_ramp_steps": 1000,
        "skip_nonfinite": True,
    }
    monitor = hydra.utils.instantiate(cfg.callbacks.blowup_monitor)
    assert monitor.pattern.search("atom_line.layers.1.product")
    losses = {loss.name: loss for loss in cfg.training.losses}
    assert losses["forces"].error == "norm" and losses["forces"].fn == "mse"
    assert losses["energy"].fn == "mse"
    # exact dataset-level MAE, and never on OOD
    assert cfg.checkpoint_manager.top_k.metric == "val_forces/forces_mae_atomw"
    # MFC: on in the GMD configs, off on the single-system benchmarks
    assert cfg.model.backbone.mfc.enabled == ("gmd26" in name)


GMD_E0 = {1: -13.595298, 6: -1028.481145, 7: -1483.895483, 8: -2041.117735}
BPA_E0 = {
    1: -13.587222780835477,
    6: -1029.4889999855063,
    7: -1484.9814568572233,
    8: -2041.9816003861047,
}


@pytest.mark.parametrize("name", TASKS)
def test_element_references_match_the_labels(name: str) -> None:
    """GMD's isolated atoms only on GMD; 3BPA's shipped atoms; least squares on rMD17 and MD22."""
    ae = OmegaConf.to_container(_compose(name).model.backbone.atomic_energies)
    if "gmd26" in name:
        assert ae == {"mode": "provided", "values": GMD_E0}
    elif name == "simurgh_dev_3bpa":
        assert ae == {"mode": "provided", "values": BPA_E0}
    else:
        assert ae == {"mode": "dataset", "values": None}


def test_least_squares_references_on_one_composition() -> None:
    """A single-molecule training set has a rank-1 composition matrix: no warning, no NaN."""
    import warnings

    import torch

    from goal.ml.data.statistics import compute_atomic_references

    class Frame:
        def __init__(self, energy: float) -> None:
            self.atomic_numbers = torch.tensor([6] * 9 + [1] * 8 + [8] * 4)  # aspirin
            self.energy = torch.tensor(energy, dtype=torch.float64)

    energies = -17630.0 + torch.rand(
        50, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        refs = compute_atomic_references([Frame(float(e)) for e in energies])
    assert sorted(refs) == [1, 6, 8] and all(torch.isfinite(torch.tensor(list(refs.values()))))
    # the per-molecule baseline is the mean energy: only a constant is removed
    assert 9 * refs[6] + 8 * refs[1] + 4 * refs[8] == pytest.approx(
        float(energies.mean()), abs=1e-8
    )


GMD_TASKS = {
    "FD": "fragment_duplication",
    "FF": "fragment_fusion",
    "FCE": "fragment_chain_extension",
    "FCB": "fragment_combination",
    "FD_plus_alkane": "fragment_duplication",
}


@pytest.mark.parametrize("name", [t for t in TASKS if "gmd26" in t])
def test_gmd_configs_download_the_task_from_the_hub(name: str) -> None:
    """GMD comes from the Hugging Face Hub: validate on TestID, test on the OOD molecules."""
    data = _compose(name).data
    assert data.dataset_type == "gmd" and data.get("train_dir") is None
    assert data.get("task", "fragment_duplication") == GMD_TASKS[name.split("gmd26_")[1]]
    assert data.get("val_split", "test_id") == "test_id"
    if name.endswith("plus_alkane"):
        assert list(data.extra_train_molecules) == ["Alkanes/C8H18"]


ARMS = sorted(
    "ablations/simurgh_dev/" + os.path.basename(p)[:-5]
    for p in glob.glob(f"{CONFIGS_ML_DIR}/ablations/simurgh_dev/s*.yaml")
)


@pytest.mark.parametrize("name", ARMS)
def test_ablation_arm_builds_its_model(name: str) -> None:
    import goal.ml.nn.models  # noqa: F401
    from goal.ml.registry import MODEL_REGISTRY

    cfg = _compose(name)
    assert cfg.task_name.startswith("simurgh_dev_abl_s") and cfg.data.dataset_type == "gmd"
    backbone = OmegaConf.to_container(cfg.model.backbone, resolve=True)
    name_key = backbone.pop("name")
    channels = 8
    backbone.update(
        avg_num_neighbors=10.0, scale=1.0, num_channels=channels, atom_statistics=STATS
    )
    backbone["message_irreps"] = backbone["message_irreps"].replace("128x", f"{channels}x")
    MODEL_REGISTRY.get(name_key)(**backbone)


@pytest.mark.parametrize("name", TASKS)
def test_dataset_accepts_every_forwarded_data_key(name: str) -> None:
    """The datamodule forwards non-meta data keys to the dataset class."""
    import inspect

    import goal.ml.data.datasets  # noqa: F401  (registers the datasets)
    from goal.ml.data.datamodule import _FRAGMENT_KEYS, GOALDataModule
    from goal.ml.registry import DATASET_REGISTRY

    cfg = _compose(name)
    cls = DATASET_REGISTRY.get(cfg.data.dataset_type)
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        accepted = None
    else:
        accepted = set(params)
    forwarded = {
        k for k in cfg.data if k not in GOALDataModule._META_KEYS and k not in _FRAGMENT_KEYS
    }
    forwarded -= {"dataset_type", "root"}
    if accepted is not None:
        assert forwarded <= accepted, sorted(forwarded - accepted)
    else:  # **kwargs datasets: the benchmark loaders take named arguments only
        known = set(params) - {"self", "kwargs", "fragment_kwargs"}
        assert forwarded <= known, sorted(forwarded - known)


#: The screening arms the combined base switches on (MSE is the default loss now).
COMBINED_ARMS = (
    "s06_loss_mse",
    "s09_mfc_violation_weighting",
    "s09_pgs_beta_9.2",
    "s05_radial_env_conditioned",
    "s02_correlation5_ladder_reduced",
    "s06_weight_decay_1e-3",
)


def _changed(cfg, base: dict) -> dict:
    def leaves(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict) and v:
                yield from leaves(v, f"{prefix}{k}.")
            else:
                yield f"{prefix}{k}", v

    out = dict(leaves(OmegaConf.to_container(cfg, resolve=False)))
    return {
        k: v
        for k, v in out.items()
        if base.get(k) != v and k.split(".")[0] not in ("task_name", "experiment")
    }


def test_combined_base_is_the_screening_arms_together() -> None:
    import goal.ml.nn.models  # noqa: F401
    from goal.ml.registry import MODEL_REGISTRY

    base = _changed(_compose("simurgh_dev_gmd26_FD"), {})
    want = {"model.backbone.mfc.weight": 0.1}
    for arm in COMBINED_ARMS:
        want.update(_changed(_compose(f"ablations/simurgh_dev/{arm}"), base))
    cfg = _compose("ablations/simurgh_dev/combined_base")
    assert _changed(cfg, base) == want
    backbone = OmegaConf.to_container(cfg.model.backbone, resolve=True)
    backbone.pop("name")
    backbone.update(avg_num_neighbors=10.0, scale=1.0, num_channels=8, atom_statistics=STATS)
    backbone["message_irreps"] = backbone["message_irreps"].replace("128x", "8x")
    MODEL_REGISTRY.get(cfg.model.backbone.name)(**backbone)
