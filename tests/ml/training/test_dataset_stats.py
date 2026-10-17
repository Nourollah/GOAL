"""Dataset-derived backbone statistics are recorded and reused on resume.

``goal-train`` derives elements, atomic reference energies, scale,
avg_num_neighbors and pair counts from the training set.  These tests pin
down that they are written into the run config (``dataset_stats``), read
back from a checkpoint's ``config.yaml`` or hyper-parameters, applied before
anything is recomputed, and that a recomputed buffer value survives the
checkpoint restore.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from goal.ml.training import dataset_stats as ds
from goal.ml.training.module import GOALModule

# ----------------------------------------------------------------------
# recompute parsing
# ----------------------------------------------------------------------


class TestNormaliseRecompute:
    def test_empty_values_mean_nothing(self) -> None:
        assert ds.normalise_recompute(None) == set()
        assert ds.normalise_recompute([]) == set()
        assert ds.normalise_recompute(False) == set()

    def test_all_means_every_statistic(self) -> None:
        assert ds.normalise_recompute("all") == set(ds.STAT_PATHS)
        assert ds.normalise_recompute(True) == set(ds.STAT_PATHS)
        assert ds.normalise_recompute(["scale", "all"]) == set(ds.STAT_PATHS)

    def test_names_and_listconfig(self) -> None:
        assert ds.normalise_recompute("scale") == {"scale"}
        listcfg = OmegaConf.create({"r": ["scale", "avg_num_neighbors"]}).r
        assert ds.normalise_recompute(listcfg) == {"scale", "avg_num_neighbors"}

    def test_unknown_name_is_rejected_with_the_valid_ones(self) -> None:
        with pytest.raises(ValueError, match="avg_num_neighbours"):
            ds.normalise_recompute(["avg_num_neighbours"])


# ----------------------------------------------------------------------
# collect / apply
# ----------------------------------------------------------------------


def _backbone() -> dict:
    return {
        "name": "simurgh_arace",
        "elements": [1, 6, 8],
        "scale": 0.25,
        "avg_num_neighbors": 17.5,
        "atomic_energies": {"mode": "dataset", "values": {1: -13.6, 6: -1028.5, 8: -2041.1}},
        "artisan": {"rare_artisan": {"enabled": True}, "pair_counts": {"1_6": 10, "6_8": 3}},
        "num_rounds": 2,
    }


class TestCollectApply:
    def test_collect_finds_every_derived_value_and_nothing_else(self) -> None:
        found = ds.collect(_backbone())
        assert found == {
            "elements": [1, 6, 8],
            "scale": 0.25,
            "avg_num_neighbors": 17.5,
            "atomic_energies.values": {1: -13.6, 6: -1028.5, 8: -2041.1},
            "atomic_energies.mode": "dataset",
            "artisan.pair_counts": {"1_6": 10, "6_8": 3},
        }

    def test_mode_alone_is_a_user_setting_not_a_statistic(self) -> None:
        found = ds.collect({"atomic_energies": {"mode": "learned", "values": None}})
        assert found == {}

    def test_dressing_kwargs_location_is_collected(self) -> None:
        found = ds.collect({"dressing_kwargs": {"avg_num_neighbors": 9.0, "lmax": 2}})
        assert found == {"dressing_kwargs.avg_num_neighbors": 9.0}

    def test_apply_round_trips_into_a_fresh_backbone(self) -> None:
        fresh = {
            "elements": None,
            "scale": None,
            "avg_num_neighbors": None,
            "atomic_energies": OmegaConf.create(
                {"mode": "learned", "values": None, "trainable": True}
            ),
            "artisan": OmegaConf.create({"rare_artisan": {"enabled": True}, "pair_counts": None}),
        }
        applied = ds.apply(fresh, ds.collect(_backbone()), skip=set())
        assert set(applied) == {
            "elements",
            "scale",
            "avg_num_neighbors",
            "atomic_energies",
            "pair_counts",
        }
        assert fresh["elements"] == [1, 6, 8]
        assert fresh["atomic_energies"] == {
            "mode": "dataset",
            "values": {1: -13.6, 6: -1028.5, 8: -2041.1},
            "trainable": True,  # user settings next to the statistic are kept
        }
        assert fresh["artisan"]["pair_counts"] == {"1_6": 10, "6_8": 3}
        assert fresh["artisan"]["rare_artisan"] == {"enabled": True}

    def test_apply_honours_skip(self) -> None:
        fresh = {"scale": None, "avg_num_neighbors": None, "elements": None}
        applied = ds.apply(fresh, ds.collect(_backbone()), skip={"scale", "elements"})
        assert applied == ["avg_num_neighbors"]
        assert fresh["scale"] is None and fresh["elements"] is None

    def test_apply_never_creates_a_missing_sub_config(self) -> None:
        fresh = {"scale": None}  # no dressing_kwargs / artisan on this backbone
        ds.apply(
            fresh, {"dressing_kwargs.avg_num_neighbors": 9.0, "artisan.pair_counts": {}}, set()
        )
        assert fresh == {"scale": None}


# ----------------------------------------------------------------------
# fingerprint
# ----------------------------------------------------------------------


class TestFingerprint:
    def test_loader_settings_are_not_part_of_the_dataset_identity(self) -> None:
        a = OmegaConf.create(
            {"data": {"root": "x.xyz", "cutoff": 5.0, "batch_size": 8, "num_workers": 4}}
        )
        b = OmegaConf.create(
            {"data": {"root": "x.xyz", "cutoff": 5.0, "batch_size": 64, "num_workers": 0}}
        )
        assert ds.dataset_fingerprint(a, 100) == ds.dataset_fingerprint(b, 100)

    def test_changes_report_old_and_new(self) -> None:
        old = {"root": "a.xyz", "n_train": 48, "cutoff": 5.0}
        new = {"root": "b.xyz", "n_train": 72, "cutoff": 5.0}
        assert ds.fingerprint_changes(old, new) == {
            "n_train": (48, 72),
            "root": ("a.xyz", "b.xyz"),
        }


# ----------------------------------------------------------------------
# record / load
# ----------------------------------------------------------------------


def _run_cfg() -> OmegaConf:
    cfg = OmegaConf.create(
        {
            "data": {"root": "train.xyz", "cutoff": 5.0, "batch_size": 8},
            "model": {"backbone": _backbone()},
            "dataset_stats": {"reuse": True, "recompute": ["scale"]},
        }
    )
    OmegaConf.set_struct(cfg, True)
    return cfg


class TestRecordAndLoad:
    def test_record_survives_a_yaml_round_trip(self, tmp_path: Path) -> None:
        cfg = _run_cfg()
        fp = ds.dataset_fingerprint(cfg, 48)
        ds.record(cfg, _backbone(), fp, origin="computed from the training set")
        (tmp_path / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

        saved = ds.load_stored(tmp_path, None)
        values, fingerprint = ds.stored_values(saved)
        assert values == ds.collect(_backbone())
        assert fingerprint == fp
        # the user's policy settings are preserved next to the record
        assert saved.dataset_stats.reuse is True
        assert list(saved.dataset_stats.recompute) == ["scale"]

    def test_next_resume_compares_against_the_data_last_trained_on(self) -> None:
        cfg = _run_cfg()
        old, new = {"root": "a.xyz"}, {"root": "b.xyz"}
        ds.record(cfg, _backbone(), old, origin="reused", trained_on=new)
        _, fingerprint = ds.stored_values(OmegaConf.create(OmegaConf.to_yaml(cfg)))
        assert fingerprint == new
        assert cfg.dataset_stats.provenance.dataset == old

    def test_configs_from_before_the_record_still_yield_their_values(self) -> None:
        legacy = OmegaConf.create({"model": {"backbone": _backbone()}})
        values, fingerprint = ds.stored_values(legacy)
        assert values["scale"] == 0.25 and values["elements"] == [1, 6, 8]
        assert fingerprint == {}

    def test_checkpoint_hyper_parameters_are_the_fallback(self, tmp_path: Path) -> None:
        cfg = _run_cfg()
        ds.record(cfg, _backbone(), {"root": "train.xyz"}, origin="computed")
        ckpt = tmp_path / "last.ckpt"
        torch.save({"state_dict": {}, "hyper_parameters": {"config": cfg}}, ckpt)

        saved = ds.load_stored(tmp_path, str(ckpt))  # no config.yaml in tmp_path
        values, _ = ds.stored_values(saved)
        assert values["avg_num_neighbors"] == 17.5

    def test_nothing_to_load_returns_none(self, tmp_path: Path) -> None:
        assert ds.load_stored(tmp_path, None) is None


# ----------------------------------------------------------------------
# effect on a resumed model
# ----------------------------------------------------------------------


class _TinyBackbone(nn.Module):
    def __init__(self, n_elements: int, scale: float) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_elements, 4)
        self.register_buffer("scale", torch.tensor(scale))
        self.register_buffer("atomic_energies", torch.zeros(n_elements))


class _Host(nn.Module):
    def __init__(self, n_elements: int, scale: float) -> None:
        super().__init__()
        self.backbone = _TinyBackbone(n_elements, scale)


class TestResumedModule:
    def test_recomputed_buffers_survive_the_checkpoint_restore(self) -> None:
        fresh = _Host(9, scale=2.5)
        trained = _Host(9, scale=1.0).state_dict()
        trained["backbone.atomic_energies"] = torch.full((9,), -7.0)
        checkpoint = {"state_dict": {k: v.clone() for k, v in trained.items()}}
        host = SimpleNamespace(
            keep_fresh_buffers={"scale"},
            state_dict=fresh.state_dict,
            _BACKBONE_BUFFER_KEY=GOALModule._BACKBONE_BUFFER_KEY,
        )
        GOALModule._keep_fresh_buffers_on_load(host, checkpoint)
        assert checkpoint["state_dict"]["backbone.scale"].item() == pytest.approx(2.5)
        # everything not asked for is still restored from the checkpoint
        assert torch.equal(
            checkpoint["state_dict"]["backbone.embed.weight"], trained["backbone.embed.weight"]
        )
        assert torch.equal(
            checkpoint["state_dict"]["backbone.atomic_energies"],
            trained["backbone.atomic_energies"],
        )

    def test_nothing_is_kept_by_default(self) -> None:
        checkpoint = {"state_dict": _Host(9, scale=1.0).state_dict()}
        host = SimpleNamespace(
            keep_fresh_buffers=set(),
            state_dict=_Host(9, scale=2.5).state_dict,
            _BACKBONE_BUFFER_KEY=GOALModule._BACKBONE_BUFFER_KEY,
        )
        GOALModule._keep_fresh_buffers_on_load(host, checkpoint)
        assert checkpoint["state_dict"]["backbone.scale"].item() == pytest.approx(1.0)

    def test_structural_change_is_reported_per_tensor(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "last.ckpt"
        torch.save({"state_dict": _Host(9, 1.0).state_dict()}, ckpt)
        assert ds.structural_mismatches(_Host(9, 1.0), str(ckpt)) == []
        problems = ds.structural_mismatches(_Host(10, 1.0), str(ckpt))
        assert any(
            "backbone.embed.weight" in p and "(9, 4)" in p and "(10, 4)" in p for p in problems
        )

    def test_uncovered_elements(self) -> None:
        data = [
            SimpleNamespace(atomic_numbers=torch.tensor([1, 6, 7])),
            SimpleNamespace(atomic_numbers=torch.tensor([8])),
        ]
        assert ds.uncovered_elements([1, 6, 8], data) == [7]
        assert ds.uncovered_elements([1, 6, 7, 8], data) == []


# ----------------------------------------------------------------------
# end to end: train, then extend with and without recomputation
# ----------------------------------------------------------------------


def _write_lj_dataset(path: Path, elements: list[str], n: int, seed: int) -> None:
    import numpy as np
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        k = int(rng.integers(4, 8))
        atoms = Atoms(list(rng.choice(elements, size=k)), positions=rng.uniform(0, 3.2, (k, 3)))
        atoms.calc = LennardJones(sigma=1.0, epsilon=0.05, rc=6.0)
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        frames.append(atoms)
    write(path, frames)


def _goal_train(run_dir: Path, xyz: Path, *extra: str) -> subprocess.CompletedProcess:
    exe = shutil.which("goal-train") or str(Path(sys.executable).with_name("goal-train"))
    args = [
        exe,
        f"hydra.run.dir={run_dir}",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "data.dataset_type=xyz",
        "data.train_dir=null",
        "data.val_dir=null",
        "data.test_dir=null",
        f"data.root={xyz}",
        "data.compute_fragment_index=false",
        "data.batch_size=8",
        "data.monitor_test_during_training=false",
        "~model.backbone.fragment_interaction",
        "model.backbone.atomic_energies.mode=dataset",
        "model.backbone.atomic_energies.values=null",
        "logger=null",
        *extra,
    ]
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, "HYDRA_FULL_ERROR": "1"},
    )


@pytest.mark.slow
def test_extension_reuses_the_recorded_statistics(tmp_path: Path) -> None:
    """Fresh run records the statistics; an extension reuses them unchanged,
    and recomputing scale on new data trains with the new value."""
    hco, more = tmp_path / "hco.xyz", tmp_path / "hco_more.xyz"
    _write_lj_dataset(hco, ["H", "C", "O"], 60, seed=0)
    _write_lj_dataset(more, ["H", "C", "O"], 90, seed=1)

    first = _goal_train(tmp_path / "run1", hco, "trainer.max_epochs=2")
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    assert "[stats] elements discovered" in first.stdout
    saved = OmegaConf.load(tmp_path / "run1/checkpoints/config.yaml")
    recorded = dict(saved.dataset_stats["values"])
    assert {"elements", "scale", "avg_num_neighbors", "atomic_energies.values"} <= set(recorded)

    same = _goal_train(
        tmp_path / "run2", hco, f"resume.from={tmp_path / 'run1'}", "trainer.max_epochs=3"
    )
    assert same.returncode == 0, same.stdout[-3000:] + same.stderr[-3000:]
    assert "[stats] reusing elements, atomic_energies, scale, avg_num_neighbors" in same.stdout
    assert "[stats] elements discovered" not in same.stdout  # nothing recomputed
    assert (
        dict(OmegaConf.load(tmp_path / "run2/checkpoints/config.yaml").dataset_stats["values"])
        == recorded
    )

    changed = _goal_train(
        tmp_path / "run3",
        more,
        f"resume.from={tmp_path / 'run1'}",
        "trainer.max_epochs=3",
        "dataset_stats.recompute=[scale]",
    )
    assert changed.returncode == 0, changed.stdout[-3000:] + changed.stderr[-3000:]
    assert "the training data differs" in changed.stdout
    new_scale = OmegaConf.load(tmp_path / "run3/checkpoints/config.yaml").dataset_stats["values"][
        "scale"
    ]
    assert new_scale != pytest.approx(recorded["scale"])
    ckpt = torch.load(
        tmp_path / "run3/checkpoints/last.ckpt", map_location="cpu", weights_only=False
    )
    assert ckpt["state_dict"]["backbone.scale"].item() == pytest.approx(new_scale)
