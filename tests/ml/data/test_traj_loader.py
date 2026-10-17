"""Regression tests for the single ``.traj`` loading pipeline, run against a
REAL GMD trajectory (no mocks, no synthetic files).

Trajectory selection:
- ``GOAL_TEST_TRAJ_PATH`` environment variable, if set; otherwise
- the smallest bundled GMD trajectory (so the tests run in-repo); otherwise
- the whole module is skipped (keeps CI green without the dataset).

    GOAL_TEST_TRAJ_PATH=/path/to/gmd.traj pytest tests/ml/data/test_traj_loader.py

These lock in the fixes for: single-file split leakage (train==val==test),
the Python-3.14 forkserver ``num_workers>0`` failure, and graceful handling of
tiny/edge-case datasets.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import pytest
import torch

# --- Resolve the test trajectory -------------------------------------------
_ENV_TRAJ = os.environ.get("GOAL_TEST_TRAJ_PATH")
_DEFAULT_TRAJ = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "GMD"
    / "FragmentChainExtensionAugmented"
    / "TestOOD"
    / "O2C2H4_PBE.traj"
)
if _ENV_TRAJ:
    TRAJ_PATH: str | None = _ENV_TRAJ
elif _DEFAULT_TRAJ.exists():
    TRAJ_PATH = str(_DEFAULT_TRAJ)
else:
    TRAJ_PATH = None

requires_traj = pytest.mark.skipif(
    TRAJ_PATH is None,
    reason="GOAL_TEST_TRAJ_PATH not set and no bundled GMD trajectory found — skipping real-data tests",
)

CUTOFF = 5.0


# --- Helpers ---------------------------------------------------------------


def _make_cfg(root: str, **overrides):
    from omegaconf import OmegaConf

    data = {
        "dataset_type": "trajectory",
        "root": root,
        "cutoff": CUTOFF,
        "batch_size": 4,
        "num_workers": 0,
        "pin_memory": False,
        "split_ratio": [0.8, 0.1, 0.1],
        "split_seed": 42,
        "energy_key": "energy",
        "forces_key": "forces",
    }
    data.update(overrides)
    return OmegaConf.create({"data": data})


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


@pytest.fixture(scope="module")
def dataset():
    from goal.ml.data.datasets.trajectory import TrajectoryDataset

    return TrajectoryDataset(root=TRAJ_PATH, cutoff=CUTOFF, split="train")


# --- Tests -----------------------------------------------------------------


@requires_traj
def test_1_single_file_load_completes(dataset):
    assert len(dataset) >= 1
    for i in range(len(dataset)):
        g = dataset[i]
        assert g.pos is not None and g.z is not None
        assert g.energy is not None, f"frame {i} has no energy"
        assert g.forces is not None, f"frame {i} has no forces"
        for name, t in (("pos", g.pos), ("energy", g.energy), ("forces", g.forces)):
            assert _finite(t), f"frame {i}: non-finite {name}"


@requires_traj
def test_2_atomicgraph_fields_correct(dataset):
    g = dataset[0]
    n = g.pos.shape[0]
    assert g.pos.shape == (n, 3)
    assert g.z.shape == (n,)
    assert g.edge_index.shape[0] == 2 and g.edge_index.shape[1] > 0
    assert g.energy.numel() == 1  # scalar per structure
    assert g.forces.shape == (n, 3)
    # Newton's third law: net force ~ 0
    net = g.forces.sum(dim=0).abs().max().item()
    assert net < 1e-3, f"net force too large: {net}"


@requires_traj
def test_3_split_sizes(dataset):
    from goal.ml.data.datamodule import GOALDataModule

    n = len(dataset)
    dm = GOALDataModule(_make_cfg(TRAJ_PATH, split_ratio=[0.8, 0.1, 0.1]))
    dm.setup("fit")
    dm.setup("test")
    ntr, nva, nte = len(dm.data_train), len(dm.data_val), len(dm.data_test)
    assert ntr + nva + nte == n, f"frames lost: {ntr}+{nva}+{nte} != {n}"
    assert nva >= 1 and nte >= 1
    # no leakage: splits are disjoint by construction (random_split)
    assert ntr < n  # train is a proper subset, not the whole file


@requires_traj
def test_4_dataloader_iterates(dataset):
    from goal.ml.data.datamodule import GOALDataModule

    dm = GOALDataModule(_make_cfg(TRAJ_PATH, batch_size=4, num_workers=0))
    dm.setup("fit")
    loader = dm.train_dataloader()
    seen = 0
    for b in loader:
        assert b.energy is not None and b.forces is not None and b.pos is not None
        seen += int(b.num_graphs)
    assert seen == len(dm.data_train)


@requires_traj
def test_5_avg_num_neighbors_reasonable(dataset):
    from goal.ml.data.statistics import compute_avg_num_neighbors

    ann = compute_avg_num_neighbors(dataset)
    assert 1.0 <= ann <= 100.0, f"avg_num_neighbors out of range: {ann}"


@requires_traj
def test_6_num_workers_2_works():
    from goal.ml.data.datamodule import GOALDataModule

    dm0 = GOALDataModule(_make_cfg(TRAJ_PATH, batch_size=4, num_workers=0))
    dm0.setup("fit")
    seen0 = sum(int(b.num_graphs) for b in dm0.train_dataloader())

    dm2 = GOALDataModule(_make_cfg(TRAJ_PATH, batch_size=4, num_workers=2))
    dm2.setup("fit")
    seen2 = sum(int(b.num_graphs) for b in dm2.train_dataloader())

    assert seen2 == seen0, f"num_workers=2 saw {seen2}, num_workers=0 saw {seen0}"


@requires_traj
def test_7_e0_extraction(dataset):
    from goal.ml.data.statistics import (
        compute_atomic_references,
        compute_unique_elements,
    )

    elements = compute_unique_elements(dataset)
    refs = compute_atomic_references(dataset)
    assert set(refs.keys()) == set(elements), "every element must have an E0"
    assert all(math.isfinite(v) for v in refs.values())


@requires_traj
def test_8_one_frame_dataset():
    """A 1-frame dataset (real frame 0) → everything in train, val/test empty,
    handled gracefully (no crash)."""
    from ase.io import Trajectory
    from omegaconf import OmegaConf  # noqa: F401 (kept for clarity)

    from goal.ml.data.datamodule import GOALDataModule

    src = Trajectory(TRAJ_PATH, mode="r")
    with tempfile.TemporaryDirectory() as d:
        one = str(Path(d) / "one.traj")
        w = Trajectory(one, mode="w")
        w.write(src[0])  # a REAL frame, not synthetic
        w.close()
        src.close()

        dm = GOALDataModule(_make_cfg(one, split_ratio=[0.8, 0.1, 0.1]))
        with pytest.warns(UserWarning):  # positive ratio rounds to 0 → warns
            dm.setup("fit")
        assert len(dm.data_train) == 1
        assert len(dm.data_val) == 0


@requires_traj
def test_9_full_datamodule_setup():
    from goal.ml.data.datamodule import GOALDataModule

    dm = GOALDataModule(_make_cfg(TRAJ_PATH))
    dm.setup("fit")
    assert len(dm.train_dataloader()) > 0
    assert len(dm.val_dataloader()) > 0


@requires_traj
def test_10_reproducibility():
    from goal.ml.data.datamodule import GOALDataModule

    def first10_energies():
        dm = GOALDataModule(_make_cfg(TRAJ_PATH, split_seed=42))
        dm.setup("fit")
        return [float(dm.data_train[i].energy) for i in range(min(10, len(dm.data_train)))]

    assert first10_energies() == first10_energies()
