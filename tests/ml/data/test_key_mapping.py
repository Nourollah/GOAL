"""Tests for dataset label key-mapping (MACE/FairChem-style ``key_mapping``).

Covers the bug where a trajectory/xyz whose energy/forces live under a custom
key (e.g. ``REF_energy``) silently produced ``None`` labels and then crashed
downstream. Uses small in-memory ASE structures (CI-safe, no GMD data).
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# resolve_label_keys
# ---------------------------------------------------------------------------


def test_resolve_defaults():
    from goal.ml.data.keys import resolve_label_keys

    mapping, explicit = resolve_label_keys()
    assert mapping == {"energy": "energy", "forces": "forces", "stress": "stress"}
    assert explicit == set()  # nothing configured → nothing strict


def test_resolve_per_key_marks_explicit():
    from goal.ml.data.keys import resolve_label_keys

    mapping, explicit = resolve_label_keys(energy_key="E", forces_key="F")
    assert mapping["energy"] == "E" and mapping["forces"] == "F"
    assert explicit == {"energy", "forces"}


def test_key_mapping_overrides_per_key():
    from goal.ml.data.keys import resolve_label_keys

    mapping, explicit = resolve_label_keys(
        energy_key="E", key_mapping={"energy": "REF_energy", "stress": "virial"}
    )
    assert mapping["energy"] == "REF_energy"  # key_mapping wins
    assert mapping["stress"] == "virial"
    assert explicit == {"energy", "stress"}


# ---------------------------------------------------------------------------
# extract_ase_labels
# ---------------------------------------------------------------------------


def _h2o(energy_key=None, forces_key=None, stress_key=None):
    from ase.build import molecule

    a = molecule("H2O")
    if energy_key:
        a.info[energy_key] = -10.0
    if forces_key:
        a.new_array(forces_key, np.zeros((len(a), 3)))
    if stress_key:
        a.info[stress_key] = np.eye(3)
    return a


def test_extract_custom_keys():
    from goal.ml.data.keys import extract_ase_labels, resolve_label_keys

    atoms = _h2o(energy_key="REF_energy", forces_key="REF_forces", stress_key="virial")
    mapping, explicit = resolve_label_keys(
        key_mapping={"energy": "REF_energy", "forces": "REF_forces", "stress": "virial"}
    )
    labels = extract_ase_labels(atoms, mapping, explicit)
    assert labels["energy"] == -10.0
    assert labels["forces"].shape == (3, 3)
    assert labels["stress"].shape == (3, 3)


def test_extract_missing_explicit_key_raises_with_available():
    from goal.ml.data.keys import extract_ase_labels, resolve_label_keys

    atoms = _h2o(energy_key="REF_energy")  # only REF_energy present, no calc
    mapping, explicit = resolve_label_keys(energy_key="energy")  # wrong, explicit
    with pytest.raises(ValueError) as exc:
        extract_ase_labels(atoms, mapping, explicit)
    assert "REF_energy" in str(exc.value)  # error lists the key that IS present


def test_extract_missing_default_key_is_permissive():
    from goal.ml.data.keys import extract_ase_labels, resolve_label_keys

    atoms = _h2o(energy_key="REF_energy")  # nothing under default keys, no calc
    mapping, explicit = resolve_label_keys()  # defaults, not explicit
    labels = extract_ase_labels(atoms, mapping, explicit)
    assert labels["energy"] is None and labels["forces"] is None  # no crash


# ---------------------------------------------------------------------------
# ExtXYZDataset end to end
# ---------------------------------------------------------------------------


def _write_xyz(path, energy_key="REF_energy", forces_key="REF_forces"):
    from ase.io import write

    write(str(path), _h2o(energy_key=energy_key, forces_key=forces_key), format="extxyz")
    return str(path)


def test_xyz_key_mapping_loads_labels(tmp_path):
    from goal.ml.data.datasets.xyz import ExtXYZDataset

    path = _write_xyz(tmp_path / "c.xyz")
    ds = ExtXYZDataset(
        root=path, cutoff=5.0, key_mapping={"energy": "REF_energy", "forces": "REF_forces"}
    )
    g = ds[0]
    assert g.energy is not None and float(g.energy) == -10.0
    assert g.forces is not None and g.forces.shape == (3, 3)


def test_xyz_wrong_explicit_key_errors(tmp_path):
    from goal.ml.data.datasets.xyz import ExtXYZDataset

    path = _write_xyz(tmp_path / "c.xyz")
    with pytest.raises(ValueError, match="REF_energy"):
        ExtXYZDataset(root=path, cutoff=5.0, energy_key="energy", forces_key="forces")


def test_trajectory_accepts_key_mapping_and_stress(tmp_path):
    """Regression: TrajectoryDataset now accepts stress_key/key_mapping and no
    longer hard-codes the 'stress' info key. (Custom energy + stress live in
    ``atoms.info``, which round-trips through .traj; custom per-atom force
    arrays do not, so the forces remapping is covered by the xyz test.)"""
    from ase.io import Trajectory

    from goal.ml.data.datasets.trajectory import TrajectoryDataset

    p = str(tmp_path / "t.traj")
    w = Trajectory(p, mode="w")
    w.write(_h2o(energy_key="REF_energy", stress_key="virial"))
    w.close()

    ds = TrajectoryDataset(
        root=p,
        cutoff=5.0,
        key_mapping={"energy": "REF_energy", "stress": "virial"},
    )
    g = ds[0]
    assert float(g.energy) == -10.0
    assert g.stress is not None and tuple(g.stress.shape) == (3, 3)


def test_datamodule_forwards_key_mapping(tmp_path):
    from omegaconf import OmegaConf

    from goal.ml.data.datamodule import GOALDataModule

    path = _write_xyz(tmp_path / "c.xyz")
    cfg = OmegaConf.create(
        {
            "data": {
                "dataset_type": "xyz",
                "root": path,
                "cutoff": 5.0,
                "batch_size": 1,
                "num_workers": 0,
                "split_ratio": [1.0, 0.0],
                "key_mapping": {"energy": "REF_energy", "forces": "REF_forces"},
            }
        }
    )
    dm = GOALDataModule(cfg)
    dm.setup("fit")
    g = dm.data_train[0]
    assert g.energy is not None and float(g.energy) == -10.0
