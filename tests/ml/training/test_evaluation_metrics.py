"""Dataset-level evaluation metrics and per-element force MAE."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from goal.ml.training.metrics import evaluation_metrics, mlip_metrics


def _two_molecules() -> dict[str, np.ndarray]:
    """CH4 (2 conformers, residuals +1, +3) and H2O (2 conformers, residuals -1, -1)."""
    ch4, h2o = [6, 1, 1, 1, 1], [8, 1, 1]
    numbers = np.array(ch4 + ch4 + h2o + h2o)
    ref = np.array([-10.0, -11.0, -20.0, -21.0])
    rng = np.random.default_rng(0)
    forces_ref = rng.normal(size=(len(numbers), 3))
    forces = forces_ref.copy()
    forces[numbers == 8] += 0.3  # only oxygen is wrong
    return {
        "energy": ref + np.array([1.0, 3.0, -1.0, -1.0]),
        "energy_ref": ref,
        "num_atoms": np.array([5, 5, 3, 3]),
        "atomic_numbers": numbers,
        "forces": forces,
        "forces_ref": forces_ref,
    }


def test_energy_offset_decomposition() -> None:
    m = evaluation_metrics(_two_molecules())
    assert m["energy_mae"] == pytest.approx(1.5)
    assert m["energy_centred_mae"] == pytest.approx(0.5)  # |r - δ_m|: 1, 1, 0, 0
    assert m["energy_offset_mae"] == pytest.approx(1.5)  # |δ|: 2 (CH4), 1 (H2O)
    assert m["energy_mae_per_atom"] == pytest.approx((1 / 5 + 3 / 5 + 1 / 3 + 1 / 3) / 4)
    assert m["energy_offset_mae_per_atom"] == pytest.approx((2 / 5 + 1 / 3) / 2)
    rows = m["per_molecule"]
    assert [r["molecule"] for r in rows] == ["H2O", "CH4"]  # sorted by carbon count
    assert rows[1]["mean_signed_error"] == pytest.approx(2.0)
    assert rows[0]["mean_signed_error_per_atom"] == pytest.approx(-1 / 3)


def test_force_metrics_and_tails() -> None:
    m = evaluation_metrics(_two_molecules())
    assert m["forces_mae_O"] == pytest.approx(0.3)
    assert m["forces_mae_H"] == m["forces_mae_C"] == 0.0
    assert m["forces_mae"] == pytest.approx(0.3 * 6 / (16 * 3))
    assert m["forces_max"] == pytest.approx(0.3)
    assert m["forces_q95"] <= m["forces_q99"] <= m["forces_max"]


def test_energy_only_set_skips_forces() -> None:
    data = _two_molecules()
    del data["forces"], data["forces_ref"]
    assert "forces_mae" not in evaluation_metrics(data)


def test_mlip_metrics_per_element_force_mae() -> None:
    data = _two_molecules()
    batch = SimpleNamespace(
        atomic_numbers=torch.as_tensor(data["atomic_numbers"]),
        forces=torch.as_tensor(data["forces_ref"]),
        batch=torch.repeat_interleave(torch.arange(4), torch.as_tensor(data["num_atoms"])),
    )
    out = mlip_metrics({"forces": torch.as_tensor(data["forces"])}, batch)
    assert float(out["forces_mae_O"]) == pytest.approx(0.3)
    assert float(out["forces_mae_C"]) == 0.0
