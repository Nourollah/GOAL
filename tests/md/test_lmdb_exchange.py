"""Tests for the LMDB exchange adapter (GOAL ↔ Belero)."""

from __future__ import annotations

import numpy as np
import pytest

from goal.md.adapters.lmdb_exchange import LMDBExchange, atoms_to_record, encode_record


def test_atoms_roundtrip(tmp_path, ethanol_atoms):
    ethanol_atoms.info = {"charge": 0, "method": "EMT", "spin": np.int64(1)}
    energy = ethanol_atoms.get_potential_energy()
    forces = ethanol_atoms.get_forces()

    with LMDBExchange(tmp_path / "db") as db:
        out = db.get_atoms(db.put(ethanol_atoms))

    np.testing.assert_array_equal(out.numbers, ethanol_atoms.numbers)
    np.testing.assert_array_equal(out.positions, ethanol_atoms.positions)
    np.testing.assert_array_equal(out.get_forces(), forces)
    assert out.get_potential_energy() == energy
    assert out.info == {"charge": 0, "method": "EMT", "spin": 1}


def test_request_response_and_key_order(tmp_path, water_atoms):
    request = {"atomic_numbers": water_atoms.numbers, "positions": water_atoms.positions}
    with LMDBExchange(tmp_path / "db") as db:
        for key in (256, 2, 10):
            db.put(request, key=key)
        assert db.keys() == [2, 10, 256]  # big-endian keys sort numerically
        assert db.keys(start=11) == [256]
        assert db.keys(start=257) == []
        assert db.put(request) == 257

        assert "energy" not in db.get(10)
        energy = water_atoms.get_potential_energy()
        db.put(water_atoms, key=10)  # answer the request in place
        assert db.get(10)["energy"] == energy
        assert len(db) == 4
        with pytest.raises(KeyError):
            db.get(3)


def test_blob_is_valid_safetensors(ethanol_atoms):
    st = pytest.importorskip("safetensors.numpy")  # the Rust parser Belero links against
    ethanol_atoms.get_forces()
    record = atoms_to_record(ethanol_atoms)

    loaded = st.load(encode_record(record))

    assert set(loaded) == {"atomic_numbers", "positions", "cell", "pbc", "energy", "forces"}
    np.testing.assert_array_equal(loaded["forces"], record["forces"])
    np.testing.assert_array_equal(loaded["pbc"], record["pbc"])
    assert loaded["energy"].shape == ()
