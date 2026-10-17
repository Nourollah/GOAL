"""Tests for goal.md factory classes.

These tests cover:
- MoleculeFactory: from_smiles, from_file
- CalculatorFactory: registration, creation, error handling
- DynamicsFactory: registration, creation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

# ---------------------------------------------------------------------------
# MoleculeFactory
# ---------------------------------------------------------------------------


class TestMoleculeFactory:
    """Tests for MoleculeFactory."""

    def test_factory_is_importable(self) -> None:
        """Test MoleculeFactory is importable and has create method."""
        from goal.md.core.molecule_factory import MoleculeFactory

        assert hasattr(MoleculeFactory, "create")

    def test_from_file(self, tmp_path: Path) -> None:
        """Test MoleculeFactory.create creates Atoms from XYZ file."""
        from goal.md.core.molecule_factory import MoleculeFactory

        xyz = tmp_path / "mol.xyz"
        xyz.write_text("3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n")
        atoms = MoleculeFactory.create("from_file", path=str(xyz))
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 3

    @pytest.mark.skipif(
        not pytest.importorskip("rdkit", reason="rdkit not installed"),
        reason="rdkit not installed",
    )
    def test_from_smiles_water(self) -> None:
        from goal.md.core.molecule_factory import MoleculeFactory

        atoms = MoleculeFactory.create("from_smiles", smiles="O")
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 3  # O + 2H

    def test_from_file_missing_raises(self, tmp_path: Path) -> None:
        """Test MoleculeFactory raises when file does not exist."""
        from goal.md.core.molecule_factory import MoleculeFactory

        with pytest.raises((FileNotFoundError, Exception)):
            MoleculeFactory.create("from_file", path=str(tmp_path / "missing.xyz"))

    def test_unknown_method_raises(self) -> None:
        """Test MoleculeFactory raises for unknown creation method."""
        from goal.md.core.molecule_factory import MoleculeFactory

        with pytest.raises((ValueError, KeyError)):
            MoleculeFactory.create("unknown_method_xyz")


# ---------------------------------------------------------------------------
# CalculatorFactory
# ---------------------------------------------------------------------------


class TestCalculatorFactory:
    """Tests for CalculatorFactory."""

    def test_factory_is_importable(self) -> None:
        """Test CalculatorFactory is importable and has create method."""
        from goal.md.core.calculator_factory import CalculatorFactory

        assert hasattr(CalculatorFactory, "create")

    def test_registered_keys(self) -> None:
        """Test CalculatorFactory has all expected calculator types registered."""
        from goal.md.core.calculator_factory import CalculatorFactory

        keys = list(CalculatorFactory._builders.keys())
        assert "goal_model" in keys
        assert "flashmd" in keys
        assert "mace" in keys
        assert "orca" in keys
        assert "cp2k" in keys
        assert "espresso" in keys
        assert "tblite" in keys
        assert "nequip" in keys

    def test_unknown_key_raises(self) -> None:
        """Test CalculatorFactory raises for unregistered calculator."""
        from goal.md.core.calculator_factory import CalculatorFactory

        with pytest.raises(ValueError, match="not registered"):
            CalculatorFactory.create("nonexistent_calc_xyz")

    def test_custom_registration(self) -> None:
        """Test custom calculators can be registered with decorator."""
        from goal.md.core.calculator_factory import (
            CalculatorBuilder,
            CalculatorFactory,
            register_calculator,
        )

        @register_calculator("_test_emt")
        class _TestEMTBuilder(CalculatorBuilder):
            def build(self, **kwargs):  # type: ignore[override]
                return EMT()

        calc = CalculatorFactory.create("_test_emt")
        assert isinstance(calc, EMT)

    def test_flashmd_requires_import(self) -> None:
        """Test CalculatorFactory handles missing flashmd gracefully."""
        from goal.md.core.calculator_factory import CalculatorFactory

        try:
            calc = CalculatorFactory.create("flashmd")
        except ImportError:
            pass  # expected when flashmd not installed
        except Exception as exc:
            pytest.fail(f"Unexpected exception: {exc}")


class TestTBLiteBuilder:
    """Tests for the tblite (GFN-xTB) builder."""

    def test_gfn2_energy_and_forces(self) -> None:
        """GFN2-xTB returns an energy and analytic forces that match finite differences."""
        pytest.importorskip("tblite")
        from ase.build import molecule

        from goal.md.core.calculator_factory import CalculatorFactory

        atoms = molecule("H2O")
        atoms.calc = CalculatorFactory.create("tblite", method="GFN2-xTB")
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        assert np.isfinite(energy)
        assert forces.shape == (3, 3)

        # finite-difference check on one coordinate
        h = 1e-4
        displaced = atoms.copy()
        displaced.calc = CalculatorFactory.create("tblite", method="GFN2-xTB")
        displaced.positions[0, 2] += h
        e_plus = displaced.get_potential_energy()
        displaced.positions[0, 2] -= 2 * h
        e_minus = displaced.get_potential_energy()
        assert forces[0, 2] == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=1e-3)

    def test_hydra_config_instantiates(self) -> None:
        """configs/md/calculators/tblite.yaml builds a working calculator."""
        pytest.importorskip("tblite")
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(
            Path(__file__).parents[2] / "src/goal/configs/md/calculators/tblite.yaml"
        )
        calc = instantiate(cfg)
        assert "forces" in calc.implemented_properties


class TestORCABuilder:
    """Tests for the ORCA builder's generated input (no ORCA run)."""

    def test_xtb2_input_requests_gradient(self, tmp_path: Path) -> None:
        """ORCA's built-in GFN2-xTB gets EnGrad added and a %pal block."""
        from ase.build import molecule

        from goal.md.core.calculator_factory import CalculatorFactory

        calc = CalculatorFactory.create(
            "orca",
            orca_path="/opt/orca/orca",
            keywords="XTB2",
            num_threads=4,
            directory=str(tmp_path),
        )
        calc.template.write_input(
            calc.profile, tmp_path, molecule("H2O"), calc.parameters, ["energy", "forces"]
        )
        text = (tmp_path / "orca.inp").read_text()

        assert text.startswith("! EnGrad XTB2")
        assert "%pal nprocs 4 end" in text
        assert "*xyz 0 1" in text

    def test_engrad_not_duplicated(self, tmp_path: Path) -> None:
        """An explicit EnGrad keyword is kept once."""
        from goal.md.core.calculator_factory import CalculatorFactory

        calc = CalculatorFactory.create(
            "orca",
            orca_path="/opt/orca/orca",
            keywords="EnGrad B3LYP def2-SVP",
            directory=str(tmp_path),
        )
        assert calc.parameters["orcasimpleinput"].lower().split().count("engrad") == 1


class TestEspressoBuilder:
    """Tests for the Quantum ESPRESSO builder's input handling (no pw.x run)."""

    _PSEUDOS = {"H": "H.UPF", "O": "O.UPF"}

    def test_hydra_config_keeps_cutoffs_with_molecular_preset(self, tmp_path: Path) -> None:
        """DictConfig input_data from Hydra must merge with the preset, not be replaced by it."""
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(
            Path(__file__).parents[2] / "src/goal/configs/md/calculators/espresso.yaml"
        )
        cfg.command = "pw.x"
        cfg.pseudo_dir = str(tmp_path)
        cfg.pseudopotentials = self._PSEUDOS
        cfg.directory = str(tmp_path / "run")
        calc = instantiate(cfg)

        system = calc.parameters["input_data"]["system"]
        assert system["ecutwfc"] == 80
        assert system["ecutrho"] == 640
        assert system["assume_isolated"] == "mt"
        assert calc.parameters["input_data"]["control"]["tprnfor"] is True

    def test_bulk_preset_requests_stress_and_kpts(self, tmp_path: Path) -> None:
        """The bulk preset enables tstress and default k-points unless overridden."""
        from goal.md.core.calculator_factory import CalculatorFactory

        calc = CalculatorFactory.create(
            "espresso",
            command="pw.x",
            pseudo_dir=str(tmp_path),
            pseudopotentials=self._PSEUDOS,
            preset="bulk",
            directory=str(tmp_path / "run"),
        )
        assert calc.parameters["input_data"]["control"]["tstress"] is True
        assert tuple(calc.parameters["kpts"]) == (2, 2, 2)

        calc = CalculatorFactory.create(
            "espresso",
            command="pw.x",
            pseudo_dir=str(tmp_path),
            pseudopotentials=self._PSEUDOS,
            preset="bulk",
            input_data={"control": {"tstress": False}},
            directory=str(tmp_path / "run2"),
        )
        assert calc.parameters["input_data"]["control"]["tstress"] is False


# ---------------------------------------------------------------------------
# DynamicsFactory
# ---------------------------------------------------------------------------


class TestDynamicsFactory:
    """Tests for DynamicsFactory."""

    def test_factory_is_importable(self) -> None:
        """Test DynamicsFactory is importable and has create method."""
        from goal.md.core.md_factory import DynamicsFactory

        assert hasattr(DynamicsFactory, "create")

    def test_registered_keys(self) -> None:
        """Test DynamicsFactory has all expected dynamics types registered."""
        from goal.md.core.md_factory import DynamicsFactory

        keys = list(DynamicsFactory._builders.keys())
        assert "langevin_ase" in keys
        assert "langevin_flashmd" in keys

    def test_langevin_ase_creation(self, water_atoms: Atoms) -> None:
        """Test DynamicsFactory creates Langevin dynamics successfully."""
        from ase import units

        from goal.md.core.md_factory import DynamicsFactory

        dyn = DynamicsFactory.create(
            "langevin_ase",
            atoms=water_atoms,
            timestep=1.0 * units.fs,
            temperature_K=300.0,
            friction=0.01,
        )
        assert hasattr(dyn, "run")
        assert dyn.atoms is water_atoms

    def test_unknown_key_raises(self, water_atoms: Atoms) -> None:
        """Test DynamicsFactory raises for unregistered dynamics type."""
        from goal.md.core.md_factory import DynamicsFactory

        with pytest.raises(ValueError, match="not registered"):
            DynamicsFactory.create("nonexistent_dynamics_xyz", atoms=water_atoms)
