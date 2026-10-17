"""The shipped MD configs compose to the schema the code reads, and run.

``langevin_with_model.yaml`` / ``mts_respa.yaml`` used to (a) lack
``# @package _global_`` — Hydra nested them under ``md.simulations`` /
``md.mts.simulations`` so no ``molecule:`` was found — and (b) name their
factory keys ``method:`` where ``CalculatorFactory.create(key, ...)``
needs ``key``.  The Langevin template's ``timestep: 1.0`` was ~10 fs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ase import units
from ase.build import molecule
from ase.calculators.emt import EMT
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from goal import CONFIGS_DIR
from goal.md.core.calculator_factory import CalculatorBuilder, CalculatorFactory
from goal.md.core.md_factory import DynamicsFactory
from goal.md.core.molecule_factory import MoleculeFactory

SIMULATION_CONFIGS = [
    "md/simulations/langevin_with_model",
    "md/mts/simulations/mts_respa",
    "md/mts/simulations/online_learner",
    "md/mts/simulations/orca_ml_learner",
]


def _compose(name: str, overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base="1.3"):
        return compose(config_name=name, overrides=overrides or [])


@pytest.fixture()
def emt_builder():
    class _EMTBuilder(CalculatorBuilder):
        def build(self):
            return EMT()

    CalculatorFactory.register("test_emt", _EMTBuilder())
    yield
    CalculatorFactory._builders.pop("test_emt", None)


@pytest.mark.parametrize("name", SIMULATION_CONFIGS)
def test_simulation_configs_compose_at_top_level(name: str) -> None:
    cfg = _compose(name)
    assert "molecule" in cfg, f"{name} composed to {list(cfg)} — missing '# @package _global_'?"
    assert "md" not in cfg


@pytest.mark.parametrize(
    "path", sorted(Path(CONFIGS_DIR, "md").rglob("*.yaml")), ids=lambda p: p.name
)
def test_factory_templates_name_their_key_with_key(path: Path) -> None:
    """``method`` is a real argument for tblite / cp2k; everywhere else the
    factory key must be ``key`` (or ``type``), or instantiate() fails."""
    raw = OmegaConf.load(path)
    target = str(raw.get("_target_", ""))
    if target.endswith("Factory.create"):
        assert "key" in raw, f"{path.name}: factory template without 'key:'"


class TestFactoriesFromConfig:
    def test_molecule_accepts_key_type_or_method_and_ignores_hydra_keys(self) -> None:
        for spelling in ("key", "type", "method"):
            atoms = MoleculeFactory.from_config(
                {"_target_": "x", "_convert_": "none", spelling: "from_smiles", "smiles": "O"}
            )
            assert atoms.get_chemical_formula() == "H2O"

    def test_calculator_keeps_method_as_builder_argument(self, monkeypatch) -> None:
        seen = {}

        class _Builder(CalculatorBuilder):
            def build(self, method="GFN2-xTB"):
                seen["method"] = method
                return EMT()

        monkeypatch.setitem(CalculatorFactory._builders, "probe", _Builder())
        CalculatorFactory.from_config({"_target_": "x", "key": "probe", "method": "GFN1-xTB"})
        assert seen["method"] == "GFN1-xTB"

    def test_group_template_instantiates_with_hydra(self, tmp_path) -> None:
        import hydra

        cfg = OmegaConf.load(Path(CONFIGS_DIR, "md/molecules/ethanol.yaml"))
        atoms = hydra.utils.instantiate(cfg)
        assert len(atoms) == 9


class TestGoalSimulateConfig:
    def test_default_config_runs_with_a_stub_calculator(self, emt_builder, tmp_path) -> None:
        from goal.md.core.simulation import Simulation

        cfg = _compose(
            "md/simulations/langevin_with_model",
            overrides=[
                "~calculator",
                "+calculator={key: test_emt}",
                "molecule.smiles=C",
                "steps=5",
                "chunk_size=5",
                f"output_dir={tmp_path}",
                "+show_progress=false",
            ],
        )
        sim = Simulation.from_config(cfg)
        assert sim.timestep_fs == 0.5
        result = sim.run()
        assert result.steps_completed == 5
        assert (tmp_path / "trajectory.traj").is_file()

    def test_num_steps_alias(self, emt_builder) -> None:
        from goal.md.core.simulation import Simulation

        cfg = OmegaConf.create(
            {
                "molecule": {"key": "from_smiles", "smiles": "O"},
                "calculator": {"key": "test_emt"},
                "num_steps": 7,
            }
        )
        assert Simulation.from_config(cfg).steps == 7

    def test_nested_config_gives_a_pointed_error(self) -> None:
        from goal.md.core.simulation import Simulation

        cfg = OmegaConf.create({"md": {"simulations": {"molecule": {}}}})
        with pytest.raises(ValueError, match="@package _global_"):
            Simulation.from_config(cfg)

    def test_goal_model_template_points_at_goal_calculator(self) -> None:
        cfg = _compose("md/simulations/langevin_with_model")
        assert cfg.calculator.key == "goal_model"
        assert OmegaConf.is_missing(cfg.calculator, "checkpoint")


class TestMTSConfig:
    def test_respa_config_builds_fast_and_slow(self, emt_builder, tmp_path) -> None:
        from goal.md.mts.ipi_adapter import MTSSimulation

        cfg = _compose(
            "md/mts/simulations/mts_respa",
            overrides=[
                "~fast_calculator",
                "+fast_calculator={type: test_emt}",
                "~slow_calculator",
                "+slow_calculator={type: test_emt}",
                "molecule.smiles=O",
                "mts.total_steps=2",
                "mts.show_progress=false",
                f"mts.output_dir={tmp_path}",
            ],
        )
        sim = MTSSimulation.from_config(cfg)
        assert sim.slow_calc is not None
        assert sim.run()["steps_completed"] == 2


class TestLangevinUnits:
    def test_fs_arguments_are_converted(self) -> None:
        atoms = molecule("H2O")
        dyn = DynamicsFactory.create(
            "langevin_ase", atoms=atoms, timestep_fs=0.5, friction_per_fs=0.02
        )
        assert dyn.dt == pytest.approx(0.5 * units.fs)
        assert dyn.fr == pytest.approx(0.02 / units.fs)

    def test_raw_ase_units_still_accepted(self) -> None:
        dyn = DynamicsFactory.create(
            "langevin_ase", atoms=molecule("H2O"), timestep=0.5 * units.fs
        )
        assert dyn.dt == pytest.approx(0.5 * units.fs)

    def test_defaults_unchanged(self) -> None:
        dyn = DynamicsFactory.create("langevin_ase", atoms=molecule("H2O"))
        assert dyn.dt == pytest.approx(1.0 * units.fs)
        assert dyn.fr == pytest.approx(0.5)

    def test_both_spellings_rejected(self) -> None:
        with pytest.raises(ValueError, match="either"):
            DynamicsFactory.create(
                "langevin_ase", atoms=molecule("H2O"), timestep=1.0, timestep_fs=1.0
            )

    def test_template_is_one_femtosecond(self) -> None:
        import hydra

        cfg = OmegaConf.load(Path(CONFIGS_DIR, "md/dynamics/langevin.yaml"))
        dyn = hydra.utils.instantiate(cfg, atoms=molecule("H2O"))
        assert dyn.dt == pytest.approx(1.0 * units.fs)


class TestORCABuilder:
    """ASE ≥ 3.23 replaced ORCA's ``command`` with a ``profile``; the builder
    uses ``OrcaProfile`` and constructs cleanly on the installed ASE."""

    def test_factory_builds_orca_calculator(self, tmp_path) -> None:
        pytest.importorskip("ase.calculators.orca")
        calc = CalculatorFactory.create(
            "orca", orca_path="/opt/orca/orca", keywords="XTB2", directory=str(tmp_path)
        )
        assert type(calc).__name__ == "ORCA"
        assert calc.parameters["orcasimpleinput"].split()[0] == "EnGrad"
        assert calc.profile.command == "/opt/orca/orca"

    def test_mts_reference_template_builds(self, tmp_path) -> None:
        cfg = OmegaConf.load(Path(CONFIGS_DIR, "md/mts/base_calculators/orca_reference.yaml"))
        cfg.orca_path = "/opt/orca/orca"
        calc = CalculatorFactory.from_config(cfg)
        assert type(calc).__name__ == "ORCA"


def test_goal_model_builder_matches_calculator_loading(tmp_path) -> None:
    """``goal_model`` forwards selector / EMA and sniffs the dtype instead of
    forcing float64."""
    import torch

    from tests.ml.test_archive import _build_module, _tiny_config

    cfg = _tiny_config()
    module = _build_module(cfg).float()
    ckpt = tmp_path / "m.ckpt"
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": {"config": cfg}}, ckpt)
    calc = CalculatorFactory.create("goal_model", checkpoint=str(ckpt))
    assert calc.dtype == torch.float32
    assert calc.weights == "raw"


def test_get_model_cutoff_reads_config_without_building(tmp_path) -> None:
    import torch

    from goal.md.adapters.model_loader import get_model_cutoff

    cfg = OmegaConf.create(
        {"data": {"cutoff": 6.5}, "model": {"backbone": {"name": "does_not_exist"}}}
    )
    ckpt = tmp_path / "m.ckpt"
    torch.save({"state_dict": {}, "hyper_parameters": {"config": cfg}}, ckpt)
    assert get_model_cutoff(ckpt) == 6.5
