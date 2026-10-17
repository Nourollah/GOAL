"""Simulation — central orchestration class for goal.md.

Ties together a molecule (ASE Atoms), a force calculator, and a dynamics
integrator into a single runnable object.  Can be built from:

- Programmatic factory calls (direct-library usage)
- A Hydra :class:`omegaconf.DictConfig` (config-first usage)

Usage patterns
--------------
**Direct (library)**::

    from goal.md.core.simulation import Simulation
    from goal.md.core.molecule_factory import MoleculeFactory
    from goal.md.core.calculator_factory import CalculatorFactory

    atoms = MoleculeFactory.create("from_smiles", smiles="CCO")
    calc  = CalculatorFactory.create("goal_model", checkpoint="model.ckpt")
    sim   = Simulation(atoms, calc, steps=5000, temperature_K=300.0)
    result = sim.run()

**Config-first (Hydra)**::

    from hydra import compose, initialize_config_dir
    from goal import CONFIGS_DIR
    from goal.md.core.simulation import simulate_from_config

    with initialize_config_dir(config_dir=CONFIGS_DIR):
        cfg = compose("md/simulations/langevin_with_model")
    result = simulate_from_config(cfg)
"""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from goal.md.core.observers import setup_observers
from goal.md.core.runner import RunResult, run_dynamics


@dataclasses.dataclass
class SimulationConfig:
    """Flat configuration for a single MD run.

    All time-related parameters use ASE/physical units:
    - temperature in Kelvin
    - timestep in femtoseconds
    """

    steps: int = 1000
    temperature_K: float = 300.0
    timestep_fs: float = 1.0
    friction: float = 0.5
    chunk_size: int = 100
    restart_thermostat: bool = False
    log_interval: int = 10
    trajectory_file: str | None = None
    log_file: str | None = None
    output_dir: str = "outputs/md"
    show_progress: bool = True
    initialise_velocities: bool = True


class Simulation:
    """A complete MD simulation: molecule + calculator + dynamics + observers.

    Parameters
    ----------
    atoms : ase.Atoms
        The molecular system to simulate.
    calculator : ase.calculators.calculator.Calculator
        Force/energy calculator.
    steps : int
        Number of MD steps.
    temperature_K : float
        Target temperature in Kelvin.
    timestep_fs : float
        Integration timestep in femtoseconds.
    friction : float
        Langevin friction coefficient (1/fs).
    chunk_size : int
        Steps per ``dynamics.run()`` call (for progress and restarts).
    restart_thermostat : bool
        Re-sample velocities after each chunk.
    log_interval : int
        How often to write trajectory/log entries.
    trajectory_file : str, optional
        Path for ``.traj`` output.  Relative paths are resolved under
        ``output_dir``.
    log_file : str, optional
        Path for plain-text MD log.
    output_dir : str
        Base directory for outputs when relative paths are given.
    show_progress : bool
        Show Rich progress bar during ``run()``.
    initialise_velocities : bool
        Draw initial velocities from Maxwell–Boltzmann before running.
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator: Calculator,
        steps: int = 1000,
        temperature_K: float = 300.0,
        timestep_fs: float = 1.0,
        friction: float = 0.5,
        chunk_size: int = 100,
        restart_thermostat: bool = False,
        log_interval: int = 10,
        trajectory_file: str | None = None,
        log_file: str | None = None,
        output_dir: str = "outputs/md",
        show_progress: bool = True,
        initialise_velocities: bool = True,
    ) -> None:
        self.atoms = atoms
        self.atoms.calc = calculator
        self.calculator = calculator
        self.steps = steps
        self.temperature_K = temperature_K
        self.timestep_fs = timestep_fs
        self.friction = friction
        self.chunk_size = chunk_size
        self.restart_thermostat = restart_thermostat
        self.log_interval = log_interval
        self.output_dir = Path(output_dir)
        self.show_progress = show_progress
        self.initialise_velocities = initialise_velocities

        self._traj_file = self._resolve_path(trajectory_file)
        self._log_file = self._resolve_path(log_file)

        self._dynamics: Langevin | None = None
        self._observers: dict[str, typing.Any] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, path: str | None) -> str | None:
        if path is None:
            return None
        p = Path(path)
        if not p.is_absolute():
            return str(self.output_dir / p)
        return str(p)

    def _build_dynamics(self) -> Langevin:
        dyn = Langevin(
            self.atoms,
            timestep=self.timestep_fs * units.fs,
            temperature_K=self.temperature_K,
            friction=self.friction / units.fs,
        )
        return dyn

    def _setup(self) -> None:
        """Prepare dynamics and observers (idempotent)."""
        if self._dynamics is not None:
            return

        if self.initialise_velocities:
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.temperature_K)

        self._dynamics = self._build_dynamics()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._observers = setup_observers(
            dynamics=self._dynamics,
            atoms=self.atoms,
            trajectory_file=self._traj_file,
            log_file=self._log_file,
            log_interval=self.log_interval,
            collect_energies=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dynamics(self) -> Langevin:
        """The underlying ASE dynamics object (created on first access)."""
        if self._dynamics is None:
            self._setup()
        return self._dynamics  # type: ignore[return-value]

    def run(self) -> RunResult:
        """Execute the simulation.

        Returns
        -------
        RunResult
            Summary: steps run, final energy, final temperature, energy history.
        """
        self._setup()
        return run_dynamics(
            dynamics=self._dynamics,
            steps=self.steps,
            chunk_size=self.chunk_size,
            restart_thermostat=self.restart_thermostat,
            temperature_K=self.temperature_K,
            progress=self.show_progress,
            label=f"MD {self.steps} steps @ {self.temperature_K:.0f} K",
            energy_collector=self._observers.get("energy_collector"),
        )

    # ------------------------------------------------------------------
    # Config-first factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: typing.Any) -> Simulation:
        """Build a :class:`Simulation` from a Hydra DictConfig.

        The config must contain the following top-level keys:

        - ``molecule`` — passed to :meth:`MoleculeFactory.from_config`
          (builder named by ``key``, ``type`` or ``method``)
        - ``calculator`` — passed to :meth:`CalculatorFactory.from_config`
          (builder named by ``key`` or ``type``)
        - All optional fields of :class:`SimulationConfig` (``steps``,
          ``timestep_fs``, ``temperature_K``, ``friction`` …).  The older
          spelling ``num_steps`` is accepted for ``steps``.

        The integrator is always ASE Langevin; a ``dynamics:`` section is
        not read.

        Parameters
        ----------
        cfg : DictConfig
            Hydra config object.

        Returns
        -------
        Simulation
        """
        from goal.md.core.calculator_factory import CalculatorFactory
        from goal.md.core.molecule_factory import MoleculeFactory

        if cfg.get("molecule") is None or cfg.get("calculator") is None:
            raise ValueError(
                "A simulation config needs top-level 'molecule:' and 'calculator:' "
                f"sections; got top-level keys {sorted(cfg.keys())}.  A config "
                "stored in a sub-directory must start with '# @package _global_', "
                "otherwise Hydra nests it (e.g. under 'md.simulations')."
            )

        atoms: Atoms = MoleculeFactory.from_config(cfg.molecule, default_key="from_smiles")
        calculator = CalculatorFactory.from_config(cfg.calculator, default_key="goal_model")

        # Simulation parameters
        sim_keys = {f.name for f in dataclasses.fields(SimulationConfig)}
        sim_params = {k: v for k, v in cfg.items() if k in sim_keys}
        if "steps" not in sim_params and cfg.get("num_steps") is not None:
            sim_params["steps"] = int(cfg.num_steps)

        return cls(atoms=atoms, calculator=calculator, **sim_params)


def simulate_from_config(cfg: typing.Any) -> RunResult:
    """Hydra-compatible entrypoint: build and run a simulation from config.

    Designed to be the ``_target_`` in simulation config files::

        # src/goal/configs/md/simulations/langevin_with_model.yaml
        _target_: goal.md.core.simulation.simulate_from_config

    Parameters
    ----------
    cfg : DictConfig
        Full simulation config.

    Returns
    -------
    RunResult
        Simulation summary.
    """
    sim = Simulation.from_config(cfg)
    return sim.run()
