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
    sim   = Simulation(atoms, calc, dynamics_key="langevin_ase", steps=5000, temperature_K=300.0, friction=1.0)
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
import warnings

import dataclasses
import typing
from pathlib import Path

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from goal.md.core.observers import setup_observers
from goal.md.core.runner import RunResult, run_dynamics

from goal.md.core import ASE_DYNAMICS_PARAMS, FLASHMD_DYNAMICS_PARAMS

SUPPORTED_DYNAMICS_PARAMS = {
    "langevin_ase": ASE_DYNAMICS_PARAMS,
    "langevin_flashmd": FLASHMD_DYNAMICS_PARAMS
}

@dataclasses.dataclass
class SimulationConfig:
    """Flat configuration for a single MD run."""

    steps: int = 1000
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

    All time-related parameters use ASE/physical units:
    - temperature in Kelvin
    - timestep in femtoseconds
    - time constant in femtoseconds

    Parameters
    ----------
    atoms : ase.Atoms
        The molecular system to simulate.
    calculator : ase.calculators.calculator.Calculator
        Force/energy calculator.
    dynamics_key : str
        The dynamics method to be used for the simulation ("langevin_ase" or "langevin_flashmd")
    dynamics_cfg : dict
        The configuration parameters used to create a dynamics object.
    steps : int
        Number of MD steps.
    temperature_K : float
        Target temperature in Kelvin.
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
    timestep : float
        Integration timestep in ASE units.
    timeste_fs : float
        Integration timestep in femtoseconds - for 'langevin_ase' dynamics method.
    fixcm : bool
        Fix Centre of Mass. Passed to `DynamicsFactory.create`.
    friction : float
        Friction coefficient in inverse ASE time units - for 'langevin_ase' dynamics method.
    friction_per_fs : float
        Friction coefficient in 1/fs - for 'langevin_ase' dynamics method.
    time_constant: float
        Thermostat time constant in fs - for 'langevin_flashmd' dynamics method.
    model_name : str
        FlashMD model name - for 'langevin_flashmd' dynamics method.
    device : str
        Torch device ("cuda" or "cpu") - for 'langevin_flashmd' dynamics method.
    **kwargs:
        Additional arguments for building the dynamics object

    """

    def __init__(
        self,
        atoms: Atoms,
        calculator: Calculator,
        dynamics_key: str | None = None,
        dynamics_cfg: dict = dict(),
        steps: int = 1000,
        temperature_K: float = 300.0,
        chunk_size: int = 100,
        restart_thermostat: bool = False,
        log_interval: int = 10,
        trajectory_file: str | None = None,
        log_file: str | None = None,
        output_dir: str = "outputs/md",
        show_progress: bool = True,
        initialise_velocities: bool = True,
        timestep: float | None = None,
        timestep_fs: float | None = None,
        fixcm: bool | None = None,
        friction: float | None = None,
        friction_per_fs: float | None = None,
        time_constant: float | None = None,
        model_name: str | None = None,
        device: str | None = None,
        **kwargs: typing.Any

    ) -> None:
        self.atoms = atoms
        self.atoms.calc = calculator
        self.calculator = calculator
        self.dynamics_key = dynamics_key or "langevin_ase" # Set 'langevin_ase' if no value is provided.
        self.dynamics_cfg = dict(dynamics_cfg) # Create another dictionary instance to keep the original configurations intact.
        self.steps = steps
        self.temperature_K = temperature_K
        self.chunk_size = chunk_size
        self.restart_thermostat = restart_thermostat
        self.log_interval = log_interval
        self.output_dir = Path(output_dir)
        self.show_progress = show_progress
        self.initialise_velocities = initialise_velocities
        self.dyn_kwargs = kwargs

        self._traj_file = self._resolve_path(trajectory_file)
        self._log_file = self._resolve_path(log_file)

        self._dynamics: Object | None = None
        self._observers: dict[str, typing.Any] = {}

        assert self.dynamics_key in SUPPORTED_DYNAMICS_PARAMS.keys(), f"`dynamics_key` must be either 'langevin_ase' or 'langevin_flashmd'. Got '{self.dynamics_method}'"

        self.dynamics_cfg |= self.dyn_kwargs

        # Override the dynamics configuration values when they are provided by a library call.
        overrides = {
        # Values common for both 'langevin_ase' and 'langevin_flashmd':
                "timestep": timestep,
                "fixcm": fixcm,
                # Values for 'langevin_ase' dynamics method:
                "timestep_fs": timestep_fs,
                "friction": friction,
                "friction_fs": friction_fs,
                # Values for 'langevin_flashmd' dynamics method:
                "time_constant": time_constant,
                "model_name": model_name,
                "device": device
                }

        for key, value in overrides.items():
            if value is not None:
                self.dynamics_cfg[key] = value

        self._validate_dynamics_params()

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

    def _build_dynamics(self) -> Object:
        from goal.md.core.md_factory import DynamicsFactory

        dyn = DynamicsFactory.create(
            self.dynamics_key,
            self.atoms,
            temperature_K=self.temperature_K,
            **self.dynamics_cfg
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

    def _validate_dynamics_params(self) -> None:
        """Checks if the input dynamics parameters are supported by the relevant method. Otherwise, raises ValueError"""

        # Change 'model', used by `flashmd.ase.langevin.Langevin`, to 'model_name' as it is handled by `goal.md.core.md_factory.DynamicsFactory`.
        if self.dynamics_key == 'langevin_flashmd' and 'model' in self.dynamics_cfg.keys():
            self.dynamics_cfg['model_name'] = self.dynamics_cfg.pop('model')

            warning_msg = f"Use 'model_name' instead of 'model' as `goal.md.core.md_factory.DynamicsFactory` internally handles it. Refer to FlashMD part of `goal.md.core.md_factory.DynamicsFactory` for more information."
            warnings.warn(warning_msg)

        supported = SUPPORTED_DYNAMICS_PARAMS[self.dynamics_key]
        unsupported = set(self.dynamics_cfg.keys()) - supported

        if unsupported:
            names = ', '.join(sorted(unsupported))
            raise ValueError(
                f"Unsupported parameter{'' if len(unsupported) == 1 else 's'} for {self.dynamics_key!r}: {names}"
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
        - ``dynamics`` - passed to :meth:`DynamicsFactory.create`
          (builder named by ``key`` or ``method``)
        - All optional fields of :class:`SimulationConfig` (``steps``,
          ``chunk_size``, ``restart_thermostat``, ``log_interval`` …).  
          The older spelling ``num_steps`` is accepted for ``steps``.

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

        if cfg.get("molecule") is None or cfg.get("calculator") is None or cfg.get("dynamics") is None:
            raise ValueError(
                    "A simulation config needs top-level 'molecule:', 'calculator:', and the optional 'dynamics:' "
                f"sections; got top-level keys {sorted(cfg.keys())}.  A config "
                "stored in a sub-directory must start with '# @package _global_', "
                "otherwise Hydra nests it (e.g. under 'md.simulations')."
            )

        atoms: Atoms = MoleculeFactory.from_config(cfg.molecule, default_key="from_smiles")
        calculator = CalculatorFactory.from_config(cfg.calculator, default_key="goal_model")
        
        # Dynamics Parameters
        dyn_cfg = dict(cfg.get("dynamics", {}))

        # Make a copy of the dynamics config to avoid changes to the original config and remove keys Hydra reserves for ``instantiate``.
        dynamics_cfg: dict[str, typing.Any] = {
                str(k): v
                for k, v in dyn_cfg.items()
                if not (str(k).startswith('_') and str(k).startswith('_'))
                }

        temperature_K: float = dynamics_cfg.pop('temperature_K', 300.0)
        dynamics_key: str = dynamics_cfg.pop('key', None) or dynamics_cfg.pop('method', 'langevin_ase')

        # Simulation parameters
        sim_keys = {f.name for f in dataclasses.fields(SimulationConfig)}
        sim_params = {k: v for k, v in cfg.items() if k in sim_keys}
        if "steps" not in sim_params and cfg.get("num_steps") is not None:
            sim_params["steps"] = int(cfg.num_steps)

        return cls(
                atoms=atoms,
                calculator=calculator,
                dynamics_key=dynamics_key,
                dynamics_cfg=dynamics_cfg,
                temperature_K=temperature_K,
                **sim_params
                )


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
