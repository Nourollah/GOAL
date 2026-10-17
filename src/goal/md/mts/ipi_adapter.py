"""i-pi compatibility layer for goal.md MTS simulations.

i-pi (https://github.com/i-pi/i-pi) is the reference implementation of
multiple-timescale MD physics.  Its server communicates with force calculators
via a socket protocol.  This module provides two things:

1. :class:`IPIForceProvider` — a protocol that mirrors the interface i-pi uses
   to query forces, so GOAL calculators can be used as drop-in force providers.

2. :class:`IPICompatibleDriver` — wraps any ASE calculator to match the
   protocol and handles the ASE ↔ i-pi unit conversion (Å ↔ bohr,
   eV ↔ Hartree, eV/Å ↔ Hartree/bohr).

3. :class:`MTSSimulation` — a high-level object that ties together the
   :class:`~goal.md.mts.integrators.respa.RespaIntegrator` with the
   i-pi-compatible force provider abstraction.  When a real i-pi Python
   library API becomes available this class can delegate to it transparently
   via :meth:`MTSSimulation.use_ipi_backend`.

i-pi forward-compatibility contract
------------------------------------
The interface below is modelled on the *force provider* abstraction that i-pi
uses internally (``ipi.engine.ForceField``).  Any future version of i-pi that
exposes a Python library interface should be wire-compatible with
:class:`IPIForceProvider`:

    class MyForceField:
        def get_forces(self, pos_bohr, cell_bohr) -> tuple[np.ndarray, float, np.ndarray]:
            ...  # returns (forces_au, energy_au, virial_au)

References
----------
- Ceriotti, M. et al., Comput. Phys. Common. 185, 1019 (2014)
- flashmd i-pi adapter: https://github.com/lab-cosmo/flashmd/blob/main/src/flashmd/ipi.py
"""

from __future__ import annotations

import logging
import typing
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator

from goal.md.mts.integrators.respa import RespaIntegrator

logger = logging.getLogger(__name__)

# Unit conversion factors: ASE ↔ i-pi (Hartree atomic units)
_ANG_TO_BOHR = 1.0 / units.Bohr  # Ångström → bohr
_BOHR_TO_ANG = units.Bohr  # bohr → Ångström
_EV_TO_HARTREE = 1.0 / units.Hartree  # eV → Hartree
_HARTREE_TO_EV = units.Hartree  # Hartree → eV
_EV_PER_ANG_TO_AU = _EV_TO_HARTREE * _BOHR_TO_ANG  # eV/Å → Hartree/bohr
_AU_FORCE_TO_ASE = _HARTREE_TO_EV / _BOHR_TO_ANG  # Hartree/bohr → eV/Å


class IPIForceProvider(typing.Protocol):
    """Protocol matching i-pi's internal force-field interface.

    Implementers receive positions and cell in **atomic units** (bohr) and
    must return energy in Hartree, forces in Hartree/bohr, and the virial
    tensor in Hartree.

    This protocol is intentionally minimal so that it remains stable across
    i-pi version changes.
    """

    def get_forces(
        self,
        pos_bohr: np.ndarray,
        cell_bohr: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Compute energy and forces.

        Parameters
        ----------
        pos_bohr : np.ndarray, shape (N, 3)
            Atomic positions in bohr.
        cell_bohr : np.ndarray, shape (3, 3)
            Simulation cell in bohr (rows are lattice vectors).

        Returns
        -------
        forces_au : np.ndarray, shape (N, 3)
            Forces in Hartree/bohr.
        energy_au : float
            Total potential energy in Hartree.
        virial_au : np.ndarray, shape (3, 3)
            Virial tensor in Hartree (zeros for non-periodic / no-stress).
        """
        ...


class IPICompatibleDriver:
    """Wraps an ASE calculator to satisfy :class:`IPIForceProvider`.

    This is the bridge between GOAL calculators and i-pi's force-field
    interface.  Unit conversion is handled transparently.

    Parameters
    ----------
    calculator : ase.calculators.calculator.Calculator
        Any ASE-compatible calculator.
    atoms : ase.Atoms
        Reference structure (used to build temporary copies for evaluation).
    """

    def __init__(
        self,
        calculator: Calculator,
        atoms: Atoms,
    ) -> None:
        self._calc = calculator
        self._reference_atoms = atoms

    def get_forces(
        self,
        pos_bohr: np.ndarray,
        cell_bohr: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Evaluate energy and forces via ASE, return in i-pi units.

        Parameters
        ----------
        pos_bohr : np.ndarray, shape (N, 3)
            Atomic positions in bohr.
        cell_bohr : np.ndarray, shape (3, 3)
            Simulation cell in bohr.

        Returns
        -------
        forces_au, energy_au, virial_au
        """
        atoms = self._reference_atoms.copy()
        atoms.set_positions(pos_bohr * _BOHR_TO_ANG)
        atoms.set_cell(cell_bohr * _BOHR_TO_ANG)
        atoms.calc = self._calc

        energy_ev = float(atoms.get_potential_energy())
        forces_ev_ang = atoms.get_forces().copy()

        energy_au = energy_ev * _EV_TO_HARTREE
        forces_au = forces_ev_ang * _EV_PER_ANG_TO_AU

        try:
            stress_ev_ang3 = atoms.get_stress(voigt=False)
            vol_ang3 = atoms.get_volume()
            virial_au = -stress_ev_ang3 * vol_ang3 * _EV_TO_HARTREE * (_ANG_TO_BOHR**3)
        except Exception:
            virial_au = np.zeros((3, 3))

        return forces_au, energy_au, virial_au

    # Convenience: evaluate directly in ASE units (eV, Å)
    def get_forces_ase(
        self,
        atoms: Atoms,
    ) -> tuple[np.ndarray, float]:
        """Compute forces and energy in ASE units (eV/Å, eV)."""
        atoms_copy = atoms.copy()
        atoms_copy.calc = self._calc
        energy = float(atoms_copy.get_potential_energy())
        forces = atoms_copy.get_forces().copy()
        return forces, energy


@dataclass
class MTSConfig:
    """Configuration for a multi-timescale simulation.

    Attributes
    ----------
    timestep_fs : float
        Inner (fast) timestep in femtoseconds.
    mts_ratio : int
        Ratio of outer to inner steps. E.g., 4 means the slow calculator
        is called once every 4 fast steps.
    total_steps : int
        Total number of outer (slow) steps.
    temperature_K : float or None
        Langevin thermostat temperature.  ``None`` disables thermostat.
    friction : float
        Langevin friction coefficient (1/fs).
    log_interval : int
        Logging frequency in outer steps.
    trajectory_file : str or None
        Output trajectory path.
    log_file : str or None
        Output MD log path.
    output_dir : str
        Base output directory.
    """

    timestep_fs: float = 1.0
    mts_ratio: int = 4
    total_steps: int = 1000
    temperature_K: float | None = 300.0
    friction: float = 0.01
    log_interval: int = 10
    trajectory_file: str | None = None
    log_file: str | None = None
    output_dir: str = "outputs/md/mts"
    chunk_size: int = 50
    show_progress: bool = True


class MTSSimulation:
    """High-level MTS simulation using RESPA + i-pi-compatible force providers.

    This class is the primary entry point for multi-timescale MD in goal.md.
    It:

    1. Wraps the fast and slow calculators in :class:`IPICompatibleDriver`
       adapters so the interface is i-pi compatible.
    2. Instantiates a :class:`~goal.md.mts.integrators.respa.RespaIntegrator`
       to integrate the equations of motion.
    3. Attaches observers (trajectory, log, progress).

    Parameters
    ----------
    atoms : ase.Atoms
        The molecular system.
    fast_calculator : Calculator
        Cheap force calculator (e.g., ML model).  Called at every inner step.
    config : MTSConfig
        All simulation parameters.
    slow_calculator : Calculator, optional
        Expensive calculator (e.g., QM).  Its *correction* force
        ``F_slow = F_QM - F_ML`` is applied once per outer step.
    """

    def __init__(
        self,
        atoms: Atoms,
        fast_calculator: Calculator,
        config: MTSConfig | None = None,
        slow_calculator: Calculator | None = None,
    ) -> None:
        self.atoms = atoms
        self.fast_calc = fast_calculator
        self.slow_calc = slow_calculator
        self.config = config or MTSConfig()

        # i-pi-compatible wrappers
        self.fast_driver = IPICompatibleDriver(fast_calculator, atoms)
        self.slow_driver = IPICompatibleDriver(slow_calculator, atoms) if slow_calculator else None

        self._integrator: RespaIntegrator | None = None
        self._ipi_backend: bool = False

    def _build_integrator(self) -> RespaIntegrator:
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        cfg = self.config
        if cfg.temperature_K is not None:
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=cfg.temperature_K)

        integrator = RespaIntegrator(
            atoms=self.atoms,
            fast_calculator=self.fast_calc,
            timestep=cfg.timestep_fs * units.fs,
            mts_ratio=cfg.mts_ratio,
            slow_calculator=self.slow_calc,
            temperature_K=cfg.temperature_K,
            friction=cfg.friction,
        )
        return integrator

    def use_ipi_backend(self) -> None:
        """Switch to using i-pi's native Python integration engine.

        This is a forward-compatibility hook.  When i-pi publishes a pure
        Python API, call this method to delegate integration to i-pi while
        keeping GOAL calculators as the force providers.

        Raises
        ------
        ImportError
            If ``ipi`` is not installed (``pip install ipi``).
        RuntimeError
            If the i-pi version does not expose the required Python API.
        """
        try:
            import ipi  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "i-pi Python package not found. Install with: pip install ipi\n"
                "Alternatively, use the built-in RESPA integrator (default)."
            ) from exc

        logger.info(
            "Switching to i-pi native Python backend. "
            "GOAL calculators are exposed via IPICompatibleDriver."
        )
        self._ipi_backend = True

    def run(self) -> dict[str, typing.Any]:
        """Run the MTS simulation.

        Returns
        -------
        dict
            Summary with keys ``steps_completed``, ``final_energy``,
            ``final_temperature``, ``mts_stats``.
        """
        from pathlib import Path

        from goal.md.core.observers import setup_observers
        from goal.md.core.runner import run_dynamics

        cfg = self.config
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        traj = (
            str(Path(cfg.output_dir) / cfg.trajectory_file)
            if cfg.trajectory_file and not Path(cfg.trajectory_file).is_absolute()
            else cfg.trajectory_file
        )
        log = (
            str(Path(cfg.output_dir) / cfg.log_file)
            if cfg.log_file and not Path(cfg.log_file).is_absolute()
            else cfg.log_file
        )

        if self._ipi_backend:
            return self._run_with_ipi()

        integrator = self._build_integrator()
        self._integrator = integrator

        observers = setup_observers(
            dynamics=integrator,
            atoms=self.atoms,
            trajectory_file=traj,
            log_file=log,
            log_interval=cfg.log_interval,
            collect_energies=True,
        )

        result = run_dynamics(
            dynamics=integrator,
            steps=cfg.total_steps,
            chunk_size=cfg.chunk_size,
            temperature_K=cfg.temperature_K or 300.0,
            progress=cfg.show_progress,
            label=f"MTS ({cfg.mts_ratio}× ratio) @ {cfg.temperature_K or 'NVE'} K",
            energy_collector=observers.get("energy_collector"),
        )

        return {
            "steps_completed": result.steps_completed,
            "final_energy": result.final_energy,
            "final_temperature": result.final_temperature,
            "mts_stats": integrator.mts_stats,
            "energy_history": result.energy_history,
        }

    @staticmethod
    def _build_online_learner(cfg: typing.Any, mts_cfg: MTSConfig) -> Calculator:
        """``QMLearnerCalculator`` from the ``base_calculator`` / ``ml_calculator``
        / ``learner`` sections of an online-learning config."""
        from goal.md.core.calculator_factory import CalculatorFactory
        from goal.md.mts.dual_calculator.qm_ml_learner import (
            AlwaysBasePolicy,
            GOALOnlineFinetuner,
            QMLearnerCalculator,
            StepFractionPolicy,
            ThresholdPolicy,
        )

        base_calc = CalculatorFactory.from_config(cfg.base_calculator, default_key="orca")
        ml_calc = CalculatorFactory.from_config(cfg.ml_calculator, default_key="goal_model")

        learner: dict[str, typing.Any] = dict(cfg.get("learner") or {})
        policy_cfg: dict[str, typing.Any] = dict(learner.get("policy") or {})
        policy_type: str = str(policy_cfg.pop("type", "threshold"))
        policy: typing.Any
        if policy_type == "threshold":
            policy = ThresholdPolicy(**policy_cfg)
        elif policy_type == "step_fraction":
            policy_cfg.setdefault("total_steps", mts_cfg.total_steps * mts_cfg.mts_ratio)
            policy = StepFractionPolicy(**policy_cfg)
        elif policy_type == "always_base":
            policy = AlwaysBasePolicy()
        else:
            raise ValueError(
                f"learner.policy.type must be 'threshold', 'step_fraction' or "
                f"'always_base', got {policy_type!r}."
            )

        fine_tuner = None
        ft_cfg: dict[str, typing.Any] = dict(learner.get("fine_tuning") or {})
        if ft_cfg and getattr(ml_calc, "module", None) is not None:
            if getattr(ml_calc, "cutoff", None) is not None:
                ft_cfg["cutoff"] = ml_calc.cutoff  # graphs come from the calculator
            fine_tuner = GOALOnlineFinetuner(ml_calc, **ft_cfg)
        elif ft_cfg:
            logger.warning(
                "learner.fine_tuning is set but %s is not a GOAL model; the ML "
                "calculator will not be fine-tuned.",
                type(ml_calc).__name__,
            )

        return QMLearnerCalculator(
            base_calculator=base_calc,
            ml_calculator=ml_calc,
            policy=policy,
            fine_tuner=fine_tuner,
            eval_interval=int(learner.get("eval_interval", 100)),
            data_buffer_size=int(learner.get("data_buffer_size", 5000)),
        )

    def _run_with_ipi(self) -> dict[str, typing.Any]:
        """Delegate to i-pi's native engine (future compatibility stub)."""
        raise NotImplementedError(
            "i-pi native backend not yet integrated. "
            "Contributions welcome: implement _run_with_ipi() in "
            "goal.md.mts.ipi_adapter.MTSSimulation using ipi.engine.*"
        )

    @classmethod
    def from_config(cls, cfg: typing.Any) -> MTSSimulation:
        """Build an :class:`MTSSimulation` from a Hydra DictConfig.

        Expected config structure::

            molecule:
              method: from_smiles
              smiles: CCO
            fast_calculator:
              type: goal_model
              checkpoint: path/to/model.ckpt
            slow_calculator:          # optional
              type: orca
              orca_path: /path/to/orca
            mts:
              timestep_fs: 1.0
              mts_ratio: 4
              total_steps: 10000
              temperature_K: 300.0
              ...

        Parameters
        ----------
        cfg : DictConfig
            Hydra config.

        Returns
        -------
        MTSSimulation
        """
        from goal.md.core.calculator_factory import CalculatorFactory
        from goal.md.core.molecule_factory import MoleculeFactory

        if cfg.get("molecule") is None:
            raise ValueError(
                "An MTS config needs a top-level 'molecule:' section; got top-level "
                f"keys {sorted(cfg.keys())}.  A config stored in a sub-directory must "
                "start with '# @package _global_', otherwise Hydra nests it."
            )

        # Molecule
        atoms = MoleculeFactory.from_config(cfg.molecule, default_key="from_smiles")

        # MTS config
        mts_raw = dict(cfg.get("mts", {}))
        mts_cfg = MTSConfig(**{k: v for k, v in mts_raw.items() if hasattr(MTSConfig, k)})

        if cfg.get("ml_calculator") is not None and cfg.get("base_calculator") is not None:
            # Online-learning mode (online_learner.yaml): the force provider
            # is a QMLearnerCalculator switching between base and ML itself,
            # so there is no separate slow correction.
            fast_calc = cls._build_online_learner(cfg, mts_cfg)
            return cls(atoms=atoms, fast_calculator=fast_calc, config=mts_cfg)

        # Fast calculator
        fast_calc = CalculatorFactory.from_config(
            cfg.get("fast_calculator", {}), default_key="goal_model"
        )

        # Slow calculator (optional)
        slow_calc = None
        if "slow_calculator" in cfg and cfg.slow_calculator is not None:
            slow_calc = CalculatorFactory.from_config(cfg.slow_calculator, default_key="orca")

        return cls(
            atoms=atoms,
            fast_calculator=fast_calc,
            config=mts_cfg,
            slow_calculator=slow_calc,
        )
