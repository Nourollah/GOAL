"""Observer protocol and implementations for attaching to MD dynamics.

Observers are callables attached to an ASE dynamics object via
``dynamics.attach(observer, interval=n)``.  This module provides
typed implementations plus a single convenience factory ``setup_observers``
that wires everything up from a plain config dict / DictConfig.
"""

from __future__ import annotations

import typing
from pathlib import Path

import ase
from ase import io as ase_io
from ase.md import MDLogger


class TrajectoryObserver:
    """Writes structures to an ASE trajectory file.

    Parameters
    ----------
    atoms : ase.Atoms
        The simulated system (reference, not a copy).
    path : str or Path
        Destination ``.traj`` file.
    mode : str
        ``"w"`` to overwrite, ``"a"`` to append.
    """

    def __init__(self, atoms: ase.Atoms, path: str | Path, mode: str = "w") -> None:
        self._traj = ase_io.Trajectory(str(path), mode=mode, atoms=atoms)

    def __call__(self) -> None:
        self._traj.write()

    def close(self) -> None:
        """Close the underlying ASE trajectory file (also called by ``__del__``)."""
        self._traj.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class MDLogObserver:
    """Writes energy, temperature, and stress to a plain-text log.

    Wraps ASE's :class:`ase.md.MDLogger`.

    Parameters
    ----------
    dynamics : object
        ASE dynamics instance (must have ``.atoms`` attribute).
    atoms : ase.Atoms
        The simulated system.
    path : str or Path
        Destination log file.
    header : bool
        Write column header.
    stress : bool
        Include stress tensor in log.
    per_atom : bool
        Report energy per atom.
    mode : str
        ``"a"`` append, ``"w"`` overwrite.
    """

    def __init__(
        self,
        dynamics: object,
        atoms: ase.Atoms,
        path: str | Path,
        header: bool = True,
        stress: bool = False,
        per_atom: bool = True,
        mode: str = "a",
    ) -> None:
        self._logger = MDLogger(
            dynamics,
            atoms,
            logfile=str(path),
            header=header,
            stress=stress,
            peratom=per_atom,
            mode=mode,
        )

    def __call__(self) -> None:
        self._logger()


class EnergyCollector:
    """Collects energy values during a simulation for in-memory analysis.

    Parameters
    ----------
    atoms : ase.Atoms
        The simulated system.
    """

    def __init__(self, atoms: ase.Atoms) -> None:
        self._atoms = atoms
        self.energies: list[float] = []
        self.steps: list[int] = []
        self._step = 0

    def __call__(self) -> None:
        try:
            self.energies.append(float(self._atoms.get_potential_energy()))
        except Exception:
            self.energies.append(float("nan"))
        self.steps.append(self._step)
        self._step += 1


def setup_observers(
    dynamics: object,
    atoms: ase.Atoms,
    trajectory_file: str | None = None,
    log_file: str | None = None,
    log_interval: int = 10,
    trajectory_interval: int | None = None,
    log_stress: bool = False,
    log_per_atom: bool = True,
    trajectory_mode: str = "w",
    log_mode: str = "a",
    collect_energies: bool = False,
) -> dict[str, typing.Any]:
    """Attach observers to a dynamics object and return them.

    Parameters
    ----------
    dynamics : object
        ASE dynamics instance.
    atoms : ase.Atoms
        The simulated system.
    trajectory_file : str, optional
        Path for trajectory output.
    log_file : str, optional
        Path for plain-text MD log.
    log_interval : int
        How often (in steps) to write to all logs.
    trajectory_interval : int, optional
        Override interval for trajectory (falls back to ``log_interval``).
    log_stress : bool
        Include stress in MD log.
    log_per_atom : bool
        Report per-atom energies in MD log.
    trajectory_mode : str
        File mode for trajectory (``"w"``/``"a"``).
    log_mode : str
        File mode for MD log.
    collect_energies : bool
        Attach an :class:`EnergyCollector` to accumulate energies in memory.

    Returns
    -------
    dict
        Mapping of observer name to observer instance. Keys:
        ``"trajectory"``, ``"log"``, ``"energy_collector"``.
    """
    traj_interval = trajectory_interval if trajectory_interval is not None else log_interval
    observers: dict[str, typing.Any] = {}

    if trajectory_file:
        Path(trajectory_file).parent.mkdir(parents=True, exist_ok=True)
        obs = TrajectoryObserver(atoms, trajectory_file, mode=trajectory_mode)
        dynamics.attach(obs, interval=traj_interval)
        observers["trajectory"] = obs

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        obs = MDLogObserver(
            dynamics,
            atoms,
            log_file,
            stress=log_stress,
            per_atom=log_per_atom,
            mode=log_mode,
        )
        dynamics.attach(obs, interval=log_interval)
        observers["log"] = obs

    if collect_energies:
        collector = EnergyCollector(atoms)
        dynamics.attach(collector, interval=log_interval)
        observers["energy_collector"] = collector

    return observers
