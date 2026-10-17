"""Batched MD and geometry relaxation of many structures with one GOAL model.

Two backends, one interface:

* ``"nvalchemi"`` — every structure in one GPU-resident batch, integrated
  by nvalchemi-toolkit (``NVTLangevin`` for MD, ``FIRE2`` for relaxation)
  through :class:`~goal.ml.adapters.nvalchemi_wrapper.SIMURGHNvalchemiWrapper`.
  NVIDIA GPUs only.
* ``"ase"`` — the structures one after another with ASE (``Langevin`` /
  ``FIRE``) and :class:`~goal.ml.utils.calculator.GOALCalculator`.  Runs
  everywhere: CPU, AMD ROCm, Apple MPS, NVIDIA.

``backend="auto"`` (default) picks ``"nvalchemi"`` when nvalchemi-toolkit and
an NVIDIA GPU are present, otherwise ``"ase"``, and logs the choice.

Units follow the rest of goal.md: time in fs, friction in 1/fs,
temperature in K, energies in eV, forces in eV/Å.

Usage::

    from goal.md.core.batched import BatchSimulationConfig, run_batched_simulation

    result = run_batched_simulation(
        BatchSimulationConfig(
            checkpoint="logs/.../checkpoints",
            structures=["data/frames.xyz"],
            steps=2000, temperature_K=300.0, timestep_fs=0.5,
        )
    )
    result.final_structures[0].get_potential_energy()

or ``goal-simulate-batch --checkpoint ... --structures data/frames.xyz``.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import typing
from pathlib import Path

import numpy as np
from ase import Atoms, units

log = logging.getLogger(__name__)

MODES: tuple[str, ...] = ("md", "relax")
BACKENDS: tuple[str, ...] = ("auto", "nvalchemi", "ase")
SAMPLERS: tuple[str, ...] = ("langevin", "metadynamics", "remd")
TRAJECTORY_FORMATS: tuple[str, ...] = ("extxyz", "zarr")


@dataclasses.dataclass
class BatchSimulationConfig:
    """Settings for :func:`run_batched_simulation`.

    Parameters
    ----------
    checkpoint : str
        GOAL checkpoint: ``.simurgh`` archive, checkpoint directory or ``.ckpt``.
    structures : list of str
        Structure files read with ``ase.io.read(path, index=":")`` — every
        frame of every file is simulated.
    steps : int
        MD steps, or the maximum number of optimiser steps for ``relax``.
    timestep_fs : float
        MD timestep in fs (passed to ASE as ``timestep_fs * ase.units.fs``).
    temperature_K : float
        Langevin target temperature (K); also the initial
        Maxwell–Boltzmann temperature.
    friction : float
        Langevin friction in 1/fs.
    fmax : float
        Force convergence for ``relax`` (eV/Å).
    mode : str
        ``"md"`` or ``"relax"``.
    backend : str
        ``"auto"``, ``"nvalchemi"`` or ``"ase"``.
    sampler : str
        ``"langevin"``; ``"metadynamics"`` / ``"remd"`` are roadmap
        placeholders (see :mod:`goal.ml.utils.enhanced_sampling`).
    output_dir : str
        Where trajectories and the final structures go.
    write_trajectory : bool
        Write frames every ``log_every`` steps.
    trajectory_format : str
        ``"extxyz"`` (one file per structure) or ``"zarr"`` (one store,
        GOAL's Zarr layout — needs ``zarr``).
    log_every : int
        Logging / trajectory interval in steps.
    device : str
        Torch device for the model (``"auto"``: CUDA when available).
    checkpoint_select : str
        Selector inside a checkpoint directory (``best``, ``last``, …).
    seed : int
        Seed for the initial velocities and the thermostat.
    """

    checkpoint: str = ""
    structures: list[str] = dataclasses.field(default_factory=list)
    steps: int = 1000
    timestep_fs: float = 0.5
    temperature_K: float = 300.0
    friction: float = 0.01
    fmax: float = 0.05
    mode: str = "md"
    backend: str = "auto"
    sampler: str = "langevin"
    output_dir: str = "outputs/md/batch"
    write_trajectory: bool = True
    trajectory_format: str = "extxyz"
    log_every: int = 100
    device: str = "auto"
    checkpoint_select: str = "best"
    seed: int = 42

    def validate(self) -> None:
        """Check the settings; called by :func:`run_batched_simulation`.

        Raises
        ------
        ValueError
            If ``mode``, ``backend``, ``sampler`` or ``trajectory_format``
            is not one of the allowed values, ``checkpoint`` or
            ``structures`` is empty, or ``steps`` or ``log_every`` is
            below 1.
        """
        for name, value, allowed in (
            ("mode", self.mode, MODES),
            ("backend", self.backend, BACKENDS),
            ("sampler", self.sampler, SAMPLERS),
            ("trajectory_format", self.trajectory_format, TRAJECTORY_FORMATS),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}.")
        if not self.checkpoint:
            raise ValueError("checkpoint is required.")
        if not self.structures:
            raise ValueError("structures is required (one or more structure files).")
        if self.steps < 1 or self.log_every < 1:
            raise ValueError("steps and log_every must be >= 1.")


@dataclasses.dataclass
class BatchSimulationResult:
    """Outcome of a batched run.

    ``steps_completed`` is the number of steps actually taken: the largest
    per-structure count for ASE (a relaxation stops at convergence), the
    integrated step count for nvalchemi.
    """

    backend: str
    final_structures: list[Atoms]
    final_energies: list[float]
    steps_completed: int
    wall_time_s: float
    output_files: list[str] = dataclasses.field(default_factory=list)


def resolve_backend(requested: str) -> str:
    """``"auto"`` → ``"nvalchemi"`` on NVIDIA with nvalchemi-toolkit, else ``"ase"``."""
    from goal.ml.utils.hardware import nvalchemi_available
    from goal.ml.utils.hardware import resolve_backend as _resolve

    return _resolve(
        requested,
        fast="nvalchemi",
        fallback="ase",
        available=nvalchemi_available,
        feature="batched simulation",
    )


def load_structures(paths: typing.Sequence[str]) -> list[Atoms]:
    """Every frame of every file (``ase.io.read(path, index=":")``)."""
    from ase.io import read

    frames: list[Atoms] = []
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Structure file not found: {path}")
        read_frames = read(path, index=":")
        frames.extend(read_frames if isinstance(read_frames, list) else [read_frames])
    if not frames:
        raise ValueError(f"No structures found in {list(paths)}.")
    return frames


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def run_batched_simulation(config: BatchSimulationConfig) -> BatchSimulationResult:
    """Run MD or relaxation for every structure in ``config.structures``.

    Raises
    ------
    NotImplementedError
        For the roadmap samplers (``metadynamics``, ``remd``).
    """
    config.validate()
    if config.sampler != "langevin":
        from goal.ml.utils.enhanced_sampling import SAMPLERS as _ENHANCED

        _ENHANCED[config.sampler](config)
    frames = load_structures(config.structures)
    backend = resolve_backend(config.backend)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Batched %s of %d structure(s), %d steps, backend=%s",
        config.mode,
        len(frames),
        config.steps,
        backend,
    )
    start = time.perf_counter()
    if backend == "nvalchemi":
        try:
            final, files, steps = _run_nvalchemi(frames, config, out_dir)
        except NotImplementedError as exc:  # e.g. a fragment-aware checkpoint
            if config.backend != "auto":
                raise ValueError(f"backend 'nvalchemi' cannot run this checkpoint: {exc}") from exc
            log.warning("nvalchemi backend unavailable for this checkpoint (%s); using ASE.", exc)
            backend = "ase"
    if backend == "ase":
        final, files, steps = _run_ase(frames, config, out_dir)
    from ase.io import write

    final_path = out_dir / "final.extxyz"
    write(final_path, final, format="extxyz")
    files.append(str(final_path))
    energies = [float(a.info.get("energy", np.nan)) for a in final]
    return BatchSimulationResult(
        backend=backend,
        final_structures=final,
        final_energies=energies,
        steps_completed=steps,
        wall_time_s=time.perf_counter() - start,
        output_files=files,
    )


# ---------------------------------------------------------------------------
# Trajectory writing (shared)
# ---------------------------------------------------------------------------


class _TrajectoryWriter:
    """Append frames per structure, as extxyz files or one Zarr store."""

    def __init__(self, config: BatchSimulationConfig, out_dir: Path, n_structures: int) -> None:
        self.enabled = config.write_trajectory
        self.format = config.trajectory_format
        self.out_dir = out_dir
        self.n = n_structures
        self._zarr_frames: list[Atoms] = []
        self.files: list[str] = []
        if self.enabled and self.format == "extxyz":
            self.files = [str(out_dir / f"traj_{i:04d}.extxyz") for i in range(n_structures)]
            for f in self.files:
                Path(f).unlink(missing_ok=True)

    def add(self, index: int, atoms: Atoms) -> None:
        if not self.enabled:
            return
        if self.format == "extxyz":
            from ase.io import write

            write(self.files[index], atoms, format="extxyz", append=True)
        else:
            frame = atoms.copy()
            frame.info["structure_index"] = index
            self._zarr_frames.append(frame)

    def close(self) -> list[str]:
        if self.enabled and self.format == "zarr" and self._zarr_frames:
            from goal.ml.data.zarr_dataset import write_zarr

            # Same order for every backend: by structure, then by step.
            frames = sorted(
                self._zarr_frames, key=lambda a: (a.info["structure_index"], a.info["step"])
            )
            store = self.out_dir / "trajectory.zarr"
            write_zarr(
                frames,
                store,
                split="trajectory",
                overwrite=True,
                info_keys=("structure_index", "step"),
            )
            self.files = [str(store)]
        return self.files


def _labelled(atoms: Atoms, energy: float, forces: np.ndarray, step: int) -> Atoms:
    frame = atoms.copy()
    frame.calc = None
    frame.info["energy"] = float(energy)
    frame.info["step"] = int(step)
    frame.arrays["forces"] = np.asarray(forces, dtype=float)
    return frame


# ---------------------------------------------------------------------------
# ASE backend (portable)
# ---------------------------------------------------------------------------


def _run_ase(
    frames: list[Atoms], config: BatchSimulationConfig, out_dir: Path
) -> tuple[list[Atoms], list[str], int]:
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.optimize import FIRE

    from goal.ml.utils.calculator import GOALCalculator

    calc = GOALCalculator(
        checkpoint_path=config.checkpoint,
        checkpoint=config.checkpoint_select,
        device=_device(config.device),
    )
    writer = _TrajectoryWriter(config, out_dir, len(frames))
    rng = np.random.default_rng(config.seed)
    final: list[Atoms] = []
    steps_taken = 0
    for index, source in enumerate(frames):
        atoms = source.copy()
        atoms.calc = calc
        if config.mode == "md":
            MaxwellBoltzmannDistribution(atoms, temperature_K=config.temperature_K, rng=rng)
            dyn: typing.Any = Langevin(
                atoms,
                timestep=config.timestep_fs * units.fs,
                temperature_K=config.temperature_K,
                friction=config.friction / units.fs,
                rng=rng,
            )
        else:
            dyn = FIRE(atoms, logfile=None)

        def _record(atoms: Atoms = atoms, index: int = index, dyn: typing.Any = dyn) -> None:
            if dyn.nsteps > 0:  # ASE also calls observers before the first step
                writer.add(
                    index,
                    _labelled(atoms, atoms.get_potential_energy(), atoms.get_forces(), dyn.nsteps),
                )

        dyn.attach(_record, interval=config.log_every)
        if config.mode == "md":
            dyn.run(config.steps)
        else:
            dyn.run(fmax=config.fmax, steps=config.steps)
        steps_taken = max(steps_taken, int(dyn.nsteps))
        final.append(
            _labelled(atoms, atoms.get_potential_energy(), atoms.get_forces(), int(dyn.nsteps))
        )
        log.info(
            "[ase] structure %d/%d done: E = %.6f eV",
            index + 1,
            len(frames),
            final[-1].info["energy"],
        )
    return final, writer.close(), steps_taken


# ---------------------------------------------------------------------------
# nvalchemi backend (NVIDIA GPUs)
# ---------------------------------------------------------------------------


def _to_atomic_data(
    atoms: Atoms, device: str, dtype: typing.Any, velocities: np.ndarray | None
) -> typing.Any:
    """ASE ``Atoms`` → nvalchemi ``AtomicData`` with every field present.

    ``cell`` / ``pbc`` are always set (identity cell, ``pbc=False`` for
    molecules): ``Batch.from_data_list`` keeps only the fields of the first
    structure, so a molecule first in a mixed batch would otherwise drop
    the periodic cells of the rest.  Velocities are in ASE units
    (Å per 10.18 fs), which is nvalchemi's internal velocity unit.
    """
    import torch
    from nvalchemi.data import AtomicData

    n = len(atoms)
    periodic = bool(np.any(atoms.pbc))
    cell = np.array(atoms.cell) if periodic else np.eye(3)
    fields = {
        "positions": torch.tensor(atoms.positions, dtype=dtype, device=device),
        "atomic_numbers": torch.tensor(atoms.numbers, dtype=torch.long, device=device),
        "atomic_masses": torch.tensor(atoms.get_masses(), dtype=dtype, device=device),
        "cell": torch.tensor(cell, dtype=dtype, device=device).reshape(1, 3, 3),
        "pbc": torch.tensor(atoms.pbc, dtype=torch.bool, device=device).reshape(1, 3),
        "forces": torch.zeros(n, 3, dtype=dtype, device=device),
        "energy": torch.zeros(1, 1, dtype=dtype, device=device),
    }
    if velocities is not None:
        fields["velocities"] = torch.tensor(velocities, dtype=dtype, device=device)
    return AtomicData(**fields)


def _from_batch(batch: typing.Any, templates: list[Atoms], step: int) -> list[Atoms]:
    """nvalchemi ``Batch`` → labelled ASE frames (positions, velocities, E, F)."""
    positions = batch.positions.detach().cpu().numpy()
    forces = batch.forces.detach().cpu().numpy()
    energies = batch.energy.detach().cpu().numpy().reshape(-1)
    velocities = getattr(batch, "velocities", None)
    velocities = velocities.detach().cpu().numpy() if velocities is not None else None
    ptr = batch.batch_ptr.detach().cpu().numpy()
    frames: list[Atoms] = []
    for g, template in enumerate(templates):
        lo, hi = int(ptr[g]), int(ptr[g + 1])
        atoms = template.copy()
        atoms.positions = positions[lo:hi]
        if velocities is not None:
            atoms.set_velocities(velocities[lo:hi])
        frames.append(_labelled(atoms, energies[g], forces[lo:hi], step))
    return frames


def _run_nvalchemi(
    frames: list[Atoms], config: BatchSimulationConfig, out_dir: Path
) -> tuple[list[Atoms], list[str], int]:
    import torch
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from nvalchemi.data import Batch
    from nvalchemi.dynamics import FIRE2, ConvergenceHook, NVTLangevin
    from nvalchemi.dynamics.hooks import NaNDetectorHook

    from goal.ml.adapters.nvalchemi_wrapper import SIMURGHNvalchemiWrapper

    device = _device(config.device)
    model = SIMURGHNvalchemiWrapper(
        config.checkpoint, device=device, checkpoint=config.checkpoint_select
    )
    dtype = next(model.parameters()).dtype
    rng = np.random.default_rng(config.seed)

    data = []
    for atoms in frames:
        velocities = None
        if config.mode == "md":
            atoms = atoms.copy()
            MaxwellBoltzmannDistribution(atoms, temperature_K=config.temperature_K, rng=rng)
            velocities = atoms.get_velocities()
        data.append(_to_atomic_data(atoms, device, dtype, velocities))
    batch = Batch.from_data_list(data, device=torch.device(device))

    hooks = [NaNDetectorHook()]
    if config.mode == "md":
        engine = NVTLangevin(
            model=model,
            dt=config.timestep_fs,  # fs
            temperature=config.temperature_K,
            friction=config.friction,  # 1/fs
            random_seed=config.seed,
            hooks=hooks,
        )
    else:
        engine = FIRE2(
            model=model,
            dt=0.05,
            hooks=hooks,
            convergence_hook=ConvergenceHook.from_fmax(config.fmax),
        )

    writer = _TrajectoryWriter(config, out_dir, len(frames))
    done = 0
    while done < config.steps:
        chunk = min(config.log_every, config.steps - done)
        batch = engine.run(batch, n_steps=chunk)
        done += chunk
        for index, frame in enumerate(_from_batch(batch, frames, done)):
            writer.add(index, frame)
        log.info("[nvalchemi] step %d/%d", done, config.steps)
    return _from_batch(batch, frames, done), writer.close(), done
