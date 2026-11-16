"""RESPA (Reference System Propagator Algorithm) for multi-timescale MD.

This module implements the velocity-Verlet RESPA integrator (Tuckerman et al.,
J. Chem. Phys. 97, 1990–2001, 1992) that forms the physics core of i-pi's
MTS integration.  The design deliberately mirrors i-pi's internal separation
of ``motion`` (integrator) from ``forcefields`` (calculators) so that the
implementation stays forward-compatible: when i-pi exposes a native Python
library API the integration step can delegate to it transparently (see
:mod:`goal.md.mts.ipi_adapter`).

MTS overview
------------
The system Hamiltonian is split into fast and slow parts::

    H = H_fast + H_slow

Typical assignment for QM/ML MTS:

- **Fast forces** (inner loop, cheap):  ML model forces, computed at every
  inner step.
- **Slow forces** (outer loop, expensive):  QM correction
  ``F_slow = F_QM - F_ML_reference``, computed every ``mts_ratio`` steps.

One outer (slow) step of velocity-Verlet RESPA looks like::

    v  +=  F_slow / m  *  dt_outer / 2          # slow half-kick
    for _ in range(mts_ratio):
        v  +=  F_fast / m  *  dt_inner / 2      # fast half-kick
        x  +=  v  *  dt_inner                   # drift
        update F_fast
        v  +=  F_fast / m  *  dt_inner / 2      # fast half-kick
    update F_slow
    v  +=  F_slow / m  *  dt_outer / 2          # slow half-kick

When a Langevin thermostat is desired each full-step velocity update is
replaced by the stochastic Ornstein–Uhlenbeck half-kick (B-A-O-B-A-B
splitting, as in i-pi's ``barolangevin``).
"""

from __future__ import annotations

import logging
import typing

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.md.md import MolecularDynamics

logger = logging.getLogger(__name__)


class RespaIntegrator(MolecularDynamics):
    """Velocity-Verlet RESPA multi-timescale integrator.

    Integrates the equations of motion with two force levels:

    - **fast_calculator** — cheap potential, called at every inner step.
    - **slow_calculator** — expensive potential (or correction), called once
      per outer step.

    If ``slow_calculator`` is ``None`` the integrator reduces to standard
      single-timescale velocity-Verlet (useful for testing / debugging).

    Parameters
    ----------
    atoms : ase.Atoms
        The system to simulate.  Must already have masses set.
    fast_calculator : Calculator
        Inexpensive force calculator (ML model).
    timestep : float
        Inner (fast) timestep in ASE internal units.  Typically
        ``n_fs * ase.units.fs``.
    mts_ratio : int
        Number of fast steps per slow step.  ``slow_timestep = mts_ratio * timestep``.
    slow_calculator : Calculator or None
        Expensive force calculator.  When ``None``, the outer loop runs with
        zero slow forces (degenerates to normal Verlet).
    temperature_K : float or None
        If set, apply a Langevin thermostat at this temperature.
    friction : float
        Langevin friction coefficient (1/fs).  Used only when
        ``temperature_K`` is set.
    trajectory : str, optional
        Path to trajectory file for ASE MolecularDynamics base class.
    logfile : str, optional
        Path to log file for ASE MolecularDynamics base class.
    loginterval : int
        Log interval.
    """

    def __init__(
        self,
        atoms: Atoms,
        fast_calculator: Calculator,
        timestep: float,
        mts_ratio: int = 4,
        slow_calculator: Calculator | None = None,
        temperature_K: float | None = None,
        friction: float = 0.01,
        trajectory: str | None = None,
        logfile: str | None = None,
        loginterval: int = 1,
    ) -> None:
        # ASE's dt is the time per step(), i.e. the outer step (log time, get_time()).
        super().__init__(
            atoms,
            timestep=mts_ratio * timestep,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
        )
        self.fast_calc = fast_calculator
        self.slow_calc = slow_calculator
        self.mts_ratio = mts_ratio
        self.temperature_K = temperature_K
        self.friction = friction / units.fs

        # Observers (MDLogger, Trajectory) read atoms.calc: the slow level after
        # each outer step (its energy is the conserved one), the fast one without it.
        self.atoms.calc = fast_calculator

        self._dt_inner = timestep
        self._dt_outer = mts_ratio * timestep

        # Cached force arrays
        self._fast_forces: np.ndarray | None = None
        self._slow_forces: np.ndarray | None = None
        self.n_fast_calls = 0
        self.n_slow_calls = 0

        # Langevin pre-computed coefficients
        self._lang_c1: np.ndarray | None = None
        self._lang_c2: np.ndarray | None = None
        if temperature_K is not None:
            self._precompute_langevin(atoms)

    # ------------------------------------------------------------------
    # Langevin helpers
    # ------------------------------------------------------------------

    def _precompute_langevin(self, atoms: Atoms) -> None:
        """Pre-compute Ornstein–Uhlenbeck coefficients (per atom, per axis).

        In ASE's momentum-based unit system:
        - momenta p have units of sqrt(amu * eV)
        - masses m are in amu
        - kB in eV/K
        - c2 = sqrt(m * kB * T * (1 - exp(-2*gamma*dt))) has units of sqrt(amu*eV) = [p]
        """
        masses = atoms.get_masses()[:, np.newaxis]  # (N, 1), amu
        gamma = self.friction
        dt = self._dt_outer / 2.0  # the O-update runs twice per outer step

        self._lang_c1 = np.exp(-gamma * dt)
        self._lang_c2 = np.sqrt(
            masses * units.kB * self.temperature_K * (1.0 - np.exp(-2.0 * gamma * dt))
        )

    def _langevin_kick(self, momenta: np.ndarray) -> np.ndarray:
        """Apply one Ornstein–Uhlenbeck half-step to momenta."""
        assert self._lang_c1 is not None and self._lang_c2 is not None
        noise = np.random.standard_normal(momenta.shape)
        return self._lang_c1 * momenta + self._lang_c2 * noise

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    def _compute_fast_forces(self) -> np.ndarray:
        """Compute forces from the fast (ML) calculator."""
        atoms = self.atoms
        atoms.calc = self.fast_calc
        f = atoms.get_forces().copy()
        self._fast_forces = f
        self.n_fast_calls += 1
        return f

    def _compute_slow_forces(self) -> np.ndarray:
        """Compute slow (RESPA correction) forces: F_QM - F_ML.

        If ``slow_calculator`` is ``None`` the correction is zero.

        RESPA force decomposition:
          F_total = F_fast + F_slow_correction
          F_slow_correction = F_QM - F_ML_reference

        Both terms are evaluated at the same positions so the decomposition
        is exact.
        """
        if self.slow_calc is None:
            n = len(self.atoms)
            self._slow_forces = np.zeros((n, 3))
            return self._slow_forces

        # Expensive reference forces at the current positions; slow_calc stays
        # attached so observers log E_QM and F_QM (= F_fast + F_slow here).
        self.atoms.calc = self.slow_calc
        f_qm = self.atoms.get_forces().copy()
        self.n_slow_calls += 1

        # Use the already-cached fast forces (from last inner step)
        # to avoid redundant ML evaluation
        if self._fast_forces is None:
            self._compute_fast_forces()
        f_ml = self._fast_forces  # type: ignore[assignment]

        self._slow_forces = f_qm - f_ml
        return self._slow_forces

    # ------------------------------------------------------------------
    # Integration step
    # ------------------------------------------------------------------

    def step(self, forces: np.ndarray | None = None) -> bool:
        """Execute one outer (slow) RESPA step.

        Uses ASE's momentum-based integration (same convention as
        ``ase.md.verlet.VelocityVerlet``).  In ASE's unit system:
        - momenta p  : sqrt(amu * eV)
        - forces F   : eV/Å
        - timestep dt: ASE time units (set via ``units.fs``)
        - F * dt     : sqrt(amu * eV)  [same units as p]  ✓
        - p/m * dt   : Å               [position update]   ✓

        Parameters
        ----------
        forces : np.ndarray, optional
            Unused (kept for ASE interface compatibility).

        Returns
        -------
        bool
            Always ``False`` (required by ASE MolecularDynamics interface).
        """
        atoms = self.atoms
        masses = atoms.get_masses()[:, np.newaxis]  # (N, 1), amu

        # Momenta (ASE native integration variable)
        p = atoms.get_momenta()  # shape (N, 3), units sqrt(amu·eV)
        x = atoms.get_positions()  # shape (N, 3), Å

        dt_i = self._dt_inner
        dt_o = self._dt_outer

        # Initialise forces on first call
        if self._fast_forces is None:
            self._compute_fast_forces()
        if self._slow_forces is None:
            self._compute_slow_forces()

        f_fast = self._fast_forces  # type: ignore[assignment]
        f_slow = self._slow_forces  # type: ignore[assignment]

        # ── Outer half-kick (slow forces) ──────────────────────────────
        p += f_slow * (dt_o / 2.0)
        if self.temperature_K is not None:
            p = self._langevin_kick(p)

        # ── Inner Verlet loop ──────────────────────────────────────────
        for _ in range(self.mts_ratio):
            p += f_fast * (dt_i / 2.0)  # fast half-kick
            x += p / masses * dt_i  # drift
            atoms.set_positions(x)
            f_fast = self._compute_fast_forces()  # recompute at new x
            p += f_fast * (dt_i / 2.0)  # fast half-kick

        atoms.set_momenta(p, apply_constraint=False)
        atoms.set_positions(x)

        # ── Recompute slow forces at final positions ───────────────────
        f_slow = self._compute_slow_forces()

        # ── Outer half-kick (slow forces) ──────────────────────────────
        p = atoms.get_momenta()
        p += f_slow * (dt_o / 2.0)
        if self.temperature_K is not None:
            p = self._langevin_kick(p)
        atoms.set_momenta(p, apply_constraint=False)

        return False

    # ------------------------------------------------------------------
    # Run override to expose force caching
    # ------------------------------------------------------------------

    def run(self, steps: int = 1) -> None:
        """Run for ``steps`` outer RESPA steps.

        Forces are computed once at step 0 and then carried over, so running in
        chunks costs no extra calls.  Observers see step 0, then each outer step
        (ASE's convention).
        """
        if self.nsteps == 0 and self._fast_forces is None:
            self._compute_fast_forces()
            self._compute_slow_forces()
            self.call_observers()
        for _ in range(steps):
            self.step()
            self.nsteps += 1
            self.call_observers()

    @property
    def mts_stats(self) -> dict[str, typing.Any]:
        """Runtime statistics: call counts per calculator."""
        return {
            "fast_calculator_calls": self.n_fast_calls,
            "slow_calculator_calls": self.n_slow_calls,
            "mts_ratio": self.mts_ratio,
            "inner_timestep_fs": self._dt_inner / units.fs,
            "outer_timestep_fs": self._dt_outer / units.fs,
        }
