"""Base classes for dual-calculator MTS system."""

from __future__ import annotations

import abc
import typing
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


@dataclass
class DualCalculatorState:
    """State tracking for dual-calculator system."""

    step: int = 0
    base_forces: np.ndarray | None = None
    base_energy: float | None = None
    ml_forces: np.ndarray | None = None
    ml_energy: float | None = None
    force_rmse: float | None = None
    energy_mae: float | None = None
    active_calculator: str = "base"  # "base" or "ml"
    ml_uncertainty: float = float("inf")
    metrics: dict = field(default_factory=dict)


class DualCalculator(Calculator, abc.ABC):
    """Abstract base for dual-calculator systems.

    Manages two calculators (e.g., QM base + ML learner)
    with switching logic and monitoring.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        base_calculator: Calculator,
        **kwargs: typing.Any,
    ) -> None:
        """Initialize dual calculator.

        Parameters
        ----------
        base_calculator : Calculator
            High-accuracy base calculator (reference)
        **kwargs
            Additional arguments for Calculator base class
        """
        super().__init__(**kwargs)
        self.base = base_calculator
        self.state = DualCalculatorState()

    @abc.abstractmethod
    def compute_base(self, atoms: Atoms) -> dict[str, typing.Any]:
        """Compute with base calculator."""
        pass

    @abc.abstractmethod
    def compute_ml(self, atoms: Atoms) -> dict[str, typing.Any]:
        """Compute with ML calculator."""
        pass

    @abc.abstractmethod
    def decide_calculator(self, atoms: Atoms) -> str:
        """Decide which calculator to use.

        Returns
        -------
        str
            "base" or "ml"
        """
        pass

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        """Calculate properties using appropriate calculator.

        Implements ASE Calculator interface.
        """
        if properties is None:
            properties = self.implemented_properties

        super().calculate(atoms, properties, system_changes)

        # Decide first, so the expensive base calculator runs only when
        # this step needs it: on base steps, and on the re-calibration
        # steps a subclass asks for through ``reference_needed``.
        active = self.decide_calculator(self.atoms)
        self.state.active_calculator = active

        base_results: dict[str, typing.Any] | None = None
        if active == "base" or self.reference_needed(active):
            base_results = self.compute_base(self.atoms)
            self.state.metrics["base_calls"] = self.state.metrics.get("base_calls", 0) + 1
        ml_results = self.compute_ml(self.atoms)
        self.state.metrics["ml_calls"] = self.state.metrics.get("ml_calls", 0) + 1

        # Update state.  A step without a reference clears the base fields
        # so a stale reference from an older geometry is never compared
        # against the current ML prediction.
        self.state.base_forces = base_results.get("forces") if base_results else None
        self.state.base_energy = base_results.get("energy") if base_results else None
        self.state.ml_forces = ml_results.get("forces")
        self.state.ml_energy = ml_results.get("energy")

        # Compute error metrics (only when a reference was computed)
        self._compute_errors()

        # Return results from active calculator
        if active == "base" and base_results is not None:
            self.results.update(base_results)
        else:
            self.results.update(ml_results)

    def reference_needed(self, active: str) -> bool:
        """Whether to also run the base calculator on an ``"ml"`` step.

        The default never does: once the ML model is active the base
        (e.g. QM) calculator is skipped entirely.  Subclasses override
        this to request periodic re-calibration points.
        """
        return False

    def _compute_errors(self) -> None:
        """Compute force RMSE and energy MAE."""
        if self.state.base_forces is not None and self.state.ml_forces is not None:
            self.state.force_rmse = np.sqrt(
                np.mean((self.state.ml_forces - self.state.base_forces) ** 2)
            )

        if self.state.base_energy is not None and self.state.ml_energy is not None:
            self.state.energy_mae = np.abs(self.state.ml_energy - self.state.base_energy)

    def get_state(self) -> DualCalculatorState:
        """Get current state."""
        return self.state

    def get_metrics(self) -> dict[str, typing.Any]:
        """Get monitoring metrics."""
        return {
            "step": self.state.step,
            "force_rmse": self.state.force_rmse,
            "energy_mae": self.state.energy_mae,
            "active_calculator": self.state.active_calculator,
            "ml_uncertainty": self.state.ml_uncertainty,
            "base_calls": self.state.metrics.get("base_calls", 0),
            "ml_calls": self.state.metrics.get("ml_calls", 0),
        }
