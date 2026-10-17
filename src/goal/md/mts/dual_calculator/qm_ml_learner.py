"""QM + ML online-learning dual-calculator for adaptive MTS-MD.

This module implements the adaptive simulation loop described in the project
goals: a cheap ML model is fine-tuned on-the-fly from expensive QM reference
data generated during the simulation.  Once the ML model reaches the required
accuracy it takes over as the primary force calculator.  QM calls are then
used only periodically for re-calibration.

Decision policies
-----------------
Two built-in policies are provided; custom ones can be supplied as callables:

``ThresholdPolicy``
    Switch to ML when both force RMSE and energy MAE fall below configurable
    thresholds.  This is the default.

``AlwaysBasePolicy``
    Always use the base (QM) calculator — useful for data collection without
    switching.

``StepFractionPolicy``
    Switch to ML after a fixed fraction of total steps regardless of accuracy.
    Useful as a baseline comparison.

Fine-tuning
-----------
When the ``fine_tuner`` parameter is provided (a :class:`OnlineFinetuner`),
it is called after every ``eval_interval`` steps with the current data buffer.
This updates the ML model weights in-place.  The :class:`GOALOnlineFinetuner`
implementation uses goal.ml's training infrastructure for this.
"""

from __future__ import annotations

import logging
import typing
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from goal.md.mts.dual_calculator.base import DualCalculator, DualCalculatorState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision policies
# ---------------------------------------------------------------------------


class SwitchPolicy(typing.Protocol):
    """Protocol for switch-decision policies."""

    def should_switch_to_ml(self, state: DualCalculatorState) -> bool:
        """Return True when the ML model should be used."""
        ...

    def should_return_to_base(self, state: DualCalculatorState) -> bool:
        """Return True when QM should resume control."""
        ...


@dataclass
class ThresholdPolicy:
    """Switch based on configurable force/energy thresholds.

    Parameters
    ----------
    force_rmse : float
        Maximum acceptable force RMSE (eV/Å).
    energy_mae : float
        Maximum acceptable energy MAE (eV/atom).
    min_steps : int
        Minimum training steps before any switch is allowed.
    return_threshold_scale : float
        Multiply thresholds by this factor for the return condition
        (hysteresis).  Default 2.0 means switch back only when error is
        twice the switching threshold.
    """

    force_rmse: float = 0.05
    energy_mae: float = 0.005
    min_steps: int = 500
    return_threshold_scale: float = 2.0

    def should_switch_to_ml(self, state: DualCalculatorState) -> bool:
        """Return True when the ML model is accurate enough to take over.

        Requires ``state.step >= min_steps`` and both ``state.force_rmse``
        and ``state.energy_mae`` to be set and strictly below
        ``force_rmse`` and ``energy_mae`` respectively.
        """
        if state.step < self.min_steps:
            return False
        if state.force_rmse is None or state.energy_mae is None:
            return False
        return state.force_rmse < self.force_rmse and state.energy_mae < self.energy_mae

    def should_return_to_base(self, state: DualCalculatorState) -> bool:
        """Return True when the base (QM) calculator should resume control.

        True if either error estimate is still ``None``, or if
        ``state.force_rmse > force_rmse * return_threshold_scale`` or
        ``state.energy_mae > energy_mae * return_threshold_scale``.
        ``min_steps`` is not checked here.
        """
        if state.force_rmse is None or state.energy_mae is None:
            return True
        scale = self.return_threshold_scale
        return (
            state.force_rmse > self.force_rmse * scale
            or state.energy_mae > self.energy_mae * scale
        )


@dataclass
class AlwaysBasePolicy:
    """Never switch to ML — use base (QM) throughout."""

    def should_switch_to_ml(self, state: DualCalculatorState) -> bool:
        """Always return False: the ML model is never used."""
        return False

    def should_return_to_base(self, state: DualCalculatorState) -> bool:
        """Always return False (the base calculator is never left)."""
        return False


@dataclass
class StepFractionPolicy:
    """Switch after a fixed fraction of total estimated steps.

    Parameters
    ----------
    switch_fraction : float
        Fraction of ``total_steps`` after which the ML model takes over.
    total_steps : int
        Estimated total steps (used to compute absolute switch step).
    """

    switch_fraction: float = 0.1
    total_steps: int = 10000

    @property
    def switch_step(self) -> int:
        """Absolute step of the switch, ``int(switch_fraction * total_steps)``."""
        return int(self.switch_fraction * self.total_steps)

    def should_switch_to_ml(self, state: DualCalculatorState) -> bool:
        """Return True once ``state.step >= switch_step``, regardless of accuracy."""
        return state.step >= self.switch_step

    def should_return_to_base(self, state: DualCalculatorState) -> bool:
        """Always return False: once switched, the ML model stays in control."""
        return False


# ---------------------------------------------------------------------------
# Fine-tuner protocol
# ---------------------------------------------------------------------------


class OnlineFinetuner(typing.Protocol):
    """Protocol for on-the-fly ML model fine-tuning."""

    def update(
        self,
        buffer: deque[tuple[Atoms, np.ndarray, float]],
    ) -> dict[str, float]:
        """Fine-tune the ML model on the current data buffer.

        Parameters
        ----------
        buffer : deque of (Atoms, forces, energy)
            Training data collected so far.

        Returns
        -------
        dict
            Training metrics (e.g., ``{"loss": 0.01, "force_rmse": 0.03}``).
        """
        ...

    @property
    def is_ready(self) -> bool:
        """True when the fine-tuner has been initialised and can be called."""
        ...


class GOALOnlineFinetuner:
    """Fine-tunes a goal.ml GOALCalculator on collected QM data.

    Uses :class:`~goal.ml.utils.mini_trainer.MiniTrainer` with
    :func:`~goal.ml.utils.mini_trainer.graph_step` and the project's
    composable loss system (:class:`~goal.ml.training.loss.CompositeLoss`)
    so that fine-tuning is consistent with how models are trained in goal.ml.

    The trainer (and its optimizer state) is kept across ``update()`` calls so
    that AdamW accumulates proper momentum/variance estimates over time —
    beneficial for continual fine-tuning.

    Parameters
    ----------
    calculator : GOALCalculator
        The ML calculator whose underlying ``module`` (a ``GOALModule``) will
        be fine-tuned in-place.
    cutoff : float
        Neighbor-list cutoff in Ångström (must match the model's training
        cutoff).
    learning_rate : float
        AdamW learning rate for fine-tuning steps.
    epochs_per_update : int
        Number of full passes over the current buffer per ``update()`` call.
        One epoch = one pass through all sampled frames.
    batch_size : int
        Number of structures per gradient step inside each epoch.
    min_buffer_size : int
        Minimum buffer frames required before fine-tuning starts.
    device : str or None
        Torch device (``"cpu"``, ``"cuda"``, etc.).  ``None`` (default)
        uses the calculator's own device, so fine-tuning never moves the
        model away from where inference runs.
    grad_clip : float or None
        Maximum gradient norm for clipping. ``None`` disables.
    energy_weight : float
        Weight of the energy term in :class:`~goal.ml.training.loss.CompositeLoss`.
    forces_weight : float
        Weight of the forces term.
    """

    def __init__(
        self,
        calculator: typing.Any,  # GOALCalculator — avoid circular import
        cutoff: float = 5.0,
        learning_rate: float = 1e-4,
        epochs_per_update: int = 3,
        batch_size: int = 8,
        min_buffer_size: int = 20,
        device: str | None = None,
        grad_clip: float | None = 1.0,
        energy_weight: float = 1.0,
        forces_weight: float = 10.0,
    ) -> None:
        self._calc = calculator
        self.cutoff = cutoff
        self.lr = learning_rate
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        if device is None:
            device = str(getattr(calculator, "device", "cpu"))
        self.device = device
        self.grad_clip = grad_clip
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight

        self._trainer: typing.Any = None  # MiniTrainer, created lazily
        self._update_count = 0

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def model(self) -> typing.Any:
        """The ``nn.Module`` being fine-tuned (shared with the calculator)."""
        for attr in ("module", "_module"):
            module = getattr(self._calc, attr, None)
            if module is not None:
                return module
        raise AttributeError(
            f"{type(self._calc).__name__} exposes no model to fine-tune; "
            f"GOALOnlineFinetuner needs a GOALCalculator (attribute 'module')."
        )

    def _graph(self, atoms: Atoms, forces: np.ndarray, energy: float) -> typing.Any:
        """A training graph built exactly like the calculator's inference graphs."""
        build = getattr(self._calc, "build_graph", None)
        if callable(build):
            return build(atoms, energy=float(energy), forces=forces)

        from goal.md.adapters.ase_converter import atomic_graph_from_ase

        dtype = next(self.model.parameters()).dtype
        return atomic_graph_from_ase(
            atoms, cutoff=self.cutoff, energy=float(energy), forces=forces, dtype=dtype
        )

    def _ensure_trainer(self) -> None:
        """Create the MiniTrainer on first use (lazy — avoids import at module level)."""
        if self._trainer is not None:
            return

        import torch.optim as optim

        from goal.ml.training.loss import (
            CompositeLoss,
            EnergyLoss,
            ForcesLoss,
            WeightedLoss,
        )
        from goal.ml.utils.mini_trainer import MiniTrainer, graph_step

        module = self.model
        loss_fn = WeightedLoss(EnergyLoss(), weight=self.energy_weight) + WeightedLoss(
            ForcesLoss(), weight=self.forces_weight
        )
        optimizer = optim.AdamW(module.parameters(), lr=self.lr)

        self._trainer = MiniTrainer(
            model=module,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=self.device,
            step_fn=graph_step,
            grad_clip=self.grad_clip,
            enable_progress=False,
        )

    def update(
        self,
        buffer: deque[tuple[Atoms, np.ndarray, float]],
    ) -> dict[str, float]:
        """Fine-tune the ML model using goal.ml's MiniTrainer.

        Converts the data buffer to a PyG DataLoader of
        :class:`~goal.ml.data.graph.AtomicGraph` objects, then calls
        :meth:`~goal.ml.utils.mini_trainer.MiniTrainer.fit` with
        ``graph_step`` and the project's ``CompositeLoss``.

        Parameters
        ----------
        buffer : deque of (Atoms, forces_ev_ang, energy_ev)
            Training data collected from the base (QM) calculator.

        Returns
        -------
        dict
            Keys: ``loss`` (final epoch train loss), ``update_count``,
            ``buffer_size``.
        """
        if len(buffer) < self.min_buffer_size:
            return {
                "loss": float("nan"),
                "update_count": self._update_count,
                "buffer_size": len(buffer),
            }

        from torch_geometric.loader import DataLoader as PyGDataLoader

        self._ensure_trainer()

        # Build AtomicGraph dataset from the buffer — same cutoff, dtype and
        # fragment labels as the calculator uses for inference.
        graphs = [
            self._graph(atoms_snap, forces_ref, energy_ref)
            for atoms_snap, forces_ref, energy_ref in buffer
        ]

        loader = PyGDataLoader(graphs, batch_size=self.batch_size, shuffle=True)

        # Delegate to MiniTrainer — uses graph_step + CompositeLoss internally
        try:
            history = self._trainer.fit(loader, epochs=self.epochs_per_update, verbose=False)
        finally:
            # Back to inference mode: the calculator shares this module, and
            # train mode would make every MD step build a double-backward graph.
            self.model.eval()

        self._update_count += 1
        last_loss = history.train_loss[-1] if history.train_loss else float("nan")

        return {
            "loss": last_loss,
            "update_count": self._update_count,
            "buffer_size": len(buffer),
        }


# ---------------------------------------------------------------------------
# QMLearnerCalculator
# ---------------------------------------------------------------------------


class QMLearnerCalculator(DualCalculator):
    """Dual calculator: QM reference + on-the-fly ML fine-tuning.

    The ML model is trained from QM data collected during the simulation.
    A configurable *switch policy* decides when the ML model is accurate
    enough to take over.  An optional *fine-tuner* updates the ML model
    weights between evaluations.

    Parameters
    ----------
    base_calculator : Calculator
        High-accuracy calculator (QM or expensive ML).
    ml_calculator : Calculator
        ML model that will be fine-tuned during the simulation.
    policy : SwitchPolicy, optional
        Decision policy for switching.  Defaults to
        :class:`ThresholdPolicy` with ``force_rmse=0.05``.
    fine_tuner : OnlineFinetuner, optional
        If provided, called every ``eval_interval`` steps to update the
        ML model weights.
    eval_interval : int
        Steps between accuracy evaluations and fine-tuning calls.
    data_buffer_size : int
        Maximum number of training frames to keep in memory.
    log_switch_events : bool
        Emit log messages on calculator switches.
    """

    def __init__(
        self,
        base_calculator: Calculator,
        ml_calculator: Calculator,
        policy: SwitchPolicy | None = None,
        fine_tuner: OnlineFinetuner | None = None,
        eval_interval: int = 100,
        data_buffer_size: int = 5000,
        log_switch_events: bool = True,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(base_calculator, **kwargs)
        self.ml = ml_calculator
        self.policy: SwitchPolicy = policy or ThresholdPolicy()
        self.fine_tuner = fine_tuner
        self.eval_interval = eval_interval
        self.data_buffer: deque[tuple[Atoms, np.ndarray, float]] = deque(maxlen=data_buffer_size)
        self.log_switch_events = log_switch_events

        self._fine_tune_metrics: list[dict[str, float]] = []
        self._last_eval_step = 0

    # ------------------------------------------------------------------
    # DualCalculator interface
    # ------------------------------------------------------------------

    def compute_base(self, atoms: Atoms) -> dict[str, typing.Any]:
        """Compute with QM (base) calculator, store in data buffer."""
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.base
        try:
            energy = float(atoms_copy.get_potential_energy())
            forces = atoms_copy.get_forces().copy()
        except Exception as exc:
            logger.warning("Base calculator failed: %s", exc)
            n = len(atoms)
            return {
                "energy": float("nan"),
                "forces": np.full((n, 3), float("nan")),
            }

        # Always store QM data for fine-tuning
        self.data_buffer.append((atoms.copy(), forces, energy))
        return {"energy": energy, "forces": forces}

    def compute_ml(self, atoms: Atoms) -> dict[str, typing.Any]:
        """Compute with ML calculator."""
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.ml
        try:
            return {
                "energy": float(atoms_copy.get_potential_energy()),
                "forces": atoms_copy.get_forces().copy(),
            }
        except Exception as exc:
            logger.debug("ML calculator failed (may be expected early): %s", exc)
            n = len(atoms)
            return {
                "energy": float("nan"),
                "forces": np.full((n, 3), float("nan")),
            }

    def reference_needed(self, active: str) -> bool:
        """Run QM on an ML step only when an evaluation happened this step.

        While the ML model is active the QM calculator is otherwise
        skipped.  Every ``eval_interval`` steps one QM reference of the
        *current* geometry is computed: it refreshes the training buffer
        and gives :meth:`should_return_to_base` a fresh error estimate on
        the region the trajectory is actually exploring.
        """
        return active == "ml" and self._last_eval_step == self.state.step

    def decide_calculator(self, atoms: Atoms) -> str:
        """Return ``"base"`` or ``"ml"`` based on the switch policy."""
        self.state.step += 1

        # Run fine-tuning and evaluation periodically
        if (self.state.step - self._last_eval_step) >= self.eval_interval:
            self._run_fine_tuning_and_eval()
            self._last_eval_step = self.state.step

        # Apply policy
        if self.state.active_calculator == "ml":
            if self.policy.should_return_to_base(self.state):
                if self.log_switch_events:
                    logger.info(
                        "Step %d: ML accuracy degraded (F-RMSE=%.4f, E-MAE=%.4f) "
                        "→ switching back to base calculator.",
                        self.state.step,
                        self.state.force_rmse or float("nan"),
                        self.state.energy_mae or float("nan"),
                    )
                self.state.active_calculator = "base"
                return "base"
            return "ml"
        else:
            if self.policy.should_switch_to_ml(self.state):
                if self.log_switch_events:
                    logger.info(
                        "Step %d: ML model accurate enough (F-RMSE=%.4f, E-MAE=%.4f) "
                        "→ switching to ML calculator. QM will only be used for "
                        "re-calibration every %d steps.",
                        self.state.step,
                        self.state.force_rmse or float("nan"),
                        self.state.energy_mae or float("nan"),
                        self.eval_interval,
                    )
                self.state.active_calculator = "ml"
                return "ml"
            return "base"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_fine_tuning_and_eval(self) -> None:
        """Call fine-tuner (if set) and update error metrics."""
        if self.fine_tuner is not None and self.fine_tuner.is_ready:
            try:
                metrics = self.fine_tuner.update(self.data_buffer)
                self._fine_tune_metrics.append(metrics)
            except Exception as exc:
                logger.warning("Fine-tuning step failed: %s", exc)

        # Recompute error metrics from the last buffered data points
        self._update_accuracy_metrics()

    def _update_accuracy_metrics(self) -> None:
        """Estimate force RMSE and energy MAE on recent buffer frames."""
        if len(self.data_buffer) < 2:
            return

        # Use up to 20 most recent frames for error estimation
        recent = list(self.data_buffer)[-20:]
        f_errors: list[float] = []
        e_errors: list[float] = []

        for atoms_snap, f_ref, e_ref in recent:
            ml_result = self.compute_ml(atoms_snap)
            f_ml = ml_result["forces"]
            e_ml = ml_result["energy"]

            if not np.any(np.isnan(f_ml)):
                rmse = float(np.sqrt(np.mean((f_ml - f_ref) ** 2)))
                f_errors.append(rmse)
            if not np.isnan(e_ml):
                n = len(atoms_snap)
                e_errors.append(abs(e_ml - e_ref) / n)

        if f_errors:
            self.state.force_rmse = float(np.mean(f_errors))
        if e_errors:
            self.state.energy_mae = float(np.mean(e_errors))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_training_history(self) -> dict[str, typing.Any]:
        """Return training and switching history."""
        return {
            "data_buffer_size": len(self.data_buffer),
            "fine_tune_steps": len(self._fine_tune_metrics),
            "fine_tune_metrics": self._fine_tune_metrics[-10:],  # last 10
            "current_force_rmse": self.state.force_rmse,
            "current_energy_mae": self.state.energy_mae,
            "active_calculator": self.state.active_calculator,
            "total_md_steps": self.state.step,
        }
