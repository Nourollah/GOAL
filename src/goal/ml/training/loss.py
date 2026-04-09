"""Composable loss system with configurable loss functions per property.

Each property loss accepts a ``loss_fn`` parameter (e.g. ``"mse"``,
``"mae"``, ``"huber"``, ``"rmse"``) so the user can choose the loss
function from config without modifying code.

Supports **composite losses per property** — multiple loss functions
with independent weights for the same property, each logged separately::

    losses:
      - name: forces
        fn:
          - name: mse
            weight: 4.0
          - name: rmse
            weight: 8.0

This produces three logged metrics: ``forces_mse``, ``forces_rmse``,
and ``forces`` (their sum).

Supports dotted import paths for custom or ``torchmetrics`` functions::

    fn: torchmetrics.functional.mean_squared_error
"""

from __future__ import annotations

import importlib
import typing

import torch
import torch.nn as nn

from goal.ml.registry import LOSS_REGISTRY


# ---------------------------------------------------------------------------
# Loss function lookup
# ---------------------------------------------------------------------------


def _rmse_loss(
    input: torch.Tensor, target: torch.Tensor, **kwargs: typing.Any,
) -> torch.Tensor:
    """Root mean squared error — √MSE."""
    return torch.sqrt(nn.functional.mse_loss(input, target, **kwargs))


_LOSS_FN_MAP: typing.Dict[str, typing.Callable[..., torch.Tensor]] = {
    "mse": nn.functional.mse_loss,
    "mae": nn.functional.l1_loss,
    "l1": nn.functional.l1_loss,
    "huber": nn.functional.huber_loss,
    "smooth_l1": nn.functional.smooth_l1_loss,
    "rmse": _rmse_loss,
}


def resolve_loss_fn(name: str) -> typing.Callable[..., torch.Tensor]:
    """Resolve a loss function name to a callable.

    Supports three resolution modes:

    1. **Built-in name** — ``"mse"``, ``"mae"``, ``"rmse"``, etc.
    2. **Dotted import path** — ``"torchmetrics.functional.mean_squared_error"``
       or any fully-qualified callable.
    3. **Short alias** — ``"l1"`` is an alias for ``"mae"``.

    Parameters
    ----------
    name : str
        A built-in name, dotted import path, or alias.

    Raises
    ------
    ValueError
        If the name cannot be resolved.
    """
    fn: typing.Optional[typing.Callable[..., torch.Tensor]] = _LOSS_FN_MAP.get(name)
    if fn is not None:
        return fn
    # Dotted import path: e.g. "torchmetrics.functional.mean_squared_error"
    if "." in name:
        module_path: str
        attr_name: str
        module_path, attr_name = name.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            return getattr(module, attr_name)
        except (ImportError, AttributeError) as exc:
            raise ValueError(
                f"Cannot resolve loss function '{name}': {exc}"
            ) from exc
    available: str = ", ".join(sorted(_LOSS_FN_MAP.keys()))
    raise ValueError(
        f"Unknown loss function '{name}'. "
        f"Built-in: {available}. "
        f"Or use a dotted import path (e.g. 'torchmetrics.functional.mean_squared_error')."
    )


class WeightedLoss(nn.Module):
    """Wrap any loss with a scalar weight and a logging label.

    Parameters
    ----------
    loss : nn.Module
        The inner property loss (e.g. ``EnergyLoss``).
    weight : float
        Scalar multiplier applied to the loss value.
    label : str or None
        Logging key. Defaults to ``loss.__class__.__name__``.
    group : str or None
        Property group name for aggregated logging. When multiple
        ``WeightedLoss`` entries share the same group, ``CompositeLoss``
        emits an additional ``group`` key with their sum.
    """

    def __init__(
        self,
        loss: nn.Module,
        weight: float,
        label: typing.Optional[str] = None,
        group: typing.Optional[str] = None,
    ) -> None:
        super().__init__()
        self.loss: nn.Module = loss
        self.weight: float = weight
        self.label: str = label or loss.__class__.__name__
        self.group: typing.Optional[str] = group

    def forward(
        self,
        predictions: typing.Dict[str, torch.Tensor],
        targets: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.weight * self.loss(predictions, targets)

    def __add__(self, other: WeightedLoss) -> CompositeLoss:
        return CompositeLoss([self, other])


class CompositeLoss(nn.Module):
    """Sum of multiple weighted losses with per-component logging.

    Supports the ``+`` operator for clean composition::

        composite = weighted_a + weighted_b + weighted_c

    When multiple components share the same ``group``, the forward
    dict includes an additional entry for the group total.  Example
    output for forces with MSE + RMSE sub-losses::

        {"total": 12.0, "forces_mse": 4.0, "forces_rmse": 8.0, "forces": 12.0, "energy": 2.5}
    """

    def __init__(self, losses: typing.List[WeightedLoss]) -> None:
        super().__init__()
        self.losses: nn.ModuleList = nn.ModuleList(losses)

    def __add__(self, other: WeightedLoss) -> CompositeLoss:
        return CompositeLoss([*self.losses, other])

    def forward(
        self,
        predictions: typing.Dict[str, torch.Tensor],
        targets: typing.Dict[str, torch.Tensor],
    ) -> typing.Dict[str, torch.Tensor]:
        """Compute all constituent losses and return a breakdown dict.

        Returns
        -------
        typing.Dict[str, Tensor]
            Always contains ``'total'``.  Each component is keyed by
            its ``label``.  Groups with multiple members get an extra
            aggregated entry keyed by the group name.
        """
        device: torch.device = next(iter(predictions.values())).device
        total: torch.Tensor = torch.tensor(0.0, device=device)
        breakdown: typing.Dict[str, torch.Tensor] = {}
        group_sums: typing.Dict[str, torch.Tensor] = {}
        group_counts: typing.Dict[str, int] = {}

        for loss in self.losses:
            val: torch.Tensor = loss(predictions, targets)
            breakdown[loss.label] = val
            total = total + val

            if loss.group is not None:
                if loss.group not in group_sums:
                    group_sums[loss.group] = torch.tensor(0.0, device=device)
                    group_counts[loss.group] = 0
                group_sums[loss.group] = group_sums[loss.group] + val
                group_counts[loss.group] += 1

        # Emit group totals only when a group has 2+ members
        for grp, grp_total in group_sums.items():
            if group_counts[grp] > 1 and grp not in breakdown:
                breakdown[grp] = grp_total

        breakdown["total"] = total
        return breakdown


# ---------------------------------------------------------------------------
# Built-in loss functions
# ---------------------------------------------------------------------------


@LOSS_REGISTRY.register("energy")
class EnergyLoss(nn.Module):
    """Loss on per-atom energy with configurable loss function."""

    def __init__(self, loss_fn: str = "mse") -> None:
        super().__init__()
        self.loss_fn: typing.Callable[..., torch.Tensor] = resolve_loss_fn(loss_fn)

    def forward(
        self,
        pred: typing.Dict[str, torch.Tensor],
        target: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.loss_fn(
            pred["energy"] / pred["num_atoms"],
            target["energy"] / target["num_atoms"],
        )


@LOSS_REGISTRY.register("forces")
class ForcesLoss(nn.Module):
    """Loss on atomic forces with configurable loss function."""

    def __init__(self, loss_fn: str = "mse") -> None:
        super().__init__()
        self.loss_fn: typing.Callable[..., torch.Tensor] = resolve_loss_fn(loss_fn)

    def forward(
        self,
        pred: typing.Dict[str, torch.Tensor],
        target: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.loss_fn(pred["forces"], target["forces"])


@LOSS_REGISTRY.register("stress")
class StressLoss(nn.Module):
    """Loss on the stress tensor with configurable loss function."""

    def __init__(self, loss_fn: str = "mse") -> None:
        super().__init__()
        self.loss_fn: typing.Callable[..., torch.Tensor] = resolve_loss_fn(loss_fn)

    def forward(
        self,
        pred: typing.Dict[str, torch.Tensor],
        target: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.loss_fn(pred["stress"], target["stress"])


@LOSS_REGISTRY.register("dipole")
class DipoleLoss(nn.Module):
    """Loss on the dipole moment vector with configurable loss function."""

    def __init__(self, loss_fn: str = "mse") -> None:
        super().__init__()
        self.loss_fn: typing.Callable[..., torch.Tensor] = resolve_loss_fn(loss_fn)

    def forward(
        self,
        pred: typing.Dict[str, torch.Tensor],
        target: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.loss_fn(pred["dipole"], target["dipole"])


@LOSS_REGISTRY.register("charge")
class ChargeLoss(nn.Module):
    """Loss on total charge with configurable loss function."""

    def __init__(self, loss_fn: str = "mse") -> None:
        super().__init__()
        self.loss_fn: typing.Callable[..., torch.Tensor] = resolve_loss_fn(loss_fn)

    def forward(
        self,
        pred: typing.Dict[str, torch.Tensor],
        target: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        target_charge: torch.Tensor = target.get(
            "total_charge",
            torch.zeros_like(pred["total_charge"]),
        )
        return self.loss_fn(pred["total_charge"], target_charge)
