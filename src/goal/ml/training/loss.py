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
    input: torch.Tensor,
    target: torch.Tensor,
    **kwargs: typing.Any,
) -> torch.Tensor:
    """Root mean squared error — √MSE."""
    return torch.sqrt(nn.functional.mse_loss(input, target, **kwargs))


_LOSS_FN_MAP: dict[str, typing.Callable[..., torch.Tensor]] = {
    "mse": nn.functional.mse_loss,
    "mae": nn.functional.l1_loss,
    "l1": nn.functional.l1_loss,
    "huber": nn.functional.huber_loss,
    "smooth_l1": nn.functional.smooth_l1_loss,
    "rmse": _rmse_loss,
}

# Per-element (un-reduced) form of each built-in loss.  Used by the
# per-sample weighting path so we can compute a weighted mean
# afterwards.  RMSE is not point-wise (it is sqrt-of-mean) so it is
# special-cased in :func:`_weighted_loss_value` rather than appearing
# here.
_ELEMENTWISE_FN_MAP: dict[str, typing.Callable[..., torch.Tensor]] = {
    "mse": lambda p, t: (p - t) ** 2,
    "mae": lambda p, t: (p - t).abs(),
    "l1": lambda p, t: (p - t).abs(),
    "huber": lambda p, t: nn.functional.huber_loss(p, t, reduction="none"),
    "smooth_l1": lambda p, t: nn.functional.smooth_l1_loss(p, t, reduction="none"),
}


def _weighted_mean(
    values: torch.Tensor,
    weight: torch.Tensor,
    batch_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample-weighted mean of an arbitrary-shape per-element tensor.

    ``weight`` is per-graph (shape ``(G,)``) — for per-atom tensors
    pass the PyG ``batch`` vector as ``batch_index`` so each element
    inherits its graph's weight.  Trailing dims are broadcast.
    Divides by ``weight.sum()`` (the convention MACE uses), not by
    element count, so a uniform 1.0 weight reproduces the plain
    arithmetic mean only when ``weight.sum() == len(values)``.  Use
    ``weight = ones / G`` if you want a true average.
    """
    w: torch.Tensor = weight.to(values.dtype).to(values.device)
    if batch_index is not None:
        w = w[batch_index]
    while w.dim() < values.dim():
        w = w.unsqueeze(-1)
    w_exp: torch.Tensor = w.expand_as(values)
    return (w_exp * values).sum() / w_exp.sum().clamp_min(1.0e-12)


def _weighted_loss_value(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    fn_name: str,
    batch_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample-weighted scalar loss for one of the built-in fns.

    Falls back to the un-weighted reduced form when ``fn_name`` is not
    a recognised built-in (e.g. a dotted torchmetrics path) — those
    loss fns don't accept ``reduction='none'`` uniformly so we can't
    decompose them safely.  RMSE is special-cased: weighted MSE then
    sqrt.
    """
    if fn_name == "rmse":
        sq: torch.Tensor = (pred - target) ** 2
        return torch.sqrt(_weighted_mean(sq, weight, batch_index).clamp_min(0.0))
    elw: typing.Callable[..., torch.Tensor] | None = _ELEMENTWISE_FN_MAP.get(fn_name)
    if elw is None:
        # Unknown loss fn (e.g. a torchmetrics path).  Honest fall-back
        # rather than silently dropping the weighting.
        fn: typing.Callable[..., torch.Tensor] = resolve_loss_fn(fn_name)
        return fn(pred, target)
    return _weighted_mean(elw(pred, target), weight, batch_index)


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
    fn: typing.Callable[..., torch.Tensor] | None = _LOSS_FN_MAP.get(name)
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
            raise ValueError(f"Cannot resolve loss function '{name}': {exc}") from exc
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
        label: str | None = None,
        group: str | None = None,
    ) -> None:
        super().__init__()
        self.loss: nn.Module = loss
        self.weight: float = weight
        self.label: str = label or loss.__class__.__name__
        self.group: str | None = group

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate the wrapped loss and scale it by ``self.weight``.

        Parameters
        ----------
        predictions : dict[str, Tensor]
            Model output dict, passed unchanged to the inner loss.
        targets : dict[str, Tensor]
            Reference labels (typically the PyG batch), passed unchanged
            to the inner loss.

        Returns
        -------
        Tensor
            Scalar ``weight * loss(predictions, targets)``.
        """
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

    def __init__(self, losses: list[WeightedLoss]) -> None:
        super().__init__()
        self.losses: nn.ModuleList = nn.ModuleList(losses)

    def __add__(self, other: WeightedLoss) -> CompositeLoss:
        return CompositeLoss([*self.losses, other])

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
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
        breakdown: dict[str, torch.Tensor] = {}
        group_sums: dict[str, torch.Tensor] = {}
        group_counts: dict[str, int] = {}

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
    """Loss on per-atom energy with configurable loss function.

    Picks up an optional **per-sample weight** from ``target.weight``
    (shape ``(num_graphs,)``).  When present, the per-graph squared /
    absolute errors are averaged with that weight (MACE convention,
    matches its ``per_sample_weights`` switch).  When absent, the
    plain ``loss_fn(p, t)`` reduction is used.
    """

    def __init__(self, loss_fn: str = "mse") -> None:
        super().__init__()
        self._loss_fn_name: str = loss_fn
        self.loss_fn: typing.Callable[..., torch.Tensor] = resolve_loss_fn(loss_fn)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the loss on per-atom energies.

        Both predicted and reference total energies are divided by
        ``pred["num_atoms"]`` (eV → eV/atom) before comparison, so
        structures of different size are on the same scale.  Without a
        per-graph ``target.weight`` attribute the configured ``loss_fn``
        is applied with its default (mean) reduction; with one, the
        per-graph errors are averaged with those weights (divided by
        ``weight.sum()``, MACE convention).

        Parameters
        ----------
        pred : dict[str, Tensor]
            Model outputs; reads ``'energy'`` ``(G,)`` and
            ``'num_atoms'`` ``(G,)``.
        target : dict[str, Tensor]
            Reference labels; reads ``'energy'`` ``(G,)`` and, if
            present, the ``weight`` attribute ``(G,)``.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        # ``num_atoms`` is a property of the input batch, so it is identical
        # in ``pred`` and ``target``.  The PyG ``Batch`` used as ``target``
        # doesn't carry it as an indexable key — only the head's output dict
        # does — so we read it from ``pred`` for both denominators.
        n_atoms: torch.Tensor = pred["num_atoms"]
        p: torch.Tensor = pred["energy"] / n_atoms
        t: torch.Tensor = target["energy"] / n_atoms

        weight: torch.Tensor | None = getattr(target, "weight", None)
        if weight is None or weight.numel() == 0:
            return self.loss_fn(p, t)
        # Energy is one element per graph — no batch_index needed.
        return _weighted_loss_value(p, t, weight, self._loss_fn_name)


def _per_structure_force_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fn_name: str,
    batch_index: torch.Tensor,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    """Force loss normalised per structure — equal gradient contribution per molecule.

    For each structure m with ``N_m`` atoms:

        loss_m = mean_{i ∈ m, d ∈ {x,y,z}} L(F_pred_{i,d}, F_target_{i,d})

    where ``L`` is the per-element loss (absolute error for MAE, squared for MSE,
    etc.).  The returned scalar is the mean (or weighted mean) of ``loss_m`` over
    all structures in the batch, so every molecule contributes equally regardless
    of size.

    For RMSE, per-structure MSE is computed first, then the per-structure
    square-root is taken, then the mean over structures.

    Falls back to the flat global loss for unknown dotted-path ``fn_name``
    values that cannot be decomposed per-element.

    Parameters
    ----------
    pred, target : Tensor ``(N, 3)``
        Predicted and target forces, all atoms concatenated.
    fn_name : str
        Loss function name (``"mae"``, ``"mse"``, ``"rmse"``, ``"huber"``, etc.).
    batch_index : Tensor ``(N,)``
        PyG ``batch`` vector — integer index of the graph each atom belongs to.
    weight : Tensor ``(G,)`` or None
        Optional per-graph sample weights (MACE convention).  When ``None``,
        a plain mean over structures is returned.
    """
    if fn_name == "rmse":
        per_element: torch.Tensor = (pred - target) ** 2  # (N, 3)
    else:
        elw: typing.Callable[..., torch.Tensor] | None = _ELEMENTWISE_FN_MAP.get(fn_name)
        if elw is None:
            # Unknown fn (e.g. a dotted torchmetrics path) — can't decompose.
            return resolve_loss_fn(fn_name)(pred, target)
        per_element = elw(pred, target)  # (N, 3)

    # Average over the three force components → one error per atom  (N,)
    per_atom: torch.Tensor = per_element.mean(dim=-1)

    # Scatter-mean over atoms in the same structure → (G,)
    num_structs: int = int(batch_index.max().item()) + 1
    per_struct: torch.Tensor = torch.zeros(
        num_structs, dtype=per_atom.dtype, device=per_atom.device
    )
    counts: torch.Tensor = torch.zeros(num_structs, dtype=per_atom.dtype, device=per_atom.device)
    per_struct.index_add_(0, batch_index, per_atom)
    counts.index_add_(0, batch_index, torch.ones_like(per_atom))
    per_struct = per_struct / counts.clamp_min(1.0)

    if fn_name == "rmse":
        per_struct = torch.sqrt(per_struct.clamp_min(0.0))

    if weight is None:
        return per_struct.mean()
    # Weighted mean over structures (one weight per graph, no batch broadcast needed)
    return _weighted_mean(per_struct, weight)


@LOSS_REGISTRY.register("forces")
class ForcesLoss(nn.Module):
    """Loss on atomic forces with configurable loss function.

    Like :class:`EnergyLoss`, supports an optional per-graph
    ``target.weight``.  Forces are per atom, so the per-graph weight
    is broadcast through ``target.batch`` (the PyG node-to-graph
    index) before reduction.

    Parameters
    ----------
    loss_fn : str
        Loss function — ``"mse"`` (default), ``"mae"``, ``"rmse"``,
        ``"huber"``, ``"smooth_l1"``, or a dotted import path.
    normalize_by_n_atoms : bool
        When ``True`` (default), compute the loss per-structure and then
        average over structures — every molecule contributes equally to the
        gradient regardless of how many atoms it contains.  Concretely,
        for each structure ``m``:

            loss_m = mean over atoms in m and xyz components of L(F_pred, F_target)

        and the returned scalar is ``mean_m(loss_m)``.

        When ``False``, the old behaviour is preserved: the loss is a flat
        mean over all ``(N_total × 3)`` force components in the batch, which
        gives larger structures proportionally more gradient weight.

        Has no effect when ``target.batch`` is not available (e.g. plain
        dict targets in unit tests) — the flat mean is used in that case.
    """

    def __init__(self, loss_fn: str = "mse", normalize_by_n_atoms: bool = True) -> None:
        super().__init__()
        self._loss_fn_name: str = loss_fn
        self.loss_fn: typing.Callable[..., torch.Tensor] = resolve_loss_fn(loss_fn)
        self._normalize_by_n_atoms: bool = bool(normalize_by_n_atoms)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the loss between predicted and reference forces.

        With ``normalize_by_n_atoms=True`` and a ``target.batch``
        attribute available, the element-wise loss is averaged over the
        xyz components and atoms of each structure and then over
        structures (weighted by ``target.weight`` when present), so each
        structure contributes equally.  Otherwise the loss is a flat mean
        over all ``N x 3`` components, per-atom weighted through
        ``target.batch`` when ``target.weight`` is given.

        Parameters
        ----------
        pred : dict[str, Tensor]
            Model outputs; reads ``'forces'`` ``(N, 3)`` in eV/Å.
        target : dict[str, Tensor]
            Reference labels; reads ``'forces'`` ``(N, 3)`` and, if
            present, the ``batch`` ``(N,)`` and ``weight`` ``(G,)``
            attributes (an empty ``weight`` is treated as absent).

        Returns
        -------
        Tensor
            Scalar loss.
        """
        p: torch.Tensor = pred["forces"]
        t: torch.Tensor = target["forces"]
        batch_index: torch.Tensor | None = getattr(target, "batch", None)
        weight: torch.Tensor | None = getattr(target, "weight", None)
        if weight is not None and weight.numel() == 0:
            weight = None

        if self._normalize_by_n_atoms and batch_index is not None:
            return _per_structure_force_loss(p, t, self._loss_fn_name, batch_index, weight)

        # Flat mean over all atoms (normalize_by_n_atoms=False, or no batch info).
        if weight is None:
            return self.loss_fn(p, t)
        return _weighted_loss_value(p, t, weight, self._loss_fn_name, batch_index=batch_index)


@LOSS_REGISTRY.register("stress")
class StressLoss(nn.Module):
    """Loss on the stress tensor with configurable loss function."""

    def __init__(self, loss_fn: str = "mse") -> None:
        super().__init__()
        self.loss_fn: typing.Callable[..., torch.Tensor] = resolve_loss_fn(loss_fn)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply ``loss_fn`` to ``pred["stress"]`` and ``target["stress"]``.

        No per-atom normalisation or sample weighting is applied; the
        reduction is the default of the configured ``loss_fn`` (mean over
        all ``(G, 3, 3)`` components for the built-ins).

        Parameters
        ----------
        pred : dict[str, Tensor]
            Model outputs; reads ``'stress'``.
        target : dict[str, Tensor]
            Reference labels; reads ``'stress'`` (same shape and units).

        Returns
        -------
        Tensor
            Scalar loss.
        """
        return self.loss_fn(pred["stress"], target["stress"])


@LOSS_REGISTRY.register("dipole")
class DipoleLoss(nn.Module):
    """Loss on the dipole moment vector with configurable loss function."""

    def __init__(self, loss_fn: str = "mse") -> None:
        super().__init__()
        self.loss_fn: typing.Callable[..., torch.Tensor] = resolve_loss_fn(loss_fn)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply ``loss_fn`` to ``pred["dipole"]`` and ``target["dipole"]``.

        No normalisation or sample weighting is applied; the reduction is
        the default of the configured ``loss_fn`` (mean over all
        ``(G, 3)`` components for the built-ins).

        Parameters
        ----------
        pred : dict[str, Tensor]
            Model outputs; reads ``'dipole'`` ``(G, 3)`` in e·Å.
        target : dict[str, Tensor]
            Reference labels; reads ``'dipole'`` ``(G, 3)``.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        return self.loss_fn(pred["dipole"], target["dipole"])


@LOSS_REGISTRY.register("charge")
class ChargeLoss(nn.Module):
    """Loss on total charge with configurable loss function."""

    def __init__(self, loss_fn: str = "mse") -> None:
        super().__init__()
        self.loss_fn: typing.Callable[..., torch.Tensor] = resolve_loss_fn(loss_fn)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply ``loss_fn`` to the predicted and reference total charge.

        When ``target`` has no ``'total_charge'`` entry the reference
        defaults to zeros, i.e. the loss becomes a charge-neutrality
        penalty on the predicted per-graph charge sum.

        Parameters
        ----------
        pred : dict[str, Tensor]
            Model outputs; reads ``'total_charge'`` ``(G,)`` in e.
        target : dict[str, Tensor]
            Reference labels; reads ``'total_charge'`` ``(G,)`` if
            present.

        Returns
        -------
        Tensor
            Scalar loss (default reduction of ``loss_fn``).
        """
        target_charge: torch.Tensor = target.get(
            "total_charge",
            torch.zeros_like(pred["total_charge"]),
        )
        return self.loss_fn(pred["total_charge"], target_charge)


@LOSS_REGISTRY.register("gate_reg")
class GateRegLoss(nn.Module):
    """L2 regularisation on the artisan gate scalars.

    Penalises large gate magnitudes: ``L = sum(P_AB ** 2)`` over all
    expert gates in the batch.  The gate values are injected into the
    predictions dict by the training module under the key
    ``"gate_values"`` (a 1-D tensor of all expert gate scalars stacked).

    When ``"gate_values"`` is absent from ``pred`` (e.g. for non-SIMURGH
    models), the loss returns ``0.0`` so the same config works for all
    model families.

    The ``loss_fn`` kwarg is accepted but ignored — ``gate_reg`` has no
    configurable loss function (it is always L2).  This keeps it
    compatible with the generic ``_build_loss`` builder which passes
    ``loss_fn`` to every registered loss class.
    """

    def __init__(self, loss_fn: str = "l2", **_kwargs: typing.Any) -> None:
        super().__init__()

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return ``sum(gate_values ** 2)`` over all gate scalars.

        The sum is not normalised by the number of gates or graphs.

        Parameters
        ----------
        pred : dict[str, Tensor]
            Model outputs; reads ``'gate_values'`` (1-D) if present.
        target : dict[str, Tensor]
            Unused; accepted for interface compatibility.

        Returns
        -------
        Tensor
            Scalar penalty, or ``0.0`` on the device of the first
            prediction when ``'gate_values'`` is missing or empty.
        """
        gate_values: torch.Tensor | None = pred.get("gate_values", None)
        if gate_values is None or gate_values.numel() == 0:
            device: torch.device = next(iter(pred.values())).device
            return torch.tensor(0.0, device=device)
        return (gate_values**2).sum()


@LOSS_REGISTRY.register("scalar_property")
class ScalarPropertyLoss(nn.Module):
    """Generic loss on any named scalar property.

    Unlike the hard-coded ``EnergyLoss`` or ``ForcesLoss``, this loss
    reads a configurable key from the predictions and targets dicts.
    Use it for arbitrary properties (HOMO, LUMO, band gap, etc.)
    without writing a custom loss class for each.

    Parameters
    ----------
    property_name : str
        Key to look up in both ``pred`` and ``target`` dicts.
    loss_fn : str
        Loss function name — ``"mse"``, ``"mae"``, ``"huber"``, etc.
    per_atom : bool
        If ``True``, normalise by number of atoms before computing loss
        (like ``EnergyLoss`` does for energy).
    """

    def __init__(
        self,
        property_name: str,
        loss_fn: str = "mse",
        per_atom: bool = False,
    ) -> None:
        super().__init__()
        self.property_name: str = property_name
        self.per_atom: bool = per_atom
        self.loss_fn: typing.Callable[..., torch.Tensor] = resolve_loss_fn(loss_fn)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply ``loss_fn`` to the ``property_name`` entry of both dicts.

        When ``per_atom`` is ``True`` both prediction and target are first
        divided by ``pred["num_atoms"]``.  No sample weighting is applied;
        the reduction is the default of the configured ``loss_fn``.

        Parameters
        ----------
        pred : dict[str, Tensor]
            Model outputs; reads ``property_name`` and, with
            ``per_atom=True``, ``'num_atoms'`` ``(G,)``.
        target : dict[str, Tensor]
            Reference labels; reads ``property_name``.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        p: torch.Tensor = pred[self.property_name]
        t: torch.Tensor = target[self.property_name]
        if self.per_atom:
            n_atoms: torch.Tensor = pred["num_atoms"]
            p = p / n_atoms
            t = t / n_atoms
        return self.loss_fn(p, t)
