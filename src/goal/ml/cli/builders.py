"""Component builders shared by the ``goal-*`` training-type commands.

``goal-train``, ``goal-eval``, ``goal-tune`` and ``goal-finetune`` all turn
the same config sections into the same objects; keeping one copy means a
config that trains also evaluates, tunes and fine-tunes.
"""

from __future__ import annotations

import typing

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from goal.ml.nn.models.base import MonolithicModel
from goal.ml.registry import HEAD_REGISTRY, LOSS_REGISTRY
from goal.ml.training.loss import CompositeLoss, WeightedLoss


def instantiate_callbacks(cfg: DictConfig | None) -> list[Callback]:
    """Instantiate Lightning callbacks from Hydra config."""
    if not cfg:
        return []
    callbacks: list[Callback] = []
    for cb_conf in cfg.values():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def instantiate_loggers(cfg: DictConfig | None) -> list[Logger]:
    """Instantiate Lightning loggers from Hydra config."""
    if not cfg:
        return []
    loggers: list[Logger] = []
    for lg_conf in cfg.values():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers


def build_loss(cfg: DictConfig) -> CompositeLoss:
    """Construct the composite loss from config.

    Each loss entry may specify ``fn`` as either:

    - A **string** (default ``"mse"``) — single loss function.
    - A **list** of ``{name, weight}`` dicts — multiple loss functions
      for the same property, each logged and weighted independently.

    Examples::

        # Single fn (backward compatible)
        losses:
          - name: energy
            weight: 4.0
            fn: mse

        # Composite fn per property
        losses:
          - name: forces
            fn:
              - name: mse
                weight: 4.0
              - name: rmse
                weight: 8.0
    """
    import inspect

    losses: list[WeightedLoss] = []
    for loss_cfg in cfg.training.losses:
        loss_cls: typing.Any = LOSS_REGISTRY.get(loss_cfg.name)
        fn_spec: typing.Any = loss_cfg.get("fn", "mse")

        # Forward only the kwargs that the loss class actually accepts.
        # This lets the config uniformly annotate entries (e.g. normalize_by_n_atoms)
        # without every loss class needing to declare it.
        sig = inspect.signature(loss_cls.__init__)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        accepted: set[str] = set(sig.parameters) - {"self"}
        loss_kwargs: dict[str, typing.Any] = {
            k: v
            for k, v in loss_cfg.items()
            if k not in ("name", "weight", "fn") and (has_var_keyword or k in accepted)
        }

        if isinstance(fn_spec, str):
            # Single loss function — backward compatible
            losses.append(
                WeightedLoss(
                    loss_cls(loss_fn=fn_spec, **loss_kwargs),
                    weight=loss_cfg.weight,
                    label=loss_cfg.name,
                )
            )
        else:
            # Composite: list of {name, weight} sub-fns
            for sub in fn_spec:
                sub_name: str = sub["name"] if isinstance(sub, dict) else sub.name
                sub_weight: float = float(sub["weight"] if isinstance(sub, dict) else sub.weight)
                fn_label: str = sub_name.rsplit(".", 1)[-1]
                losses.append(
                    WeightedLoss(
                        loss_cls(loss_fn=sub_name),
                        weight=sub_weight,
                        label=f"{loss_cfg.name}_{fn_label}",
                        group=loss_cfg.name,
                    )
                )
    return CompositeLoss(losses)


def build_head(cfg: DictConfig, backbone: typing.Any) -> typing.Any:
    """Build the task head from config, or ``None`` for monolithic backbones.

    The pairing rule is enforced from the backbone's protocol:

    * A backbone satisfying the :class:`MonolithicModel` protocol
      *must* have ``head: null`` — it already returns the property
      dict directly, so a head would be silently ignored.
    * Any other backbone *must* declare a ``head:`` block — without one
      the backbone's ``NodeFeatures`` would have no consumer and the
      loss would see no predictions.

    Either mismatch raises ``ValueError`` here rather than failing
    deep inside the training loop.
    """
    head_cfg = cfg.model.get("head", None)
    is_monolithic: bool = isinstance(backbone, MonolithicModel)
    backbone_name: str = cfg.model.backbone.name

    if is_monolithic:
        if head_cfg is not None:
            raise ValueError(
                f"Backbone '{backbone_name}' is monolithic (returns a "
                f"property dict directly) and is incompatible with a "
                f"task head.  Set 'head: null' in the model config."
            )
        return None

    if head_cfg is None:
        raise ValueError(
            f"Backbone '{backbone_name}' is modular (produces "
            f"NodeFeatures) and requires a task head.  Declare a "
            f"'head:' block in the model config, or switch to a "
            f"monolithic backbone (e.g. 'simurgh_monolithic', "
            f"'monolithic_example')."
        )

    head_cls: typing.Any = HEAD_REGISTRY.get(head_cfg.name)
    head_kwargs: dict[str, typing.Any] = {k: v for k, v in head_cfg.items() if k != "name"}
    return head_cls(**head_kwargs)
