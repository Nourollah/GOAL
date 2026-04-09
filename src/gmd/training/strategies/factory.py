"""Strategy factory — the single entry point for all strategy construction.

Replaces any hard-coded strategy logic in ``gmd/cli/train.py``.  Called
as ``build_strategy(cfg)`` with the full Hydra config, reads from the
``cfg.strategy`` subtree.
"""

from __future__ import annotations

import typing

from omegaconf import DictConfig

if typing.TYPE_CHECKING:
    from lightning.pytorch.strategies import Strategy


_KNOWN_STRATEGIES: typing.List[str] = [
    "auto",
    "single",
    "ddp",
    "fsdp",
    "fsdp2",
    "deepspeed_zero1",
    "deepspeed_zero2",
    "deepspeed_zero3",
    "deepspeed_zero3_offload",
]


def build_strategy(cfg: DictConfig) -> "typing.Union[Strategy, str]":
    """Build a Lightning strategy from Hydra config.

    Single entry point for all strategy construction.  Called from
    ``gmd/cli/train.py`` — replaces any existing hard-coded strategy
    construction there.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.  Expects ``cfg.strategy`` subtree with at
        least a ``name`` key.

    Returns
    -------
    Strategy or str
        Instantiated Lightning ``Strategy`` object, or a string
        shorthand for simple cases (``'auto'``, ``'single'``).

    Raises
    ------
    ValueError
        If the strategy name is not recognised.
    ImportError
        If DeepSpeed is requested but not installed.
    """
    strategy_cfg: typing.Any = cfg.get("strategy")
    if strategy_cfg is None:
        return "auto"

    name: str = strategy_cfg.get("name", "auto")

    if name == "auto":
        return "auto"

    if name == "single":
        from lightning.pytorch.strategies import SingleDeviceStrategy
        return SingleDeviceStrategy()

    if name == "ddp":
        from gmd.training.strategies.ddp import build_ddp_strategy
        return build_ddp_strategy(strategy_cfg)

    if name == "fsdp":
        from gmd.training.strategies.fsdp import build_fsdp_strategy
        return build_fsdp_strategy(strategy_cfg)

    if name == "fsdp2":
        from gmd.training.strategies.fsdp import build_fsdp2_strategy
        return build_fsdp2_strategy(strategy_cfg)

    if name.startswith("deepspeed"):
        from gmd.training.strategies.deepspeed import build_deepspeed_strategy
        return build_deepspeed_strategy(strategy_cfg)

    available: str = ", ".join(_KNOWN_STRATEGIES)
    raise ValueError(
        f"Unknown strategy '{name}'.  Available strategies: {available}\n"
        f"Set strategy.name to one of these in your config."
    )
