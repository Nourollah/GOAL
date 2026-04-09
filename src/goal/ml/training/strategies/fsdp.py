"""FSDP and FSDP2 strategy builders.

FSDP (Fully Sharded Data Parallel) shards model parameters, gradients,
and optimiser states across all GPUs.  FSDP2 is the modernised rewrite
built on DTensor with per-parameter sharding and ``torch.compile``
compatibility.
"""

from __future__ import annotations

import typing

from omegaconf import DictConfig

if typing.TYPE_CHECKING:
    from lightning.pytorch.strategies import FSDPStrategy


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _resolve_sharding_strategy(name: str) -> typing.Any:
    """Map a human-readable sharding name to a ``ShardingStrategy`` enum value."""
    from torch.distributed.fsdp import ShardingStrategy

    mapping: dict[str, ShardingStrategy] = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown sharding_strategy '{name}'.  " f"Choose from: {list(mapping.keys())}"
        )
    return mapping[name]


def _resolve_state_dict_type(name: str) -> typing.Any:
    """Map a human-readable state-dict name to a ``StateDictType`` enum value."""
    from torch.distributed.fsdp import StateDictType

    mapping: dict[str, StateDictType] = {
        "full": StateDictType.FULL_STATE_DICT,
        "sharded": StateDictType.SHARDED_STATE_DICT,
        "local": StateDictType.LOCAL_STATE_DICT,
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown state_dict_type '{name}'.  " f"Choose from: {list(mapping.keys())}"
        )
    return mapping[name]


def _resolve_auto_wrap_policy(
    policy_name: str | None,
    min_params: int,
) -> typing.Any:
    """Build an auto-wrap policy callable from config."""
    if policy_name is None:
        return None

    if policy_name == "size":
        import functools

        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=min_params,
        )

    if policy_name == "transformer":
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        return transformer_auto_wrap_policy

    raise ValueError(
        f"Unknown auto_wrap_policy '{policy_name}'.  "
        f"Choose from: 'size', 'transformer', or null."
    )


# ------------------------------------------------------------------
# FSDP1
# ------------------------------------------------------------------


def build_fsdp_strategy(cfg: DictConfig) -> FSDPStrategy:
    """Build ``FSDPStrategy`` (FSDP1) from config.

    FSDP1 shards model parameters, gradients, and optimiser states
    across all GPUs.  Each GPU holds 1/N of the model at rest,
    gathering full layers only during forward/backward.

    Parameters
    ----------
    auto_wrap_policy : str or None
        ``'size'`` wraps modules above ``size_based_min_params``
        parameters.  ``'transformer'`` wraps transformer-style blocks.
        ``None`` wraps the entire model as a single FSDP unit.
    size_based_min_params : int
        Minimum parameter count for size-based wrapping.
    activation_checkpointing : bool
        Re-compute activations during backward instead of storing
        them.  Trades ~30 % extra compute for ~40 % less activation
        memory.
    cpu_offload : bool
        Move parameters to CPU RAM when not in use.  Very slow —
        last resort only.
    sharding_strategy : str
        ``'full_shard'`` | ``'shard_grad_op'`` | ``'no_shard'``.
    state_dict_type : str
        ``'full'`` | ``'sharded'`` | ``'local'``.  Use ``'full'``
        for compatibility with existing MACE checkpoints.
    """
    from lightning.pytorch.strategies import FSDPStrategy

    policy_name: str | None = cfg.get("auto_wrap_policy", None)
    min_params: int = cfg.get("size_based_min_params", 1_000_000)
    activation_ckpt: bool = cfg.get("activation_checkpointing", False)
    cpu_offload: bool = cfg.get("cpu_offload", False)

    kwargs: dict[str, typing.Any] = {}

    auto_wrap: typing.Any = _resolve_auto_wrap_policy(policy_name, min_params)
    if auto_wrap is not None:
        kwargs["auto_wrap_policy"] = auto_wrap

    if cpu_offload:
        from torch.distributed.fsdp import CPUOffload

        kwargs["cpu_offload"] = CPUOffload(offload_params=True)

    if activation_ckpt and policy_name == "size":
        kwargs["activation_checkpointing_policy"] = _resolve_auto_wrap_policy(
            "size",
            min_params,
        )

    return FSDPStrategy(**kwargs)


# ------------------------------------------------------------------
# FSDP2 (via ModelParallelStrategy)
# ------------------------------------------------------------------


def build_fsdp2_strategy(cfg: DictConfig) -> typing.Any:
    """Build ``ModelParallelStrategy`` with FSDP2 backend from config.

    FSDP2 is the modernised rewrite of FSDP with per-parameter
    sharding, cleaner composability, improved memory efficiency,
    and native ``torch.compile`` support.  Prefer over FSDP1 for
    new projects.

    Parameters
    ----------
    Same config fields as ``build_fsdp_strategy``.

    Notes
    -----
    FSDP2 wrapping is controlled via ``configure_model()`` in the
    ``GOALModule``, not via auto-wrap policies.  The strategy itself
    uses ``ModelParallelStrategy``.
    """
    from lightning.pytorch.strategies import ModelParallelStrategy

    return ModelParallelStrategy()
