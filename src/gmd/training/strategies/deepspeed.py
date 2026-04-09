"""DeepSpeed ZeRO strategy builder.

Builds ``DeepSpeedStrategy`` with ZeRO stages 1/2/3 and optional CPU
offload.  All imports are deferred — DeepSpeed is never imported at
module load time.
"""

from __future__ import annotations

import logging
import typing

from omegaconf import DictConfig

if typing.TYPE_CHECKING:
    pass


def _require_deepspeed() -> None:
    """Guard import — raises a helpful error if ``deepspeed`` is absent."""
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        raise ImportError(
            "DeepSpeed is not installed.  Install it with:\n"
            "  pip install gmd[deepspeed]\n"
            "  pixi add --pypi deepspeed   (for pixi users)\n\n"
            "Minimum version required: deepspeed>=0.14"
        ) from None


def build_deepspeed_strategy(cfg: DictConfig) -> typing.Any:
    """Build ``DeepSpeedStrategy`` from config.

    DeepSpeed implements ZeRO (Zero Redundancy Optimiser) with three
    progressive stages of memory reduction:

    **Stage 1 — Optimiser state partitioning.**
        Shards Adam moments and fp32 master weights.  ~4× memory
        reduction for optimiser states with no communication overhead
        during forward/backward.

    **Stage 2 — Gradient partitioning.**
        Stage 1 + shards gradients.  ~8× memory reduction vs DDP.
        Each GPU only stores gradients for its own parameter shard.

    **Stage 3 — Parameter partitioning.**
        Stage 2 + shards model parameters.  Equivalent to FSDP.
        Maximum memory reduction at the cost of extra all-gathers.

    **Stage 3 + CPU offload.**
        Moves parameters and optimiser states to CPU RAM.  Enables
        training models larger than total GPU VRAM.

    Parameters
    ----------
    stage : int
        ZeRO stage (1, 2, or 3).
    offload_optimizer : bool
        Move optimiser states to CPU.  Stage 2+ only.
    offload_parameters : bool
        Move parameters to CPU.  Stage 3 only.
    allgather_bucket_size : int
        Bucket size for parameter all-gather in Stage 3.
    reduce_bucket_size : int
        Bucket size for gradient reduce-scatter.
    overlap_comm : bool
        Overlap gradient communication with backward pass.
    contiguous_gradients : bool
        Copy gradients into a contiguous buffer.
    logging_level : int
        DeepSpeed internal logging verbosity.
    loss_scale : float
        Loss scaling for mixed precision.  0 = dynamic.
    initial_scale_power : int
        Initial loss scale = ``2 ** initial_scale_power``.
    """
    _require_deepspeed()

    from lightning.pytorch.strategies import DeepSpeedStrategy

    name: str = cfg.get("name", "deepspeed_zero2")
    stage: int = cfg.get("stage", 2)
    offload_optimizer: bool = cfg.get("offload_optimizer", False)
    offload_parameters: bool = cfg.get("offload_parameters", False)
    allgather_bucket: int = cfg.get("allgather_bucket_size", 200_000_000)
    reduce_bucket: int = cfg.get("reduce_bucket_size", 200_000_000)
    overlap: bool = cfg.get("overlap_comm", True)
    contiguous: bool = cfg.get("contiguous_gradients", True)
    log_level: int = cfg.get("logging_level", logging.WARNING)
    loss_scale: float = cfg.get("loss_scale", 0)
    initial_scale: int = cfg.get("initial_scale_power", 16)

    # Override from name shorthand
    if name == "deepspeed_zero1":
        stage = 1
    elif name == "deepspeed_zero2":
        stage = 2
    elif name == "deepspeed_zero3":
        stage = 3
    elif name == "deepspeed_zero3_offload":
        stage = 3
        offload_optimizer = True
        offload_parameters = True

    return DeepSpeedStrategy(
        stage=stage,
        offload_optimizer=offload_optimizer,
        offload_parameters=offload_parameters,
        allgather_bucket_size=allgather_bucket,
        reduce_bucket_size=reduce_bucket,
        zero_allow_untested_optimizer=True,
        logging_level=log_level,
        loss_scale=loss_scale,
        initial_scale_power=initial_scale,
    )
