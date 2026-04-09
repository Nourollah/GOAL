"""DDP strategy builder.

Builds ``DDPStrategy`` with sensible defaults for HPC and research use.
"""

from __future__ import annotations

import typing

from omegaconf import DictConfig

if typing.TYPE_CHECKING:
    from lightning.pytorch.strategies import DDPStrategy


def build_ddp_strategy(cfg: DictConfig) -> DDPStrategy:
    """Build ``DDPStrategy`` from config.

    Parameters (all optional with defaults)
    ----------------------------------------
    find_unused_parameters : bool
        Set ``True`` only if the model has conditional computation paths
        that leave some parameters unused.  Adds overhead.  Equivariant
        GNNs (MACE, HyperSpec) should keep this ``False``.
    static_graph : bool
        Set ``True`` when the computation graph is identical every
        forward pass.  Enables internal DDP optimisations.  Safe for
        most GNN architectures.
    gradient_as_bucket_view : bool
        Reduces peak memory by sharing gradient storage with the DDP
        communication buckets.
    bucket_cap_mb : int
        Size of gradient all-reduce buckets in MB.  Reduce to 10 if
        hitting OOM during the gradient sync phase.
    process_group_backend : str
        ``'nccl'`` for GPU, ``'gloo'`` for CPU.  Never change for GPU
        training.
    """
    from lightning.pytorch.strategies import DDPStrategy

    find_unused: bool = cfg.get("find_unused_parameters", False)
    static_graph: bool = cfg.get("static_graph", False)
    gradient_bucket_view: bool = cfg.get("gradient_as_bucket_view", True)
    bucket_cap: int = cfg.get("bucket_cap_mb", 25)
    backend: str = cfg.get("process_group_backend", "nccl")

    return DDPStrategy(
        find_unused_parameters=find_unused,
        static_graph=static_graph,
        gradient_as_bucket_view=gradient_bucket_view,
        bucket_cap_mb=bucket_cap,
        process_group_backend=backend,
    )
