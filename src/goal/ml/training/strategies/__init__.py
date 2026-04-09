"""Distributed training strategy builders.

Provides a unified ``build_strategy()`` entry point that maps a Hydra
config subtree to a Lightning ``Strategy`` object (or ``'auto'`` string).

Supported strategies:
    - ``ddp``                     — DDPStrategy
    - ``fsdp``                    — FSDPStrategy (FSDP1)
    - ``fsdp2``                   — ModelParallelStrategy (FSDP2)
    - ``deepspeed_zero1``         — DeepSpeedStrategy stage 1
    - ``deepspeed_zero2``         — DeepSpeedStrategy stage 2
    - ``deepspeed_zero3``         — DeepSpeedStrategy stage 3
    - ``deepspeed_zero3_offload`` — DeepSpeedStrategy stage 3 + CPU offload
    - ``single``                  — SingleDeviceStrategy
    - ``auto``                    — Lightning auto-selects
"""

from __future__ import annotations

from goal.ml.training.strategies.factory import build_strategy

__all__: list[str] = ["build_strategy"]
