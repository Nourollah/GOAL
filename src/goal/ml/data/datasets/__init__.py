"""GOAL dataset implementations.

Drop a new ``.py`` file into this folder with
``@DATASET_REGISTRY.register("name")`` on its class and it is discovered
automatically — no edit to this file required.
"""

from __future__ import annotations

from goal.ml.registry import DATASET_REGISTRY, auto_discover

auto_discover(__name__)

# Zarr stores (optional zarr>=3 dependency; imported only when used).
DATASET_REGISTRY.register_lazy("zarr", "goal.ml.data.zarr_dataset:ZarrDataset")

# Benchmark adapters (md17/rmd17, md22, qm9, ani1/ani1x, spice) — lazy, so
# nothing heavy is imported until one is requested.
import goal.ml.data.benchmarks  # noqa: F401
