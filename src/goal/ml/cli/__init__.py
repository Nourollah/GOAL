"""GOAL command-line interface — Hydra-based entry points."""

from __future__ import annotations

import os as _os

# PyTorch 2.6+ changed torch.load to weights_only=True by default.
# GOAL checkpoints include omegaconf DictConfig and other non-tensor
# objects in hyper_parameters, making weights_only=True incompatible.
# Patch Lightning's TorchCheckpointIO to always use weights_only=False
# for checkpoints that originate from this project.
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO as _TorchCheckpointIO
from lightning.fabric.utilities.cloud_io import _load as _pl_load

from goal import CONFIGS_DIR as _goal_configs_dir


def _patched_load_checkpoint(
    self: _TorchCheckpointIO,
    path: str,
    map_location: object = None,
    weights_only: bool = False,
) -> dict:
    return _pl_load(path, map_location=map_location, weights_only=False)


_TorchCheckpointIO.load_checkpoint = _patched_load_checkpoint  # type: ignore[method-assign]

# Absolute path to the bundled ``goal/configs/`` directory.  Used by every
# ``@hydra.main`` decorator in this package.
#
# Why absolute? Hydra's relative ``config_path`` is resolved differently
# depending on how the entrypoint is invoked (``python -m`` uses filesystem
# resolution, console scripts use module-based resolution).  An absolute
# filesystem path works identically for both, from a source checkout and
# from an installed wheel.
CONFIGS_DIR: str = _goal_configs_dir

# Absolute path to src/goal/configs/ml/ — self-contained single-file experiment configs.
CONFIGS_ML_DIR: str = _os.path.join(CONFIGS_DIR, "ml")


import goal.ml.adapters  # noqa: F401,E402
import goal.ml.data.datasets  # noqa: F401,E402
import goal.ml.nn.heads  # noqa: F401,E402

# ---------------------------------------------------------------------------
# Populate the global registries used by every CLI entry point.
#
# Importing these subpackages runs their ``__init__.py`` side-effects, which
# call ``MODEL_REGISTRY.register_lazy(...)`` / ``HEAD_REGISTRY.register_lazy(...)``
# / ``DATASET_REGISTRY.register_lazy(...)``.  Without these imports the
# registries are empty and the CLIs cannot resolve names like
# ``cfg.model.backbone.name = "simurgh"``.
# ---------------------------------------------------------------------------
import goal.ml.nn.models  # noqa: F401,E402
