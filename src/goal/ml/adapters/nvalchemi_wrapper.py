"""SIMURGH / any GOAL checkpoint as an nvalchemi-toolkit model.

`nvalchemi-toolkit <https://github.com/NVIDIA/nvalchemi-toolkit>`_ runs
batched, GPU-resident MD and geometry optimisation for models that follow
its ``BaseModelMixin`` protocol.  :class:`SIMURGHNvalchemiWrapper` exposes a
trained GOAL checkpoint through that protocol, so it can drive
``NVTLangevin``, ``FIRE2``, ``FusedStage`` pipelines, …::

    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics import NVTLangevin
    from goal.ml.adapters.nvalchemi_wrapper import SIMURGHNvalchemiWrapper

    model = SIMURGHNvalchemiWrapper("logs/.../checkpoints")
    batch = Batch.from_data_list([...], device="cuda")
    batch = NVTLangevin(model=model, dt=0.5, temperature=300.0, friction=0.01, n_steps=1000).run(batch)

Requirements
------------
* ``nvalchemi-toolkit`` — on Python 3.14 install it from git ``main``
  (the PyPI releases up to 0.2.0 declare ``requires-python <3.14``)::

      pip install "nvalchemi-toolkit @ git+https://github.com/NVIDIA/nvalchemi-toolkit"

* an NVIDIA CUDA GPU.  AMD ROCm and Apple MPS are not supported by
  nvalchemi — use :class:`~goal.ml.utils.calculator.GOALCalculator` (ASE)
  or ``goal-simulate-batch --backend ase`` there.

Importing this module always works; constructing the wrapper without
nvalchemi or an NVIDIA GPU raises ``RuntimeError`` with these instructions.

How it works
------------
nvalchemi hands the model an ``AtomicData`` / ``Batch``.  The wrapper builds
the GOAL neighbour list itself on the GPU (the batched
``nvalchemi-toolkit-ops`` kernel, see
:func:`~goal.ml.data.neighbor_list.build_neighbor_list_batch`), assembles the
same PyG ``Batch`` of ``AtomicGraph``s the model was trained on, runs it, and
returns ``energy [B, 1]`` / ``forces [N, 3]`` in eV and eV/Å — nvalchemi's
units, which are GOAL's.  The model therefore declares no
``neighbor_config``: no ``NeighborListHook`` is needed.
"""

from __future__ import annotations

import collections
import logging
import typing
from functools import cache
from pathlib import Path

import torch
from torch import nn

from goal.ml.utils.hardware import nvalchemi_available

log = logging.getLogger(__name__)

INSTALL_HINT: str = (
    "SIMURGHNvalchemiWrapper requires nvalchemi-toolkit and an NVIDIA CUDA GPU.\n"
    "  Python 3.14 (GOAL's environments): "
    'pip install "nvalchemi-toolkit @ git+https://github.com/NVIDIA/nvalchemi-toolkit"\n'
    "  Python <= 3.13: pip install 'nvalchemi-toolkit[cu12]'   # or [cu13]\n"
    "AMD ROCm and Apple MPS are not supported by nvalchemi; use "
    "goal.ml.utils.calculator.GOALCalculator (ASE) or `goal-simulate-batch --backend ase` there."
)


def atomic_data_to_goal_batch(
    data: typing.Any,
    cutoff: float,
    dtype: torch.dtype,
    neighbor_backend: str = "auto",
) -> tuple[typing.Any, torch.Tensor]:
    """nvalchemi ``AtomicData`` / ``Batch`` → (GOAL PyG ``Batch``, positions leaf).

    Field mapping (nvalchemi → GOAL): ``positions [V,3]`` → ``pos``,
    ``atomic_numbers [V]`` → ``z``, ``batch_idx [V]`` → ``batch``,
    ``cell [B,3,3]`` → ``cell``, ``pbc [B,3]`` → ``pbc``; the neighbour list
    and integer ``unit_shifts`` are rebuilt for GOAL's cutoff.

    The returned positions tensor requires grad, and the batch's ``pos``
    is computed from it, so ``-dE/d(positions)`` are the forces.
    """
    from goal.ml.data.collate import batch_from_tensors

    positions = data.positions.detach().to(dtype).clone().requires_grad_(True)
    numbers = data.atomic_numbers
    batch_idx = getattr(data, "batch_idx", None)
    batch = batch_idx.long() if batch_idx is not None else None
    cell = getattr(data, "cell", None)
    pbc = getattr(data, "pbc", None)
    if cell is not None and pbc is None:
        pbc = torch.ones(cell.shape[0], 3, dtype=torch.bool, device=cell.device)
    if pbc is not None and not bool(torch.as_tensor(pbc).any()):
        cell, pbc = None, None
    goal_batch = batch_from_tensors(
        positions,
        numbers,
        cutoff,
        batch=batch,
        cell=cell.to(dtype) if cell is not None else None,
        pbc=pbc,
        neighbor_backend=neighbor_backend,
    )
    return goal_batch, positions


def goal_outputs_to_model_outputs(
    out: typing.Mapping[str, torch.Tensor], active: typing.Iterable[str]
) -> collections.OrderedDict[str, torch.Tensor | None]:
    """GOAL prediction dict → nvalchemi ``ModelOutputs`` (canonical shapes)."""
    outputs: collections.OrderedDict[str, torch.Tensor | None] = collections.OrderedDict()
    active = set(active)
    if "energy" in active:
        outputs["energy"] = out["energy"].reshape(-1, 1)
    if "forces" in active:
        outputs["forces"] = out["forces"]
    if "stress" in active and "stress" in out:
        outputs["stress"] = out["stress"].reshape(-1, 3, 3)
    return outputs


@cache
def _wrapper_class() -> type:
    """Build the concrete ``nn.Module + BaseModelMixin`` class (needs nvalchemi)."""
    from nvalchemi.models.base import BaseModelMixin, ModelConfig

    class _SIMURGHNvalchemiModel(nn.Module, BaseModelMixin, SIMURGHNvalchemiWrapper):
        def __init__(
            self,
            checkpoint_path: str | Path,
            device: str | torch.device = "cuda",
            dtype: torch.dtype | None = None,
            checkpoint: str = "best",
            use_ema: bool = True,
            neighbor_backend: str = "auto",
        ) -> None:
            nn.Module.__init__(self)
            from goal.ml.utils.calculator import (
                fragment_options_from_config,
                load_goal_checkpoint,
            )

            loaded = load_goal_checkpoint(
                checkpoint_path, checkpoint=checkpoint, device=device, use_ema=use_ema
            )
            if fragment_options_from_config(loaded.config):
                raise NotImplementedError(
                    "This checkpoint was trained with fragment labels "
                    "(data.compute_fragment_index); fragment perception is not available "
                    "on the batched GPU path.  Use GOALCalculator."
                )
            self.model = loaded.module.eval()
            self._tmpdir = loaded.tmpdir
            if dtype is not None:
                self.model = self.model.to(dtype)
            self._dtype: torch.dtype = next(self.model.parameters()).dtype
            self.cutoff: float = loaded.cutoff
            self.neighbor_backend = neighbor_backend
            self.weights: str = loaded.weights
            self.model_config = ModelConfig(
                outputs=frozenset({"energy", "forces"}),
                autograd_outputs=frozenset({"forces"}),
                autograd_inputs=frozenset({"positions"}),
                required_inputs=frozenset(),
                optional_inputs=frozenset({"cell", "pbc"}),
                supports_pbc=True,
                needs_pbc=False,
                neighbor_config=None,  # the GOAL neighbour list is built in forward()
            )

        @property
        def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
            return {}

        def compute_embeddings(self, data: typing.Any, **kwargs: typing.Any) -> typing.Any:
            raise NotImplementedError("GOAL checkpoints expose no embeddings to nvalchemi.")

        def forward(self, data: typing.Any, **kwargs: typing.Any) -> typing.Any:
            goal_batch, _ = atomic_data_to_goal_batch(
                data, self.cutoff, self._dtype, self.neighbor_backend
            )
            with torch.enable_grad():
                out = self.model(goal_batch)
            active = self.model_config.active_outputs or self.model_config.outputs
            return goal_outputs_to_model_outputs(
                {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in out.items()},
                active,
            )

    return _SIMURGHNvalchemiModel


class SIMURGHNvalchemiWrapper:
    """A trained GOAL checkpoint as an nvalchemi ``BaseModelMixin`` model.

    Constructing it returns an instance of a class that derives from
    ``torch.nn.Module``, ``nvalchemi.models.base.BaseModelMixin`` *and*
    this class, so ``isinstance(model, SIMURGHNvalchemiWrapper)`` holds.

    Parameters
    ----------
    checkpoint_path : str or Path
        ``.simurgh`` archive, checkpoint directory or ``.ckpt``.
    device : str, optional
        CUDA device, e.g. ``"cuda"`` or ``"cuda:1"``.
    dtype : torch.dtype, optional
        Cast the model (``None`` keeps the checkpoint's precision).
    checkpoint : str
        Selector inside a checkpoint directory.
    use_ema : bool
        Use the EMA weights when the checkpoint has them.
    neighbor_backend : str
        ``"auto"`` (batched Warp kernel), ``"nvalchemiops"`` or ``"torch"``.

    Raises
    ------
    RuntimeError
        Without nvalchemi-toolkit or an NVIDIA CUDA GPU.
    """

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # Allocate the concrete nvalchemi-backed subclass; ``type.__call__``
        # then runs its ``__init__`` exactly once (the result is an instance
        # of ``cls``).
        target: type = cls
        if cls is SIMURGHNvalchemiWrapper:
            if not nvalchemi_available():
                raise RuntimeError(INSTALL_HINT)
            target = _wrapper_class()
        return super().__new__(target)
