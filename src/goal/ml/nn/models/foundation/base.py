"""Base class for trainable foundation-model backbones.

A ``FoundationModelBackbone`` loads a pre-trained interatomic potential
(MACE, UMA, …) as a **real** ``torch.nn.Module`` and exposes it to GOAL as a
*monolithic* backbone — it returns the property dict ``{energy, forces,
num_atoms}`` directly, so it plugs into :class:`goal.ml.training.module.GOALModule`
with ``head: null`` and reuses the loss, stage curriculum, optimiser
parameter-grouping, ``GOALCheckpointManager``, DDP/FSDP and HPO **unchanged**.

Why a real ``nn.Module``?  Assigned as ``GOALModule.backbone``, only an
``nn.Module`` has its parameters registered in the module ``state_dict`` (so
``trainer.save_checkpoint`` persists the fine-tuned weights), seen by
``configure_optimizers`` (``self.parameters()``), and moved to the right
device/dtype by Lightning.  The historical ``MACEAdapter``/``UMAAdapter``
were plain classes and therefore could not actually be fine-tuned.

Three fine-tuning strategies are shared here:

* ``head_only`` — freeze everything except the final readout layers.  Fastest,
  most stable; recommended for small datasets (< ~5000 structures).
* ``full`` — train all parameters.  Use a much smaller LR than ``head_only``
  (≈1e-5 vs 1e-4); risk of catastrophic forgetting on small datasets.
* ``lora`` — freeze the backbone and inject LoRA adapters into every
  ``nn.Linear`` via ``peft`` (model-agnostic).

Subclasses implement the model-specific pieces: :meth:`_load_model`,
:meth:`forward`, :meth:`_readout_name_fragments`, and
:meth:`set_atomic_energies`.
"""

from __future__ import annotations

import logging
import typing

import torch
import torch.nn as nn

from goal.ml.data.statistics import ATOMIC_SYMBOLS, compute_atomic_references

log = logging.getLogger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "double": torch.float64,
    "float": torch.float32,
    "half": torch.float16,
}


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Map a dtype name (or a ``torch.dtype``) to a ``torch.dtype``."""
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).replace("torch.", "").lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype {dtype!r}. Choose one of {sorted(_DTYPE_MAP)}.")
    return _DTYPE_MAP[key]


def estimate_reference_energies(
    dataset: typing.Iterable[typing.Any],
) -> dict[int, float]:
    """Ridge/least-squares per-element reference energies from a dataset.

    Thin wrapper over :func:`goal.ml.data.statistics.compute_atomic_references`
    (``E_total_i = Σ_Z n_iZ · e_Z``) so foundation-model callers have a single
    named entry point.  Returns ``{Z: e_Z}`` for every element that appears in
    the training set.

    Re-estimating E0s is the most common cause of fine-tuning failure: a
    foundation model's atomic energies are fit to *its* training distribution
    (e.g. MPTrj), not yours, so the interaction network is handed an O(eV)
    baseline error to absorb.
    """
    return compute_atomic_references(dataset)


class FoundationModelBackbone(nn.Module):
    """Base class for trainable foundation-model backbones (monolithic).

    Parameters
    ----------
    strategy : str
        ``"head_only"`` (default), ``"full"``, or ``"lora"``.
    dtype : str or torch.dtype
        Compute precision for the wrapped model (foundation models are
        typically ``float64``).
    reestimate_e0s : bool
        Advisory flag read by :class:`goal.ml.training.callbacks.foundation.FoundationE0Callback`
        to decide whether to re-baseline atomic energies from the training set
        before fitting.  The backbone itself does not read the dataset.
    lora_rank, lora_alpha : int, float
        LoRA hyper-parameters (only used when ``strategy == "lora"``).
    lora_target_modules : list[str] or None
        Explicit LoRA target module names.  ``None`` targets every
        ``nn.Linear`` in the wrapped model.
    """

    def __init__(
        self,
        strategy: str = "head_only",
        dtype: str | torch.dtype = "float64",
        reestimate_e0s: bool = True,
        lora_rank: int = 4,
        lora_alpha: float = 16.0,
        lora_target_modules: list[str] | None = None,
    ) -> None:
        super().__init__()
        if strategy not in ("head_only", "full", "lora"):
            raise ValueError(f"strategy must be 'head_only', 'full', or 'lora', got {strategy!r}.")
        self.strategy: str = strategy
        self.torch_dtype: torch.dtype = resolve_dtype(dtype)
        self.reestimate_e0s: bool = bool(reestimate_e0s)
        self.lora_rank: int = int(lora_rank)
        self.lora_alpha: float = float(lora_alpha)
        self.lora_target_modules: list[str] | None = lora_target_modules
        # Subclass sets ``self._model`` (an nn.Module) in its __init__, then
        # calls ``self._apply_strategy()``.
        self._model: nn.Module

    # ------------------------------------------------------------------
    # Monolithic backbone protocol
    # ------------------------------------------------------------------

    @property
    def output_keys(self) -> list[str]:
        """Keys this backbone's forward produces."""
        return ["energy", "forces"]

    def forward(self, graph: typing.Any) -> dict[str, torch.Tensor]:  # pragma: no cover
        """Run the wrapped model and return GOAL's monolithic property dict.

        Subclasses must convert the ``AtomicGraph`` to the upstream
        model's input format, evaluate ``self._model`` (forces are usually
        obtained by autograd inside the upstream model, so the call should
        run under ``torch.enable_grad()``), and cast the outputs back to
        ``graph.pos.dtype`` so they match the labels.

        Parameters
        ----------
        graph : AtomicGraph
            Batched input graph (``pos`` ``(N, 3)`` in Å, ``z``,
            ``edge_index``, ``batch``, optional ``cell``).

        Returns
        -------
        dict[str, Tensor]
            ``'energy'`` ``(B,)`` in eV, ``'forces'`` ``(N, 3)`` in eV/Å
            and ``'num_atoms'`` ``(B,)``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _load_model(self, **kwargs: typing.Any) -> nn.Module:  # pragma: no cover
        """Load and return the upstream model as an ``nn.Module``."""
        raise NotImplementedError

    def _readout_name_fragments(self) -> tuple[str, ...]:  # pragma: no cover
        """Substrings identifying readout (head) parameters for ``head_only``.

        A parameter is left trainable in ``head_only`` mode iff its
        fully-qualified name contains any of these fragments.
        """
        raise NotImplementedError

    def set_atomic_energies(self, references: dict[int, float]) -> None:  # pragma: no cover
        """Write per-element reference energies into the model in place."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Fine-tuning strategy
    # ------------------------------------------------------------------

    def _apply_strategy(self) -> None:
        """Apply the configured freezing / adapter strategy to ``self._model``."""
        if self.strategy == "full":
            for p in self._model.parameters():
                p.requires_grad_(True)
            log.info(
                "[finetune] strategy=full — all %d params trainable.", _count(self._model, True)
            )
            return

        if self.strategy == "head_only":
            fragments = self._readout_name_fragments()
            n_train = 0
            for name, p in self._model.named_parameters():
                keep = any(frag in name for frag in fragments)
                p.requires_grad_(keep)
                n_train += int(keep)
            log.info(
                "[finetune] strategy=head_only — %d/%d params trainable "
                "(readout fragments=%s).",
                n_train,
                _count(self._model, None),
                fragments,
            )
            return

        # lora
        self._inject_lora()

    def _inject_lora(self) -> None:
        """Freeze the base model and inject LoRA adapters into its linears."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover - exercised via mock in tests
            raise ImportError(
                "LoRA fine-tuning requires `peft`. Install with: pip install peft."
            ) from exc

        for p in self._model.parameters():
            p.requires_grad_(False)

        targets: list[str] = self.lora_target_modules or [
            name for name, m in self._model.named_modules() if isinstance(m, nn.Linear)
        ]
        if not targets:
            log.warning("[finetune] strategy=lora — no nn.Linear modules found to adapt.")

        config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=targets,
        )
        self._model = get_peft_model(self._model, config)
        log.info(
            "[finetune] strategy=lora — rank=%d alpha=%s, %d target linear modules.",
            self.lora_rank,
            self.lora_alpha,
            len(targets),
        )

    # ------------------------------------------------------------------
    # E0 re-estimation
    # ------------------------------------------------------------------

    def reestimate_atomic_energies(self, dataset: typing.Iterable[typing.Any]) -> dict[int, float]:
        """Fit per-element E0s on ``dataset`` and write them into the model.

        Returns the ``{Z: e_Z}`` dictionary that was applied.
        """
        refs = estimate_reference_energies(dataset)
        if not refs:
            log.warning("[finetune] E0 re-estimation skipped — no energies in dataset.")
            return {}
        self.set_atomic_energies(refs)
        pretty = ", ".join(
            f"{ATOMIC_SYMBOLS.get(z, '?')}({z})={e:+.4f}" for z, e in sorted(refs.items())
        )
        log.info("[finetune] re-estimated atomic energies: %s", pretty)
        return refs


def _count(model: nn.Module, requires_grad: bool | None) -> int:
    """Count parameters, optionally filtered by ``requires_grad``."""
    return sum(
        1 for p in model.parameters() if requires_grad is None or p.requires_grad == requires_grad
    )
