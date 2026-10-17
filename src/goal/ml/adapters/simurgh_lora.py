"""LoRA fine-tuning of a trained SIMURGH / ARACE checkpoint.

Low-rank adapters are injected into the **artisan** layers — the
per-element-pair networks, where e.g. mono→dicarboxylic-acid transfer has
to move — while the rest of the checkpoint stays frozen.

Which layers can carry an adapter
---------------------------------
LoRA (via `peft <https://github.com/huggingface/peft>`_) adapts plain
``nn.Linear`` layers.  Inside an artisan those are the **invariant** maps:
the element-conditioning table (``element_linear``), the radial MLP that
produces the tensor-product weights, and the scalar energy readout
(``energy_mlp``).  Their inputs and outputs are rotation-invariant, so a
low-rank update cannot break equivariance.  The equivariant maps (e3nn
``Linear`` / tensor products) are not ``nn.Linear`` and are left frozen.

Usage
-----
Through the training CLI (recommended — checkpoint manager, archives,
resume)::

    goal-train --config-name finetune/simurgh_lora \\
        model.backbone.checkpoint_path=/abs/path/to/run/checkpoints \\
        data.train_paths=[data/dicarboxylic/train.xyz] ...

or ``goal-finetune`` (same default config), or from Python::

    from goal.ml.adapters.simurgh_lora import SIMURGHLoRAAdapter, SIMURGHLoRAConfig

    adapter = SIMURGHLoRAAdapter(SIMURGHLoRAConfig(checkpoint_path="run/checkpoints", rank=8))
    print(adapter.trainable_parameters(), "/", adapter.total_parameters())
    ...  # train adapter(graph)["energy"/"forces"]
    adapter.save("dicarboxylic.lora")  # adapter delta only
    adapter = SIMURGHLoRAAdapter.load("dicarboxylic.lora")
"""

from __future__ import annotations

import dataclasses
import logging
import typing
from pathlib import Path

import torch
from torch import nn

log = logging.getLogger(__name__)

#: File suffix of a saved adapter delta.
LORA_SUFFIX: str = ".lora"

#: Format tag written into ``.lora`` files.
LORA_FORMAT: str = "goal-simurgh-lora/1"

_PEFT_HINT: str = "LoRA fine-tuning needs peft. Install peft: pip install 'peft>=0.13'"

#: Parameter-name fragments that make up the "task head" of a SIMURGH model:
#: the modular head, the artisans' scalar energy readouts and the learned
#: per-element reference energies.
DEFAULT_HEAD_PATTERNS: tuple[str, ...] = ("head.", "energy_mlp", "atomic_energies")


@dataclasses.dataclass
class SIMURGHLoRAConfig:
    """Configuration for LoRA fine-tuning of a SIMURGH checkpoint.

    Parameters
    ----------
    rank : int
        Rank of the low-rank update.  Smaller = fewer parameters; 4–64 is
        the usual range, 8 a good start for domain adaptation.
    alpha : float
        LoRA scale; the update is multiplied by ``alpha / rank``.  Keep
        ``alpha >= rank``; doubling it doubles the adapters' effective
        learning rate.
    dropout : float
        Dropout inside the adapters (0.05–0.1 helps on very small sets).
    target_modules : list of str
        Substrings of the ``nn.Linear`` names to adapt.  ``["artisan"]``
        adapts every linear inside the artisan layers; add e.g.
        ``"fragment_interaction"`` to also adapt that add-on's radial MLPs.
    bias : str
        ``"none"`` (adapters only), ``"all"`` (every bias trainable) or
        ``"lora_only"`` (biases of the adapted layers).
    checkpoint_path : str or Path
        The SIMURGH checkpoint to adapt: a ``.simurgh`` archive, a
        checkpoint directory or a ``.ckpt``.  Use an absolute path — a
        fine-tuned model is rebuilt from it when loaded later.
    checkpoint : str
        Selector inside a checkpoint directory (``"best"``, ``"last"``,
        ``"epoch=N"``).
    use_ema : bool
        Start from the base run's EMA weights when it has them.
    head_trainable : bool
        Also train the task head (energy readouts, atomic references)
        in full.  Usually converges faster on small datasets.  Head layers
        are then trained directly and get no adapter.
    head_patterns : list of str
        Name fragments that define the head (see
        :data:`DEFAULT_HEAD_PATTERNS`).
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = dataclasses.field(default_factory=lambda: ["artisan"])
    bias: str = "none"
    checkpoint_path: str | None = None
    checkpoint: str = "best"
    use_ema: bool = True
    head_trainable: bool = True
    head_patterns: list[str] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_HEAD_PATTERNS)
    )

    def __post_init__(self) -> None:
        if self.checkpoint_path is not None:
            self.checkpoint_path = str(self.checkpoint_path)
        self.target_modules = [str(t) for t in self.target_modules]
        self.head_patterns = [str(t) for t in self.head_patterns]
        if self.bias not in ("none", "all", "lora_only"):
            raise ValueError(f"bias must be 'none', 'all' or 'lora_only', got {self.bias!r}.")
        if int(self.rank) < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}.")


def _require_peft() -> typing.Any:
    try:
        import peft
    except ImportError as exc:
        raise ImportError(_PEFT_HINT) from exc
    return peft


class SIMURGHLoRAAdapter(nn.Module):
    """A frozen SIMURGH model with trainable LoRA adapters on its artisans.

    Parameters
    ----------
    config : SIMURGHLoRAConfig
        Adapter settings; ``config.checkpoint_path`` is loaded unless
        ``model`` is given.
    model : nn.Module, optional
        An already-built SIMURGH model to adapt instead of loading one —
        a ``GOALModule`` or anything with ``backbone`` / ``head``
        attributes, or a bare monolithic model.

    The forward pass is the wrapped model's: ``adapter(graph)`` returns
    the ``{"energy", "forces", ...}`` dict.
    """

    def __init__(self, config: SIMURGHLoRAConfig, model: nn.Module | None = None) -> None:
        peft = _require_peft()
        super().__init__()
        self.config = config
        self._tmpdir: typing.Any = None
        self.base_config: typing.Any = None
        self.cutoff: float | None = None

        if model is None:
            if not config.checkpoint_path:
                raise ValueError("SIMURGHLoRAConfig.checkpoint_path is required.")
            from goal.ml.utils.calculator import load_goal_checkpoint

            path = Path(config.checkpoint_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Base SIMURGH checkpoint not found: {path}.  A LoRA model is rebuilt "
                    f"from its base checkpoint, so that file must stay reachable."
                )
            loaded = load_goal_checkpoint(
                path, checkpoint=config.checkpoint, device="cpu", use_ema=config.use_ema
            )
            self._tmpdir = loaded.tmpdir
            self.base_config = loaded.config
            self.cutoff = loaded.cutoff
            model = loaded.module
            log.info("[lora] base model %s (%s weights)", path, loaded.weights)

        # Keep the backbone/head pair, not a nested LightningModule.
        if hasattr(model, "backbone") and hasattr(model, "head"):
            self.backbone: nn.Module = model.backbone
            self.head: nn.Module | None = model.head
        else:
            self.backbone, self.head = model, None

        targets = self._target_names()
        if not targets:
            linear_names = [n for n, m in self.named_modules() if isinstance(m, nn.Linear)]
            raise ValueError(
                f"No nn.Linear matches target_modules={config.target_modules}.  "
                f"Linear layers in this model (first 10): {linear_names[:10]}"
            )
        lora_cfg = peft.LoraConfig(
            r=int(config.rank),
            lora_alpha=float(config.alpha),
            lora_dropout=float(config.dropout),
            target_modules=targets,
            bias="none",  # bias trainability is set below, for every mode
        )
        peft.inject_adapter_in_model(lora_cfg, self)
        self._adapted: list[str] = targets
        self._apply_trainability()
        log.info(
            "[lora] rank=%d alpha=%s on %d linear layers; %d / %d parameters trainable",
            config.rank,
            config.alpha,
            len(targets),
            self.trainable_parameters(),
            self.total_parameters(),
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _is_head(self, name: str) -> bool:
        return any(p in name for p in self.config.head_patterns)

    def _target_names(self) -> list[str]:
        """Fully-qualified names of the ``nn.Linear`` layers to adapt."""
        names: list[str] = []
        for name, module in self.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(t in name for t in self.config.target_modules):
                continue
            if self.config.head_trainable and self._is_head(name + "."):
                continue  # trained in full instead
            names.append(name)
        return names

    def _apply_trainability(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        for name, p in self.named_parameters():
            if (
                "lora_" in name
                or self.config.head_trainable
                and self._is_head(name)
                or name.endswith(".bias")
                and (
                    self.config.bias == "all"
                    or (
                        self.config.bias == "lora_only"
                        and any(name.startswith(t + ".") for t in self._adapted)
                    )
                )
            ):
                p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Forward / introspection
    # ------------------------------------------------------------------

    def forward(self, graph: typing.Any) -> dict[str, torch.Tensor]:
        """Run the adapted model: ``head(backbone(graph), graph)``.

        For a bare monolithic model (no separate head) the backbone's own
        output dict is returned.  The LoRA deltas are applied inside the
        injected linear layers, so the outputs are those of the base
        SIMURGH model with its artisans adapted.

        Parameters
        ----------
        graph : AtomicGraph
            Batched input graph, as for the base model.

        Returns
        -------
        dict[str, Tensor]
            The wrapped model's predictions, e.g. ``'energy'`` ``(B,)`` in
            eV, ``'forces'`` ``(N, 3)`` in eV/Å and ``'num_atoms'``.
        """
        if self.head is None:
            return self.backbone(graph)
        return self.head(self.backbone(graph), graph)

    @property
    def output_keys(self) -> list[str]:
        source = self.head if self.head is not None else self.backbone
        return list(getattr(source, "output_keys", ["energy", "forces"]))

    def trainable_parameters(self) -> int:
        """Number of trainable scalars (adapters + head, if enabled)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        """Number of scalars in the whole model (frozen base + adapters)."""
        return sum(p.numel() for p in self.parameters())

    def adapted_layers(self) -> list[str]:
        """Names of the linear layers that carry an adapter."""
        return list(self._adapted)

    def merge(self) -> None:
        """Fold the adapters into the base weights (faster inference).

        Irreversible for this object; save the adapter first if you need it.
        """
        for module in self.modules():
            if hasattr(module, "merge") and hasattr(module, "lora_A"):
                module.merge()

    # ------------------------------------------------------------------
    # Persistence: the adapter delta only
    # ------------------------------------------------------------------

    def delta_state_dict(self) -> dict[str, torch.Tensor]:
        """Trainable tensors only: adapters, head (if trained), biases."""
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        return {
            k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in trainable
        }

    def save(self, path: str | Path) -> Path:
        """Write the adapter delta (not the base model) to ``path``.

        ``.lora`` is appended when missing.  The file references the base
        checkpoint, which must still exist when it is loaded.
        """
        path = Path(path)
        if path.suffix != LORA_SUFFIX:
            path = path.with_name(path.name + LORA_SUFFIX)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": LORA_FORMAT,
                "config": dataclasses.asdict(self.config),
                "adapted_layers": self._adapted,
                "state_dict": self.delta_state_dict(),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, model: nn.Module | None = None) -> SIMURGHLoRAAdapter:
        """Rebuild an adapter from a ``.lora`` file (base checkpoint + delta)."""
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if payload.get("format") != LORA_FORMAT:
            raise ValueError(
                f"{path} is not a SIMURGH LoRA file (format={payload.get('format')!r})."
            )
        adapter = cls(SIMURGHLoRAConfig(**payload["config"]), model=model)
        missing = set(payload["state_dict"]) - set(adapter.state_dict())
        if missing:
            raise ValueError(f"{path} does not match the rebuilt adapter: {sorted(missing)[:5]}")
        adapter.load_state_dict(payload["state_dict"], strict=False)
        return adapter


class SimurghLoRABackbone(SIMURGHLoRAAdapter):
    """Config-driven monolithic backbone: ``model.backbone.name: simurgh_lora``.

    Registered lazily in :mod:`goal.ml.adapters` (backbone and model
    registries).

    Same keyword arguments as :class:`SIMURGHLoRAConfig`.  Monolithic
    (``head: null``): returns the wrapped model's property dict, so
    ``goal-train`` / ``goal-finetune`` train it like any other model and
    the fine-tuned checkpoint loads in ``GOALCalculator``.
    """

    def __init__(self, cutoff: float | None = None, **kwargs: typing.Any) -> None:
        fields = {f.name for f in dataclasses.fields(SIMURGHLoRAConfig)}
        unknown = set(kwargs) - fields
        if unknown:
            raise TypeError(
                f"simurgh_lora got unknown arguments {sorted(unknown)}; valid: {sorted(fields)}"
            )
        cfg_kwargs = {
            k: (list(v) if k in ("target_modules", "head_patterns") else v)
            for k, v in kwargs.items()
        }
        super().__init__(SIMURGHLoRAConfig(**cfg_kwargs))
        # ``cutoff`` (usually ${data.cutoff}) must be the base model's: the
        # fine-tuning graphs are built with it, and a different neighbour
        # list is a different model.
        if (
            cutoff is not None
            and self.cutoff is not None
            and abs(float(cutoff) - self.cutoff) > 1e-9
        ):
            raise ValueError(
                f"data.cutoff={cutoff} differs from the base checkpoint's cutoff "
                f"{self.cutoff}; set data.cutoff={self.cutoff}."
            )
