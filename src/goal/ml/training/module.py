"""Universal Lightning module for any backbone + head combination.

Handles training, validation, testing, EMA, gradient clipping, and logging.
Compatible with Lightning 2.6+ features including:
- ``configure_model()`` hook for FSDP2 / torch.compile
- ``EMAWeightAveraging`` callback (replaces manual EMA when preferred)
- ``torch.compile`` support with configurable mode and fullgraph
- Proper multi-GPU sync via ``sync_dist``
- ``test_step`` for evaluation
- Performance: gradient accumulation, non-blocking transfers, TF32
"""

from __future__ import annotations

import re
import typing
import warnings

import lightning as L
import torch
from omegaconf import DictConfig
from torch import nn

from goal.ml.data.graph import AtomicGraph
from goal.ml.training.ema import EMAWrapper
from goal.ml.training.loss import CompositeLoss
from goal.ml.training.metrics import PROG_BAR_METRICS, metric_section, mlip_metrics


class GOALModule(L.LightningModule):
    """Universal ``LightningModule`` for any backbone + head combination.

    Handles: training, validation, testing, EMA, gradient clipping, logging.
    Compatible with Lightning >=2.6 APIs.

    Supports two wiring modes:

    **Modular** (backbone + head)
        A backbone satisfying ``EquivariantBackbone`` or ``InvariantBackbone``
        produces ``NodeFeatures``, then a ``TaskHead`` maps those features to
        a property dictionary.

    **Monolithic** (backbone only, head=None)
        A model satisfying ``MonolithicModel`` directly returns a property
        dictionary from ``forward(graph)``.  Set ``head=None`` and the
        training loop will call the backbone directly.

    Parameters
    ----------
    backbone : EquivariantBackbone | MonolithicModel
        Any model satisfying a backbone or monolithic protocol.
    head : TaskHead | None
        Output head for modular models.  ``None`` for monolithic models.
    loss : CompositeLoss
        Composable loss function built from weighted components.
    config : DictConfig
        Full Hydra configuration.
    compile_model : bool
        Whether to apply ``torch.compile`` to the backbone in
        ``configure_model()``.
    """

    def __init__(
        self,
        backbone: typing.Any,
        head: typing.Any,
        loss: CompositeLoss,
        config: DictConfig,
        compile_model: bool = False,
    ) -> None:
        super().__init__()
        self.backbone: typing.Any = backbone
        self.head: typing.Any = head
        self.loss: CompositeLoss = loss
        self.config: DictConfig = config
        self._compile_model: bool = compile_model
        self.save_hyperparameters(ignore=["backbone", "head", "loss"])

        #: Backbone buffers (``"scale"``, ``"atomic_energies"``) whose freshly
        #: built values must survive a checkpoint restore.  ``goal-train``
        #: fills this when a resumed run recomputes those statistics
        #: (``dataset_stats.recompute``); empty means restore everything.
        self.keep_fresh_buffers: set[str] = set()

        # EMA — keeps a shadow copy of weights for evaluation.
        # Lightning 2.6 provides EMAWeightAveraging callback as an alternative,
        # but we keep this for fine-grained control (e.g. per-step decay schedule).
        if config.training.ema.get("enabled", False):
            self.ema = EMAWrapper(
                self.parameters(),
                decay=config.training.ema.decay,
            )

        # ----- TASK 2: stage-based loss-weight schedule -----
        #
        # ``training.stages`` is a list of dicts ``{name, epochs,
        # loss_weights}`` defining an epoch-keyed curriculum.  At each
        # step we look up the currently-active stage and patch
        # ``WeightedLoss.weight`` for every matching label.  The set
        # of *labels* a stage can override is fixed at construction
        # time by introspecting the CompositeLoss — unknown labels in
        # the stage config are silently ignored (a stage may name
        # ``"newton"`` before the matching loss exists).
        raw_stages: typing.Any = config.training.get("stages", None)
        self._stages_cfg: list[typing.Any] = list(raw_stages or [])
        self._weighted_loss_by_label: dict[str, typing.Any] = {}
        self._base_loss_weights: dict[str, float] = {}
        for wl in self.loss.losses:
            label: str | None = getattr(wl, "label", None)
            if label is None:
                continue
            self._weighted_loss_by_label[label] = wl
            self._base_loss_weights[label] = float(getattr(wl, "weight", 1.0))
        self._current_stage_name: str | None = None

        # Warn about labels named in any stage's loss_weights that do
        # not have a matching declared component.  Forward-declaration
        # is intentionally allowed (a stage may name ``newton`` before
        # the newton loss is wired), but typos like ``force`` vs
        # ``forces`` would otherwise silently leave the base weight
        # untouched — invisible at runtime, easy to miss for hours.
        # ``strict_stage_labels: true`` upgrades the warning to an error
        # for projects that want hard validation.
        known_labels: set[str] = set(self._weighted_loss_by_label.keys())
        unknown: dict[str, list[str]] = {}
        for stage in self._stages_cfg:
            stage_name: str = str(stage.get("name", "?"))
            overrides: typing.Any = stage.get("loss_weights", None) or {}
            for label in dict(overrides).keys():
                if str(label) not in known_labels:
                    unknown.setdefault(stage_name, []).append(str(label))
        if unknown:
            details: str = "; ".join(
                f"stage {n!r} -> {sorted(labels)}" for n, labels in unknown.items()
            )
            msg: str = (
                f"training.stages references loss labels that aren't "
                f"declared in training.losses: {details}.  Declared "
                f"labels: {sorted(known_labels)}.  These overrides will "
                f"be ignored until a matching loss component is added."
            )
            if bool(config.training.get("strict_stage_labels", False)):
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

    def configure_model(self) -> None:
        """Hook called before ``configure_optimizers`` (Lightning 2.4+).

        Used for:
        - ``torch.compile`` wrapping (avoids recompilation per epoch)
        - FSDP2 / tensor parallelism wrapping via ``ModelParallelStrategy``
        - DTensor-based sharding
        """
        if self._compile_model:
            compile_cfg: dict[str, typing.Any] = self.config.training.get("compile", {})
            # torch.compile requires PyTorch 2.0+ and works best with CUDA
            if not hasattr(torch, "compile"):
                warnings.warn(
                    "torch.compile requested but unavailable (PyTorch <2.0). Skipping.",
                    stacklevel=2,
                )
            else:
                self.backbone = torch.compile(
                    self.backbone,
                    mode=compile_cfg.get("mode", "default"),
                    fullgraph=compile_cfg.get("fullgraph", False),
                    dynamic=compile_cfg.get("dynamic", None),
                )

    #: Artisan classes that own learnable gate scalars, mapped to the
    #: attribute holding them (``gate``: 0-d parameter, ``gates``: 1-d).
    #: Matched by class *name* along the MRO rather than ``isinstance``:
    #: a model rebuilt from a checkpoint's frozen source carries copies of
    #: these classes from the snapshot, which ``isinstance`` against the
    #: live classes would reject.
    GATED_ARTISAN_CLASSES: typing.ClassVar[dict[str, str]] = {
        "PotentialArtisan": "gate",  # legacy SimurghArtisanBank
        "RarePotentialArtisan": "gates",  # legacy shared rare-pair artisan
        "_AracePairArtisan": "gate",  # ARACE family (+ _AraceRarePairArtisan)
    }

    def _gated_artisans(self) -> list[tuple[nn.Module, str]]:
        """``(module, attribute)`` for every gate-carrying artisan in the backbone.

        Built once — the module tree does not change after construction —
        and the gates themselves are read live in :meth:`forward`.
        """
        cached: list[tuple[nn.Module, str]] | None = self.__dict__.get("_gated_artisan_cache")
        if cached is None:
            cached = []
            for mod in self.backbone.modules():
                for cls in type(mod).__mro__:
                    attr: str | None = self.GATED_ARTISAN_CLASSES.get(cls.__name__)
                    if attr is not None:
                        if isinstance(getattr(mod, attr, None), nn.Parameter):
                            cached.append((mod, attr))
                        break
            self.__dict__["_gated_artisan_cache"] = cached
        return cached

    def _gate_values(self) -> torch.Tensor | None:
        """All artisan gates as one live (grad-enabled) 1-D tensor, or ``None``."""
        gates: list[torch.Tensor] = [
            getattr(mod, attr).reshape(-1) for mod, attr in self._gated_artisans()
        ]
        return torch.cat(gates) if gates else None

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        """Run backbone -> head pipeline, or backbone alone for monolithic models."""
        if self.head is None:
            predictions: dict[str, torch.Tensor] = self.backbone(graph)
        else:
            features: typing.Any = self.backbone(graph)
            predictions = self.head(features, graph)
        # Inject gate values (live, grad-enabled) for GateRegLoss — the raw
        # Parameters, not ``.gates()`` (which detaches).  Every SIMURGH
        # family is covered, modular and monolithic; backbones without
        # artisans are silently skipped and ``gate_reg`` stays 0.
        gate_values: torch.Tensor | None = self._gate_values()
        if gate_values is not None:
            predictions["gate_values"] = gate_values

        # Adaptive-depth-gate side channels (ARACE add-on).  Duck-typed,
        # so every other backbone is silently skipped; both attributes are
        # ``None`` unless ``model.backbone.adaptive_gate`` is configured.
        # ``last_aux_loss`` keeps its autograd graph — :meth:`_with_aux_loss`
        # adds it to the training total.
        # A monolithic model that reports these itself keeps its own values.
        aux_loss: torch.Tensor | None = getattr(self.backbone, "last_aux_loss", None)
        if aux_loss is not None and "aux_loss" not in predictions:
            predictions["aux_loss"] = aux_loss
        round_gates: typing.Any = getattr(self.backbone, "last_gate_scores", None)
        if round_gates and "gate_scores" not in predictions:
            populated: list[torch.Tensor] = [g for g in round_gates if g is not None]
            if populated:
                predictions["gate_scores"] = torch.stack(populated, dim=0)  # (L, N)
        return predictions

    def on_before_batch_transfer(self, batch: typing.Any, dataloader_idx: int) -> typing.Any:
        """Pre-transfer hook — ensures non-blocking device transfer.

        Lightning calls this before moving the batch to GPU. We ensure
        the batch tensors are contiguous (avoids scattered reads on PCIe).
        """
        return batch

    # ------------------------------------------------------------------
    # TASK 2 — stage helpers
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> str | None:
        """Name of the active loss-weight stage, or ``None`` when no
        ``training.stages`` schedule is configured."""
        return self._current_stage_name

    def stage_display(self, epoch: int) -> str:
        """Human-readable ``"[stage_name epoch_in_stage/stage_length] Epoch X/Y"``
        prefix.  Used by :class:`GOALRichProgressBar` to prepend the stage
        status to the running epoch counter.  Returns the bare epoch
        string when no stage schedule is configured."""
        # ``self.trainer`` raises when the module hasn't been attached
        # to a Trainer yet (e.g. during pure-Python unit tests).  The
        # underlying ``_trainer`` attribute returns ``None`` instead,
        # which is what we want here.
        trainer: typing.Any = getattr(self, "_trainer", None)
        max_epochs: int | None = None
        if trainer is not None:
            max_epochs = getattr(trainer, "max_epochs", None)
        epoch_str: str = (
            f"Epoch {epoch}/{max_epochs - 1}"
            if isinstance(max_epochs, int) and max_epochs > 0
            else f"Epoch {epoch}"
        )

        if not self._stages_cfg:
            return epoch_str

        boundary: int = 0
        for stage in self._stages_cfg:
            n_ep: typing.Any = stage.get("epochs", None)
            name: str = str(stage.get("name", "?"))
            if n_ep is None:
                # Open-ended final stage
                within: int = epoch - boundary + 1
                return f"[{name} {within}/∞] {epoch_str}"
            upper: int = boundary + int(n_ep)
            if epoch < upper:
                within = epoch - boundary + 1
                return f"[{name} {within}/{int(n_ep)}] {epoch_str}"
            boundary = upper
        # Past every finite stage — fall back to the last one's name.
        last_name: str = str(self._stages_cfg[-1].get("name", "?"))
        return f"[{last_name}] {epoch_str}"

    def _get_current_stage(self, epoch: int) -> typing.Any | None:
        """Resolve the active stage dict for the given epoch.

        Stage ``i`` spans ``[start_i, start_i + epochs_i)`` where
        ``start_0 = 0`` and ``start_{i+1} = start_i + epochs_i``.  A
        stage with ``epochs: null`` captures every remaining epoch
        (used for the final, open-ended phase).  Returns ``None`` when
        no schedule is configured — callers should then leave the
        base loss weights intact.
        """
        if not self._stages_cfg:
            return None
        boundary: int = 0
        for stage in self._stages_cfg:
            n_ep: typing.Any = stage.get("epochs", None)
            if n_ep is None:
                return stage
            upper: int = boundary + int(n_ep)
            if epoch < upper:
                return stage
            boundary = upper
        # Past every finite stage — clamp to the last one defined.
        return self._stages_cfg[-1]

    def _apply_stage_weights(self, stage: typing.Any) -> None:
        """Patch ``WeightedLoss.weight`` in-place per the stage overrides.

        Labels that don't have a matching ``WeightedLoss`` in the
        ``CompositeLoss`` are silently skipped — that lets a stage
        config name ``"newton"`` before the newton loss component is
        added.
        """
        overrides: typing.Any = stage.get("loss_weights", None)
        if overrides is None:
            return
        for label, w in dict(overrides).items():
            wl: typing.Any = self._weighted_loss_by_label.get(str(label))
            if wl is not None:
                wl.weight = float(w)

    def _maybe_advance_stage(self, epoch: int) -> None:
        """Resolve the active stage, apply its weights, and log
        transitions.  Cheap to call every step."""
        stage: typing.Any = self._get_current_stage(epoch)
        if stage is None:
            return
        self._apply_stage_weights(stage)
        stage_name: str = str(stage.get("name", "?"))
        if stage_name != self._current_stage_name:
            self._current_stage_name = stage_name
            active: dict[str, float] = {
                lbl: float(wl.weight) for lbl, wl in self._weighted_loss_by_label.items()
            }
            # ``rank_zero_only`` keeps DDP runs quiet.
            from lightning.pytorch.utilities.rank_zero import rank_zero_info

            rank_zero_info(f"[stage] epoch={epoch} → {stage_name!r}  " f"active_weights={active}")

    # ------------------------------------------------------------------
    # Logger sections
    # ------------------------------------------------------------------
    #
    # Everything a run logs lands in one of a few W&B / TensorBoard
    # sections, chosen by the text before the first ``/``.  Which section
    # a given name belongs to is decided in one place — the table in
    # :mod:`goal.ml.training.metrics` — and this module only prepends the
    # split:
    #
    #   {train,val,test}/         the run's verdict: the loss ``total``,
    #                             plus any loss component that is neither
    #                             energy nor forces (``stress``, ...).
    #   {train,val,test}_energy/  the energy loss term, ``energy_mae_per_atom``,
    #                             ``energy_rmse_per_atom`` and the per-round
    #                             decomposition ``energy_round_{L}``.
    #   {train,val,test}_forces/  the forces loss term, ``forces_mae``,
    #                             ``forces_rmse``, ``forces_cosine_similarity``,
    #                             ``forces_magnitude_mae`` and the Newton
    #                             violation (``newton_violation{,_x,_y,_z}``).
    #   {train,val,test}_gate/    every gate diagnostic:
    #                             gate_mean_round_{L}   adaptive depth gate
    #                             aux_loss              its sparsity penalty
    #                             gate_reg              L2 on artisan gates
    #                             artisan/round{L}/{pair}
    #                                                   artisan gate values
    #                             artisan_load/{pair}   load balance (val
    #                             artisan_load_variance only — see below)
    #
    # Energy and forces are split apart because they live on different
    # scales and answer different questions — a per-atom energy MAE panel
    # sitting next to a force cosine similarity is two unrelated readings
    # sharing a row.  Gate curves are in their own section because they
    # are numerous — one per round, plus one per element pair per round —
    # and they are for occasional inspection, not for judging a run:
    # keeping them apart is what stops them from burying
    # ``val_forces/forces_mae`` in a wall of panels.

    @staticmethod
    def _section(prefix: str, section: str | None) -> str:
        """``("val/", "forces")`` → ``"val_forces/"``; ``None`` → ``"val/"``."""
        base: str = prefix.rstrip("/")
        return f"{base}_{section}/" if section else f"{base}/"

    @classmethod
    def _key(cls, prefix: str, name: str) -> str:
        """Full logger key for one metric or loss label.

        ``("val/", "forces_mae")`` → ``"val_forces/forces_mae"``,
        ``("train/", "total")`` → ``"train/total"``.
        """
        return f"{cls._section(prefix, metric_section(name))}{name}"

    @classmethod
    def _gate_prefix(cls, prefix: str) -> str:
        """``"train/"`` → ``"train_gate/"``, ``"val/"`` → ``"val_gate/"``."""
        return cls._section(prefix, "gate")

    def _log_artisan_gates(self, batch: AtomicGraph, prefix: str) -> None:
        """Log every artisan gate as ``{split}_gate/artisan/round{L}/{pair}``.

        These are the learnable per-element-pair gates of the artisan bank
        — ``nn.Parameter`` scalars, so reading them costs nothing and they
        can be logged on both train and validation.  A gate drifting to
        near-zero means that element pair has been switched off, which is
        the artisan equivalent of expert collapse.

        Duck-typed via ``backbone.gates()``; a silent no-op for backbones
        without an artisan bank.  These are the most numerous curves a run
        produces — ``K*(K+1)/2 x num_rounds`` of them — which is exactly
        why they are filed under the gate section instead of next to the
        energy and force metrics.
        """
        gates_fn: typing.Any = getattr(self.backbone, "gates", None)
        if not callable(gates_fn):
            return
        gate_values: dict[str, torch.Tensor] = gates_fn()
        if not gate_values:
            return
        self.log_dict(
            {self._key(prefix, f"artisan/{k}"): v for k, v in gate_values.items()},
            batch_size=batch.num_graphs,
            sync_dist=True,
            on_step=prefix.startswith("train"),
            on_epoch=True,
            add_dataloader_idx=False,
        )

    # ------------------------------------------------------------------
    # Step hooks
    # ------------------------------------------------------------------

    def _log_losses(
        self,
        losses: dict[str, torch.Tensor],
        batch: AtomicGraph,
        prefix: str,
        on_step: bool = True,
        prog_bar: bool = True,
    ) -> None:
        """Log the loss breakdown, each term in its own section.

        ``total`` (and any component that is neither energy nor forces)
        keeps the plain ``{prefix}`` section; the ``energy`` / ``forces``
        terms follow their metrics into ``{prefix}_energy/`` and
        ``{prefix}_forces/``; ``aux_loss`` / ``gate_reg`` move to
        ``{prefix}_gate/`` and never reach the progress bar.
        """
        grouped: dict[str | None, dict[str, torch.Tensor]] = {}
        for label, value in losses.items():
            grouped.setdefault(metric_section(label), {})[label] = value.detach()

        for section, values in grouped.items():
            self.log_dict(
                {f"{self._section(prefix, section)}{k}": v for k, v in values.items()},
                batch_size=batch.num_graphs,
                sync_dist=True,
                prog_bar=prog_bar and section != "gate",
                on_step=on_step,
                on_epoch=True,
                add_dataloader_idx=False,
            )

    def _log_step_metrics(
        self,
        predictions: dict[str, torch.Tensor],
        batch: AtomicGraph,
        prefix: str,
        prog_bar: bool = True,
    ) -> None:
        """Compute the standard MLIP diagnostic metrics and log them.

        Energy metrics land in ``{prefix}_energy/``, force metrics (the
        Newton violation included) in ``{prefix}_forces/``.  Headline
        metrics (per atom energy MAE, force cosine similarity, Newton
        violation) are routed to the Lightning progress bar; the rest go
        to loggers only.  ``prog_bar=False`` keeps a secondary evaluation
        set (the OOD one) off the bar entirely.
        """
        is_train = prefix.startswith("train")
        metrics: dict[str, torch.Tensor] = mlip_metrics(predictions, batch)
        for name, value in metrics.items():
            self.log(
                self._key(prefix, name),
                value,
                batch_size=batch.num_graphs,
                sync_dist=True,
                prog_bar=prog_bar and name in PROG_BAR_METRICS,
                on_step=is_train,
                on_epoch=True,
                add_dataloader_idx=False,
            )

    def _log_round_energies(
        self,
        predictions: dict[str, torch.Tensor],
        batch: AtomicGraph,
        prefix: str,
    ) -> None:
        """Log per-round ARACE energy contributions as
        ``{prefix}energy_round_{L}``.

        Duck-typed no-op for non-ARACE backbones: the per-round totals
        come either from the monolithic model's ``"layer_energies"``
        prediction key or from the modular backbone's
        ``last_round_energies`` attribute (a detached ``(L, B)``
        snapshot of the most recent forward).  Monitors whether every
        round keeps contributing — a round collapsing to zero is a
        warning sign.
        """
        layer_e: torch.Tensor | None = predictions.get("layer_energies", None)
        if layer_e is None:
            layer_e = getattr(self.backbone, "last_round_energies", None)
        if layer_e is None or layer_e.dim() != 2:
            return
        for round_idx in range(layer_e.shape[0]):
            self.log(
                self._key(prefix, f"energy_round_{round_idx}"),
                layer_e[round_idx].detach().mean(),
                batch_size=batch.num_graphs,
                sync_dist=True,
                on_step=prefix.startswith("train"),
                on_epoch=True,
                add_dataloader_idx=False,
            )

    def _log_gate_scores(
        self,
        predictions: dict[str, torch.Tensor],
        batch: AtomicGraph,
        prefix: str,
    ) -> None:
        """Log per-round adaptive-gate means to the **gate** section.

        Emits ``{train,val}_gate/gate_mean_round_{L}`` — deliberately not
        ``train/…``, so the gate curves stay out of the section holding
        the energy/force metrics (see "Logger sections" above).

        Duck-typed no-op unless the ARACE adaptive depth gate is enabled:
        the ``(L, N)`` gate tensor comes either from the monolithic
        model's ``"gate_scores"`` prediction key or from the modular
        backbone's ``last_gate_scores`` side channel (injected into
        ``predictions`` by :meth:`forward`).

        Reading the curve: ~1.0 means the gate is always open (not
        selective — raise ``aux_loss_weight``), ~0.5 means it is
        discriminating between atoms, and a collapse towards 0 means
        rounds are being skipped wholesale (lower ``aux_loss_weight``).
        """
        gates: torch.Tensor | None = predictions.get("gate_scores", None)
        if gates is None or gates.dim() != 2:
            return
        for round_idx in range(gates.shape[0]):
            self.log(
                self._key(prefix, f"gate_mean_round_{round_idx}"),
                gates[round_idx].detach().mean(),
                batch_size=batch.num_graphs,
                sync_dist=True,
                on_step=prefix.startswith("train"),
                on_epoch=True,
                add_dataloader_idx=False,
            )

    @staticmethod
    def _with_aux_loss(
        losses: dict[str, torch.Tensor],
        predictions: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Fold a model-supplied ``aux_loss`` into the loss breakdown.

        Some models emit their own regularisation term alongside the
        predictions — currently the ARACE adaptive depth gate's sparsity
        penalty ``λ · mean(g)``, already scaled by its configured weight.
        It is added to ``total`` and surfaced under the ``aux_loss`` label
        so it shows up in the logged breakdown like any other component.

        No config wiring is needed (and none should be: the weight lives
        with the module that defines the term).  Models that emit no
        ``aux_loss`` are untouched — the returned dict is the input dict.
        """
        aux: torch.Tensor | None = predictions.get("aux_loss", None)
        if aux is None or "aux_loss" in losses:
            return losses
        return {**losses, "aux_loss": aux, "total": losses["total"] + aux}

    def training_step(self, batch: AtomicGraph, batch_idx: int) -> torch.Tensor:
        """Single training step -- forward, loss, logging."""
        self._maybe_advance_stage(self.current_epoch)

        predictions: dict[str, torch.Tensor] = self(batch)
        losses: dict[str, torch.Tensor] = self._with_aux_loss(
            self.loss(predictions, batch), predictions
        )

        # Loss components — physics terms on the progress bar under
        # ``train/``, gate terms quietly under ``train_gate/``.
        self._log_losses(losses, batch, prefix="train/")
        # Standard MLIP diagnostic metrics (energy_mae_per_atom,
        # forces_{mae,rmse,cosine_similarity,magnitude_mae},
        # newton_violation).
        self._log_step_metrics(predictions, batch, prefix="train/")
        # ARACE per-round energy decomposition (no-op for other backbones).
        self._log_round_energies(predictions, batch, prefix="train/")
        # ARACE adaptive-gate means per round (no-op when the gate is off).
        self._log_gate_scores(predictions, batch, prefix="train/")
        # Per-element-pair artisan gates (no-op without an artisan bank).
        self._log_artisan_gates(batch, prefix="train/")
        return losses["total"]

    def on_train_batch_end(
        self,
        outputs: typing.Any,
        batch: AtomicGraph,
        batch_idx: int,
    ) -> None:
        """Update EMA after each optimiser step."""
        if hasattr(self, "ema"):
            self.ema.update()

    #: Section name per validation dataloader, used when the attached
    #: datamodule publishes no ``val_split_names``.  Index 0 is the
    #: primary (in-distribution) set, index 1 the secondary one — which
    #: is the OOD/test split for every datamodule in this project.
    VAL_SECTION_FALLBACK: typing.ClassVar[tuple[str, ...]] = ("val", "test")

    def _val_prefix(self, dataloader_idx: int) -> str:
        """Logger prefix for validation dataloader *dataloader_idx*.

        With a single validation set this is always ``"val/"``.  When the
        datamodule hands Lightning several validation dataloaders — the
        ID set plus an OOD set monitored during training — each one needs
        its own section so the two never average together, and the names
        come from the datamodule's ``val_split_names`` (``["val",
        "test"]``), falling back to :attr:`VAL_SECTION_FALLBACK`.
        """
        names: typing.Sequence[str] = self.VAL_SECTION_FALLBACK
        trainer: typing.Any = getattr(self, "_trainer", None)
        if trainer is not None:
            published: typing.Any = getattr(
                getattr(trainer, "datamodule", None), "val_split_names", None
            )
            if published:
                names = list(published)
        if dataloader_idx < len(names):
            return f"{names[dataloader_idx]}/"
        # More dataloaders than names: never silently merge sections.
        return f"val{dataloader_idx}/"

    def validation_step(
        self,
        batch: AtomicGraph,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Single validation step -- uses EMA weights if available.

        Runs for every validation dataloader the datamodule provides.
        ``dataloader_idx == 0`` is the primary (in-distribution) set and
        logs under ``val…/``; a second dataloader is the OOD set being
        monitored during training and logs under ``test…/``, so the two
        appear as separate curves in the same run and neither is mixed
        into the other.  The secondary set stays off the progress bar and
        skips the diagnostics that are either duplicates (the artisan
        gates are parameters, identical for both sets) or expensive (the
        artisan load balance runs an extra forward per artisan).

        Wraps the model forward in :func:`torch.enable_grad` because
        Lightning's default validation loop runs under
        ``torch.no_grad``, which would prevent autograd-derived force
        heads (``DualForcesHead`` in ``mode='autograd'`` / ``'hybrid'``)
        from computing ``-∂E/∂r``.  The :func:`torch.no_grad` outer
        context still applies to anything *outside* this block (loss
        backward, etc.), so the cost is just the per-forward grad
        tape.  The loss + metrics dicts are then detached when
        Lightning logs them, so no second-order graph survives the
        step.
        """
        # Mirror the same stage on validation so val metrics use the
        # same per-component weighting as training (i.e. a 0-weight
        # forces stage logs val_forces/forces but with that 0 baked in).
        # The OOD set gets the same weights, which is what makes its loss
        # directly comparable to the ID one epoch by epoch.  (A standalone
        # ``trainer.test()`` run deliberately differs — see
        # :meth:`test_step` — so only compare test/ curves within a phase.)
        self._maybe_advance_stage(self.current_epoch)

        prefix: str = self._val_prefix(dataloader_idx)
        primary: bool = dataloader_idx == 0

        with torch.enable_grad():
            if hasattr(self, "ema"):
                with self.ema.average_parameters():
                    predictions = self(batch)
            else:
                predictions = self(batch)

            losses = self._with_aux_loss(self.loss(predictions, batch), predictions)

        self._log_losses(losses, batch, prefix=prefix, prog_bar=primary)
        self._log_step_metrics(predictions, batch, prefix=prefix, prog_bar=primary)
        # ARACE per-round energy decomposition (no-op for other backbones).
        self._log_round_energies(predictions, batch, prefix=prefix)
        # ARACE adaptive-gate means per round (no-op when the gate is off).
        self._log_gate_scores(predictions, batch, prefix=prefix)

        if not primary:
            # Everything below is either identical for both evaluation
            # sets (the artisan gates are parameters) or costs a full
            # extra forward — no reason to pay for it twice per epoch.
            return

        # Per-element-pair artisan gates → val_gate/artisan/… (also logged
        # every train step; see :meth:`_log_artisan_gates`).
        self._log_artisan_gates(batch, prefix=prefix)

        # Artisan load-balance diagnostic (legacy ACE-first SIMURGH only —
        # the ARACE backbones do not expose ``compute_artisan_loads``, so
        # this is a no-op for them).  Stays validation-only on purpose: it
        # runs a *full extra forward of every artisan* over the batch, far
        # too expensive per training step.  Filed under the gate section as
        # ``val_gate/artisan_load/…`` alongside the gate values it explains.
        if hasattr(self.backbone, "compute_artisan_loads"):
            loads: dict[str, torch.Tensor] = self.backbone.compute_artisan_loads(batch)
            load_var: torch.Tensor | None = loads.pop("load_variance", None)
            self.log_dict(
                {self._key(prefix, f"artisan_load/{k}"): v for k, v in loads.items()},
                batch_size=batch.num_graphs,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )
            if load_var is not None:
                self.log(
                    self._key(prefix, "artisan_load_variance"),
                    load_var,
                    batch_size=batch.num_graphs,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    add_dataloader_idx=False,
                )

    def test_step(self, batch: AtomicGraph, batch_idx: int) -> None:
        """Single test step -- uses EMA weights if available.

        Same :func:`torch.enable_grad` wrapper as
        :meth:`validation_step` (see its docstring for the rationale):
        autograd-based force heads (``EnergyForcesHead``, ``DualForcesHead``
        in autograd/hybrid mode) need the grad tape even during evaluation.

        Stage weights are intentionally skipped here — the test loop always
        runs with the base loss weights so all components are reported at
        full weight regardless of which training stage the checkpoint is from.

        Note this is *not* the path that produces the ``test…/`` curves
        during training: those come from :meth:`validation_step` with a
        second validation dataloader (``data.monitor_test_during_training``),
        which mirrors the active stage weights.  The metrics (MAE, RMSE,
        cosine similarity, Newton violation) are weight-independent and
        therefore comparable across both paths; the loss values are not.
        """

        with torch.enable_grad():
            if hasattr(self, "ema"):
                with self.ema.average_parameters():
                    predictions = self(batch)
            else:
                predictions = self(batch)

            losses = self._with_aux_loss(self.loss(predictions, batch), predictions)

        # Same split as train/val: gate terms under ``test_gate/``.
        # ``prog_bar=False`` keeps the test loop as quiet as it was.
        self._log_losses(losses, batch, prefix="test/", prog_bar=False)
        self._log_step_metrics(predictions, batch, prefix="test/")
        # ARACE adaptive-gate means per round (no-op when the gate is off).
        self._log_gate_scores(predictions, batch, prefix="test/")

    def _split_param_groups(
        self,
        lr: float,
        weight_decay: float,
    ) -> list[dict[str, typing.Any]]:
        """Split parameters into equivariant (no WD) and standard (WD) groups.

        Equivariant group: parameters belonging to any module whose
        fully-qualified name contains ``"tp"`` or ``"equivariant"``, or
        whose type is ``EquivariantLinear`` or ``FullyConnectedTensorProduct``
        (from e3nn).  All other parameters go into the standard group and
        receive ``weight_decay``.

        Using ``id()`` deduplication ensures shared parameters (if any)
        appear only once.
        """
        from e3nn.o3 import FullyConnectedTensorProduct

        from goal.ml.nn.primitives.linear import EquivariantLinear

        equivariant_ids: set[int] = set()
        for name, module in self.named_modules():
            is_equiv_type = isinstance(module, (EquivariantLinear, FullyConnectedTensorProduct))
            is_equiv_name = "tp" in name or "equivariant" in name.lower()
            if is_equiv_type or is_equiv_name:
                for param in module.parameters(recurse=False):
                    equivariant_ids.add(id(param))

        equivariant_params: list[torch.nn.Parameter] = []
        standard_params: list[torch.nn.Parameter] = []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            if id(param) in equivariant_ids:
                equivariant_params.append(param)
            else:
                standard_params.append(param)

        return [
            {"params": equivariant_params, "weight_decay": 0.0, "lr": lr},
            {"params": standard_params, "weight_decay": weight_decay, "lr": lr},
        ]

    def configure_optimizers(self) -> dict[str, typing.Any]:
        """Set up optimiser and learning rate scheduler.

        Uses AdamW (PyTorch 2.x) as default. Supports cosine annealing
        and reduce-on-plateau schedulers.

        Parameters are split into two groups:
        - Equivariant parameters (TP layers, EquivariantLinear): no weight decay.
        - Standard parameters (MLP weights, gates, embeddings): weight decay applies.
        """
        opt_cfg: typing.Any = self.config.training.optimizer
        scheduler_type: str = opt_cfg.get("scheduler_type", "reduce_on_plateau")
        weight_decay: float = float(opt_cfg.get("weight_decay", 0.0))

        param_groups = self._split_param_groups(
            lr=float(opt_cfg.lr),
            weight_decay=weight_decay,
        )

        optimizer: torch.optim.AdamW = torch.optim.AdamW(
            param_groups,
            lr=opt_cfg.lr,
            amsgrad=opt_cfg.get("amsgrad", True),
        )

        # max_epochs for the scheduler: prefer trainer.max_epochs (what Lightning
        # actually uses) over training.max_epochs (legacy, scheduler-only field).
        # trainer is not attached yet when configure_optimizers runs, so read
        # from cfg.trainer first, fall back to cfg.training for old configs.
        trainer_max: int = int(
            self.config.get("trainer", {}).get("max_epochs", None)
            or self.config.training.get("max_epochs", 500)
        )

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=trainer_max,
                eta_min=opt_cfg.get("min_lr", 1e-7),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        elif scheduler_type == "cosine_warmup":
            from torch.optim.lr_scheduler import (
                CosineAnnealingLR,
                LinearLR,
                SequentialLR,
            )

            warmup_epochs = opt_cfg.get("warmup_epochs", 10)
            warmup = LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=trainer_max - warmup_epochs,
                eta_min=opt_cfg.get("min_lr", 1e-7),
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        else:
            # Default: ReduceLROnPlateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=opt_cfg.scheduler.patience,
                factor=opt_cfg.scheduler.factor,
                min_lr=opt_cfg.get("min_lr", 1e-7),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/total",
                    "interval": "epoch",
                },
            }

    # ------------------------------------------------------------------
    # EMA persistence
    # ------------------------------------------------------------------
    #
    # ``EMAWrapper`` is a plain attribute, not a submodule, so its shadow
    # tensors are absent from ``state_dict()`` and used to be lost on every
    # resume — the run came back on the raw weights while validation and
    # every packed archive report the EMA ones.  Over a crash/resume cycle
    # that is a few hundred steps of re-convergence; over an extension of a
    # finished run it throws away the averaged weights the run was judged on.

    def on_save_checkpoint(self, checkpoint: dict[str, typing.Any]) -> None:
        """Store the EMA shadow weights alongside the raw ones."""
        if hasattr(self, "ema"):
            checkpoint["goal_ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, typing.Any]) -> None:
        """Restore EMA shadow weights written by :meth:`on_save_checkpoint`.

        Silently tolerates their absence: checkpoints written before EMA
        was persisted, and runs that enable EMA on a resume, both leave
        the shadow to re-initialise from the restored weights.

        Lightning calls this hook *before* it loads the state dict, so the
        re-initialisation cannot happen here — the parameters still hold
        their random initial values.  It is deferred to
        :meth:`_reseed_ema_if_pending`, which the fit / validate / test /
        predict start hooks call once the weights are in place.
        """
        GOALModule._keep_fresh_buffers_on_load(self, checkpoint)
        state: typing.Any = checkpoint.get("goal_ema")
        if not hasattr(self, "ema"):
            return
        if state is None:
            self._ema_reseed_pending = True
            return
        shadow: typing.Any = state.get("shadow")
        expected: int = len(list(self.parameters()))
        if not isinstance(shadow, list) or len(shadow) != expected:
            found = len(shadow) if isinstance(shadow, list) else type(shadow).__name__
            warnings.warn(
                f"Checkpoint EMA state has {found} tensors but the model has "
                f"{expected} parameters — discarding it.  The EMA will "
                f"re-initialise from the restored weights.",
                UserWarning,
                stacklevel=2,
            )
            self._ema_reseed_pending = True
            return
        self.ema.load_state_dict(state)

    _BACKBONE_BUFFER_KEY: typing.ClassVar[re.Pattern[str]] = re.compile(
        r"^backbone(?:\._orig_mod)?\.(\w+)$"
    )

    def _keep_fresh_buffers_on_load(self, checkpoint: dict[str, typing.Any]) -> None:
        """Replace checkpoint entries named in :attr:`keep_fresh_buffers`.

        Lightning loads ``checkpoint["state_dict"]`` right after this hook,
        overwriting every buffer.  For a statistic the user asked to
        recompute on resume (e.g. ``scale`` after switching datasets) the
        checkpoint's copy is swapped for the module's current, freshly
        computed value, so the new value is the one that trains.
        """
        keep: set[str] = set(getattr(self, "keep_fresh_buffers", None) or ())
        if not keep:
            return
        saved: typing.Any = checkpoint.get("state_dict")
        if not isinstance(saved, dict):
            return
        own: dict[str, torch.Tensor] = self.state_dict()
        for key in list(saved):
            match = GOALModule._BACKBONE_BUFFER_KEY.match(key)
            if match and match.group(1) in keep and key in own:
                saved[key] = own[key].detach().clone()

    def _reseed_ema_if_pending(self) -> None:
        """Re-initialise the EMA shadow from the current (restored) weights.

        Without this a checkpoint with no usable ``goal_ema`` entry keeps
        the shadow cloned from the *random* initial parameters in
        ``__init__``, and every ``ema.average_parameters()`` evaluation
        (validation, test) reports those instead of the trained model.
        """
        if not getattr(self, "_ema_reseed_pending", False) or not hasattr(self, "ema"):
            return
        self.ema.load_state_dict({"shadow": [p.detach().clone() for p in self.parameters()]})
        self._ema_reseed_pending = False

    def on_fit_start(self) -> None:
        """Re-seed a pending EMA shadow before fitting starts.

        Calls :meth:`_reseed_ema_if_pending`, so a checkpoint restored
        without usable EMA state trains against an EMA cloned from the
        restored weights rather than the random initial ones.
        """
        self._reseed_ema_if_pending()

    def on_validation_start(self) -> None:
        """Re-seed a pending EMA shadow before validation starts.

        Calls :meth:`_reseed_ema_if_pending`, so ``ema.average_parameters()``
        evaluates the restored weights, not the random initial ones.
        """
        self._reseed_ema_if_pending()

    def on_test_start(self) -> None:
        """Re-seed a pending EMA shadow before testing starts.

        Calls :meth:`_reseed_ema_if_pending`, so ``ema.average_parameters()``
        evaluates the restored weights, not the random initial ones.
        """
        self._reseed_ema_if_pending()

    def on_predict_start(self) -> None:
        """Re-seed a pending EMA shadow before prediction starts.

        Calls :meth:`_reseed_ema_if_pending`, so the EMA shadow matches the
        restored weights.
        """
        self._reseed_ema_if_pending()

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Gradient clipping before optimiser step."""
        clip_val: float = self.config.training.get("gradient_clip", 0)
        if clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm="norm",
            )
